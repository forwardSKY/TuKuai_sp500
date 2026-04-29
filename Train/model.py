import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# -------------------- 工具函数 --------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转位置编码辅助变换"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """对 query 和 key 应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# -------------------- 时间戳 RoPE --------------------
class TimeStampRoPE(nn.Module):
    """根据真实自然日时间戳生成旋转嵌入，支持解耦应用"""
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, timestamps: torch.Tensor, offset: int = 0):
        """
        timestamps: (batch_size, seq_len) 浮点型自然日
        返回 cos, sin 形状均为 (batch_size, seq_len, head_dim)
        """
        device = timestamps.device
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        t = timestamps.unsqueeze(-1) * freqs  # (B, L, head_dim/2)
        emb = torch.cat((t, t), dim=-1)       # (B, L, head_dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

# -------------------- MLA 注意力（含解耦 RoPE） --------------------
class MLAAttention(nn.Module):
    """多头潜在注意力（低秩 KV 压缩）+ 解耦 RoPE"""
    def __init__(self, d_model: int, n_heads: int, d_c: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_c = d_c  # KV 压缩维度

        # 内容分支：低秩压缩 KV
        self.w_kv_c = nn.Linear(d_model, d_c, bias=False)      # 压缩
        self.w_kc_up = nn.Linear(d_c, d_model, bias=False)     # Key 上投影
        self.w_vc_up = nn.Linear(d_c, d_model, bias=False)     # Value 上投影

        # 位置分支：解耦 RoPE 使用的高维 Q/K 投影（不压缩）
        self.w_qr = nn.Linear(d_model, d_model, bias=False)    # 用于 RoPE 的 query
        self.w_kr = nn.Linear(d_model, d_model, bias=False)    # 用于 RoPE 的 key

        self.w_o = nn.Linear(d_model, d_model, bias=False)     # 输出投影
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key_value: Optional[torch.Tensor] = None,
                timestamps_q: Optional[torch.Tensor] = None, timestamps_kv: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, rope: Optional[TimeStampRoPE] = None):
        if key_value is None:
            key_value = query
        B, Lq, _ = query.shape
        _, Lk, _ = key_value.shape

        # ---- 内容分支：低秩 KV ----
        c_kv = self.w_kv_c(key_value)                        # (B, Lk, d_c)
        k_c = self.w_kc_up(c_kv).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, Lk, head_dim)
        v_c = self.w_vc_up(c_kv).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)

        # ---- 位置分支：高维 Q/K 用于 RoPE ----
        q_r = self.w_qr(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1,2)
        k_r = self.w_kr(key_value).view(B, Lk, self.n_heads, self.head_dim).transpose(1,2)

        # 应用时间戳 RoPE 到位置分支
        if rope is not None and timestamps_q is not None and timestamps_kv is not None:
            cos_q, sin_q = rope(timestamps_q)
            cos_k, sin_k = rope(timestamps_kv)
            cos_q = cos_q.unsqueeze(1); sin_q = sin_q.unsqueeze(1)
            cos_k = cos_k.unsqueeze(1); sin_k = sin_k.unsqueeze(1)
            q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos_q, cos_k)

        # ---- 注意力得分：内容 + 位置（加性融合） ----
        attn_c = torch.matmul(q_r * self.scale, k_c.transpose(-2, -1))
        attn_r = torch.matmul(q_r * self.scale, k_r.transpose(-2, -1))
        attn = attn_c + attn_r

        if attn_mask is not None:
            attn = attn + attn_mask

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 聚合 Value（使用内容分支的 v_c）
        out = torch.matmul(attn_weights, v_c)   # (B, n_heads, Lq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.w_o(out)
        return out

# -------------------- MoE-FFN（无辅助损失，专家偏置动态均衡） --------------------
class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, top_k: int = 2, 
                 dropout: float = 0.1, bias_update_speed: float = 0.001):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.update_speed = bias_update_speed

        # 专家网络（SwiGLU）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff * 2, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False)
            ) for _ in range(n_experts)
        ])

        # 门控网络
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # 专家偏置（可学习的非梯度缓冲，用于动态负载均衡）
        self.register_buffer('expert_bias', torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.view(-1, D)          # (B*L, D)
        gate_logits = self.gate(x_flat) + self.expert_bias.unsqueeze(0)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.n_experts, device=x.device)

        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_out = expert(expert_input)
                weights = topk_weights[topk_indices == i]
                output[mask] += expert_out * weights.unsqueeze(-1)
                expert_counts[i] = mask.sum().float()

        output = output.view(B, L, D)
        return output, expert_counts

    def update_bias(self, expert_counts: torch.Tensor):
        total_tokens = expert_counts.sum()
        ideal_load = total_tokens / self.n_experts
        delta = self.update_speed * (expert_counts - ideal_load).sign()
        self.expert_bias -= delta

# -------------------- Transformer 基础层 --------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_c: int, d_ff: int, n_experts: int, top_k: int,
                 dropout: float = 0.1, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = MLAAttention(d_model, n_heads, d_c, dropout)
        if is_decoder:
            self.cross_attn_norm = nn.RMSNorm(d_model)
            # 交叉注意力使用标准多头注意力（也可以改用 MLA，这里保持简洁）
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.moe = MoEFFN(d_model, d_ff, n_experts, top_k, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: Optional[torch.Tensor] = None,
                timestamps_q: Optional[torch.Tensor] = None, timestamps_kv: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, rope: Optional[TimeStampRoPE] = None):
        # 自注意力
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, timestamps_q=timestamps_q, timestamps_kv=timestamps_kv, attn_mask=attn_mask, rope=rope)
        x = self.dropout(x) + residual

        # 交叉注意力（仅解码器）
        if self.is_decoder and enc_out is not None:
            residual = x
            x = self.cross_attn_norm(x)
            x, _ = self.cross_attn(query=x, key=enc_out, value=enc_out)
            x = self.dropout(x) + residual

        # MoE-FFN
        residual = x
        x = self.ffn_norm(x)
        x, expert_counts = self.moe(x)
        x = self.dropout(x) + residual
        return x, expert_counts

# -------------------- 完整模型 --------------------
class TradeTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 512, n_heads: int = 8, d_c: int = 128, d_ff: int = 1024,
                 n_experts: int = 8, top_k: int = 2, n_enc_layers: int = 6, n_dec_layers: int = 6,
                 max_out_len: int = 5, dropout: float = 0.1, theta: float = 10000.0):
        super().__init__()
        self.max_out_len = max_out_len
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(dropout)

        # 时间戳 RoPE 生成器
        self.rope = TimeStampRoPE(d_model // n_heads, theta)

        # 教师权重投影层（修正：固定投影）
        self.teacher_proj = nn.Linear(500, d_model, bias=False)  # 将 500 维权重映射到 d_model 维

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_c, d_ff, n_experts, top_k, dropout, is_decoder=False)
            for _ in range(n_enc_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_c, d_ff, n_experts, top_k, dropout, is_decoder=True)
            for _ in range(n_dec_layers)
        ])

        # 输出头（500 维度）
        self.output_head = nn.Linear(d_model, 500)

        # BOS token 嵌入
        self.bos_embed = nn.Parameter(torch.randn(1, 1, d_model))

        # 未来时间戳嵌入
        self.time_emb = nn.Linear(1, d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, market_seq: torch.Tensor, enc_timestamps: torch.Tensor,
                dec_timestamps: torch.Tensor, teacher_weights: Optional[torch.Tensor] = None):
        """
        market_seq: (B, L, d_in)  过去 L 天的全市场特征
        enc_timestamps: (B, L)    编码器时间戳（自然日）
        dec_timestamps: (B, K)    解码器时间戳（未来 K 个交易日）
        teacher_weights: (B, K, 500) 教师权重，用于 teacher forcing；若为 None 则自回归采样
        返回: logits (B, K, 500), expert_counts_list
        """
        B, L, _ = market_seq.shape
        K = dec_timestamps.shape[1]

        # ---- Encoder ----
        x = self.input_proj(market_seq)
        x = self.dropout(x)
        expert_counts_all = []
        for layer in self.encoder_layers:
            x, exp_counts = layer(x, enc_out=None,
                                  timestamps_q=enc_timestamps, timestamps_kv=enc_timestamps,
                                  rope=self.rope)
            expert_counts_all.append(exp_counts)
        enc_out = x

        # ---- Decoder (MTP) ----
        dec_input = self.bos_embed.expand(B, 1, -1)
        t_step0 = dec_timestamps[:, 0:1].unsqueeze(-1)      # (B, 1, 1)
        dec_input = dec_input + self.time_emb(t_step0)

        logits_list = []
        for k in range(K):
            dec_out = dec_input
            cur_len = dec_out.shape[1]
            causal_mask = torch.triu(torch.ones(cur_len, cur_len, device=dec_out.device) * float('-inf'), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, cur, cur)

            for layer in self.decoder_layers:
                dec_out, exp_counts = layer(dec_out, enc_out=enc_out,
                                            timestamps_q=dec_timestamps[:, :cur_len],
                                            timestamps_kv=dec_timestamps[:, :cur_len],
                                            attn_mask=causal_mask, rope=self.rope)
                expert_counts_all.append(exp_counts)

            step_logits = self.output_head(dec_out[:, -1, :])   # (B, 500)
            logits_list.append(step_logits)

            # 准备下一步输入
            if k < K - 1:
                if teacher_weights is not None:
                    # Teacher forcing：使用真实权重投影
                    next_token = teacher_weights[:, k, :].unsqueeze(1)  # (B, 1, 500)
                    next_token_emb = self.teacher_proj(next_token)
                else:
                    # 推理：使用当前步 softmax 均值（确定性）投影
                    probs = F.softmax(step_logits, dim=-1).unsqueeze(1)  # (B, 1, 500)
                    next_token_emb = self.teacher_proj(probs)

                t_next = dec_timestamps[:, k+1:k+2].unsqueeze(-1)
                next_input = next_token_emb + self.time_emb(t_next)
                dec_input = torch.cat([dec_input, next_input], dim=1)

        logits = torch.stack(logits_list, dim=1)  # (B, K, 500)
        return logits, expert_counts_all