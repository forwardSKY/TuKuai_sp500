from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Xception and MLA
# ============================================================
class SeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            padding=kernel_size // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
    def forward(self, x): return self.pw(self.dw(x))


class XceptionBlock1D(nn.Module):
    def __init__(self, channels: int, expand: int = 1):
        super().__init__()
        mid = channels * expand
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv1d(channels, mid, 3), nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            SeparableConv1d(mid, mid, 3), nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            SeparableConv1d(mid, channels, 3), nn.BatchNorm1d(channels),
        )
    def forward(self, x): return self.block(x) + x


class MLA(nn.Module):
    def __init__(self, d_model, latent_dim, n_heads, head_dim, dropout):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, latent_dim, bias=False)
        self.up_qkv = nn.Linear(latent_dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        latent = self.down(x)
        qkv = self.up_qkv(latent).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask.bool() if mask is not None else None,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return residual + self.dropout(self.out(out))


class BackboneBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.xception = XceptionBlock1D(args.d_model, args.xception_expand)
        self.attn = MLA(args.d_model, args.latent_dim, args.n_heads,
                        args.head_dim, args.dropout)
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        x = self.xception(x)
        x = x.permute(0, 2, 1)
        x = self.attn(x, mask)
        return x


# ============================================================
# MTP Module
# ============================================================
class MTPModule(nn.Module):
    """At position i, predicts token t_{i+k+1} given:
         h^{k-1}_i  : hidden state from previous module (or main model when k=1)
         emb(t_{i+k}): embedding of the k-th future token
       Pipeline: norm both → concat → linear proj → BackboneBlock.
    """
    def __init__(self, args):
        super().__init__()
        self.norm_h = nn.LayerNorm(args.d_model)
        self.norm_e = nn.LayerNorm(args.d_model)
        self.proj   = nn.Linear(2 * args.d_model, args.d_model, bias=False)
        self.block  = BackboneBlock(args)

    def forward(self, h_prev, emb_next, mask=None):
        fused = torch.cat([self.norm_h(h_prev), self.norm_e(emb_next)], dim=-1)
        return self.block(self.proj(fused), mask)


# ============================================================
# MTP Stack: arbitrary N modules
# ============================================================
def _slice_mask(mask: Optional[torch.Tensor], size: int) -> Optional[torch.Tensor]:
    """Shrink a causal mask to (size, size). Supports (T,T) or (...,T,T)."""
    return None if mask is None else mask[..., :size, :size]


class MTPStack(nn.Module):
    """Sequential stack of N MTP modules. N is arbitrary.
       Sequence length shrinks by 1 per level (last position has no aligned future token).
    """
    def __init__(self, args, n_predict: int):
        super().__init__()
        assert n_predict >= 1
        self.n_predict = n_predict
        self.layers = nn.ModuleList([MTPModule(args) for _ in range(n_predict)])

    def forward(self, h_main, token_embeddings, mask=None) -> List[torch.Tensor]:
        """
        h_main           : (B, T,   D) main model hidden states
        token_embeddings : (B, T,   D) embeddings for tokens t_0..t_{T-1}
        Returns          : list of N tensors, k-th has shape (B, T-k, D)
        """
        outputs, h_prev = [], h_main
        for k, layer in enumerate(self.layers, start=1):
            # Align: drop last pos of h_prev, shift embeddings by k
            h_in   = h_prev[:, :-1, :]                            # (B, T-k, D)
            emb_in = token_embeddings[:, k:k + h_in.size(1), :]   # (B, T-k, D)
            h_k    = layer(h_in, emb_in, _slice_mask(mask, h_in.size(1)))
            outputs.append(h_k)
            h_prev = h_k
        return outputs


# ============================================================
# Full Model: backbone + MTP head + shared embedding/output
# ============================================================
class Model(nn.Module):
    def __init__(self, args, n_mtp: int = 0):
        super().__init__()
        self.args  = args
        self.embed = nn.Embedding(args.vocab_size, args.d_model)
        self.backbone = nn.ModuleList([BackboneBlock(args) for _ in range(args.n_layers)])
        self.norm_out = nn.LayerNorm(args.d_model)
        self.head     = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.head.weight = self.embed.weight                      # weight tying
        self.mtp = MTPStack(args, n_mtp) if n_mtp > 0 else None   # n_mtp=0 → pure NTP

    def forward(self, tokens, mask=None) -> Dict[str, torch.Tensor]:
        emb = self.embed(tokens)
        x = emb
        for block in self.backbone:
            x = block(x, mask)
        h_main = self.norm_out(x)
        main_logits = self.head(h_main)

        out = {"main_logits": main_logits, "mtp_logits": []}
        if self.mtp is not None:
            for h in self.mtp(h_main, emb, mask):
                out["mtp_logits"].append(self.head(self.norm_out(h)))
        return out



# ============================================================
# Loss: standard NTP + averaged MTP auxiliary
# ============================================================
def compute_loss(out: Dict[str, torch.Tensor], targets: torch.Tensor,
                 mtp_weight: float = 0.3) -> Dict[str, torch.Tensor]:
    """targets: (B, T) ground-truth ids aligned with input tokens.
       Main loss: position i predicts targets[i+1].
       MTP-k    : position i predicts targets[i+k+1].
    """
    # ---- Main NTP loss
    ml = out["main_logits"][:, :-1, :]
    mt = targets[:, 1:]
    main_loss = F.cross_entropy(ml.reshape(-1, ml.size(-1)), mt.reshape(-1))

    # ---- MTP losses
    mtp_losses = []
    T = targets.size(1)
    for k, logits in enumerate(out["mtp_logits"], start=1):
        valid = T - k - 1
        if valid <= 0:
            break
        l = logits[:, :valid, :]
        t = targets[:, k + 1:k + 1 + valid]
        mtp_losses.append(F.cross_entropy(l.reshape(-1, l.size(-1)), t.reshape(-1)))

    if mtp_losses:
        mtp_loss = torch.stack(mtp_losses).mean()
        total = main_loss + mtp_weight * mtp_loss
    else:
        mtp_loss = torch.zeros((), device=main_loss.device)
        total = main_loss
    return {"loss": total, "main_loss": main_loss, "mtp_loss": mtp_loss}