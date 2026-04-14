import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class ModelArgs:
    # Token / Embedding
    n_tokens: int = 30
    d_model: int = 170
    vocab_size: int = 512          # action token vocabulary

    # MLA
    latent_dim: int = 32
    n_heads: int = 10
    head_dim: int = 8

    # Xception
    xception_expand: int = 1

    # Backbone
    n_layers: int = 4
    dropout: float = 0.1

    # MTP heads value weights (daily, weekly, monthly)
    head_weights: Tuple[float, ...] = (0.6, 0.3, 0.1)

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 10000
    batch_size: int = 64
    grad_clip: float = 1.0
    log_interval: int = 50
    eval_interval: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class XceptionBlock1D(nn.Module):
    """Depthwise separable conv × 2 with residual."""
    def __init__(self, channels: int, expand: int = 1):
        super().__init__()
        mid = channels * expand
        self.conv = nn.Sequential(
            # Round 1: depthwise → pointwise
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            # Round 2: depthwise → pointwise
            nn.Conv1d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + x)


class MLA(nn.Module):
    """Multi-head Latent Attention: shared low-rank bottleneck → Q K V."""
    def __init__(self, d_model: int, latent_dim: int, n_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, latent_dim, bias=False)
        self.up_qkv = nn.Linear(latent_dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    """Xception (local) → MLA (global)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.xception = XceptionBlock1D(args.d_model, args.xception_expand)
        self.attn = MLA(args.d_model, args.latent_dim, args.n_heads, args.head_dim, args.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Xception: (B, T, D) → (B, D, T) → conv → (B, D, T) → (B, T, D)
        x = x.permute(0, 2, 1)
        x = self.xception(x)
        x = x.permute(0, 2, 1)
        # MLA: global token mixing
        x = self.attn(x, mask)
        return x
