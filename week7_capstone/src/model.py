"""Small encoder-only Transformer forecaster. The positional encoding is the only branch.

  learned     : nn.Embedding(context, d_model) added to the input embedding once
  sinusoidal  : fixed sin/cos table added to the input embedding once
  rope        : rotary rotation applied to queries and keys (not values) in every attention layer

Everything else (input projection, blocks, final norm, head) is the same code and, for a given
seed, bitwise-identical initial weights across arms. Initialization is ordered so arm-specific
parameters cannot shift the shared weights:
  1. seed -> build the shared backbone (consumes RNG)
  2. a *separate* derived seed -> build the positional module (the only arm-specific RNG use)
RoPE and sinusoidal tables are buffers computed without randomness.
"""
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

POS_SEED_OFFSET = 10_000
TRAIN_SEED_OFFSET = 20_000
POS_TYPES = ("learned", "sinusoidal", "rope")


def sinusoidal_table(length, d):
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(length, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Rotary(nn.Module):
    """Rotates feature pairs of q or k by an angle proportional to the token position."""

    def __init__(self, head_dim, max_len, base):
        super().__init__()
        assert head_dim % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        ang = torch.arange(max_len, dtype=torch.float32)[:, None] * inv[None, :]
        self.register_buffer("cos", torch.cat([ang.cos(), ang.cos()], -1), persistent=False)
        self.register_buffer("sin", torch.cat([ang.sin(), ang.sin()], -1), persistent=False)

    def forward(self, x):                       # x: (B, H, T, head_dim)
        T, half = x.shape[-2], x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * self.cos[:T] + rot * self.sin[:T]


class SelfAttention(nn.Module):
    def __init__(self, d, heads, rotary=None):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.rotary = rotary

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.rotary is not None:             # queries and keys only; values are never rotated
            q, k = self.rotary(q), self.rotary(k)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, d_ff, dropout, rotary):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads, rotary)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff(self.ln2(x)))


class Forecaster(nn.Module):
    def __init__(self, m, context, horizon, seed):
        super().__init__()
        assert m["pos_encoding"] in POS_TYPES, m["pos_encoding"]
        self.pos_type, self.context = m["pos_encoding"], context
        d, heads = m["d_model"], m["n_heads"]

        # 1. shared backbone, built from `seed`
        torch.manual_seed(seed)
        self.in_proj = nn.Linear(1, d)
        self.blocks = nn.ModuleList([
            Block(d, heads, m["d_ff"], m["dropout"],
                  Rotary(d // heads, context, m["rope_base"]) if self.pos_type == "rope" else None)
            for _ in range(m["n_layers"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, horizon)

        # 2. arm-specific positional module, built from a separate seed
        torch.manual_seed(seed + POS_SEED_OFFSET)
        if self.pos_type == "learned":
            self.pos_emb = nn.Embedding(context, d)
        elif self.pos_type == "sinusoidal":
            self.register_buffer("pe", sinusoidal_table(context, d), persistent=False)

    def forward(self, x):                       # x: (B, context), window-normalized demand
        h = self.in_proj(x.unsqueeze(-1))
        if self.pos_type == "learned":
            h = h + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        elif self.pos_type == "sinusoidal":
            h = h + self.pe[: x.shape[1]]
        for b in self.blocks:
            h = b(h)
        # mean-pool over tokens, then a linear head. Predicts the ratio to the context mean (centered on 1).
        return 1.0 + self.head(self.ln_f(h).mean(1))


def build_model(model_cfg, context, horizon, seed):
    return Forecaster(model_cfg, context, horizon, seed)


def param_counts(model):
    pos = sum(p.numel() for n, p in model.named_parameters() if n.startswith("pos_emb"))
    return {"total": sum(p.numel() for p in model.parameters()), "positional": pos}


def weights_sha256(model, shared_only=False):
    """Hash of the state dict. shared_only drops arm-specific parameters, so the value can be compared across arms."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        if shared_only and k.startswith("pos_emb"):
            continue
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
