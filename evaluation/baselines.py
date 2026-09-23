"""Baseline D: a contemporary dense reference model (Phase 12).

A small dense transformer with the architecture choices the MSA paper plans to
target -- RMSNorm, rotary position embeddings, SwiGLU -- and no adaptivity.

It is randomly initialised. Quality numbers from it are meaningless and the
harness reports them as N/A; it exists to give a latency and parameter-count
reference for the architecture family, not a quality comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


def rotary(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        shape = (batch, seq, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = context.transpose(1, 2).reshape(batch, seq, dim)

        return self.out(context)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DenseReference(nn.Module):
    """RMSNorm + RoPE + SwiGLU decoder, no adaptivity. Untrained."""

    def __init__(self, vocab_size=50257, dim=768, depth=12, heads=12, hidden=2048,
                 max_seq=1024, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.config = {
            "vocab_size": vocab_size, "dim": dim, "depth": depth,
            "heads": heads, "ffn_hidden": hidden, "norm": "RMSNorm",
            "position": "RoPE", "activation": "SwiGLU", "trained": False,
        }

        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, hidden) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        head_dim = dim // heads
        inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq).float()
        angles = torch.outer(positions, inverse)
        emb = torch.cat([angles, angles], dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, input_ids):
        seq = input_ids.shape[1]
        cos = self.cos_cached[:, :, :seq, :]
        sin = self.sin_cached[:, :, :seq, :]

        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)

        return self.head(self.norm(x))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def demo():
    model = DenseReference(vocab_size=1000, dim=128, depth=2, heads=4, hidden=256).eval()
    ids = torch.randint(0, 1000, (1, 16))

    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (1, 16, 1000)
    assert torch.isfinite(logits).all()
    assert model.parameter_count() > 0
    assert model.config["trained"] is False

    print(f"dense reference demo OK: {model.parameter_count():,} params, "
          f"logits {tuple(logits.shape)}")


if __name__ == "__main__":
    demo()
