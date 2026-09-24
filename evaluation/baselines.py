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


PRETRAINED_DENSE_ID = "HuggingFaceTB/SmolLM-135M"


class PretrainedDenseBaseline:
    """Baseline D with real weights: SmolLM-135M (Llama architecture).

    RMSNorm + RoPE + SwiGLU, no adaptivity, a similar parameter budget to GPT-2
    small. Its tokenizer differs from GPT-2's, so per-token loss/perplexity are
    not directly comparable; exact-match accuracy and latency are.
    """

    def __init__(self, model_id=PRETRAINED_DENSE_ID):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # The checkpoint defaults to bfloat16; GPT-2 runs in float32, so match it.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32).eval()

        config = self.model.config
        self.layers = config.num_hidden_layers
        self.config = {
            "model": model_id,
            "architecture": config.model_type,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "heads": config.num_attention_heads,
            "kv_heads": getattr(config, "num_key_value_heads", None),
            "ffn_hidden": config.intermediate_size,
            "norm": "RMSNorm",
            "position": "RoPE",
            "activation": f"SwiGLU ({config.hidden_act})",
            "dtype": str(self.model.dtype).replace("torch.", ""),
            "trained": True,
        }

    def _loss(self, input_ids, labels):
        import time

        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, use_cache=False).logits
        latency = time.perf_counter() - started

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        ).item()
        return loss, latency

    def run(self, record, generate=False, max_new_tokens=24):
        """One record in the same schema the GPT-2 harness emits."""
        import math

        from data.load_data import exact_match

        question = record["question"]
        answer = record.get("answer")

        if answer is None:
            ids = self.tokenizer(question, return_tensors="pt")["input_ids"]
            loss, latency = self._loss(ids, ids)
            entry = {}
        else:
            prompt = f"Question: {question}\nAnswer:"
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            ids = self.tokenizer(f"{prompt} {answer}", return_tensors="pt")["input_ids"]
            labels = ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100

            loss, latency = self._loss(ids, ids)
            entry = {"answer_loss": self._loss(ids, labels)[0]}

            if generate:
                with torch.no_grad():
                    output = self.model.generate(
                        prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                text = self.tokenizer.decode(output[0, prompt_ids.shape[1]:])
                text = text.split("\n")[0]
                entry["generated"] = text
                entry["correct"] = exact_match(text, answer)

        return {
            "question": question,
            "configuration": "dense",
            "loss": loss,
            "perplexity": math.exp(min(loss, 20)),
            "latency_seconds": latency,
            "depth": self.layers,
            "executed_depth": self.layers,
            "attention_mode": "full",
            "ffn_mode": "full",
            "layer_reduction": 0.0,
            "relative_flops": float("nan"),
            "sequence_length": ids.shape[1],
            "confidence": float("nan"),
            "fallback_applied": False,
            "rolled_back": False,
            **entry,
        }


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
