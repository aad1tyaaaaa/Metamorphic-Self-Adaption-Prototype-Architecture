import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.configurations import (
    ATTENTION_HEAD_FRACTION,
    FFN_CHUNK_FRACTION,
    FFN_CHUNKS,
)


class AdaptiveGPT2(nn.Module):
    """GPT-2 with per-inference control over depth, attention heads and FFN width.

    Three adaptation mechanisms, none of which modify the pretrained weights:

    depth      run only the first `depth` transformer blocks
    attention  slice the QKV/out projections down to a subset of heads
    ffn        slice the MLP down to a subset of its intermediate channels

    Attention and FFN adaptation slice the weight tensors rather than masking
    their outputs, so the skipped work is genuinely not performed.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transformer = model.transformer
        self.lm_head = model.lm_head

    # ------------------------------------------------------------------
    # Adapted sub-layers
    # ------------------------------------------------------------------

    @staticmethod
    def _attention(attn, hidden_states, head_fraction):
        """Self-attention over the first `head_fraction` of the heads."""
        batch, seq, _ = hidden_states.shape
        head_dim = attn.head_dim
        embed_dim = attn.embed_dim

        heads = max(1, round(attn.num_heads * head_fraction))
        active = heads * head_dim

        # Conv1D weight is (in_features, out_features); c_attn packs [q | k | v].
        columns = torch.cat([
            torch.arange(offset, offset + active, device=hidden_states.device)
            for offset in (0, embed_dim, 2 * embed_dim)
        ])

        qkv = hidden_states @ attn.c_attn.weight[:, columns] + attn.c_attn.bias[columns]
        query, key, value = qkv.split(active, dim=-1)

        shape = (batch, seq, heads, head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=attn.scaling
        )

        context = context.transpose(1, 2).reshape(batch, seq, active)

        return context @ attn.c_proj.weight[:active, :] + attn.c_proj.bias

    @staticmethod
    def _mlp(mlp, hidden_states, ffn_fraction):
        """Feed-forward over the first `ffn_fraction` of the intermediate channels."""
        inner = mlp.c_fc.weight.shape[1]
        active = max(1, round(inner * ffn_fraction))

        hidden = mlp.act(
            hidden_states @ mlp.c_fc.weight[:, :active] + mlp.c_fc.bias[:active]
        )

        return hidden @ mlp.c_proj.weight[:active, :] + mlp.c_proj.bias

    @staticmethod
    def _routed_mlp(mlp, hidden_states, routing):
        """Token-level top-k routing over FFN chunks (MoE-style baseline).

        The gate is a fixed seeded random projection, not a learned router; each
        token pays for `top_k` of `chunks` experts instead of the whole FFN.
        """
        chunks = routing["chunks"]
        top_k = routing["top_k"]
        gate = routing["gate"].to(hidden_states.dtype)

        batch, seq, embed_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, embed_dim)

        chunk_size = mlp.c_fc.weight.shape[1] // chunks
        selected = (flat @ gate).topk(top_k, dim=-1).indices

        out = mlp.c_proj.bias.expand(flat.shape[0], -1).clone()

        for chunk in range(chunks):
            tokens = (selected == chunk).any(dim=-1).nonzero(as_tuple=True)[0]
            if tokens.numel() == 0:
                continue

            low, high = chunk * chunk_size, (chunk + 1) * chunk_size

            hidden = mlp.act(
                flat[tokens] @ mlp.c_fc.weight[:, low:high] + mlp.c_fc.bias[low:high]
            )

            out.index_add_(0, tokens, hidden @ mlp.c_proj.weight[low:high, :])

        return out.reshape(batch, seq, embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, depth=12, attention_mode="full", ffn_mode="full",
                routing=None):
        total_layers = len(self.transformer.h)

        if depth < 1 or depth > total_layers:
            raise ValueError(f"depth must be between 1 and {total_layers}, got {depth}")

        head_fraction = ATTENTION_HEAD_FRACTION[attention_mode]
        ffn_fraction = FFN_CHUNK_FRACTION[ffn_mode]

        _, seq = input_ids.shape
        position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)

        hidden_states = self.transformer.drop(
            self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        )

        # Unadapted blocks go through the stock implementation, which keeps the
        # deep configuration numerically identical to the reference model.
        stock = head_fraction == 1.0 and ffn_fraction == 1.0 and routing is None

        for index in range(depth):
            block = self.transformer.h[index]

            if stock:
                # attention_mask must stay None: SDPA only applies GPT-2's causal
                # mask when no explicit mask is supplied.
                hidden_states = block(hidden_states, attention_mask=None, use_cache=False)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                continue

            hidden_states = hidden_states + self._attention(
                block.attn, block.ln_1(hidden_states), head_fraction
            )

            normed = block.ln_2(hidden_states)

            if routing is None:
                hidden_states = hidden_states + self._mlp(block.mlp, normed, ffn_fraction)
            else:
                hidden_states = hidden_states + self._routed_mlp(block.mlp, normed, routing)

        return self.lm_head(self.transformer.ln_f(hidden_states))


def make_routing_gate(top_k=2, chunks=FFN_CHUNKS, hidden_size=768, seed=42):
    """Fixed random gate for the routing baseline, seeded for reproducibility."""
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn(hidden_size, chunks, generator=generator) / hidden_size ** 0.5
    return {"chunks": chunks, "top_k": top_k, "gate": gate}
