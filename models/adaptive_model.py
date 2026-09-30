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
        # Indices of the blocks the last forward pass actually ran, so callers can
        # verify the executed depth instead of trusting the requested one.
        self.last_executed_blocks = []

    # ------------------------------------------------------------------
    # Adapted sub-layers
    # ------------------------------------------------------------------

    @staticmethod
    def _attention(attn, hidden_states, head_fraction, attn_bias=None, proj_bias=None):
        """Self-attention over the first `head_fraction` of the heads.

        `attn_bias`/`proj_bias`, when given, override the frozen backbone biases
        (Phase 0 retraining: a per-configuration adapter fine-tunes only biases
        and LayerNorm affine parameters, never the pretrained weight matrices).
        """
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

        bias = attn.c_attn.bias if attn_bias is None else attn_bias
        qkv = hidden_states @ attn.c_attn.weight[:, columns] + bias[columns]
        query, key, value = qkv.split(active, dim=-1)

        shape = (batch, seq, heads, head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=attn.scaling
        )

        context = context.transpose(1, 2).reshape(batch, seq, active)

        out_bias = attn.c_proj.bias if proj_bias is None else proj_bias
        return context @ attn.c_proj.weight[:active, :] + out_bias

    @staticmethod
    def _mlp(mlp, hidden_states, ffn_fraction, fc_bias=None, proj_bias=None):
        """Feed-forward over the first `ffn_fraction` of the intermediate channels."""
        inner = mlp.c_fc.weight.shape[1]
        active = max(1, round(inner * ffn_fraction))

        fc_bias = mlp.c_fc.bias if fc_bias is None else fc_bias
        hidden = mlp.act(
            hidden_states @ mlp.c_fc.weight[:, :active] + fc_bias[:active]
        )

        out_bias = mlp.c_proj.bias if proj_bias is None else proj_bias
        return hidden @ mlp.c_proj.weight[:active, :] + out_bias

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
                routing=None, adapter=None):
        """`adapter`, when given, is a `models.adapters.ConfigAdapter` supplying
        per-block bias and LayerNorm parameters fine-tuned for this configuration
        (Phase 0). It only ever applies on the non-stock path, so the deep/static
        configuration's pretrained weights are never touched."""
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
        executed = []

        for index in range(depth):
            block = self.transformer.h[index]
            executed.append(index)

            if stock:
                # attention_mask must stay None: SDPA only applies GPT-2's causal
                # mask when no explicit mask is supplied.
                hidden_states = block(hidden_states, attention_mask=None, use_cache=False)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                continue

            params = adapter.block_params(index) if adapter is not None else None

            if params is None:
                ln1_out = block.ln_1(hidden_states)
            else:
                ln1_out = F.layer_norm(
                    hidden_states, (hidden_states.shape[-1],), params["ln1_w"], params["ln1_b"]
                )

            hidden_states = hidden_states + self._attention(
                block.attn, ln1_out, head_fraction,
                attn_bias=None if params is None else params["attn_bias"],
                proj_bias=None if params is None else params["attn_proj_bias"],
            )

            if params is None:
                normed = block.ln_2(hidden_states)
            else:
                normed = F.layer_norm(
                    hidden_states, (hidden_states.shape[-1],), params["ln2_w"], params["ln2_b"]
                )

            if routing is None:
                hidden_states = hidden_states + self._mlp(
                    block.mlp, normed, ffn_fraction,
                    fc_bias=None if params is None else params["fc_bias"],
                    proj_bias=None if params is None else params["mlp_proj_bias"],
                )
            else:
                hidden_states = hidden_states + self._routed_mlp(block.mlp, normed, routing)

        self.last_executed_blocks = executed

        # The adapter's final LayerNorm only applies on the path it was trained
        # on; a stock (full attention + full FFN) request must stay byte-identical
        # to the unadapted model even if an adapter object was passed in.
        if adapter is None or stock:
            return self.lm_head(self.transformer.ln_f(hidden_states))

        final = F.layer_norm(
            hidden_states, (hidden_states.shape[-1],), adapter.ln_f_w, adapter.ln_f_b
        )
        return self.lm_head(final)


def make_routing_gate(top_k=2, chunks=FFN_CHUNKS, hidden_size=768, seed=42):
    """Fixed random gate for the routing baseline, seeded for reproducibility."""
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn(hidden_size, chunks, generator=generator) / hidden_size ** 0.5
    return {"chunks": chunks, "top_k": top_k, "gate": gate}
