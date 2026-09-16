import torch
import torch.nn as nn


class AdaptiveGPT2(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model

        self.transformer = model.transformer
        self.lm_head = model.lm_head

    def forward(self, input_ids, depth=12):

        # -------------------------------------------------
        # Validate depth
        # -------------------------------------------------

        total_layers = len(self.transformer.h)

        if depth < 1 or depth > total_layers:
            raise ValueError(
                f"depth must be between 1 and {total_layers}"
            )

        # -------------------------------------------------
        # Input dimensions
        # -------------------------------------------------

        batch_size, sequence_length = input_ids.shape

        device = input_ids.device

        # -------------------------------------------------
        # Position IDs
        # -------------------------------------------------

        position_ids = torch.arange(
            sequence_length,
            device=device
        ).unsqueeze(0)

        # -------------------------------------------------
        # Token + position embeddings
        # -------------------------------------------------

        inputs_embeds = self.transformer.wte(input_ids)

        position_embeds = self.transformer.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.transformer.drop(hidden_states)

        # -------------------------------------------------
        # Attention mask
        # -------------------------------------------------

        attention_mask = torch.ones(
            (batch_size, sequence_length),
            device=device
        )

        # -------------------------------------------------
        # Run selected transformer blocks
        # -------------------------------------------------

        for i in range(depth):

            block = self.transformer.h[i]

            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False
            )

            # Transformers 5.x may return a tuple.
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        # -------------------------------------------------
        # Final LayerNorm
        # -------------------------------------------------

        hidden_states = self.transformer.ln_f(
            hidden_states
        )

        # -------------------------------------------------
        # Language model head
        # -------------------------------------------------

        logits = self.lm_head(hidden_states)

        return logits