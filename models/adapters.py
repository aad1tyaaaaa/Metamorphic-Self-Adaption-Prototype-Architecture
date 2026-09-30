"""Phase 0 retraining: per-configuration bias/LayerNorm-only fine-tuning.

`AdaptiveGPT2` slices the pretrained weight matrices to run a reduced-width
configuration; that slicing was never seen during GPT-2's pretraining. This
module adds the smallest defensible adaptation that lets a configuration
compensate for slicing without touching the shared pretrained weights: a
`ConfigAdapter` holds a trainable copy of every bias and LayerNorm affine
parameter touched by the blocks it covers (initialised from the frozen
backbone, so training starts at the identity), plus its own final LayerNorm.
The underlying `Conv1D` weight matrices are never modified, and the deep/
static configuration never receives an adapter, so it stays byte-identical to
the original model.

This is Stage 6 of the paper's training procedure ("train jointly with a
small amount of backbone fine-tuning"), generalised from routing to also
cover attention/FFN slicing, implemented as the minimal bias/LayerNorm
variant the plan explicitly allows in place of LoRA (no `peft` dependency is
installed, and the repository is CPU-only).
"""

import torch
import torch.nn as nn


class ConfigAdapter(nn.Module):
    """Trainable biases + LayerNorm affine parameters for the first `depth` blocks."""

    def __init__(self, model, depth):
        super().__init__()
        self.depth = depth

        self.ln1_w = nn.ParameterList()
        self.ln1_b = nn.ParameterList()
        self.ln2_w = nn.ParameterList()
        self.ln2_b = nn.ParameterList()
        self.attn_bias = nn.ParameterList()
        self.attn_proj_bias = nn.ParameterList()
        self.fc_bias = nn.ParameterList()
        self.mlp_proj_bias = nn.ParameterList()

        for index in range(depth):
            block = model.transformer.h[index]
            self.ln1_w.append(nn.Parameter(block.ln_1.weight.detach().clone()))
            self.ln1_b.append(nn.Parameter(block.ln_1.bias.detach().clone()))
            self.ln2_w.append(nn.Parameter(block.ln_2.weight.detach().clone()))
            self.ln2_b.append(nn.Parameter(block.ln_2.bias.detach().clone()))
            self.attn_bias.append(nn.Parameter(block.attn.c_attn.bias.detach().clone()))
            self.attn_proj_bias.append(nn.Parameter(block.attn.c_proj.bias.detach().clone()))
            self.fc_bias.append(nn.Parameter(block.mlp.c_fc.bias.detach().clone()))
            self.mlp_proj_bias.append(nn.Parameter(block.mlp.c_proj.bias.detach().clone()))

        self.ln_f_w = nn.Parameter(model.transformer.ln_f.weight.detach().clone())
        self.ln_f_b = nn.Parameter(model.transformer.ln_f.bias.detach().clone())

    def block_params(self, index):
        return {
            "ln1_w": self.ln1_w[index], "ln1_b": self.ln1_b[index],
            "ln2_w": self.ln2_w[index], "ln2_b": self.ln2_b[index],
            "attn_bias": self.attn_bias[index], "attn_proj_bias": self.attn_proj_bias[index],
            "fc_bias": self.fc_bias[index], "mlp_proj_bias": self.mlp_proj_bias[index],
        }

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def build_adapter(model, config_name, depth_map):
    return ConfigAdapter(model, depth_map[config_name])


def save_adapter(adapter, path):
    torch.save({"depth": adapter.depth, "state_dict": adapter.state_dict()}, path)


def load_adapter(model, path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    adapter = ConfigAdapter(model, checkpoint["depth"])
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    return adapter


if __name__ == "__main__":
    from models.backbone import load_model

    model, _ = load_model()
    adapter = ConfigAdapter(model, depth=4)
    params = adapter.block_params(0)
    assert params["ln1_w"].shape == model.transformer.h[0].ln_1.weight.shape
    assert torch.allclose(params["ln1_w"], model.transformer.h[0].ln_1.weight)
    assert adapter.trainable_parameter_count() > 0
    print(f"ConfigAdapter demo OK: depth=4, "
          f"{adapter.trainable_parameter_count():,} trainable parameters")
