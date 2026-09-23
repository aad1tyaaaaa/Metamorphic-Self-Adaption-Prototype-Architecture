"""Dynamic Architecture Controller (Phases 6 and 9).

Owns the single execution path from a named configuration to an actual forward
pass, and emits the telemetry every downstream experiment records.
"""

import math
import time

import torch

from configs.configurations import (
    TOTAL_LAYERS,
    TOTAL_HEADS,
    FFN_CHUNKS,
    ATTENTION_HEAD_FRACTION,
    FFN_CHUNK_FRACTION,
    resolve_config,
    relative_flops,
)


class DynamicArchitectureController:
    def __init__(self, adaptive_model, tokenizer, mechanism="depth", routing=None,
                 attention=None, ffn=None):
        """`mechanism` is shorthand: "depth" disables both extra mechanisms,
        "full" enables both. `attention`/`ffn` override it individually."""
        self.model = adaptive_model
        self.tokenizer = tokenizer
        self.mechanism = mechanism
        self.routing = routing
        self.attention = (mechanism == "full") if attention is None else attention
        self.ffn = (mechanism == "full") if ffn is None else ffn

    def knobs(self, configuration):
        return resolve_config(configuration, self.attention, self.ffn)

    # ------------------------------------------------------------------

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt")["input_ids"]

    def run(self, text, configuration, input_ids=None, label_ids=None):
        """Execute one configuration and return logits plus telemetry.

        `label_ids`, when given, restricts the loss to those positions (used for
        answer-conditional scoring); otherwise the loss is next-token loss over
        the whole prompt.
        """
        knobs = self.knobs(configuration)

        if input_ids is None:
            input_ids = self.encode(text)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(
                input_ids,
                depth=knobs["depth"],
                attention_mode=knobs["attention_mode"],
                ffn_mode=knobs["ffn_mode"],
                routing=self.routing,
            )
        latency = time.perf_counter() - start

        loss = self.loss(logits, input_ids if label_ids is None else label_ids)

        return {
            "configuration": configuration,
            "logits": logits,
            "loss": loss,
            "perplexity": math.exp(min(loss, 20)),
            "latency_seconds": latency,
            **self.telemetry(knobs, input_ids.shape[1]),
        }

    # ------------------------------------------------------------------

    def telemetry(self, knobs, seq_len):
        """Per-inference telemetry required by Phase 6."""
        ffn_fraction = FFN_CHUNK_FRACTION[knobs["ffn_mode"]]

        if self.routing is not None:
            # Each token pays for top_k of the chunks instead of the full FFN.
            ffn_fraction = self.routing["top_k"] / self.routing["chunks"]

        return {
            "depth": knobs["depth"],
            "attention_mode": knobs["attention_mode"],
            "ffn_mode": knobs["ffn_mode"],
            "layers_executed": knobs["depth"],
            "layers_skipped": TOTAL_LAYERS - knobs["depth"],
            "layer_reduction": 1 - knobs["depth"] / TOTAL_LAYERS,
            "active_heads": max(
                1, round(TOTAL_HEADS * ATTENTION_HEAD_FRACTION[knobs["attention_mode"]])
            ),
            "active_ffn_chunks": max(1, round(FFN_CHUNKS * ffn_fraction)),
            "sequence_length": seq_len,
            "relative_flops": relative_flops(
                knobs["depth"], knobs["attention_mode"], knobs["ffn_mode"],
                seq_len=seq_len, ffn_fraction=ffn_fraction,
            ),
        }

    @staticmethod
    def loss(logits, target_ids):
        """Next-token cross-entropy. Positions labelled -100 are ignored."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        ).item()

    # ------------------------------------------------------------------

    def generate(self, text, configuration, max_new_tokens=32, eos_text="\n"):
        """Greedy decoding at a fixed configuration.

        No KV cache: the adaptive stack re-runs the prefix each step. That is
        slow but keeps decoding on exactly the same code path as scoring.
        """
        knobs = self.knobs(configuration)
        input_ids = self.encode(text)
        prompt_len = input_ids.shape[1]
        stop_id = self.tokenizer.encode(eos_text)[0] if eos_text else None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(
                    input_ids,
                    depth=knobs["depth"],
                    attention_mode=knobs["attention_mode"],
                    ffn_mode=knobs["ffn_mode"],
                    routing=self.routing,
                )
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=1)

                if stop_id is not None and next_id.item() == stop_id:
                    break

        return self.tokenizer.decode(input_ids[0, prompt_len:])


def demo():
    """Phase 6 test: the controller selects and executes the requested depth."""
    from models.backbone import load_model
    from models.adaptive_model import AdaptiveGPT2

    model, tokenizer = load_model()
    controller = DynamicArchitectureController(AdaptiveGPT2(model).eval(), tokenizer)

    text = "The capital of India is New Delhi and the capital of France is"
    results = {name: controller.run(text, name) for name in ("shallow", "medium", "deep")}

    for name, result in results.items():
        assert result["depth"] == {"shallow": 4, "medium": 8, "deep": 12}[name]
        assert result["layers_executed"] + result["layers_skipped"] == TOTAL_LAYERS
        assert math.isfinite(result["loss"])
        print(
            f"{name:8s} depth={result['depth']:2d} "
            f"executed={result['layers_executed']:2d} skipped={result['layers_skipped']:2d} "
            f"loss={result['loss']:.4f} flops={result['relative_flops']:.3f} "
            f"latency={result['latency_seconds']*1000:.1f}ms"
        )

    # Deeper must not be worse on the language-model objective for this prompt.
    assert results["deep"]["loss"] < results["shallow"]["loss"]
    assert results["deep"]["relative_flops"] == 1.0

    # Full mechanism must actually change computation, not just the label.
    full = DynamicArchitectureController(controller.model, tokenizer, mechanism="full")
    depth_only = controller.run(text, "medium")["logits"]
    adapted = full.run(text, "medium")["logits"]
    assert not torch.allclose(depth_only, adapted), "attention/FFN adaptation was a no-op"
    print("controller demo OK")


if __name__ == "__main__":
    demo()
