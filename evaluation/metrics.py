"""Standardised metrics for MSA evaluation records (Phase 11)."""

import math
from collections import Counter

import numpy as np

from configs.configurations import TOTAL_LAYERS


def aggregate(records, monitor_summary=None):
    """Reduce per-inference records to the standard metric set."""
    if not records:
        return {}

    losses = np.array([r["loss"] for r in records], dtype=float)
    latencies = np.array([r["latency_seconds"] for r in records], dtype=float)
    depths = np.array([r["depth"] for r in records], dtype=float)
    flops = np.array([r["relative_flops"] for r in records], dtype=float)

    distribution = Counter(r["configuration"] for r in records)
    total = len(records)

    metrics = {
        "samples": total,
        "loss": float(losses.mean()),
        "perplexity": float(math.exp(min(losses.mean(), 20))),
        "latency_mean": float(latencies.mean()),
        "latency_std": float(latencies.std(ddof=1)) if total > 1 else 0.0,
        "latency_p50": float(np.percentile(latencies, 50)),
        "throughput": float(1.0 / latencies.mean()) if latencies.mean() > 0 else float("nan"),
        "average_depth": float(depths.mean()),
        "layer_reduction": float(1 - depths.mean() / TOTAL_LAYERS),
        "relative_flops": float(flops.mean()),
        "compute_reduction": float(1 - flops.mean()),
    }

    for name in ("shallow", "medium", "deep"):
        metrics[f"share_{name}"] = distribution.get(name, 0) / total

    # Task metrics are only present when a downstream dataset was evaluated.
    if "correct" in records[0]:
        metrics["accuracy"] = float(np.mean([r["correct"] for r in records]))

    if "answer_loss" in records[0]:
        answer_losses = np.array([r["answer_loss"] for r in records], dtype=float)
        metrics["answer_loss"] = float(answer_losses.mean())
        metrics["answer_perplexity"] = float(math.exp(min(answer_losses.mean(), 20)))

    if "fallback_applied" in records[0]:
        metrics["fallback_rate"] = float(np.mean([r["fallback_applied"] for r in records]))

    if "confidence" in records[0]:
        metrics["mean_confidence"] = float(
            np.mean([r["confidence"] for r in records])
        )

    if monitor_summary:
        metrics.update({
            "switches": monitor_summary["switches"],
            "switch_rate": monitor_summary["switch_rate"],
            "volatility": monitor_summary["volatility"],
            "monitor_stability_score": monitor_summary["stability_score"],
            "rollback_count": monitor_summary["rollback_count"],
            "rollback_rate": monitor_summary["rollback_rate"],
            "stability_status": monitor_summary["status"],
        })

    return metrics


def demo():
    records = [
        {"loss": 3.0, "latency_seconds": 0.03, "depth": 12, "relative_flops": 1.0,
         "configuration": "deep"},
        {"loss": 5.0, "latency_seconds": 0.01, "depth": 4, "relative_flops": 1 / 3,
         "configuration": "shallow"},
    ]

    metrics = aggregate(records)

    assert metrics["samples"] == 2
    assert math.isclose(metrics["loss"], 4.0)
    assert math.isclose(metrics["average_depth"], 8.0)
    assert math.isclose(metrics["layer_reduction"], 1 / 3)
    assert math.isclose(metrics["share_deep"], 0.5)
    assert math.isclose(metrics["share_medium"], 0.0)
    assert math.isclose(metrics["compute_reduction"], 1 - (1 + 1 / 3) / 2)
    assert aggregate([]) == {}

    print("metrics demo OK:", {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in metrics.items()})


if __name__ == "__main__":
    demo()
