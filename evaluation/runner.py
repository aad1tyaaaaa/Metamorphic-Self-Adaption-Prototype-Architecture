"""Standardised evaluation runner (Phase 11).

One entry point evaluates any method on any dataset and writes per-sample rows
in a fixed schema to results/evaluation/<method>__<dataset>.csv.

    python -m evaluation.runner --methods static depth_adaptive msa --dataset short_qa
    python -m evaluation.runner --methods dense_reference --dataset short_qa --generate

Methods are the harness variants (evaluation/harness.py VARIANTS) plus
`dense_reference`, the pretrained Baseline D.
"""

import argparse
import os

import numpy as np
import pandas as pd

from data.load_data import load_dataset_by_name
from evaluation.harness import VARIANTS, MSASystem, run_variant, to_records
from evaluation.metrics import aggregate

OUTPUT_DIR = "results/evaluation"

# The schema every per-sample result file shares, in this column order.
SCHEMA = [
    "method", "dataset", "sample_id", "loss", "perplexity", "latency", "depth",
    "layer_reduction", "configuration", "attention_mode", "ffn_mode", "confidence",
    "rollback",
]
# Recorded when the method/dataset provides them.
OPTIONAL = [
    "executed_depth", "relative_flops", "active_parameters", "sequence_length",
    "proposed_configuration", "fallback_applied", "stability_status", "volatility",
    "stability_score", "answer_loss", "correct", "generated", "question",
]

DENSE = "dense_reference"
_dense_model = None


def standardize(records, method, dataset):
    rows = []
    for sample_id, record in enumerate(records):
        row = {
            "method": method,
            "dataset": dataset,
            "sample_id": sample_id,
            "loss": record["loss"],
            "perplexity": record["perplexity"],
            "latency": record["latency_seconds"],
            "depth": record["depth"],
            "layer_reduction": record["layer_reduction"],
            "configuration": record["configuration"],
            "attention_mode": record["attention_mode"],
            "ffn_mode": record["ffn_mode"],
            "confidence": record.get("confidence", float("nan")),
            "rollback": bool(record.get("rolled_back", False)),
        }
        row.update({key: record[key] for key in OPTIONAL if key in record})
        rows.append(row)

    return pd.DataFrame(rows, columns=SCHEMA + [c for c in OPTIONAL
                                                if any(c in r for r in records)])


def dense_model():
    global _dense_model
    if _dense_model is None:
        from evaluation.baselines import PretrainedDenseBaseline
        _dense_model = PretrainedDenseBaseline()
    return _dense_model


def run_dense(dataset, generate=False, max_new_tokens=24):
    model = dense_model()
    records = [model.run(r, generate=generate, max_new_tokens=max_new_tokens)
               for r in to_records(dataset)]

    metrics = aggregate(records)
    # Depth-derived metrics describe GPT-2's 12-layer stack; they do not apply here.
    metrics.update({
        "average_depth": model.layers, "layer_reduction": 0.0,
        "relative_flops": float("nan"), "compute_reduction": float("nan"),
        "share_shallow": float("nan"), "share_medium": float("nan"),
        "share_deep": float("nan"), "fallback_rate": float("nan"),
        "variant": DENSE,
        "label": f"Baseline D: Dense reference ({model.model_id}, "
                 f"{model.config['parameters']:,} params)",
        "parameters": model.config["parameters"],
    })
    return metrics, records


def evaluate(system, method, dataset_name, dataset, generate=False, max_new_tokens=24,
             save=True, **kwargs):
    """Run one method on one dataset; return (aggregate metrics, per-sample frame)."""
    if method == DENSE:
        metrics, records = run_dense(dataset, generate, max_new_tokens)
    else:
        metrics, records = run_variant(system, method, dataset, generate=generate,
                                       max_new_tokens=max_new_tokens, **kwargs)

    metrics["dataset"] = dataset_name
    frame = standardize(records, method, dataset_name)

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        frame.to_csv(f"{OUTPUT_DIR}/{method}__{dataset_name}.csv", index=False)

    return metrics, frame


def main(methods, dataset_name, limit=None, generate=False, max_new_tokens=24):
    unknown = [m for m in methods if m != DENSE and m not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; choose from {sorted(VARIANTS) + [DENSE]}")

    dataset = load_dataset_by_name(dataset_name, limit=limit)
    system = MSASystem() if any(m != DENSE for m in methods) else None

    rows = []
    for method in methods:
        metrics, frame = evaluate(system, method, dataset_name, dataset,
                                  generate=generate, max_new_tokens=max_new_tokens)
        rows.append(metrics)

        assert list(frame.columns[:len(SCHEMA)]) == SCHEMA
        assert len(frame) == len(dataset)
        assert np.isfinite(frame["loss"]).all()
        print(f"  {method:18s} -> {OUTPUT_DIR}/{method}__{dataset_name}.csv "
              f"({len(frame)} rows, loss {frame['loss'].mean():.4f}, "
              f"depth {frame['depth'].mean():.2f})", flush=True)

    summary = pd.DataFrame(rows)
    columns = [c for c in ("variant", "loss", "accuracy", "answer_perplexity",
                           "average_depth", "layer_reduction", "relative_flops",
                           "latency_mean") if c in summary.columns]
    print("\n" + summary[columns].round(4).to_string(index=False))
    return summary


def cli(default_methods):
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=default_methods)
    parser.add_argument("--dataset", default="short_qa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    return main(args.methods, args.dataset, args.limit, args.generate, args.max_new_tokens)


if __name__ == "__main__":
    cli(["static", "depth_adaptive", "msa"])
