"""Final research tables (Phase 21) and the claims checklist (Phase 22).

    python -m evaluation.tables

Writes results/tables.md plus one CSV per table. Everything is read back from
the result CSVs, so no number in the paper is typed by hand.
"""

import json
import os

import joblib
import pandas as pd

from configs.configurations import (
    CONFIGURATIONS,
    DEPTH_MAP,
    FEATURE_SETS,
    TOTAL_LAYERS,
)

RESULTS = "results"
OUTPUT = f"{RESULTS}/tables.md"


def read(name):
    path = f"{RESULTS}/{name}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def table1_system():
    metadata = joblib.load(f"{RESULTS}/performance_predictor.pkl")
    environment = {}
    if os.path.exists(f"{RESULTS}/environment.json"):
        with open(f"{RESULTS}/environment.json", encoding="utf-8") as handle:
            environment = json.load(handle)

    rows = [
        ("Backbone", "GPT-2 small (HuggingFace `gpt2`)"),
        ("Parameters", "124,439,808"),
        ("Layers", TOTAL_LAYERS),
        ("Attention heads", 12),
        ("Hidden size", 768),
        ("FFN inner size", 3072),
        ("Feature set", f"{metadata['feature_set']} "
                        f"({len(FEATURE_SETS[metadata['feature_set']])} features)"),
        ("Features", ", ".join(metadata["features"])),
        ("Configurations", "; ".join(
            f"{name}: depth {knobs['depth']}, attention {knobs['attention_mode']}, "
            f"ffn {knobs['ffn_mode']}" for name, knobs in CONFIGURATIONS.items())),
        ("Configuration depths", ", ".join(f"{k}={v}" for k, v in DEPTH_MAP.items())),
        ("Predictor architecture", metadata["architecture"]),
        ("Predictor held-out accuracy", f"{metadata['accuracy']:.3f}"),
        ("Predictor 5-fold CV accuracy",
         f"{metadata['cv_accuracy_mean']:.3f} +/- {metadata['cv_accuracy_std']:.3f}"),
        ("Calibration tolerance", metadata["tolerance"]),
        ("Seed", metadata["seed"]),
        ("Device", environment.get("device", "cpu")),
        ("PyTorch", environment.get("torch", "see results/environment.json")),
        ("Transformers", environment.get("transformers", "see results/environment.json")),
    ]

    return pd.DataFrame(rows, columns=["Item", "Value"])


COLUMNS = {
    "label": "Method",
    "loss": "Prompt loss",
    "perplexity": "Perplexity",
    "answer_perplexity": "Answer PPL",
    "accuracy": "Accuracy",
    "latency_mean": "Latency (s)",
    "latency_std": "Latency std",
    "average_depth": "Avg depth",
    "layer_reduction": "Layer red.",
    "relative_flops": "Rel. FLOPs",
    "compute_reduction": "Compute red.",
}


def project(frame, columns):
    available = [c for c in columns if c in frame.columns]
    out = frame[available].copy()
    return out.rename(columns=COLUMNS).round(4)


def table2_main(baselines, datasets, dense):
    if baselines is None:
        return None

    main = project(baselines, ["label", "loss", "perplexity", "latency_mean",
                               "latency_std", "average_depth", "layer_reduction",
                               "relative_flops", "compute_reduction"])

    if dense is not None and len(dense):
        row = dense.iloc[0]
        main = pd.concat([main, pd.DataFrame([{
            "Method": "Baseline D: Dense reference (RMSNorm/RoPE/SwiGLU, untrained)",
            "Prompt loss": float("nan"),
            "Perplexity": float("nan"),
            "Latency (s)": round(float(row["latency_mean"]), 4),
            "Latency std": round(float(row["latency_std"]), 4),
            "Avg depth": row["depth"],
            "Layer red.": 0.0,
            "Rel. FLOPs": float("nan"),
            "Compute red.": float("nan"),
        }])], ignore_index=True)

    return main


def table3_ablation(ablations):
    if ablations is None:
        return None
    return project(ablations, ["label", "loss", "perplexity", "latency_mean",
                               "average_depth", "layer_reduction", "relative_flops",
                               "compute_reduction"])


def table4_stability(stability):
    if stability is None:
        return None

    columns = ["label", "switches", "switch_rate", "volatility", "rollback_count",
               "rollback_rate", "loss", "average_depth"]
    available = [c for c in columns if c in stability.columns]

    return stability[available].rename(columns={
        "label": "Variant", "switches": "Switches", "switch_rate": "Switch rate",
        "volatility": "Volatility", "rollback_count": "Rollbacks",
        "rollback_rate": "Rollback rate", "loss": "Prompt loss",
        "average_depth": "Avg depth",
    }).round(4)


def table5_datasets(datasets):
    if datasets is None:
        return None
    columns = ["dataset", "label", "answer_perplexity", "accuracy", "loss",
               "average_depth", "layer_reduction", "relative_flops", "latency_mean"]
    return project(datasets, columns).rename(columns={"dataset": "Dataset"})


CLAIMS = """
Every statement below is written against a measured result. Design objectives
and hypotheses are labelled as such, and are not presented as findings.

- MEASURED: attention-head and FFN-chunk adaptation change the computed output;
  they slice the weight tensors rather than masking outputs, so the skipped work
  is not performed (verified in `test_msa.py` against the reference model).
- MEASURED: the depth-only mechanism trades quality for compute monotonically
  across calibration tolerances (`results/quality_compute_frontier.csv`).
- MEASURED: the stability monitor detects a synthetic alternating sequence and
  performs rollbacks (`results/stability_experiment.csv`).
- LIMITATION: calibration targets come from next-token loss over the prompt.
  That is a prototype signal, not downstream task quality. Downstream numbers
  are reported separately in `results/dataset_evaluation.csv`.
- LIMITATION: GPT-2 small cannot solve GSM8K. Reported accuracy on it is near
  zero for every variant, including the static baseline, so that dataset
  separates compute behaviour, not task quality.
- LIMITATION: attention and FFN adaptation are applied to pretrained weights
  with no retraining or distillation, so quality degrades markedly. The measured
  loss increase is reported and not explained away.
- LIMITATION: the routing baseline uses a fixed seeded gate, not a learned
  router, so it is a compute-reduction reference and not a trained MoE.
- LIMITATION: Baseline D is randomly initialised. Only its parameter count and
  latency are meaningful; its quality is reported as N/A.
- LIMITATION: latency is measured on CPU. Slicing overhead can exceed the saved
  work at short sequence lengths, so FLOP reduction does not always translate
  into wall-clock reduction. Both are reported.
- HYPOTHESIS (not established here): a predictor trained on downstream task
  quality rather than prompt loss would select configurations more usefully.
- FUTURE WORK: calibrate attention/FFN configurations with retrained or
  distilled weights so the non-depth mechanisms are competitive.

Claims deliberately NOT made: that MSA always improves performance, that it
guarantees lower latency, or that it preserves quality. The experiments in this
repository do not establish any of those.
"""


def main():
    baselines = read("baselines")
    ablations = read("ablations")
    stability = read("stability_experiment")
    datasets = read("dataset_evaluation")
    dense = read("dense_reference")
    features = read("feature_ablation")

    tables = {
        "table1_system_configuration": ("Table 1 -- Model and system configuration",
                                        table1_system()),
        "table2_main_results": ("Table 2 -- Main results",
                                table2_main(baselines, datasets, dense)),
        "table3_ablation": ("Table 3 -- Ablation studies", table3_ablation(ablations)),
        "table4_stability": ("Table 4 -- Stability", table4_stability(stability)),
        "table5_datasets": ("Table 5 -- Downstream dataset evaluation",
                            table5_datasets(datasets)),
        "table6_feature_ablation": ("Table 6 -- Task Analyzer feature ablation",
                                    features),
    }

    lines = ["# MSA-GPT-2 -- Final Research Tables", ""]

    for name, (title, frame) in tables.items():
        lines += [f"## {title}", ""]

        if frame is None or not len(frame):
            lines += ["_Not available: run `python main.py --mode evaluate` first._", ""]
            print(f"  {name}: skipped (no data)")
            continue

        frame.to_csv(f"{RESULTS}/{name}.csv", index=False)
        lines += [frame.to_markdown(index=False), ""]
        print(f"  {name}: {len(frame)} rows")

    lines += ["## Research claims checklist (Phase 22)", CLAIMS.strip(), ""]

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
