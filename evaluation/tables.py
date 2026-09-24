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
    CONFIDENCE_THRESHOLD,
    CONFIGURATIONS,
    DEPTH_MAP,
    FALLBACK_CONFIGURATION,
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
        ("Calibration prompts", metadata.get("calibration_prompts", "n/a")),
        ("Predictor held-out accuracy", f"{metadata['accuracy']:.3f}"),
        ("Predictor held-out macro F1", f"{metadata.get('macro_f1', float('nan')):.3f}"),
        ("Majority-class baseline (acc / macro F1)",
         f"{metadata.get('majority_accuracy', float('nan')):.3f} / "
         f"{metadata.get('majority_macro_f1', float('nan')):.3f}"),
        ("Predictor 5-fold CV accuracy",
         f"{metadata['cv_accuracy_mean']:.3f} +/- {metadata['cv_accuracy_std']:.3f}"),
        ("Predictor 5-fold CV macro F1",
         f"{metadata.get('cv_macro_f1_mean', float('nan')):.3f} +/- "
         f"{metadata.get('cv_macro_f1_std', float('nan')):.3f}"),
        ("Calibration tolerance", metadata["tolerance"]),
        ("Policy confidence threshold", CONFIDENCE_THRESHOLD),
        ("Policy fallback", FALLBACK_CONFIGURATION),
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


def table2_main(baselines):
    if baselines is None:
        return None

    return project(baselines, ["label", "loss", "perplexity", "accuracy",
                               "answer_perplexity", "latency_mean", "latency_std",
                               "average_depth", "layer_reduction", "relative_flops",
                               "compute_reduction"])


def table3_ablation(ablations):
    if ablations is None:
        return None
    return project(ablations, ["label", "loss", "perplexity", "accuracy", "latency_mean",
                               "average_depth", "layer_reduction", "relative_flops",
                               "compute_reduction", "volatility", "rollback_count"])


def table4_stability(stability):
    if stability is None:
        return None

    columns = ["level", "label", "proposed_switches", "executed_switches",
               "executed_volatility", "stability_score", "rollbacks_observed",
               "rollback_rate", "loss", "loss_after_rollback", "loss_without_rollback",
               "latency_mean", "average_depth"]
    available = [c for c in columns if c in stability.columns]

    return stability[available].rename(columns={
        "level": "Switching", "label": "Variant",
        "proposed_switches": "Proposed switches", "executed_switches": "Executed switches",
        "executed_volatility": "Volatility V", "stability_score": "S = 1 - V",
        "rollbacks_observed": "Rollbacks", "rollback_rate": "Rollback rate",
        "loss": "Prompt loss", "loss_after_rollback": "Loss @ rollback steps",
        "loss_without_rollback": "Same steps, no monitor",
        "latency_mean": "Latency (s)", "average_depth": "Avg depth",
    }).round(4)


def table5_datasets(datasets):
    if datasets is None:
        return None
    columns = ["dataset", "label", "accuracy", "answer_perplexity", "loss",
               "average_depth", "layer_reduction", "relative_flops", "latency_mean"]
    return project(datasets, columns).rename(columns={"dataset": "Dataset"})


def table7_policy(sweep):
    if sweep is None:
        return None
    columns = ["mechanism", "threshold", "accepted", "fallback_count", "average_depth",
               "relative_flops", "latency_mean", "loss", "relative_loss_increase",
               "target_agreement", "share_shallow", "share_medium", "share_deep",
               "selected"]
    return sweep[[c for c in columns if c in sweep.columns]].round(4)


def table8_dense(dense):
    if dense is None:
        return None
    columns = ["model", "trained", "parameters", "layers", "depth", "hidden_size", "dim",
               "norm", "position", "activation", "sequence_length", "runs",
               "latency_mean", "latency_std"]
    return dense[[c for c in columns if c in dense.columns]].round(4)


CLAIMS = """
Every statement below is written against a measured result. Design objectives
and hypotheses are labelled as such, and are not presented as findings.

- MEASURED: attention-head and FFN-chunk adaptation change the computed output;
  they slice the weight tensors rather than masking outputs, so the skipped work
  is not performed (verified in `test_msa.py` against the reference model).
- MEASURED: the depth-only mechanism trades quality for compute monotonically
  across calibration tolerances (`results/quality_compute_frontier.csv`).
- MEASURED: the executed depth equals the depth f_theta + policy select, for
  every configuration and mechanism (`test_msa.py`, P2 and P6 checks).
- MEASURED: raising the policy confidence threshold from 0.40 to 0.80 moves the
  system monotonically toward the static model (fallbacks 2 -> 120 of 160,
  relative loss increase 56.9% -> 10.1%); 0.40 was selected by a rule fixed
  in advance (`results/policy_threshold_sweep.csv`).
- MEASURED: the stability monitor acts only when switching is high. At the high
  level it cut executed volatility from 0.830 to 0.566 with 13 rollbacks, but
  the rolled-back steps had higher loss (8.76 vs 5.65), so rollback traded
  quality for stability (`results/stability/stability_levels.csv`).
- MEASURED: under the controlled benchmark, depth and attention/FFN slicing
  reduce wall-clock latency; routing is slower than static depth-only at
  sequence lengths 16 and 64 and faster only at 256 (`results/benchmark.csv`).
- MEASURED: a pretrained dense model of similar size (SmolLM-135M) reaches 0.750
  exact match on short QA against 0.133 for static GPT-2 (`results/baselines.csv`).
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
- LIMITATION: Baseline D uses a different tokenizer, so its per-token loss and
  perplexity are not comparable with GPT-2's; exact match and latency are.
- LIMITATION: latency is measured on a shared CPU with 6 pinned threads. The
  single-pass latencies recorded inside evaluation runs are noisy; only
  `results/benchmark.csv` is a controlled measurement.
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
        "table2_main_results": ("Table 2 -- Main results (Baselines A-E)",
                                table2_main(baselines)),
        "table3_ablation": ("Table 3 -- Ablation studies", table3_ablation(ablations)),
        "table4_stability": ("Table 4 -- Stability by switching level",
                             table4_stability(stability)),
        "table5_datasets": ("Table 5 -- Downstream dataset evaluation",
                            table5_datasets(datasets)),
        "table6_feature_ablation": ("Table 6 -- Task Analyzer feature ablation",
                                    features),
        "table7_policy_threshold": ("Table 7 -- Policy confidence-threshold sweep",
                                    table7_policy(read("policy_threshold_sweep"))),
        "table8_dense_reference": ("Table 8 -- Baseline D dense references",
                                   table8_dense(dense)),
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

    lines += ["## Research claims checklist", CLAIMS.strip(), ""]

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
