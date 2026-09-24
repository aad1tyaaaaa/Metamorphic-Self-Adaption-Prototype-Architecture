"""Export measured results and the trained predictor for the UI (Phase 20).

    python -m ui.export_ui_data

Writes ui/msa_data.json. The predictor weights are exported so the browser runs
the real f_theta forward pass rather than a mock; the GPT-2 execution figures are
exported as recorded measurements, which the UI labels as replayed rather than
live.
"""

import json
import os

import joblib
import pandas as pd

from configs.configurations import CONFIGURATIONS, DEPTH_MAP, FEATURE_SETS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "msa_data.json")
TEMPLATE = os.path.join(HERE, "index.template.html")
PAGE = os.path.join(HERE, "index.html")


def export_predictor():
    payload = joblib.load("results/performance_predictor.pkl")
    pipeline = payload["model"]

    scaler = pipeline.named_steps["scaler"]
    mlp = pipeline.named_steps["mlp"]

    return {
        "features": payload["features"],
        "feature_set": payload["feature_set"],
        "classes": [str(c) for c in mlp.classes_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": [w.tolist() for w in mlp.coefs_],
        "biases": [b.tolist() for b in mlp.intercepts_],
        "activation": mlp.activation,
        "output_activation": mlp.out_activation_,
        "accuracy": payload["accuracy"],
        "cv_accuracy_mean": payload["cv_accuracy_mean"],
        "cv_accuracy_std": payload["cv_accuracy_std"],
        "tolerance": payload["tolerance"],
        "confusion_matrix": payload["confusion_matrix"],
        "confusion_labels": payload["confusion_labels"],
        "target_distribution": payload["target_distribution"],
    }


def frame(path, columns=None):
    if not os.path.exists(path):
        return []
    data = pd.read_csv(path)
    if columns:
        data = data[[c for c in columns if c in data.columns]]
    return json.loads(data.to_json(orient="records"))


def export_calibration():
    data = pd.read_csv("results/calibration_results.csv")

    summary = (
        data.groupby(["mechanism", "configuration"])
        [["loss", "perplexity", "latency_seconds", "relative_flops", "depth"]]
        .mean().reset_index()
    )

    categories = (
        data.drop_duplicates("sample_id")["category"]
        .value_counts().to_dict()
    )

    return {
        "summary": json.loads(summary.to_json(orient="records")),
        "categories": categories,
        "experiments": int(len(data)),
        "prompts": int(data["sample_id"].nunique()),
    }


def main():
    payload = {
        "generated_note": (
            "GPT-2 execution figures are recorded measurements replayed from "
            "results/. The Task Analyzer, predictor and policy run live in the "
            "browser from exported weights."
        ),
        "configurations": CONFIGURATIONS,
        "depth_map": DEPTH_MAP,
        "feature_sets": FEATURE_SETS,
        "predictor": export_predictor(),
        "calibration": export_calibration(),
        "frontier": frame("results/quality_compute_frontier.csv"),
        "baselines": frame("results/baselines.csv"),
        "ablations": frame("results/ablations.csv"),
        "stability": frame("results/stability_experiment.csv"),
        "datasets": frame("results/dataset_evaluation.csv"),
        "benchmark": frame("results/benchmark.csv"),
        "feature_ablation": frame("results/feature_ablation.csv"),
        "dense_reference": frame("results/dense_reference.csv"),
    }

    with open("results/environment.json", encoding="utf-8") as handle:
        payload["environment"] = json.load(handle)

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    # Inline the payload so the page opens straight from disk (no fetch, no CORS).
    with open(TEMPLATE, encoding="utf-8") as handle:
        template = handle.read()

    inline = json.dumps(payload, separators=(",", ":")).replace("</", r"<\/")

    with open(PAGE, "w", encoding="utf-8") as handle:
        handle.write(template.replace("__MSA_DATA__", inline))

    size = os.path.getsize(OUTPUT) / 1024
    print(f"Wrote {OUTPUT} ({size:.0f} KB)")
    print(f"Wrote {PAGE} ({os.path.getsize(PAGE) / 1024:.0f} KB, self-contained)")
    print(f"  predictor: {payload['predictor']['feature_set']}, "
          f"{len(payload['predictor']['features'])} features, "
          f"layers {[len(w) for w in payload['predictor']['weights']]}")
    print(f"  calibration: {payload['calibration']['experiments']} experiments")
    for key in ("frontier", "baselines", "ablations", "stability", "datasets",
                "benchmark"):
        print(f"  {key}: {len(payload[key])} rows")

    return payload


if __name__ == "__main__":
    main()
