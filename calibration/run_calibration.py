"""Calibration sweep (Phase 4).

Runs every prompt through every configuration under both adaptation mechanisms
and records the features, quality and cost of each run.

    python -m calibration.run_calibration [--limit N]

The `mechanism` column separates depth-only runs (Phase 6) from full
depth+attention+FFN runs (Phase 9); downstream code filters on it.
"""

import argparse
import os

import pandas as pd
import torch

from analyzer.task_analyzer import TaskAnalyzer
from configs.configurations import CONFIG_NAMES, FEATURES_V2
from controller.complexity import ComplexityScorer
from controller.controller import DynamicArchitectureController
from data.calibration_dataset import load
from models.adaptive_model import AdaptiveGPT2
from models.backbone import load_model

OUTPUT_PATH = "results/calibration_results.csv"


def main(limit=None, seed=42, output_path=OUTPUT_PATH):
    torch.manual_seed(seed)

    _, entries = load()
    if limit:
        entries = entries[:limit]

    model, tokenizer = load_model()
    adaptive_model = AdaptiveGPT2(model).eval()

    analyzer = TaskAnalyzer("v2")
    scorer = ComplexityScorer()

    controllers = {
        mechanism: DynamicArchitectureController(adaptive_model, tokenizer, mechanism)
        for mechanism in ("depth", "full")
    }

    rows = []
    total = len(entries) * len(controllers) * len(CONFIG_NAMES)

    for sample_id, entry in enumerate(entries):
        text = entry["text"]

        features = analyzer.analyze_all(text)
        complexity = scorer.calculate(features)
        input_ids = controllers["depth"].encode(text)

        for mechanism, controller in controllers.items():
            for configuration in CONFIG_NAMES:
                result = controller.run(text, configuration, input_ids=input_ids)

                rows.append({
                    "sample_id": sample_id,
                    "text": text,
                    "category": entry["category"],
                    "mechanism": mechanism,
                    "complexity": complexity,
                    **{name: features[name] for name in FEATURES_V2},
                    **{
                        key: value for key, value in result.items()
                        if key not in ("logits",)
                    },
                })

        if (sample_id + 1) % 25 == 0 or sample_id + 1 == len(entries):
            print(f"[{len(rows):5d}/{total}] sample {sample_id + 1}/{len(entries)}", flush=True)

    frame = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(f"\nCALIBRATION COMPLETE: {len(frame)} experiments -> {output_path}")
    print("\nMean by mechanism and configuration:")
    print(
        frame.groupby(["mechanism", "configuration"])[
            ["loss", "perplexity", "latency_seconds", "relative_flops"]
        ].mean().round(4)
    )

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    main(limit=args.limit, seed=args.seed, output_path=args.output)
