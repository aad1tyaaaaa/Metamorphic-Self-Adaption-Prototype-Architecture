"""Phase 0: re-run baselines and ablations with the retrained shallow/medium
configurations substituted in.

Loads the adapters saved by `calibration.finetune_adapters` and reruns the
exact same evaluation methodology as `python main.py --mode evaluate`
(`--suite baselines`/`--suite ablations`, dataset short_qa, 60 samples,
generation on) so the output is directly diff-able against the pre-retraining
`results/baselines.csv` / `results/ablations.csv`.

    python -m evaluation.evaluate_phase0
"""

import argparse
import os

import torch

from configs.configurations import CONFIDENCE_THRESHOLD
from data.load_data import load_dataset_by_name
from evaluation.evaluate import suite_ablations, suite_baselines
from evaluation.harness import MSASystem
from models.adapters import load_adapter

ADAPTER_DIR = "results/phase0_retraining"
PER_SAMPLE_DIR = "results/evaluation_retrained"


def load_adapters(system, configs=("shallow", "medium")):
    model = system.adaptive_model.model
    adapters = {}
    for name in configs:
        path = f"{ADAPTER_DIR}/{name}_adapter.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}; run `python -m calibration.finetune_adapters` first"
            )
        adapters[name] = load_adapter(model, path)
    return adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="short_qa")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    system = MSASystem(confidence_threshold=args.confidence_threshold, seed=args.seed)
    adapters = load_adapters(system)

    dataset = load_dataset_by_name(args.dataset, limit=args.limit)
    print(f"\nPhase 0 retrained evaluation: {args.dataset} ({len(dataset)} samples), "
          f"adapters for {sorted(adapters)}")

    # Per-sample frames go to a separate directory so results/evaluation/ (the
    # pre-retraining per-sample files) is never overwritten.
    suite_baselines(system, args.dataset, dataset, generate=True, adapters=adapters,
                    output_name="baselines_retrained", log_name="inference_log_retrained",
                    output_dir=PER_SAMPLE_DIR)
    suite_ablations(system, args.dataset, dataset, generate=True, adapters=adapters,
                    output_name="ablations_retrained", output_dir=PER_SAMPLE_DIR)

    print("\nPHASE 0 RETRAINED EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
