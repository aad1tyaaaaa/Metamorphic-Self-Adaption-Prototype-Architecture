"""Baseline A: static GPT-2 at a fixed depth (Phase 12).

    python run_static.py --depth 12

The quality/compute reference every adaptive variant is compared against.
"""

import argparse

import pandas as pd

from configs.configurations import DEPTH_MAP
from data.load_data import load_dataset_by_name
from evaluation.harness import MSASystem, run_variant

VARIANT_FOR_DEPTH = {4: "static_shallow", 8: "static_medium", 12: "static"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=12, choices=sorted(VARIANT_FOR_DEPTH))
    parser.add_argument("--dataset", default="short_qa")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--all-depths", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset_by_name(args.dataset, limit=args.limit)
    system = MSASystem()

    depths = sorted(VARIANT_FOR_DEPTH) if args.all_depths else [args.depth]
    rows = []

    print(f"\nStatic baseline on {args.dataset} ({len(dataset)} samples)\n")

    for depth in depths:
        metrics, _ = run_variant(system, VARIANT_FOR_DEPTH[depth], dataset)
        rows.append(metrics)

        assert metrics["average_depth"] == depth
        print(f"  depth {depth:2d}  loss={metrics['loss']:.4f}  "
              f"perplexity={metrics['perplexity']:10.2f}  "
              f"latency={metrics['latency_mean']*1000:6.2f} ms  "
              f"flops={metrics['relative_flops']:.3f}")

    frame = pd.DataFrame(rows)
    frame.to_csv("results/static_baseline.csv", index=False)
    print("\nSaved: results/static_baseline.csv")

    assert set(DEPTH_MAP.values()) >= set(depths)


if __name__ == "__main__":
    main()
