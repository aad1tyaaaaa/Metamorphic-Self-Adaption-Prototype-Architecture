"""Quality-compute frontier: sweep the calibration quality tolerance.

    python -m controller.policy_sweep

The tolerance is the maximum relative loss increase, against the 12-layer
model, that the calibration policy will accept in exchange for a shallower
configuration. It is an operating point to be chosen, not a constant.
"""

import pandas as pd

from controller.policy import CalibrationPolicy

CALIBRATION_PATH = "results/calibration_results.csv"
OUTPUT_PATH = "results/quality_compute_frontier.csv"

TOLERANCES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]


def sweep(frame, tolerances=None, mechanism="depth"):
    rows = []

    for tolerance in tolerances or TOLERANCES:
        targets = CalibrationPolicy(tolerance).create_targets(frame, mechanism=mechanism)
        distribution = targets["target_configuration"].value_counts().to_dict()

        rows.append({
            "tolerance": tolerance,
            "average_depth": targets["target_depth"].mean(),
            "layer_reduction_percent": (1 - targets["target_depth"].mean() / 12) * 100,
            "relative_flops": targets["relative_flops"].mean(),
            "compute_reduction_percent": (1 - targets["relative_flops"].mean()) * 100,
            "average_latency": targets["selected_latency"].mean(),
            "average_loss": targets["selected_loss"].mean(),
            "deep_loss": targets["deep_loss"].mean(),
            "relative_loss_increase": (
                targets["selected_loss"].mean() / targets["deep_loss"].mean() - 1
            ),
            "shallow": distribution.get("shallow", 0),
            "medium": distribution.get("medium", 0),
            "deep": distribution.get("deep", 0),
        })

    return pd.DataFrame(rows)


def main():
    frame = pd.read_csv(CALIBRATION_PATH)
    summary = sweep(frame)

    print("=" * 78)
    print("QUALITY-COMPUTE FRONTIER (depth-only mechanism)")
    print("=" * 78)
    print(summary.round(4).to_string(index=False))

    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Tolerances must trade quality for compute monotonically.
    assert summary["average_depth"].is_monotonic_decreasing, "frontier is not monotonic"

    usable = summary[summary["shallow"] + summary["medium"] > 0]
    if len(usable):
        print(f"\nShallowest tolerance that ever selects below depth 12: "
              f"{usable.iloc[0]['tolerance']}")

    return summary


if __name__ == "__main__":
    main()
