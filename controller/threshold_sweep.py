"""Confidence-threshold sweep for the configuration policy (Phase 3).

    python -m controller.threshold_sweep

For every threshold, the policy is applied to f_theta's predictions on the
predictor's held-out calibration prompts (never seen in training). The loss,
latency and FLOPs of the configuration the policy selects are read from the
measured calibration runs of that same prompt, so every number is a recorded
GPT-2 forward pass rather than an estimate.

Operating-point rule (fixed before looking at the results): pick the threshold
whose decisions agree most often with the calibration target c*(x); break ties
toward lower relative FLOPs.
"""

import argparse

import numpy as np
import pandas as pd

from configs.configurations import CONFIG_NAMES, FEATURES_V2
from controller.policy import CalibrationPolicy, ConfigurationPolicy
from controller.predictor import PerformancePredictor

CALIBRATION_PATH = "results/calibration_results.csv"
OUTPUT_PATH = "results/policy_threshold_sweep.csv"
THRESHOLDS = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80]


def sweep(frame, predictor, thresholds=None, mechanism="depth", fallback="deep"):
    frame = frame[frame["mechanism"] == mechanism]
    held_out = predictor.metadata.get("test_sample_ids")
    if not held_out:
        raise RuntimeError("predictor metadata has no test_sample_ids; retrain it")

    frame = frame[frame["sample_id"].isin(held_out)]
    lookup = frame.set_index(["sample_id", "configuration"])
    features = frame.groupby("sample_id")[FEATURES_V2].first()

    targets = CalibrationPolicy(predictor.metadata["tolerance"]).create_targets(
        frame, mechanism=mechanism
    ).set_index("sample_id")["target_configuration"]

    predictions = {
        sample_id: predictor.predict(row.to_dict())
        for sample_id, row in features.iterrows()
    }

    rows = []
    for threshold in thresholds or THRESHOLDS:
        policy = ConfigurationPolicy(threshold, fallback)
        decisions = []

        for sample_id, prediction in predictions.items():
            decision = policy.decide(prediction)
            measured = lookup.loc[(sample_id, decision["configuration"])]
            deep = lookup.loc[(sample_id, "deep")]
            decisions.append({
                "configuration": decision["configuration"],
                "depth": decision["depth"],
                "fallback": decision["fallback_applied"],
                "loss": float(measured["loss"]),
                "deep_loss": float(deep["loss"]),
                "latency": float(measured["latency_seconds"]),
                "flops": float(measured["relative_flops"]),
                "agrees": decision["configuration"] == targets[sample_id],
            })

        data = pd.DataFrame(decisions)
        counts = data["configuration"].value_counts()

        rows.append({
            "threshold": threshold,
            "mechanism": mechanism,
            "samples": len(data),
            "accepted": int((~data["fallback"]).sum()),
            "fallback_count": int(data["fallback"].sum()),
            "fallback_rate": float(data["fallback"].mean()),
            "average_depth": float(data["depth"].mean()),
            "layer_reduction": float(1 - data["depth"].mean() / 12),
            "relative_flops": float(data["flops"].mean()),
            "latency_mean": float(data["latency"].mean()),
            "loss": float(data["loss"].mean()),
            "deep_loss": float(data["deep_loss"].mean()),
            "relative_loss_increase": float(data["loss"].mean() / data["deep_loss"].mean() - 1),
            "target_agreement": float(data["agrees"].mean()),
            **{f"share_{name}": float(counts.get(name, 0) / len(data)) for name in CONFIG_NAMES},
        })

    return pd.DataFrame(rows)


def select_operating_point(table):
    ranked = table.sort_values(["target_agreement", "relative_flops"],
                               ascending=[False, True])
    return ranked.iloc[0]


def main(output_path=OUTPUT_PATH):
    frame = pd.read_csv(CALIBRATION_PATH)
    predictor = PerformancePredictor()

    tables = [sweep(frame, predictor, mechanism=m) for m in ("depth", "full")]
    table = pd.concat(tables, ignore_index=True)

    print("=" * 78)
    print(f"POLICY CONFIDENCE-THRESHOLD SWEEP  ({int(table['samples'].iloc[0])} held-out prompts, "
          "fallback = deep)")
    print("=" * 78)
    columns = ["mechanism", "threshold", "accepted", "fallback_count", "average_depth",
               "relative_flops", "latency_mean", "loss", "relative_loss_increase",
               "target_agreement", "share_shallow", "share_medium", "share_deep"]
    print(table[columns].round(4).to_string(index=False))

    chosen = select_operating_point(table[table["mechanism"] == "depth"])
    table["selected"] = (table["mechanism"] == "depth") & np.isclose(
        table["threshold"], chosen["threshold"])

    table.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Selected operating point: threshold {chosen['threshold']:.2f} "
          f"(agreement {chosen['target_agreement']:.3f}, "
          f"relative FLOPs {chosen['relative_flops']:.3f}, "
          f"fallback rate {chosen['fallback_rate']:.3f})")

    # Raising the threshold can only move decisions toward the fallback.
    depth_rows = table[table["mechanism"] == "depth"].sort_values("threshold")
    assert depth_rows["fallback_count"].is_monotonic_increasing
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    main(parser.parse_args().output)
