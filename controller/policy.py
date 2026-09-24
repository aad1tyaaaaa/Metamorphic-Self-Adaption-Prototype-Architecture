"""Configuration policies.

`CalibrationPolicy` derives training targets from a calibration sweep
(quality-constrained depth selection).

`ConfigurationPolicy` is the Phase 3 inference-time policy: it turns predictor
probabilities into a final configuration, separating the neural estimate from
the safety/compute decision so both stay inspectable.
"""

import pandas as pd

from configs.configurations import (
    CONFIDENCE_THRESHOLD,
    CONFIG_NAMES,
    DEPTH_MAP,
    FALLBACK_CONFIGURATION,
)


class ConfigurationPolicy:
    """Decides the final configuration from predicted probabilities.

    The confidence threshold is a tunable knob, not an established constant;
    `controller/threshold_sweep.py` sweeps it rather than assuming a value.
    """

    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD,
                 fallback=FALLBACK_CONFIGURATION):
        if fallback not in CONFIG_NAMES:
            raise ValueError(f"fallback must be one of {CONFIG_NAMES}, got {fallback!r}")

        self.confidence_threshold = confidence_threshold
        self.fallback = fallback

    def decide(self, prediction):
        predicted = prediction["configuration"]
        confidence = prediction["confidence"]

        low_confidence = confidence < self.confidence_threshold
        selected = self.fallback if low_confidence else predicted

        return {
            "probabilities": prediction["probabilities"],
            "predicted_configuration": predicted,
            "configuration": selected,
            "depth": DEPTH_MAP[selected],
            "confidence": confidence,
            "confidence_threshold": self.confidence_threshold,
            "fallback_applied": low_confidence,
            "fallback_configuration": self.fallback if low_confidence else None,
        }


class CalibrationPolicy:
    """Picks the shallowest configuration whose loss stays within tolerance."""

    def __init__(self, quality_tolerance=0.10):
        self.quality_tolerance = quality_tolerance

    def create_targets(self, source, mechanism="depth"):
        frame = pd.read_csv(source) if isinstance(source, str) else source

        if "mechanism" in frame.columns:
            frame = frame[frame["mechanism"] == mechanism]

        targets = []

        for sample_id, group in frame.groupby("sample_id"):
            group = group.sort_values("depth")

            deep_rows = group[group["depth"] == 12]
            if deep_rows.empty:
                continue

            deep_row = deep_rows.iloc[0]
            baseline_loss = float(deep_row["loss"])
            maximum_allowed = baseline_loss * (1 + self.quality_tolerance)

            acceptable = group[group["loss"] <= maximum_allowed]
            selected = (
                acceptable.sort_values("depth").iloc[0]
                if len(acceptable) else deep_row
            )

            targets.append({
                "sample_id": sample_id,
                "target_depth": int(selected["depth"]),
                "target_configuration": selected["configuration"],
                "selected_loss": float(selected["loss"]),
                "deep_loss": baseline_loss,
                "selected_latency": float(selected["latency_seconds"]),
                "deep_latency": float(deep_row["latency_seconds"]),
                "relative_flops": float(selected["relative_flops"]),
                "relative_loss_increase": float(selected["loss"]) / baseline_loss - 1,
            })

        return pd.DataFrame(targets)


def demo():
    policy = ConfigurationPolicy(confidence_threshold=0.6, fallback="deep")

    confident = policy.decide({
        "configuration": "shallow", "confidence": 0.8,
        "probabilities": {"shallow": 0.8, "medium": 0.15, "deep": 0.05},
    })
    assert confident["configuration"] == "shallow"
    assert confident["depth"] == 4
    assert confident["fallback_applied"] is False

    unsure = policy.decide({
        "configuration": "shallow", "confidence": 0.4,
        "probabilities": {"shallow": 0.4, "medium": 0.35, "deep": 0.25},
    })
    assert unsure["configuration"] == "deep"
    assert unsure["depth"] == 12
    assert unsure["fallback_applied"] is True
    assert unsure["predicted_configuration"] == "shallow"

    print("policy demo OK: confident ->", confident["configuration"],
          "| low confidence ->", unsure["configuration"], "(fallback)")


if __name__ == "__main__":
    demo()
