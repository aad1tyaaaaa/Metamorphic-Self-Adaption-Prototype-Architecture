"""Phase 1: validate the configuration predictor independently of MSA.

    python test_predictor.py
"""

import math

from analyzer.task_analyzer import TaskAnalyzer
from configs.configurations import DEPTH_MAP
from controller.policy import ConfigurationPolicy
from controller.predictor import PerformancePredictor

SAMPLES = [
    "What is the capital of France?",
    "Explain why TCP and UDP are different.",
    "Calculate the probability of drawing two aces from a deck of cards.",
    "If a train travels 240 km in 3 hours and increases its speed by 20 percent, "
    "calculate the new travel time and explain each step of your reasoning.",
]


def main():
    predictor = PerformancePredictor()
    feature_set = predictor.metadata.get("feature_set", "v1")
    analyzer = TaskAnalyzer(feature_set)
    policy = ConfigurationPolicy()

    print("=" * 72)
    print("PHASE 1 -- PREDICTOR VALIDATION")
    print("=" * 72)
    print(f"Model:       {predictor.model_path}")
    print(f"Feature set: {feature_set} ({len(predictor.features)} features)")
    print(f"Tolerance:   {predictor.metadata.get('tolerance')}")
    print(f"Held-out:    {predictor.metadata.get('accuracy')}")

    for text in SAMPLES:
        features = analyzer.analyze(text)
        result = predictor.predict(features)
        decision = policy.decide(result)

        # Model loads and accepts the feature vector.
        assert len(features) == len(predictor.features)
        # Configuration is one of shallow / medium / deep.
        assert result["configuration"] in DEPTH_MAP
        # Configuration maps to depth 4 / 8 / 12.
        assert result["depth"] == DEPTH_MAP[result["configuration"]]
        assert result["depth"] in (4, 8, 12)
        # Probabilities are produced, valid, and free of NaN.
        assert set(result["probabilities"]) == set(DEPTH_MAP)
        assert all(math.isfinite(p) and 0.0 <= p <= 1.0
                   for p in result["probabilities"].values())
        assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-6)
        assert math.isfinite(result["confidence"])
        assert result["confidence"] == max(result["probabilities"].values())
        # The policy always yields a runnable configuration.
        assert decision["configuration"] in DEPTH_MAP

        print("\n" + "-" * 72)
        print(f"INPUT: {text}")
        print("FEATURES: " + "  ".join(f"{k}={v:.3f}" for k, v in features.items()))
        print(f"CONFIGURATION: {result['configuration']}  "
              f"DEPTH: {result['depth']}/12  "
              f"CONFIDENCE: {result['confidence']:.3f}")
        print("PROBABILITIES: " + "  ".join(
            f"{k}={v:.3f}" for k, v in sorted(result['probabilities'].items())))
        print(f"POLICY: {decision['configuration']}"
              + ("  (fallback applied)" if decision["fallback_applied"] else ""))

    # Malformed input must fail loudly rather than silently mispredict.
    try:
        predictor.predict({"input_length": 0.5})
        raise AssertionError("missing features should raise")
    except KeyError:
        pass

    print("\n" + "=" * 72)
    print("PHASE 1 PASSED: predictor loads, accepts features, and produces")
    print("valid probabilities and depths with no NaN or invalid values.")


if __name__ == "__main__":
    main()
