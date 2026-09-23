"""Configuration predictor f_theta (Phase 1).

Loads the trained MLP and turns Task Analyzer features into configuration
probabilities. The feature list is read from the metadata saved at training
time, so a v1 and a v2 model can be loaded through the same interface.
"""

import math

import joblib
import numpy as np

from configs.configurations import DEPTH_MAP, FEATURES_V1

DEFAULT_MODEL_PATH = "results/performance_predictor.pkl"


class PerformancePredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path

        payload = joblib.load(model_path)

        # Models saved before feature-set metadata existed are plain pipelines.
        if isinstance(payload, dict):
            self.model = payload["model"]
            self.metadata = payload
            self.features = payload["features"]
        else:
            self.model = payload
            self.metadata = {}
            self.features = FEATURES_V1

        self.depth_map = DEPTH_MAP

    def predict(self, features):
        missing = [name for name in self.features if name not in features]
        if missing:
            raise KeyError(
                f"predictor needs features {self.features}; missing {missing}. "
                "Use TaskAnalyzer(version=...) matching the trained model."
            )

        vector = np.array(
            [[float(features[name]) for name in self.features]], dtype=np.float64
        )

        configuration = str(self.model.predict(vector)[0])
        probabilities = self.model.predict_proba(vector)[0]

        probability_map = {
            str(label): float(value)
            for label, value in zip(self.model.classes_, probabilities)
        }

        return {
            "configuration": configuration,
            "depth": self.depth_map[configuration],
            "confidence": float(max(probability_map.values())),
            "probabilities": probability_map,
        }


def demo():
    from analyzer.task_analyzer import TaskAnalyzer

    predictor = PerformancePredictor()
    analyzer = TaskAnalyzer(predictor.metadata.get("feature_set", "v1"))

    samples = [
        "What is the capital of France?",
        "Explain why TCP and UDP are different.",
        "Calculate the probability of drawing two aces from a deck of cards.",
        "If a train travels 240 km in 3 hours and increases its speed by 20 percent, "
        "calculate the new travel time and explain each step.",
    ]

    for text in samples:
        result = predictor.predict(analyzer.analyze(text))

        assert result["configuration"] in DEPTH_MAP
        assert result["depth"] == DEPTH_MAP[result["configuration"]]
        assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-6)
        assert all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in result["probabilities"].values()
        )
        assert result["confidence"] == max(result["probabilities"].values())

        probabilities = " ".join(
            f"{name}={value:.3f}" for name, value in sorted(result["probabilities"].items())
        )
        print(
            f"{result['configuration']:8s} depth={result['depth']:2d} "
            f"conf={result['confidence']:.3f}  {probabilities}  | {text[:48]}"
        )

    print("predictor demo OK")


if __name__ == "__main__":
    demo()
