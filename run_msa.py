"""End-to-end MSA inference (Phases 2, 3, 9, 10).

    python run_msa.py
    python run_msa.py --text "Explain why the sky appears blue."

Pipeline:
    Input -> Task Analyzer -> f_theta -> Policy -> Stability Monitor
          -> Dynamic Architecture Controller -> Adaptive GPT-2 -> Output
"""

import argparse
import json

import torch

from analyzer.task_analyzer import TaskAnalyzer
from controller.controller import DynamicArchitectureController
from controller.policy import ConfigurationPolicy
from controller.predictor import PerformancePredictor
from models.adaptive_model import AdaptiveGPT2
from models.backbone import load_model
from monitor.stability_monitor import StabilityMonitor

# One prompt per category the plan asks the integration to cover.
TEST_INPUTS = [
    "What is the capital of France?",
    "Explain why the sky appears blue.",
    "Calculate the percentage increase from 50 to 75.",
    "John has 5 apples. He buys 3 more apples. He then gives 2 apples to Sarah. "
    "Explain how many apples John has remaining.",
    "What is the difference between a process and a thread?",
    "You are given a report about renewable energy adoption. First, identify the "
    "three most important claims. Second, explain which claim is best supported by "
    "evidence. Third, describe what additional information would be needed to "
    "evaluate the weakest claim.",
]


class MSA:
    """The whole system behind one `run(text)` call."""

    def __init__(self, mechanism="full", confidence_threshold=0.5, fallback="deep",
                 monitor=True, seed=42):
        torch.manual_seed(seed)

        model, tokenizer = load_model()
        self.model = AdaptiveGPT2(model).eval()

        self.predictor = PerformancePredictor()
        self.analyzer = TaskAnalyzer(self.predictor.metadata.get("feature_set", "v1"))
        self.policy = ConfigurationPolicy(confidence_threshold, fallback)
        self.controller = DynamicArchitectureController(self.model, tokenizer, mechanism)
        self.monitor = StabilityMonitor() if monitor else None

    def run(self, text, return_logits=False):
        features = self.analyzer.analyze(text)
        prediction = self.predictor.predict(features)
        decision = self.policy.decide(prediction)

        configuration = decision["configuration"]
        stability = self.monitor.observe(configuration) if self.monitor else None

        if stability:
            configuration = stability["configuration"]

        outcome = self.controller.run(text, configuration)
        logits = outcome.pop("logits")

        record = {
            "text": text,
            "features": features,
            "probabilities": decision["probabilities"],
            "predicted_configuration": decision["predicted_configuration"],
            "confidence": decision["confidence"],
            "fallback_applied": decision["fallback_applied"],
            "stability": {
                "status": stability["status"],
                "volatility": stability["volatility"],
                "rolled_back": stability["rolled_back"],
                "rollback_count": stability["rollback_count"],
            } if stability else None,
            **outcome,
            "logits_shape": tuple(logits.shape),
        }

        if return_logits:
            record["logits"] = logits

        return record


def show(record):
    print("\n" + "=" * 74)
    print(f"INPUT: {record['text'][:200]}")

    print("\nFEATURES:")
    for name, value in record["features"].items():
        print(f"  {name:18s}: {value:.3f}")

    print("\nPREDICTOR:")
    for name, value in sorted(record["probabilities"].items()):
        print(f"  P({name:8s}) = {value:.3f}")
    print(f"  predicted  : {record['predicted_configuration']} "
          f"(confidence {record['confidence']:.3f})")
    print(f"  fallback   : {record['fallback_applied']}")

    print("\nEXECUTION:")
    print(f"  configuration : {record['configuration'].upper()}")
    print(f"  depth         : {record['depth']}/12  "
          f"(executed {record['layers_executed']}, skipped {record['layers_skipped']})")
    print(f"  attention     : {record['attention_mode']} "
          f"({record['active_heads']}/12 heads)")
    print(f"  ffn           : {record['ffn_mode']} "
          f"({record['active_ffn_chunks']}/4 chunks)")
    print(f"  relative flops: {record['relative_flops']:.3f}")
    print(f"  loss          : {record['loss']:.4f}   "
          f"perplexity: {record['perplexity']:.2f}")
    print(f"  latency       : {record['latency_seconds']*1000:.2f} ms")
    print(f"  logits        : {record['logits_shape']}")

    if record["stability"]:
        stability = record["stability"]
        print(f"\nSTABILITY: {stability['status']}  "
              f"volatility={stability['volatility']:.2f}  "
              f"rollbacks={stability['rollback_count']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="*", default=None)
    parser.add_argument("--mechanism", default="full", choices=["depth", "full"])
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("Loading MSA system...")
    msa = MSA(mechanism=args.mechanism,
              confidence_threshold=args.confidence_threshold,
              monitor=not args.no_monitor)

    for text in (args.text or TEST_INPUTS):
        record = msa.run(text)

        if args.json:
            print(json.dumps(record, indent=2, default=str))
        else:
            show(record)

    print("\n" + "=" * 74)
    print("MSA run complete: Analyzer -> f_theta -> Policy -> Monitor -> "
          "Controller -> Adaptive GPT-2")


if __name__ == "__main__":
    main()
