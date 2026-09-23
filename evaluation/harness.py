"""Evaluation harness (Phases 11, 12, 15).

Every baseline and every ablation is the same execution loop under a different
set of flags, so they cannot drift apart. `VARIANTS` is the whole definition.

    python -m evaluation.harness --variants static msa --dataset short_qa
"""

import torch

from analyzer.task_analyzer import TaskAnalyzer
from configs.configurations import CONFIG_NAMES
from controller.complexity import ComplexityScorer
from controller.controller import DynamicArchitectureController
from controller.policy import ConfigurationPolicy
from controller.predictor import PerformancePredictor
from controller.selector import ConfigurationSelector
from data.load_data import exact_match
from evaluation.metrics import aggregate
from models.adaptive_model import AdaptiveGPT2, make_routing_gate
from models.backbone import load_model
from monitor.stability_monitor import StabilityMonitor

# selector:  static | rule | predictor
# attention/ffn: enable that adaptation mechanism
# routing:   use the MoE-style token routing baseline instead of FFN slicing
# monitor:   run decisions through the stability monitor
VARIANTS = {
    # ---- Phase 12 baselines ----------------------------------------
    "static": dict(selector="static", static_config="deep",
                   label="Baseline A: Static GPT-2 (12 layers)"),
    "depth_adaptive": dict(selector="predictor",
                           label="Baseline B: Depth-adaptive"),
    "routing": dict(selector="static", static_config="deep", routing=True,
                    label="Baseline C: Routing / MoE-style FFN"),
    "msa": dict(selector="predictor", attention=True, ffn=True, monitor=True,
                label="Baseline E: Full MSA"),

    # ---- Phase 15 ablations ----------------------------------------
    "A1_static": dict(selector="static", static_config="deep",
                      label="A1: Static GPT-2"),
    "A2_analyzer": dict(selector="rule", label="A2: Analyzer + rule policy"),
    "A3_predictor": dict(selector="predictor", label="A3: Analyzer + f_theta"),
    "A4_depth": dict(selector="predictor", label="A4: Depth only"),
    "A5_depth_attention": dict(selector="predictor", attention=True,
                               label="A5: Depth + attention"),
    "A6_depth_ffn": dict(selector="predictor", ffn=True,
                         label="A6: Depth + FFN"),
    "A7_no_monitor": dict(selector="predictor", attention=True, ffn=True,
                          label="A7: Full MSA without stability monitor"),
    "A8_full": dict(selector="predictor", attention=True, ffn=True, monitor=True,
                    label="A8: Full MSA"),

    # ---- Reference points ------------------------------------------
    "static_shallow": dict(selector="static", static_config="shallow",
                           label="Static 4-layer reference"),
    "static_medium": dict(selector="static", static_config="medium",
                          label="Static 8-layer reference"),
}

BASELINES = ["static", "depth_adaptive", "routing", "msa"]
ABLATIONS = ["A1_static", "A2_analyzer", "A3_predictor", "A4_depth",
             "A5_depth_attention", "A6_depth_ffn", "A7_no_monitor", "A8_full"]


class MSASystem:
    """Shared, loaded-once components for every variant."""

    def __init__(self, model_path=None, confidence_threshold=0.5, fallback="deep",
                 seed=42):
        torch.manual_seed(seed)

        model, self.tokenizer = load_model()
        self.adaptive_model = AdaptiveGPT2(model).eval()

        self.predictor = (
            PerformancePredictor() if model_path is None
            else PerformancePredictor(model_path)
        )
        self.analyzer = TaskAnalyzer(self.predictor.metadata.get("feature_set", "v1"))
        self.analyzer_all = TaskAnalyzer("v2")

        self.scorer = ComplexityScorer()
        self.selector = ConfigurationSelector()
        self.policy = ConfigurationPolicy(confidence_threshold, fallback)
        self.seed = seed


def to_records(dataset):
    """Accept plain prompts or {question, answer} records."""
    records = []
    for item in dataset:
        if isinstance(item, str):
            records.append({"question": item, "answer": None})
        else:
            records.append(dict(item))
    return records


def run_variant(system, name, dataset, generate=False, max_new_tokens=24,
                monitor_kwargs=None):
    spec = VARIANTS[name]
    records = to_records(dataset)

    controller = DynamicArchitectureController(
        system.adaptive_model,
        system.tokenizer,
        routing=make_routing_gate(seed=system.seed) if spec.get("routing") else None,
        attention=spec.get("attention", False),
        ffn=spec.get("ffn", False),
    )

    monitor = None
    if spec.get("monitor"):
        monitor = StabilityMonitor(**(monitor_kwargs or {}))

    # A7 observes instability without acting, so the two can be compared.
    elif name == "A7_no_monitor":
        monitor = StabilityMonitor(enabled=False, **(monitor_kwargs or {}))

    results = []

    for record in records:
        question = record["question"]

        # --- selection ------------------------------------------------
        decision = {"fallback_applied": False, "confidence": float("nan"),
                    "probabilities": {}}

        if spec["selector"] == "static":
            configuration = spec["static_config"]
        elif spec["selector"] == "rule":
            complexity = system.scorer.calculate(system.analyzer_all.analyze_all(question))
            configuration = system.selector.select(complexity)
        else:
            prediction = system.predictor.predict(system.analyzer.analyze(question))
            decision = system.policy.decide(prediction)
            configuration = decision["configuration"]

        proposed = configuration
        stability = None

        if monitor is not None:
            stability = monitor.observe(configuration)
            configuration = stability["configuration"]

        # --- execution ------------------------------------------------
        input_ids, label_ids = build_inputs(system.tokenizer, record)

        outcome = controller.run(question, configuration, input_ids=input_ids)
        entry = {key: value for key, value in outcome.items() if key != "logits"}

        entry.update({
            "question": question,
            "proposed_configuration": proposed,
            "fallback_applied": decision["fallback_applied"],
            "confidence": decision["confidence"],
            "rolled_back": bool(stability and stability["rolled_back"]),
            "stability_status": stability["status"] if stability else None,
            "volatility": stability["volatility"] if stability else None,
            # Phase 3 requires the predicted probabilities on every record.
            **{f"p_{name}": decision["probabilities"].get(name, float("nan"))
               for name in CONFIG_NAMES},
        })

        # --- downstream metrics ---------------------------------------
        if record.get("answer") is not None:
            scored = controller.run(question, configuration, input_ids=input_ids,
                                    label_ids=label_ids)
            entry["answer_loss"] = scored["loss"]

            if generate:
                generated = controller.generate(
                    f"Question: {question}\nAnswer:", configuration,
                    max_new_tokens=max_new_tokens,
                )
                entry["generated"] = generated
                entry["correct"] = exact_match(generated, record["answer"])

        results.append(entry)

    metrics = aggregate(results, monitor.summary() if monitor else None)
    metrics["variant"] = name
    metrics["label"] = spec["label"]

    return metrics, results


def build_inputs(tokenizer, record):
    """Tokenise the prompt and, when an answer exists, mask it for scoring."""
    question = record["question"]

    if record.get("answer") is None:
        ids = tokenizer(question, return_tensors="pt")["input_ids"]
        return ids, ids

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(f"{prompt} {record['answer']}", return_tensors="pt")["input_ids"]

    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    return full_ids, labels


def run_many(system, names, dataset, **kwargs):
    rows, per_sample = [], {}

    for name in names:
        metrics, records = run_variant(system, name, dataset, **kwargs)
        rows.append(metrics)
        per_sample[name] = records

        print(
            f"  {name:20s} loss={metrics['loss']:.4f} "
            f"depth={metrics['average_depth']:.2f} "
            f"reduction={metrics['layer_reduction']*100:5.1f}% "
            f"flops={metrics['relative_flops']:.3f} "
            f"latency={metrics['latency_mean']*1000:6.2f}ms",
            flush=True,
        )

    return rows, per_sample


if __name__ == "__main__":
    import argparse

    import pandas as pd

    from data.load_data import load_dataset_by_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=BASELINES)
    parser.add_argument("--dataset", default="short_qa")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset_by_name(args.dataset, limit=args.limit)

    system = MSASystem()
    print(f"\nDataset: {args.dataset} ({len(dataset)} samples)\n")

    rows, _ = run_many(system, args.variants, dataset, generate=args.generate)

    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))

    assert all(row["samples"] == len(dataset) for row in rows)
    assert all(config in CONFIG_NAMES for config in ("shallow", "medium", "deep"))
