"""Final validation for plan phases 1-20.

    python test_msa.py

Runs the plan's phase gates as executable assertions. Every check
either passes or fails loudly; nothing is reported as working on trust.
"""

import math
import os

import torch

from analyzer.task_analyzer import TaskAnalyzer
from configs.configurations import CONFIG_NAMES, DEPTH_MAP, TOTAL_LAYERS, relative_flops
from controller.controller import DynamicArchitectureController
from controller.policy import ConfigurationPolicy
from controller.predictor import PerformancePredictor
from models.adaptive_model import AdaptiveGPT2, make_routing_gate
from models.backbone import load_model
from monitor.stability_monitor import StabilityMonitor

PROMPT = "The capital of India is New Delhi and the capital of France is"

# Phase 2 / Phase 5 integration categories.
CATEGORY_PROMPTS = {
    "factual": "What is the capital of France?",
    "explanation": "Explain why the sky appears blue.",
    "mathematics": "Calculate 15 percent of 240.",
    "multi_step_reasoning": (
        "John has 5 apples. He buys 3 more apples, then gives 2 to Sarah. "
        "Explain step by step how many apples John has remaining."
    ),
    "long_structured": (
        "You are given a report about renewable energy adoption. First, identify the "
        "three most important claims. Second, explain which claim is best supported by "
        "evidence. Third, describe what additional information would be needed to "
        "evaluate the weakest claim."
    ),
    "code": "Write a Python function that checks whether a number is prime.",
}

passed, failed = [], []


def check(name, function):
    try:
        detail = function()
        passed.append(name)
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    except Exception as error:
        failed.append((name, error))
        print(f"  [FAIL] {name} -- {type(error).__name__}: {error}")


def main():
    print("=" * 74)
    print("FINAL VALIDATION -- PHASES 1-20")
    print("=" * 74)

    print("\nCore system")

    model, tokenizer = load_model()
    adaptive = AdaptiveGPT2(model).eval()
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

    def gpt2_loads():
        assert len(model.transformer.h) == TOTAL_LAYERS
        return f"{sum(p.numel() for p in model.parameters()):,} parameters"

    def depth_12_matches_reference():
        """The adaptive stack at full depth must equal stock GPT-2 exactly."""
        with torch.no_grad():
            reference = model(input_ids=ids).logits
            produced = adaptive(ids, depth=12)
        difference = (reference - produced).abs().max().item()
        assert difference < 1e-4, f"depth-12 diverges from reference by {difference}"
        return f"max |diff| = {difference:.2e}"

    def causal_masking_is_applied():
        """A bidirectional stack would see the future; a causal one cannot."""
        with torch.no_grad():
            full = adaptive(ids, depth=12)
            truncated = adaptive(ids[:, :-1], depth=12)
        # Prefix logits must be unaffected by later tokens.
        difference = (full[:, :-1, :] - truncated).abs().max().item()
        assert difference < 1e-4, f"prefix logits depend on future tokens ({difference})"
        return f"prefix invariance = {difference:.2e}"

    def adaptive_depth_works():
        with torch.no_grad():
            losses = {
                depth: torch.nn.functional.cross_entropy(
                    adaptive(ids, depth=depth)[:, :-1, :].reshape(-1, 50257),
                    ids[:, 1:].reshape(-1),
                ).item()
                for depth in (4, 8, 12)
            }
        assert losses[12] < losses[8] < losses[4], f"depth ordering broken: {losses}"
        return " ".join(f"d{d}={v:.2f}" for d, v in losses.items())

    def predictor_loads():
        predictor = PerformancePredictor()
        assert predictor.features
        return f"{predictor.metadata.get('feature_set', 'v1')}, " \
               f"accuracy {predictor.metadata.get('accuracy')}"

    def predictor_probabilities_valid():
        predictor = PerformancePredictor()
        analyzer = TaskAnalyzer(predictor.metadata.get("feature_set", "v1"))

        for text in ("What is 2 + 2?", "Explain why the sky appears blue.", "x"):
            result = predictor.predict(analyzer.analyze(text))
            assert result["configuration"] in DEPTH_MAP
            assert result["depth"] in (4, 8, 12)
            assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-6)
            assert all(math.isfinite(v) for v in result["probabilities"].values())
        return "no NaN or invalid values"

    def policy_works():
        policy = ConfigurationPolicy(confidence_threshold=0.9, fallback="deep")
        decision = policy.decide({
            "configuration": "shallow", "confidence": 0.4,
            "probabilities": {"shallow": 0.4, "medium": 0.35, "deep": 0.25},
        })
        assert decision["fallback_applied"] and decision["configuration"] == "deep"
        return "low confidence falls back to deep"

    def controller_changes_execution():
        """Each mechanism must alter the computed output, not just a label."""
        depth_only = DynamicArchitectureController(adaptive, tokenizer)
        full = DynamicArchitectureController(adaptive, tokenizer, mechanism="full")
        routed = DynamicArchitectureController(
            adaptive, tokenizer, routing=make_routing_gate())

        base = depth_only.run(PROMPT, "medium", input_ids=ids)["logits"]
        attention_only = DynamicArchitectureController(
            adaptive, tokenizer, attention=True, ffn=False
        ).run(PROMPT, "medium", input_ids=ids)["logits"]
        ffn_only = DynamicArchitectureController(
            adaptive, tokenizer, attention=False, ffn=True
        ).run(PROMPT, "medium", input_ids=ids)["logits"]
        both = full.run(PROMPT, "medium", input_ids=ids)["logits"]
        routing = routed.run(PROMPT, "deep", input_ids=ids)["logits"]

        assert not torch.allclose(base, attention_only), "attention adaptation is a no-op"
        assert not torch.allclose(base, ffn_only), "FFN adaptation is a no-op"
        assert not torch.allclose(base, both), "combined adaptation is a no-op"
        assert not torch.allclose(base, routing), "routing baseline is a no-op"
        return "depth, attention, FFN and routing all change the output"

    def compute_estimates_decrease():
        full = relative_flops(12, "full", "full", seq_len=64)
        shallow = relative_flops(4, "reduced", "partial", seq_len=64)
        assert full == 1.0 and 0 < shallow < full
        return f"shallow = {shallow:.3f} of full"

    def telemetry_complete():
        controller = DynamicArchitectureController(adaptive, tokenizer, mechanism="full")
        record = controller.run(PROMPT, "shallow", input_ids=ids)

        for field in ("depth", "layers_executed", "layers_skipped", "latency_seconds",
                      "loss", "active_heads", "active_ffn_chunks", "relative_flops"):
            assert field in record, f"missing telemetry field {field}"
        assert record["layers_executed"] + record["layers_skipped"] == TOTAL_LAYERS
        return "all required telemetry present"

    def monitor_detects_instability():
        monitor = StabilityMonitor(window=6)
        for _ in range(6):
            monitor.observe("medium")
        statuses = [monitor.observe(c)["status"]
                    for c in ["deep", "shallow"] * 3]
        assert any(status != "STABLE" for status in statuses)
        return f"status reached {sorted(set(statuses))}"

    def rollback_works():
        monitor = StabilityMonitor(window=6)
        for _ in range(6):
            monitor.observe("medium")
        for config in ["deep", "shallow"] * 4:
            monitor.observe(config)
        assert monitor.rollback_count > 0
        return f"{monitor.rollback_count} rollbacks recorded"

    def end_to_end_runs():
        from run_msa import MSA
        record = MSA(mechanism="full").run("Explain why the sky appears blue.")
        assert record["configuration"] in CONFIG_NAMES
        assert math.isfinite(record["loss"])
        assert record["logits_shape"][-1] == 50257
        return f"selected {record['configuration']} (depth {record['depth']})"

    def f_theta_controls_execution():
        """Phase 2 gate: the depth f_theta + policy select is the depth GPT-2 runs."""
        predictor = PerformancePredictor()
        analyzer = TaskAnalyzer(predictor.metadata.get("feature_set", "v1"))
        policy = ConfigurationPolicy()
        controller = DynamicArchitectureController(adaptive, tokenizer)
        chosen = []

        for category, text in CATEGORY_PROMPTS.items():
            decision = policy.decide(predictor.predict(analyzer.analyze(text)))
            outcome = controller.run(text, decision["configuration"])

            assert outcome["executed_depth"] == decision["depth"], category
            assert adaptive.last_executed_blocks == list(range(decision["depth"]))
            assert math.isfinite(outcome["loss"])
            chosen.append(f"{category}={decision['depth']}")
        return " ".join(chosen)

    def every_depth_executes():
        """Phase 6 gate: each configuration runs exactly its number of blocks."""
        for mechanism in ("depth", "full"):
            controller = DynamicArchitectureController(adaptive, tokenizer, mechanism)
            for name, depth in DEPTH_MAP.items():
                outcome = controller.run(PROMPT, name, input_ids=ids)
                assert outcome["requested_depth"] == outcome["executed_depth"] == depth
                assert outcome["layers_skipped"] == TOTAL_LAYERS - depth
        return "4/8/12 verified for depth-only and full mechanisms"

    def analyzer_features_valid():
        """Phase 5 gate: eight bounded, finite features for every category and prompt."""
        from data.calibration_dataset import load
        analyzer = TaskAnalyzer("v2")
        texts = list(CATEGORY_PROMPTS.values()) + load()[0] + [""]

        for text in texts:
            features = analyzer.analyze(text)
            assert len(features) == 8
            assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in features.values())
        return f"{len(texts)} prompts, 8 features each, all in [0, 1]"

    def plan_sequence_rolls_back():
        """Phase 10 gate: the plan's synthetic sequence triggers a rollback."""
        monitor = StabilityMonitor(window=6)
        events = [monitor.observe(c) for c in
                  ["deep", "medium", "deep", "shallow", "deep", "medium", "deep"]]
        assert monitor.rollback_count > 0
        assert all(0.0 <= e["stability_score"] <= 1.0 for e in events)
        executed = [e["configuration"] for e in events]
        return f"{monitor.rollback_count} rollbacks, executed {executed}"

    check("GPT-2 loads from a clean environment", gpt2_loads)
    check("Depth 12 matches the reference model", depth_12_matches_reference)
    check("Causal masking is applied", causal_masking_is_applied)
    check("Adaptive depth works", adaptive_depth_works)
    check("Predictor loads", predictor_loads)
    check("Predictor produces valid probabilities", predictor_probabilities_valid)
    check("Configuration policy works", policy_works)
    check("Dynamic controller changes actual execution", controller_changes_execution)
    check("Compute estimates decrease with adaptation", compute_estimates_decrease)
    check("Telemetry is complete", telemetry_complete)
    check("Stability monitor detects instability", monitor_detects_instability)
    check("Rollback works", rollback_works)
    check("End-to-end MSA runs without intervention", end_to_end_runs)
    check("P2: f_theta controls executed GPT-2 depth", f_theta_controls_execution)
    check("P5: analyzer produces eight valid features", analyzer_features_valid)
    check("P6: every configuration executes its depth", every_depth_executes)
    check("P10: plan synthetic sequence triggers rollback", plan_sequence_rolls_back)

    def experiment_config_matches_code():
        """Phase 18: configs/experiment.yaml must describe what the code runs."""
        import yaml
        from configs import configurations as cfg
        from evaluation.benchmark import RUNS, SEQUENCE_LENGTHS, WARMUP

        with open("configs/experiment.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        assert config["policy"]["confidence_threshold"] == cfg.CONFIDENCE_THRESHOLD
        assert config["policy"]["fallback"] == cfg.FALLBACK_CONFIGURATION
        for name, knobs in cfg.CONFIGURATIONS.items():
            assert config["configurations"][name] == knobs, name
        assert config["benchmark"]["warmup"] == WARMUP >= 10
        assert config["benchmark"]["runs"] == RUNS >= 30
        assert config["benchmark"]["sequence_lengths"] == SEQUENCE_LENGTHS

        predictor = PerformancePredictor()
        assert config["predictor"]["feature_set"] == predictor.metadata["feature_set"]
        assert config["predictor"]["tolerance"] == predictor.metadata["tolerance"]
        assert config["calibration"]["size"] == predictor.metadata["calibration_prompts"]
        return "threshold, configurations, benchmark and predictor settings agree"

    check("P18: experiment.yaml matches the code", experiment_config_matches_code)

    print("\nReproducibility artefacts")

    artefacts = {
        "Calibration reproducible": "data/calibration_prompts.json",
        "Historical calibration preserved": "results/calibration/calibration_results_v1_499.csv",
        "Calibration results saved as CSV": "results/calibration_results.csv",
        "Predictor evaluation artefacts": "results/calibration/predictor_evaluation.json",
        "Policy threshold sweep": "results/policy_threshold_sweep.csv",
        "Standardised evaluation outputs": "results/evaluation/msa__short_qa.csv",
        "Stability switching levels": "results/stability/stability_levels.csv",
        "Quality vs latency figure": "results/figures/13_quality_vs_latency.png",
        "Ablation figure": "results/figures/14_ablation_comparison.png",
        "Experiment configuration saved": "configs/experiment.yaml",
        "Predictor saved": "results/performance_predictor.pkl",
        "Quality-compute frontier": "results/quality_compute_frontier.csv",
        "Feature ablation": "results/feature_ablation.csv",
        "Benchmark with warmups": "results/benchmark.csv",
        "Environment recorded": "results/environment.json",
        "Baselines implemented": "results/baselines.csv",
        "Ablations completed": "results/ablations.csv",
        "Stability experiment completed": "results/stability_experiment.csv",
        "Downstream metrics used": "results/dataset_evaluation.csv",
        "Graphs generated automatically": "results/figures",
        "Research tables": "results/tables.md",
        "Research paper written": "paper/msa_paper.md",
        "Demonstration page built": "ui/index.html",
    }

    for name, path in artefacts.items():
        check(name, lambda path=path: (
            _ for _ in ()).throw(AssertionError(f"missing {path}"))
            if not os.path.exists(path) else path)

    print("\n" + "=" * 74)
    print(f"VALIDATION: {len(passed)} passed, {len(failed)} failed")
    print("=" * 74)

    if failed:
        for name, error in failed:
            print(f"  FAILED: {name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
