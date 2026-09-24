"""Single evaluation entry point (Phases 11, 12, 14, 15, 16).

    python -m evaluation.evaluate --suite all
    python -m evaluation.evaluate --suite baselines --dataset short_qa

Suites
    baselines   Baselines A-E on one dataset (D = pretrained SmolLM-135M)
    ablations   A1-A8 on one dataset
    datasets    Static / depth-adaptive / full MSA / dense reference per dataset
    stability   Low / medium / high switching sequences, monitor on vs off
    dense       Baseline D parameter count and controlled latency
    all         Everything above

Every GPT-2 run goes through evaluation/runner.py, which also writes the
standard per-sample schema to results/evaluation/<method>__<dataset>.csv.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from configs.configurations import CONFIDENCE_THRESHOLD
from data.load_data import load_dataset_by_name
from evaluation.harness import ABLATIONS, BASELINES, MSASystem, run_variant
from evaluation.runner import DENSE, evaluate
from monitor.stability_monitor import StabilityMonitor

RESULTS = "results"
STABILITY_DIR = f"{RESULTS}/stability"

# Probability that the next prompt is drawn from a different decision bucket.
SWITCHING_LEVELS = {"low": 0.1, "medium": 0.4, "high": 0.8}


def save(frame, name):
    path = f"{RESULTS}/{name}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def run_methods(system, methods, dataset_name, dataset, **kwargs):
    rows, per_sample = [], {}

    for method in methods:
        started = time.perf_counter()
        try:
            metrics, frame = evaluate(system, method, dataset_name, dataset, **kwargs)
        except OSError as error:
            # Only Baseline D needs a download; report it instead of hiding it.
            print(f"  {method}: FAILED to load ({error}); not included")
            continue

        metrics["wall_clock_seconds"] = time.perf_counter() - started
        rows.append(metrics)
        per_sample[method] = frame
        print(
            f"  {method:20s} loss={metrics['loss']:.4f} "
            f"acc={metrics.get('accuracy', float('nan')):.3f} "
            f"depth={metrics['average_depth']:.2f} "
            f"reduction={metrics['layer_reduction']*100:5.1f}% "
            f"flops={metrics['relative_flops']:.3f} "
            f"latency={metrics['latency_mean']*1000:6.2f}ms",
            flush=True,
        )

    return rows, per_sample


def suite_baselines(system, dataset_name, dataset, generate=False):
    print("\n=== BASELINES (Phase 12) ===")
    rows, per_sample = run_methods(system, BASELINES + [DENSE], dataset_name, dataset,
                                   generate=generate)

    frame = pd.DataFrame(rows)
    save(frame, "baselines")

    # Per-inference decision log (Phase 3 requirement), GPT-2 variants only.
    log = pd.concat([
        f.drop(columns=["generated"], errors="ignore").assign(variant=m)
        for m, f in per_sample.items() if m != DENSE
    ], ignore_index=True)
    log["latency_seconds"] = log["latency"]
    save(log, "inference_log")

    return frame


def suite_ablations(system, dataset_name, dataset, generate=False):
    print("\n=== ABLATIONS (Phase 15) ===")
    rows, _ = run_methods(system, ABLATIONS, dataset_name, dataset, generate=generate)

    frame = pd.DataFrame(rows)
    save(frame, "ablations")
    return frame


def suite_datasets(system, limits, generate=True, max_new_tokens=24):
    print("\n=== DATASET EVALUATION (Phase 14) ===")
    variants = ["static", "depth_adaptive", "msa", DENSE]
    rows = []

    for name, limit in limits.items():
        try:
            dataset = load_dataset_by_name(name, limit=limit)
        except Exception as error:
            print(f"  skipping {name}: {type(error).__name__}: {error}")
            continue

        print(f"\n{name} ({len(dataset)} samples, generation={'on' if generate else 'off'})")

        rows.extend(run_methods(system, variants, name, dataset, generate=generate,
                                max_new_tokens=max_new_tokens)[0])

    frame = pd.DataFrame(rows)
    if len(frame):
        save(frame, "dataset_evaluation")
    return frame


def decision_buckets(system):
    """Calibration prompts grouped by the configuration the policy actually selects.

    Picking "easy" and "hard" prompts by hand does not work: the predictor maps
    most of them to one configuration regardless, so the sequence barely switches.
    """
    from data.calibration_dataset import load

    candidates, _ = load()
    buckets = {}
    for text in candidates:
        decision = system.policy.decide(system.predictor.predict(system.analyzer.analyze(text)))
        buckets.setdefault(decision["configuration"], []).append(text)

    populated = [name for name in ("shallow", "medium", "deep") if buckets.get(name)]
    if len(populated) < 2:
        raise RuntimeError(
            "cannot build a switching sequence: the policy selects only "
            f"{populated} across {len(candidates)} prompts"
        )
    return {name: buckets[name] for name in populated}


def switching_sequence(buckets, switch_probability, length=60, seed=42):
    """Markov sequence over decision buckets with a controlled switch rate."""
    rng = np.random.default_rng(seed)
    names = list(buckets)
    current = names[0]
    sequence = []

    for index in range(length):
        if index and rng.random() < switch_probability:
            current = rng.choice([n for n in names if n != current])
        pool = buckets[current]
        sequence.append(pool[rng.integers(len(pool))])

    return sequence


def executed_volatility(configurations, window=6):
    """Mean V(t) of the *executed* configurations, same definition as the monitor."""
    monitor = StabilityMonitor(window=window, enabled=False)
    for configuration in configurations:
        monitor.observe(configuration)
    return monitor.summary()["volatility_mean"]


def suite_stability(system, length=60, seed=42):
    print("\n=== STABILITY EXPERIMENTS (Phase 16) ===")
    buckets = decision_buckets(system)
    print("  decision buckets: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    rows, logs = [], []

    for level, probability in SWITCHING_LEVELS.items():
        sequence = switching_sequence(buckets, probability, length=length, seed=seed)
        runs = {}

        for variant in ("A7_no_monitor", "A8_full"):
            metrics, records = run_variant(system, variant, sequence)
            runs[variant] = records

            executed = [r["configuration"] for r in records]
            metrics.update({
                "level": level,
                "switch_probability": probability,
                "monitor": variant == "A8_full",
                "proposed_switches": sum(
                    a != b for a, b in zip([r["proposed_configuration"] for r in records],
                                           [r["proposed_configuration"] for r in records][1:])),
                "executed_switches": sum(a != b for a, b in zip(executed, executed[1:])),
                "executed_volatility": executed_volatility(executed),
                "rollbacks_observed": sum(r["rolled_back"] for r in records),
            })
            metrics["stability_score"] = 1 - metrics["executed_volatility"]
            rows.append(metrics)

            logs.extend({"level": level, "variant": variant, "step": step, **record}
                        for step, record in enumerate(records))

        # Quality after rollback: the executed (rolled-back) configuration vs what
        # the unmonitored run executed on the same input at the same step.
        rolled = [i for i, r in enumerate(runs["A8_full"]) if r["rolled_back"]]
        monitored = rows[-1]
        monitored["loss_after_rollback"] = (
            float(np.mean([runs["A8_full"][i]["loss"] for i in rolled])) if rolled else np.nan)
        monitored["loss_without_rollback"] = (
            float(np.mean([runs["A7_no_monitor"][i]["loss"] for i in rolled])) if rolled else np.nan)

        for row in rows[-2:]:
            print(
                f"  {level:6s} {row['variant']:14s} proposed_sw={row['proposed_switches']:3d} "
                f"executed_sw={row['executed_switches']:3d} "
                f"V={row['executed_volatility']:.3f} rollbacks={row['rollbacks_observed']:3d} "
                f"loss={row['loss']:.4f} latency={row['latency_mean']*1000:.1f}ms"
            )
        if rolled:
            print(f"         rolled-back steps: loss {monitored['loss_after_rollback']:.4f} "
                  f"vs {monitored['loss_without_rollback']:.4f} without rollback")

    frame = pd.DataFrame(rows)
    log = pd.DataFrame(logs)

    save(frame, "stability/stability_levels")
    for level in SWITCHING_LEVELS:
        save(log[log["level"] == level], f"stability/stability_log_{level}")
    save(frame, "stability_experiment")
    save(log, "stability_log")

    # The monitor must never increase executed switching, and must act at high switching.
    for level in SWITCHING_LEVELS:
        pair = frame[frame["level"] == level].set_index("variant")
        assert pair.loc["A8_full", "executed_switches"] <= pair.loc["A7_no_monitor", "executed_switches"]
    high = frame[(frame["level"] == "high") & frame["monitor"]].iloc[0]
    assert high["rollbacks_observed"] > 0, "rollback never fired at high switching"

    # Synthetic control from the plan: a forced alternation must trigger rollback.
    forced = StabilityMonitor(window=6)
    for config in ["deep", "medium", "deep", "shallow", "deep", "medium", "deep"]:
        forced.observe(config)
    print(f"\n  synthetic control -> {forced.rollback_count} rollbacks "
          f"from {forced.summary()['switches']} switches")
    assert forced.rollback_count > 0, "rollback never fired on the synthetic control"

    return frame


def suite_dense(seq_len=64, warmup=10, runs=30, seed=42):
    """Baseline D: parameter count and controlled latency for both dense references."""
    print("\n=== DENSE REFERENCE (Baseline D) ===")
    from evaluation.baselines import DenseReference
    from evaluation.benchmark import THREADS
    from evaluation.runner import dense_model

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(THREADS)

    def timed(forward):
        with torch.no_grad():
            for _ in range(warmup):
                forward()
            samples = []
            for _ in range(runs):
                started = time.perf_counter()
                forward()
                samples.append(time.perf_counter() - started)
        return np.array(samples)

    rows = []

    untrained = DenseReference(seed=seed).eval()
    ids = torch.randint(0, 50000, (1, seq_len),
                        generator=torch.Generator().manual_seed(seed))
    samples = timed(lambda: untrained(ids))
    rows.append({
        "model": "DenseReference (RMSNorm + RoPE + SwiGLU), GPT-2-matched dims",
        "parameters": untrained.parameter_count(),
        "trained": False,
        "quality_metric": "N/A (randomly initialised)",
        **{k: v for k, v in untrained.config.items() if k != "trained"},
    })
    rows[-1].update(latency_mean=float(samples.mean()), latency_std=float(samples.std(ddof=1)),
                    latency_p50=float(np.median(samples)))

    try:
        pretrained = dense_model()
        vocab = pretrained.model.config.vocab_size
        ids = torch.randint(0, vocab, (1, seq_len),
                            generator=torch.Generator().manual_seed(seed))
        samples = timed(lambda: pretrained.model(input_ids=ids, use_cache=False))
        rows.append({
            **pretrained.config,
            "quality_metric": "exact match / answer loss (see dataset_evaluation.csv)",
            "latency_mean": float(samples.mean()),
            "latency_std": float(samples.std(ddof=1)),
            "latency_p50": float(np.median(samples)),
        })
    except OSError as error:
        print(f"  pretrained dense reference unavailable: {error}")

    for row in rows:
        row.update(sequence_length=seq_len, warmup=warmup, runs=runs, torch_threads=THREADS)
        print(json.dumps(row, indent=2, default=str))

    torch.set_num_threads(previous_threads)
    frame = pd.DataFrame(rows)
    save(frame, "dense_reference")
    return frame


def main(args):
    torch.manual_seed(args.seed)
    system = MSASystem(confidence_threshold=args.confidence_threshold, seed=args.seed)

    dataset = load_dataset_by_name(args.dataset, limit=args.limit)
    print(f"\nEvaluation dataset: {args.dataset} ({len(dataset)} samples)")
    print(f"Predictor: {system.predictor.metadata.get('feature_set', 'v1')} features, "
          f"tolerance {system.predictor.metadata.get('tolerance')}, "
          f"policy threshold {args.confidence_threshold}")

    suite = args.suite

    if suite in ("baselines", "all"):
        suite_baselines(system, args.dataset, dataset, generate=args.generate)

    if suite in ("ablations", "all"):
        suite_ablations(system, args.dataset, dataset, generate=args.generate)

    if suite in ("datasets", "all"):
        suite_datasets(
            system,
            {"short_qa": args.dataset_limit, "gsm8k": args.dataset_limit},
            generate=not args.no_generate,
            max_new_tokens=args.max_new_tokens,
        )

    if suite in ("stability", "all"):
        suite_stability(system, length=args.stability_length, seed=args.seed)

    if suite in ("dense", "all"):
        suite_dense(seed=args.seed)

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all",
                        choices=["baselines", "ablations", "datasets", "stability",
                                 "dense", "all"])
    parser.add_argument("--dataset", default="short_qa")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--dataset-limit", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--stability-length", type=int, default=60)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--generate", action="store_true",
                        help="also generate text during baseline/ablation suites")
    parser.add_argument("--no-generate", action="store_true",
                        help="skip generation in the dataset suite (much faster)")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
