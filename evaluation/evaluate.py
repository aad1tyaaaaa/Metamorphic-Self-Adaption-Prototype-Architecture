"""Single evaluation entry point (Phases 11, 12, 14, 15, 16).

    python -m evaluation.evaluate --suite all
    python -m evaluation.evaluate --suite baselines --dataset short_qa

Suites
    baselines   Baselines A-E on one dataset
    ablations   A1-A8 on one dataset
    datasets    Static / depth-adaptive / full MSA on each downstream dataset
    stability   Controlled instability sequence, monitor on vs off
    dense       Baseline D parameter and latency reference
    all         Everything above
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from data.load_data import load_dataset_by_name
from evaluation.harness import ABLATIONS, BASELINES, MSASystem, run_many, run_variant
from monitor.stability_monitor import StabilityMonitor

RESULTS = "results"


def save(frame, name):
    path = f"{RESULTS}/{name}.csv"
    frame.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def suite_baselines(system, dataset, generate=False):
    print("\n=== BASELINES (Phase 12) ===")
    rows, per_sample = run_many(system, BASELINES, dataset, generate=generate)

    frame = pd.DataFrame(rows)
    save(frame, "baselines")

    # Per-inference decision log (Phase 3 requirement).
    log = pd.DataFrame([
        {"variant": variant, **{k: v for k, v in record.items() if k != "generated"}}
        for variant, records in per_sample.items() for record in records
    ])
    save(log, "inference_log")

    return frame


def suite_ablations(system, dataset, generate=False):
    print("\n=== ABLATIONS (Phase 15) ===")
    rows, _ = run_many(system, ABLATIONS, dataset, generate=generate)

    frame = pd.DataFrame(rows)
    save(frame, "ablations")
    return frame


def suite_datasets(system, limits, generate=True, max_new_tokens=24):
    print("\n=== DATASET EVALUATION (Phase 14) ===")
    variants = ["static", "depth_adaptive", "msa"]
    rows = []

    for name, limit in limits.items():
        try:
            dataset = load_dataset_by_name(name, limit=limit)
        except Exception as error:
            print(f"  skipping {name}: {type(error).__name__}: {error}")
            continue

        print(f"\n{name} ({len(dataset)} samples, generation={'on' if generate else 'off'})")

        for variant in variants:
            started = time.perf_counter()
            metrics, _ = run_variant(system, variant, dataset, generate=generate,
                                     max_new_tokens=max_new_tokens)
            metrics["dataset"] = name
            metrics["wall_clock_seconds"] = time.perf_counter() - started
            rows.append(metrics)

            print(
                f"  {variant:16s} answer_ppl={metrics.get('answer_perplexity', float('nan')):9.2f} "
                f"accuracy={metrics.get('accuracy', float('nan')):.3f} "
                f"depth={metrics['average_depth']:.2f} "
                f"flops={metrics['relative_flops']:.3f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    if len(frame):
        save(frame, "dataset_evaluation")
    return frame


def volatile_sequence(system, length=60, seed=42):
    """Build a sequence that genuinely forces configuration switching.

    Picking "easy" and "hard" prompts by hand does not work: the predictor maps
    most of them to `medium` regardless, so the sequence barely switches. Instead
    the candidate pool is bucketed by what the predictor *actually* selects, and
    the sequence alternates between distinct buckets.
    """
    from data.calibration_dataset import load

    rng = np.random.default_rng(seed)
    candidates, _ = load()

    buckets = {}
    for text in candidates:
        configuration = system.predictor.predict(
            system.analyzer.analyze(text)
        )["configuration"]
        buckets.setdefault(configuration, []).append(text)

    populated = [name for name in ("shallow", "medium", "deep") if buckets.get(name)]

    if len(populated) < 2:
        raise RuntimeError(
            "cannot build a volatile sequence: the predictor selects only "
            f"{populated} across {len(candidates)} prompts, so no prompt "
            "ordering can induce configuration switching"
        )

    print(f"  volatile pool: " + ", ".join(
        f"{name}={len(buckets[name])}" for name in populated))

    sequence = []
    for index in range(length):
        pool = buckets[populated[index % len(populated)]]
        sequence.append(pool[rng.integers(len(pool))])

    return sequence


def suite_stability(system, length=60, seed=42):
    print("\n=== STABILITY EXPERIMENT (Phase 16) ===")
    sequence = volatile_sequence(system, length=length, seed=seed)

    rows, log = [], []
    for variant in ("A7_no_monitor", "A8_full"):
        metrics, records = run_variant(system, variant, sequence)
        metrics["rollbacks_observed"] = sum(r["rolled_back"] for r in records)
        rows.append(metrics)
        log.extend({"variant": variant, **record} for record in records)

        print(
            f"  {variant:16s} switches={metrics.get('switches', 0):3d} "
            f"volatility={metrics.get('volatility', 0):.2f} "
            f"rollbacks={metrics.get('rollback_count', 0):3d} "
            f"rollback_rate={metrics.get('rollback_rate', 0):.3f} "
            f"loss={metrics['loss']:.4f} depth={metrics['average_depth']:.2f}"
        )

    frame = pd.DataFrame(rows)
    save(frame, "stability_experiment")
    save(pd.DataFrame(log), "stability_log")

    # Synthetic control: a forced alternation must trigger rollback.
    forced = StabilityMonitor(window=6)
    for config in ["medium"] * 6 + ["deep", "shallow"] * 4:
        forced.observe(config)

    print(f"\n  synthetic control -> {forced.summary()['rollback_count']} rollbacks "
          f"from {forced.summary()['switches']} switches")
    assert forced.rollback_count > 0, "rollback never fired on the synthetic control"

    return frame


def suite_dense(seq_len=64, runs=20, seed=42):
    """Baseline D: architecture-family reference, untrained."""
    print("\n=== DENSE REFERENCE (Baseline D) ===")
    from evaluation.baselines import DenseReference

    model = DenseReference(seed=seed).eval()
    ids = torch.randint(0, 50000, (1, seq_len),
                        generator=torch.Generator().manual_seed(seed))

    with torch.no_grad():
        for _ in range(5):
            model(ids)

        samples = []
        for _ in range(runs):
            started = time.perf_counter()
            model(ids)
            samples.append(time.perf_counter() - started)

    samples = np.array(samples)

    row = {
        "model": "DenseReference (RMSNorm + RoPE + SwiGLU)",
        "parameters": model.parameter_count(),
        "trained": False,
        "quality_metric": "N/A (randomly initialised)",
        "sequence_length": seq_len,
        "runs": runs,
        "latency_mean": float(samples.mean()),
        "latency_std": float(samples.std(ddof=1)),
        **model.config,
    }

    print(json.dumps({k: v for k, v in row.items()}, indent=2, default=str))

    frame = pd.DataFrame([row])
    save(frame, "dense_reference")
    return frame


def main(args):
    torch.manual_seed(args.seed)
    system = MSASystem(confidence_threshold=args.confidence_threshold, seed=args.seed)

    dataset = load_dataset_by_name(args.dataset, limit=args.limit)
    print(f"\nEvaluation dataset: {args.dataset} ({len(dataset)} samples)")
    print(f"Predictor: {system.predictor.metadata.get('feature_set', 'v1')} features, "
          f"tolerance {system.predictor.metadata.get('tolerance')}")

    suite = args.suite

    if suite in ("baselines", "all"):
        suite_baselines(system, dataset, generate=args.generate)

    if suite in ("ablations", "all"):
        suite_ablations(system, dataset, generate=args.generate)

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
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--generate", action="store_true",
                        help="also generate text during baseline/ablation suites")
    parser.add_argument("--no-generate", action="store_true",
                        help="skip generation in the dataset suite (much faster)")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
