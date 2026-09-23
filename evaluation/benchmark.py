"""Reproducible latency benchmarking (Phase 13).

Fixed sequence lengths, warm-up runs, repeated measured runs, mean and standard
deviation. Replaces the single noisy timings taken during calibration.

    python -m evaluation.benchmark [--warmup 10] [--runs 50]
"""

import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch

from configs.configurations import CONFIG_NAMES, relative_flops
from controller.controller import DynamicArchitectureController
from models.adaptive_model import AdaptiveGPT2, make_routing_gate
from models.backbone import load_model

WARMUP = 10
RUNS = 50
SEQUENCE_LENGTHS = [16, 64, 256]

OUTPUT_PATH = "results/benchmark.csv"
ENVIRONMENT_PATH = "results/environment.json"


def environment():
    import sklearn
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
    }


def trimmed_mean(samples, proportion=0.1):
    """Mean after discarding the slowest and fastest `proportion` of runs."""
    ordered = np.sort(samples)
    cut = int(len(ordered) * proportion)
    kept = ordered[cut:len(ordered) - cut] if cut and len(ordered) > 2 * cut else ordered
    return float(kept.mean())


def time_configuration(controller, input_ids, configuration, warmup, runs):
    for _ in range(warmup):
        controller.run("", configuration, input_ids=input_ids)

    samples = [
        controller.run("", configuration, input_ids=input_ids)["latency_seconds"]
        for _ in range(runs)
    ]

    return np.array(samples)


def main(warmup=WARMUP, runs=RUNS, sequence_lengths=None, seed=42,
         output_path=OUTPUT_PATH):
    sequence_lengths = sequence_lengths or SEQUENCE_LENGTHS
    torch.manual_seed(seed)

    model, tokenizer = load_model()
    adaptive_model = AdaptiveGPT2(model).eval()

    variants = {
        "depth_only": DynamicArchitectureController(adaptive_model, tokenizer),
        "depth_attention_ffn": DynamicArchitectureController(
            adaptive_model, tokenizer, mechanism="full"),
        "routing": DynamicArchitectureController(
            adaptive_model, tokenizer, routing=make_routing_gate(seed=seed)),
    }

    info = environment()
    print(json.dumps(info, indent=2))
    print(f"\nwarmup={warmup} runs={runs} sequence_lengths={sequence_lengths}\n")

    rows = []

    for seq_len in sequence_lengths:
        # Fixed synthetic input so length is controlled exactly.
        input_ids = torch.randint(0, 50000, (1, seq_len), generator=
                                  torch.Generator().manual_seed(seed))

        for mechanism, controller in variants.items():
            for configuration in CONFIG_NAMES:
                samples = time_configuration(
                    controller, input_ids, configuration, warmup, runs
                )
                knobs = controller.knobs(configuration)
                telemetry = controller.telemetry(knobs, seq_len)

                trimmed = trimmed_mean(samples)
                median = float(np.percentile(samples, 50))

                rows.append({
                    "mechanism": mechanism,
                    "configuration": configuration,
                    "sequence_length": seq_len,
                    "depth": knobs["depth"],
                    "warmup": warmup,
                    "runs": runs,
                    "latency_mean": float(samples.mean()),
                    "latency_std": float(samples.std(ddof=1)),
                    "latency_trimmed_mean": trimmed,
                    "latency_p50": median,
                    "latency_p95": float(np.percentile(samples, 95)),
                    # A CPU benchmark occasionally catches an OS stall. Flag it
                    # rather than letting one outlier set the headline number.
                    "outlier_suspected": bool(samples.mean() > 1.5 * median),
                    "throughput": float(1.0 / trimmed),
                    "relative_flops": telemetry["relative_flops"],
                    "layer_reduction": telemetry["layer_reduction"],
                    "device": info["device"],
                })

                print(
                    f"{mechanism:20s} {configuration:8s} seq={seq_len:4d} "
                    f"trimmed {trimmed*1000:7.2f} ms  "
                    f"median {median*1000:7.2f} ms  "
                    f"mean {samples.mean()*1000:7.2f} +/- {samples.std(ddof=1)*1000:6.2f} ms "
                    f"flops={telemetry['relative_flops']:.3f}"
                    + ("  [outlier suspected]" if samples.mean() > 1.5 * median else ""),
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)

    with open(ENVIRONMENT_PATH, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)

    print(f"\nSaved: {output_path} and {ENVIRONMENT_PATH}")

    flagged = frame[frame["outlier_suspected"]]
    if len(flagged):
        print(f"
{len(flagged)} cell(s) flagged for an outlying run; use "
              "latency_trimmed_mean or latency_p50 for those:")
        for _, row in flagged.iterrows():
            print(f"  {row['mechanism']} / {row['configuration']} / "
                  f"seq={row['sequence_length']}: mean {row['latency_mean']*1000:.1f} ms "
                  f"vs median {row['latency_p50']*1000:.1f} ms")

    # Sanity: within a mechanism and length, deeper must not be faster.
    for (mechanism, seq_len), group in frame.groupby(["mechanism", "sequence_length"]):
        ordered = group.sort_values("depth")["latency_trimmed_mean"].to_numpy()
        if not np.all(np.diff(ordered) > -ordered[:-1] * 0.25):
            print(f"  note: non-monotonic latency for {mechanism} at seq={seq_len}")

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=None)
    args = parser.parse_args()

    main(warmup=args.warmup, runs=args.runs, sequence_lengths=args.sequence_lengths)
