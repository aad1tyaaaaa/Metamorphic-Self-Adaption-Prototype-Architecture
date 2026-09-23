"""Generate the research figures (Phase 17).

    python -m evaluation.generate_plots

Every figure is regenerated from the CSVs in results/, so nothing is hand-edited.
Figures whose source CSV is missing are skipped with a note.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "results"
FIGURES = "results/figures"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
})

ORDER = ["shallow", "medium", "deep"]


def read(name):
    path = f"{RESULTS}/{name}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def save(fig, name):
    path = f"{FIGURES}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


# ----------------------------------------------------------------------


def depth_vs_loss(calibration):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    for mechanism, group in calibration.groupby("mechanism"):
        stats = group.groupby("depth")["loss"].agg(["mean", "std"])
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o",
                    capsize=3, label=mechanism)

    ax.set(xlabel="Depth (layers executed)", ylabel="Prompt LM loss",
           title="Depth vs loss", xticks=[4, 8, 12])
    ax.legend(title="mechanism")
    return save(fig, "01_depth_vs_loss")


def depth_vs_latency(calibration):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    for mechanism, group in calibration.groupby("mechanism"):
        stats = group.groupby("depth")["latency_seconds"].agg(["mean", "std"])
        ax.errorbar(stats.index, stats["mean"] * 1000, yerr=stats["std"] * 1000,
                    marker="s", capsize=3, label=mechanism)

    ax.set(xlabel="Depth (layers executed)", ylabel="Latency (ms)",
           title="Depth vs latency", xticks=[4, 8, 12])
    ax.legend(title="mechanism")
    return save(fig, "02_depth_vs_latency")


def frontier(frame):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.plot(frame["compute_reduction_percent"], frame["relative_loss_increase"] * 100,
            marker="o", color="#c2410c")

    for _, row in frame.iterrows():
        ax.annotate(f"{row['tolerance']:.2f}",
                    (row["compute_reduction_percent"], row["relative_loss_increase"] * 100),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set(xlabel="Compute reduction (%)", ylabel="Relative loss increase (%)",
           title="Quality-compute frontier (labels = tolerance)")
    return save(fig, "03_quality_compute_frontier")


def configuration_distribution(baselines):
    fig, ax = plt.subplots(figsize=(6, 3.2))

    shares = baselines.set_index("variant")[[f"share_{name}" for name in ORDER]]
    bottom = np.zeros(len(shares))

    for name in ORDER:
        values = shares[f"share_{name}"].to_numpy()
        ax.bar(shares.index, values, bottom=bottom, label=name)
        bottom += values

    ax.set(ylabel="Share of inferences", title="Configuration distribution", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    return save(fig, "04_configuration_distribution")


def average_depth_per_dataset(datasets):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    pivot = datasets.pivot_table(index="dataset", columns="variant",
                                 values="average_depth")
    pivot.plot(kind="bar", ax=ax, rot=0)

    ax.set(ylabel="Average depth", title="Average depth per dataset", ylim=(0, 12.5))
    ax.legend(fontsize=7)
    return save(fig, "05_average_depth_per_dataset")


def layer_reduction(baselines):
    fig, ax = plt.subplots(figsize=(6, 3.2))

    ax.bar(baselines["variant"], baselines["layer_reduction"] * 100, color="#0369a1",
           label="layer reduction")
    ax.bar(baselines["variant"], baselines["compute_reduction"] * 100, color="#f59e0b",
           alpha=0.7, width=0.5, label="FLOP reduction")

    ax.set(ylabel="Reduction (%)", title="Layer and compute reduction")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    return save(fig, "06_layer_reduction")


def latency_distribution(log):
    fig, ax = plt.subplots(figsize=(6, 3.2))

    variants = list(log["variant"].unique())
    data = [log[log["variant"] == variant]["latency_seconds"] * 1000
            for variant in variants]

    ax.boxplot(data, tick_labels=variants, showfliers=False)
    ax.set(ylabel="Latency (ms)", title="Latency distribution per variant")
    ax.tick_params(axis="x", rotation=20)
    return save(fig, "07_latency_distribution")


def confusion(metadata):
    matrix = np.array(metadata["confusion_matrix"])
    labels = metadata["confusion_labels"]

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    image = ax.imshow(matrix, cmap="Blues")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xlabel="Predicted", ylabel="Actual",
           title=f"Predictor confusion (acc {metadata['accuracy']:.2f})")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8)
    return save(fig, "08_predictor_confusion_matrix")


def confidence_distribution(log):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    values = log["confidence"].dropna()
    values = values[np.isfinite(values)]

    if values.empty:
        plt.close(fig)
        return None

    ax.hist(values, bins=20, color="#7c3aed", alpha=0.85)
    ax.axvline(0.5, color="#dc2626", linestyle="--", label="policy threshold")
    ax.set(xlabel="Predictor confidence", ylabel="Count",
           title="Predictor confidence distribution")
    ax.legend()
    return save(fig, "09_predictor_confidence")


def stability_over_time(log):
    fig, ax = plt.subplots(figsize=(6, 3.2))

    depth_of = {"shallow": 4, "medium": 8, "deep": 12}
    plotted = False

    for variant in ("A7_no_monitor", "A8_full"):
        subset = log[log["variant"] == variant]
        if subset.empty:
            continue
        ax.step(range(len(subset)), [depth_of[c] for c in subset["configuration"]],
                where="post", label=variant, alpha=0.8)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set(xlabel="Inference index", ylabel="Selected depth",
           title="Configuration trajectory", yticks=[4, 8, 12])
    ax.legend(fontsize=7)
    return save(fig, "10_stability_volatility")


def rollback_events(stability):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    x = np.arange(len(stability))
    width = 0.35

    ax.bar(x - width / 2, stability["switches"], width, label="switches")
    ax.bar(x + width / 2, stability["rollback_count"], width, label="rollbacks")

    ax.set(ylabel="Count", title="Switches and rollback events", xticks=x)
    ax.set_xticklabels(stability["variant"], rotation=10)
    ax.legend()
    return save(fig, "11_rollback_events")


def baseline_comparison(baselines):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))

    left.bar(baselines["variant"], baselines["loss"], color="#0f766e")
    left.set(ylabel="Prompt LM loss", title="Quality (lower is better)")
    left.tick_params(axis="x", rotation=20)

    right.bar(baselines["variant"], baselines["relative_flops"], color="#b45309")
    right.set(ylabel="Relative FLOPs", title="Compute (lower is cheaper)")
    right.tick_params(axis="x", rotation=20)

    fig.suptitle("Baseline comparison")
    return save(fig, "12_baseline_comparison")


# ----------------------------------------------------------------------


def main():
    os.makedirs(FIGURES, exist_ok=True)
    print(f"Writing figures to {FIGURES}/")

    calibration = read("calibration_results")
    baselines = read("baselines")
    log = read("inference_log")
    datasets = read("dataset_evaluation")
    stability = read("stability_experiment")
    frontier_frame = read("quality_compute_frontier")

    produced, skipped = [], []

    def attempt(name, function, *args):
        if any(arg is None or (hasattr(arg, "empty") and arg.empty) for arg in args):
            skipped.append(name)
            return
        path = function(*args)
        (produced if path else skipped).append(name)

    attempt("depth vs loss", depth_vs_loss, calibration)
    attempt("depth vs latency", depth_vs_latency, calibration)
    attempt("quality-compute frontier", frontier, frontier_frame)
    attempt("configuration distribution", configuration_distribution, baselines)
    attempt("average depth per dataset", average_depth_per_dataset, datasets)
    attempt("layer reduction", layer_reduction, baselines)
    attempt("latency distribution", latency_distribution, log)
    attempt("confidence distribution", confidence_distribution, log)
    attempt("rollback events", rollback_events, stability)
    attempt("baseline comparison", baseline_comparison, baselines)

    stability_log = read("stability_log")
    attempt("stability over time", stability_over_time,
            stability_log if stability_log is not None else log)

    try:
        import joblib
        metadata = joblib.load("results/performance_predictor.pkl")
        if isinstance(metadata, dict) and "confusion_matrix" in metadata:
            confusion(metadata)
            produced.append("predictor confusion matrix")
        else:
            skipped.append("predictor confusion matrix")
    except Exception as error:
        skipped.append(f"predictor confusion matrix ({type(error).__name__})")

    print(f"\n{len(produced)} figures written, {len(skipped)} skipped")
    for name in skipped:
        print(f"  skipped: {name} (source CSV missing -- run the evaluation first)")


if __name__ == "__main__":
    main()
