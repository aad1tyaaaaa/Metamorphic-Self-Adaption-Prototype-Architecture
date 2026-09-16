import pandas as pd


def create_targets(df, quality_tolerance):

    targets = []

    for sample_id, group in df.groupby("sample_id"):

        group = group.sort_values("depth")

        deep_row = group[group["depth"] == 12].iloc[0]

        baseline_loss = deep_row["loss"]

        maximum_allowed_loss = (
            baseline_loss * (1 + quality_tolerance)
        )

        acceptable = group[
            group["loss"] <= maximum_allowed_loss
        ]

        if len(acceptable) > 0:
            selected = acceptable.sort_values(
                "depth"
            ).iloc[0]
        else:
            selected = deep_row

        targets.append({
            "sample_id": sample_id,
            "depth": selected["depth"],
            "configuration": selected["configuration"],
            "selected_loss": selected["loss"],
            "deep_loss": baseline_loss,
            "latency": selected["latency_seconds"]
        })

    return pd.DataFrame(targets)


# --------------------------------------------------
# Load calibration data
# --------------------------------------------------

df = pd.read_csv(
    "results/calibration_results.csv"
)


# --------------------------------------------------
# Test multiple quality tolerances
# --------------------------------------------------

tolerances = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00
]


print("\n")
print("=" * 70)
print("MSA QUALITY-COMPUTE FRONTIER")
print("=" * 70)


summary = []


for tolerance in tolerances:

    targets = create_targets(
        df,
        tolerance
    )

    distribution = (
        targets["configuration"]
        .value_counts()
        .to_dict()
    )

    average_depth = targets["depth"].mean()

    layer_reduction = (
        1 - average_depth / 12
    ) * 100

    average_latency = targets["latency"].mean()

    average_loss = targets["selected_loss"].mean()

    deep_average_loss = (
        targets["deep_loss"].mean()
    )

    relative_loss = (
        average_loss / deep_average_loss
    ) - 1

    summary.append({
        "tolerance": tolerance,
        "average_depth": average_depth,
        "layer_reduction_percent": layer_reduction,
        "average_latency": average_latency,
        "average_loss": average_loss,
        "relative_loss_increase": relative_loss,
        "shallow": distribution.get(
            "shallow", 0
        ),
        "medium": distribution.get(
            "medium", 0
        ),
        "deep": distribution.get(
            "deep", 0
        )
    })


# --------------------------------------------------
# Display results
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

print(
    "\n"
    + summary_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"\nTolerance: "
        f"{row['tolerance'] * 100:.0f}%"
    )

    print(
        f"  Depth: "
        f"{row['average_depth']:.2f}"
    )

    print(
        f"  Layer reduction: "
        f"{row['layer_reduction_percent']:.2f}%"
    )

    print(
        f"  Shallow: "
        f"{int(row['shallow'])}"
    )

    print(
        f"  Medium: "
        f"{int(row['medium'])}"
    )

    print(
        f"  Deep: "
        f"{int(row['deep'])}"
    )


# --------------------------------------------------
# Save frontier
# --------------------------------------------------

summary_df.to_csv(
    "results/quality_compute_frontier.csv",
    index=False
)

print("\nSaved:")
print("results/quality_compute_frontier.csv")