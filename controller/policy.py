import pandas as pd


class CalibrationPolicy:

    def __init__(self, quality_tolerance=0.10):
        """
        Maximum allowed relative loss increase
        compared with the full 12-layer model.

        0.10 = allow up to 10% higher loss.
        """

        self.quality_tolerance = quality_tolerance

    def create_targets(self, csv_path):

        df = pd.read_csv(csv_path)

        targets = []

        # Process each input independently
        for sample_id, group in df.groupby("sample_id"):

            group = group.sort_values("depth")

            # ----------------------------------------
            # Full model = 12-layer baseline
            # ----------------------------------------

            deep_row = group[group["depth"] == 12].iloc[0]

            baseline_loss = deep_row["loss"]

            maximum_allowed_loss = (
                baseline_loss
                * (1 + self.quality_tolerance)
            )

            # ----------------------------------------
            # Select shallowest acceptable depth
            # ----------------------------------------

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
                "text": selected["text"],

                "input_length": selected["input_length"],
                "reasoning": selected["reasoning"],
                "domain": selected["domain"],
                "structure": selected["structure"],
                "complexity": selected["complexity"],

                "target_depth": selected["depth"],
                "target_configuration": selected["configuration"],

                "selected_loss": selected["loss"],
                "deep_loss": baseline_loss,

                "selected_latency": selected[
                    "latency_seconds"
                ],

                "deep_latency": deep_row[
                    "latency_seconds"
                ],

                "relative_loss_increase": (
                    selected["loss"] / baseline_loss
                ) - 1
            })

        return pd.DataFrame(targets)


if __name__ == "__main__":

    policy = CalibrationPolicy(
        quality_tolerance=0.10
    )

    targets = policy.create_targets(
        "results/calibration_results.csv"
    )

    print("\n")
    print("=" * 70)
    print("MSA QUALITY-CONSTRAINED POLICY")
    print("=" * 70)

    print(
        "\nQuality tolerance: "
        f"{policy.quality_tolerance * 100:.1f}%"
    )

    print("\nSelected configurations:\n")

    print(
        targets[
            [
                "sample_id",
                "complexity",
                "target_depth",
                "target_configuration",
                "selected_loss",
                "deep_loss",
                "relative_loss_increase"
            ]
        ].to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("CONFIGURATION DISTRIBUTION")
    print("=" * 70)

    print(
        targets["target_configuration"]
        .value_counts()
    )

    print("\n")
    print("=" * 70)
    print("AVERAGE COMPUTE SAVING")
    print("=" * 70)

    average_depth = targets["target_depth"].mean()

    compute_saving = (
        1 - average_depth / 12
    ) * 100

    print(
        f"Average selected depth: {average_depth:.2f}"
    )

    print(
        f"Estimated layer reduction: "
        f"{compute_saving:.2f}%"
    )

    print("\n")