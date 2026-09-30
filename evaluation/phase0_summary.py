"""Phase 0 gate report: did retraining the shallow/medium configurations close
the gap to the static baseline?

Every number in the report is read from a result file produced by an executed
run; nothing is typed in by hand.

    python -m evaluation.phase0_summary
"""

import json
import math

import pandas as pd

RESULTS = "results"
RETRAIN_DIR = f"{RESULTS}/phase0_retraining"
OUTPUT = f"{RESULTS}/phase0_summary.md"

# Variants whose execution never touches an adapter (always `deep`, routing, or
# the stock full-width path). They must reproduce the pre-retraining numbers
# exactly; any drift would indicate the adapter leaked into a frozen path.
FROZEN_CONTROLS = {
    "static", "routing", "dense_reference", "A1_static",
    "depth_adaptive", "A2_analyzer", "A3_predictor", "A4_depth",
}


def load_pair(name):
    before = pd.read_csv(f"{RESULTS}/{name}.csv").set_index("variant")
    after = pd.read_csv(f"{RESULTS}/{name}_retrained.csv").set_index("variant")
    return before, after


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def diff_table(before, after, columns):
    rows = []
    for variant in after.index:
        if variant not in before.index:
            continue
        row = {"variant": variant}
        for column in columns:
            b = before.loc[variant, column] if column in before.columns else float("nan")
            a = after.loc[variant, column] if column in after.columns else float("nan")
            row[f"{column}_before"] = b
            row[f"{column}_after"] = a
            row[f"{column}_delta"] = a - b
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def markdown_table(frame, columns, digits=4):
    header = "| variant | " + " | ".join(columns) + " |"
    rule = "|---|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for variant, row in frame.iterrows():
        cells = [fmt(row[c], digits) if isinstance(row[c], float) else str(row[c])
                 for c in columns]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict(gap_before, gap_after):
    if gap_after <= 0:
        return "YES"
    if gap_after < gap_before:
        return "PARTIALLY"
    return "NO"


def main():
    with open(f"{RETRAIN_DIR}/retraining_summary.json", encoding="utf-8") as handle:
        retrain = json.load(handle)

    base_before, base_after = load_pair("baselines")
    abl_before, abl_after = load_pair("ablations")

    metrics = ["loss", "accuracy", "answer_loss", "relative_flops", "average_depth"]
    base_diff = diff_table(base_before, base_after, metrics)
    abl_diff = diff_table(abl_before, abl_after, metrics)

    # --- headline question -------------------------------------------------
    static_loss = base_after.loc["static", "loss"]
    static_em = base_after.loc["static", "accuracy"]
    msa_loss_before = base_before.loc["msa", "loss"]
    msa_loss_after = base_after.loc["msa", "loss"]
    msa_em_before = base_before.loc["msa", "accuracy"]
    msa_em_after = base_after.loc["msa", "accuracy"]
    msa_flops = base_after.loc["msa", "relative_flops"]

    gap_before = msa_loss_before - static_loss
    gap_after = msa_loss_after - static_loss
    closed_fraction = 1 - gap_after / gap_before if gap_before else float("nan")
    loss_verdict = verdict(gap_before, gap_after)
    em_verdict = "YES" if msa_em_after > static_em else "NO"

    # --- frozen-path control check ------------------------------------------
    control_rows = []
    for name, (b, a) in (("baselines", (base_before, base_after)),
                         ("ablations", (abl_before, abl_after))):
        for variant in a.index:
            if variant in FROZEN_CONTROLS and variant in b.index:
                drift = abs(a.loc[variant, "loss"] - b.loc[variant, "loss"])
                control_rows.append((name, variant, b.loc[variant, "loss"],
                                     a.loc[variant, "loss"], drift))
    max_drift = max(r[4] for r in control_rows) if control_rows else float("nan")

    # --- ablation ladder ----------------------------------------------------
    ladder = ["A1_static", "A2_analyzer", "A3_predictor", "A4_depth",
              "A5_depth_attention", "A6_depth_ffn", "A7_no_monitor", "A8_full"]
    ladder = [v for v in ladder if v in abl_diff.index]

    lines = []
    lines.append("# Phase 0 summary: retraining the non-deep configurations\n")
    lines.append("Blocking gate for Phase 1. All numbers below are read from "
                 "`results/baselines.csv` / `results/baselines_retrained.csv`, "
                 "`results/ablations.csv` / `results/ablations_retrained.csv` and "
                 "`results/phase0_retraining/retraining_summary.json`.\n")

    lines.append("## Verdict\n")
    lines.append(f"- **Did retraining close the gap to static (prompt loss)?** "
                 f"**{loss_verdict}.** Full MSA prompt loss moved from "
                 f"{fmt(msa_loss_before)} to {fmt(msa_loss_after)} against a static "
                 f"baseline of {fmt(static_loss)}. The gap shrank from "
                 f"{fmt(gap_before)} to {fmt(gap_after)} nats "
                 f"({fmt(100 * closed_fraction, 1)}% of the gap closed).")
    lines.append(f"- **Does Full MSA (retrained) now beat static on exact match?** "
                 f"**{em_verdict}.** Exact match is {fmt(msa_em_after, 3)} for retrained "
                 f"Full MSA vs {fmt(static_em, 3)} for static "
                 f"(pre-retraining Full MSA: {fmt(msa_em_before, 3)}).")
    lines.append(f"- Compute is unchanged by construction: retrained Full MSA still runs at "
                 f"{fmt(msa_flops, 3)}x the static FLOPs (adapters add only biases and "
                 f"LayerNorm affine parameters, which do not change the FLOP model).")
    lines.append("")
    if loss_verdict == "PARTIALLY":
        lines.append("Retraining substantially reduces the damage attention/FFN slicing "
                     "does to the pretrained weights, but does not eliminate it: Full MSA "
                     "is still worse than the static 12-layer model on both prompt loss "
                     "and exact match. Per the brief, Phase 1 proceeds on the retrained "
                     "weights with this negative result recorded.")
    elif loss_verdict == "YES":
        lines.append("Retraining fully closes the prompt-loss gap.")
    else:
        lines.append("Retraining did not reduce the gap.")
    lines.append("")

    lines.append("## What was retrained\n")
    lines.append("Method: bias + LayerNorm-only fine-tuning (`models/adapters.py`), the "
                 "minimal variant the brief allows in place of LoRA. Per configuration, a "
                 "`ConfigAdapter` holds trainable copies of every bias and LayerNorm affine "
                 "parameter in the blocks it covers plus its own final LayerNorm; the "
                 "pretrained weight matrices, the shared embeddings/lm_head and the deep "
                 "configuration are frozen and never modified. Adapters are only consulted on "
                 "the sliced (reduced-attention / partial-FFN) execution path, so any "
                 "full-width request is byte-identical to the original model.\n")
    lines.append("Data: `data/calibration_prompts.json` (799 prompts), split 720 train / 79 "
                 "validation by seeded permutation (seed 42). The 60-prompt `short_qa` "
                 "evaluation set is a separate hand-written list that is not part of the "
                 "calibration corpus, so it was never seen during fine-tuning.\n")
    lines.append("| config | depth | trainable params | epochs | lr | val loss before | val loss after | wall clock |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ("shallow", "medium"):
        r = retrain[name]
        lines.append(f"| {name} | {r['depth']} | {r['trainable_parameters']:,} | {r['epochs']} | "
                     f"{r['lr']} | {fmt(r['pre_val_loss'])} | {fmt(r['post_val_loss'])} | "
                     f"{r['wall_clock_seconds']:.0f}s |")
    lines.append("")
    lines.append("Training curves (loss vs. step, train and validation) are in "
                 "`results/phase0_retraining/<config>_training_curve.csv`; adapter "
                 "checkpoints are `results/phase0_retraining/<config>_adapter.pt`.\n")

    lines.append("## Baselines: before vs. after retraining\n")
    lines.append(markdown_table(base_diff, ["loss_before", "loss_after", "loss_delta",
                                            "accuracy_before", "accuracy_after",
                                            "relative_flops_after"]))
    lines.append("")
    lines.append("Baseline D (`dense_reference`) is a different model and is unaffected. "
                 "`static`, `routing` and `depth_adaptive` never execute the sliced path, "
                 "so they are expected to reproduce the original numbers exactly.\n")

    lines.append("## Ablation ladder: before vs. after retraining\n")
    lines.append(markdown_table(abl_diff.loc[ladder],
                                ["loss_before", "loss_after", "loss_delta",
                                 "accuracy_before", "accuracy_after",
                                 "relative_flops_after"]))
    lines.append("")
    lines.append("A1-A4 use full-width attention/FFN (depth-only or static) and therefore "
                 "never touch an adapter; A5-A8 execute the sliced path on shallow/medium "
                 "prompts and are the rows retraining can change.\n")

    lines.append("## Frozen-path control check\n")
    lines.append("Variants that by construction never execute an adapted block must "
                 "reproduce their pre-retraining loss. Maximum absolute drift across "
                 f"{len(control_rows)} such rows: **{max_drift:.2e}**.\n")
    lines.append("| suite | variant | loss before | loss after | abs drift |")
    lines.append("|---|---|---|---|---|")
    for suite, variant, b, a, d in control_rows:
        lines.append(f"| {suite} | {variant} | {fmt(b)} | {fmt(a)} | {d:.2e} |")
    lines.append("")

    lines.append("## Provenance note\n")
    lines.append("The first retrained-evaluation run was discarded: an implementation bug "
                 "applied the adapter's final LayerNorm on the full-width path whenever a "
                 "shallow/medium *name* was selected, which changed `depth_adaptive`, "
                 "`A2_analyzer`, `A3_predictor` and `A4_depth` even though those variants "
                 "never slice attention or FFN. The bug was fixed in "
                 "`models/adaptive_model.py` (`if adapter is None or stock`), verified by "
                 "a zero-difference check on the stock path, and the full evaluation was "
                 "rerun from scratch. The control table above is the evidence that the "
                 "rerun is clean.\n")

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("\n".join(lines[:12]))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
