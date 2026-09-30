# Phase 0 summary: retraining the non-deep configurations

Blocking gate for Phase 1. All numbers below are read from `results/baselines.csv` / `results/baselines_retrained.csv`, `results/ablations.csv` / `results/ablations_retrained.csv` and `results/phase0_retraining/retraining_summary.json`.

## Verdict

- **Did retraining close the gap to static (prompt loss)?** **PARTIALLY.** Full MSA prompt loss moved from 9.0972 to 5.7089 against a static baseline of 3.8890. The gap shrank from 5.2082 to 1.8199 nats (65.1% of the gap closed).
- **Does Full MSA (retrained) now beat static on exact match?** **NO.** Exact match is 0.000 for retrained Full MSA vs 0.133 for static (pre-retraining Full MSA: 0.000).
- Compute is unchanged by construction: retrained Full MSA still runs at 0.322x the static FLOPs (adapters add only biases and LayerNorm affine parameters, which do not change the FLOP model).

Retraining substantially reduces the damage attention/FFN slicing does to the pretrained weights, but does not eliminate it: Full MSA is still worse than the static 12-layer model on both prompt loss and exact match. Per the brief, Phase 1 proceeds on the retrained weights with this negative result recorded.

## What was retrained

Method: bias + LayerNorm-only fine-tuning (`models/adapters.py`), the minimal variant the brief allows in place of LoRA. Per configuration, a `ConfigAdapter` holds trainable copies of every bias and LayerNorm affine parameter in the blocks it covers plus its own final LayerNorm; the pretrained weight matrices, the shared embeddings/lm_head and the deep configuration are frozen and never modified. Adapters are only consulted on the sliced (reduced-attention / partial-FFN) execution path, so any full-width request is byte-identical to the original model.

Data: `data/calibration_prompts.json` (799 prompts), split 720 train / 79 validation by seeded permutation (seed 42). The 60-prompt `short_qa` evaluation set is a separate hand-written list that is not part of the calibration corpus, so it was never seen during fine-tuning.

| config | depth | trainable params | epochs | lr | val loss before | val loss after | wall clock |
|---|---|---|---|---|---|---|---|
| shallow | 4 | 41,472 | 3 | 0.0003 | 8.4088 | 4.8084 | 142s |
| medium | 8 | 81,408 | 3 | 0.0003 | 9.6336 | 4.1764 | 275s |

Training curves (loss vs. step, train and validation) are in `results/phase0_retraining/<config>_training_curve.csv`; adapter checkpoints are `results/phase0_retraining/<config>_adapter.pt`.

## Baselines: before vs. after retraining

| variant | loss_before | loss_after | loss_delta | accuracy_before | accuracy_after | relative_flops_after |
|---|---|---|---|---|---|---|
| static | 3.8890 | 3.8890 | 0.0000 | 0.1333 | 0.1333 | 1.0000 |
| depth_adaptive | 5.9736 | 5.9736 | 0.0000 | 0.0167 | 0.0167 | 0.6444 |
| routing | 6.3663 | 6.3663 | 0.0000 | 0.0000 | 0.0000 | 0.6677 |
| msa | 9.0972 | 5.7089 | -3.3883 | 0.0000 | 0.0000 | 0.3222 |
| dense_reference | 1.9846 | 1.9846 | 0.0000 | 0.7500 | 0.7500 | n/a |

Baseline D (`dense_reference`) is a different model and is unaffected. `static`, `routing` and `depth_adaptive` never execute the sliced path, so they are expected to reproduce the original numbers exactly.

## Ablation ladder: before vs. after retraining

| variant | loss_before | loss_after | loss_delta | accuracy_before | accuracy_after | relative_flops_after |
|---|---|---|---|---|---|---|
| A1_static | 3.8890 | 3.8890 | 0.0000 | 0.1333 | 0.1333 | 1.0000 |
| A2_analyzer | 8.0672 | 8.0672 | 0.0000 | 0.0167 | 0.0167 | 0.3333 |
| A3_predictor | 5.9736 | 5.9736 | 0.0000 | 0.0167 | 0.0167 | 0.6444 |
| A4_depth | 5.9736 | 5.9736 | 0.0000 | 0.0167 | 0.0167 | 0.6444 |
| A5_depth_attention | 8.4026 | 5.6980 | -2.7046 | 0.0333 | 0.0000 | 0.5364 |
| A6_depth_ffn | 7.7411 | 6.0599 | -1.6812 | 0.0000 | 0.0000 | 0.4303 |
| A7_no_monitor | 9.0972 | 5.7089 | -3.3883 | 0.0000 | 0.0000 | 0.3222 |
| A8_full | 9.0972 | 5.7089 | -3.3883 | 0.0000 | 0.0000 | 0.3222 |

A1-A4 use full-width attention/FFN (depth-only or static) and therefore never touch an adapter; A5-A8 execute the sliced path on shallow/medium prompts and are the rows retraining can change.

## Frozen-path control check

Variants that by construction never execute an adapted block must reproduce their pre-retraining loss. Maximum absolute drift across 8 such rows: **5.56e-08**.

| suite | variant | loss before | loss after | abs drift |
|---|---|---|---|---|
| baselines | static | 3.8890 | 3.8890 | 0.00e+00 |
| baselines | depth_adaptive | 5.9736 | 5.9736 | 0.00e+00 |
| baselines | routing | 6.3663 | 6.3663 | 5.56e-08 |
| baselines | dense_reference | 1.9846 | 1.9846 | 0.00e+00 |
| ablations | A1_static | 3.8890 | 3.8890 | 0.00e+00 |
| ablations | A2_analyzer | 8.0672 | 8.0672 | 0.00e+00 |
| ablations | A3_predictor | 5.9736 | 5.9736 | 0.00e+00 |
| ablations | A4_depth | 5.9736 | 5.9736 | 0.00e+00 |

## Provenance note

The first retrained-evaluation run was discarded: an implementation bug applied the adapter's final LayerNorm on the full-width path whenever a shallow/medium *name* was selected, which changed `depth_adaptive`, `A2_analyzer`, `A3_predictor` and `A4_depth` even though those variants never slice attention or FFN. The bug was fixed in `models/adaptive_model.py` (`if adapter is None or stock`), verified by a zero-difference check on the stock path, and the full evaluation was rerun from scratch. The control table above is the evidence that the rerun is clean.
