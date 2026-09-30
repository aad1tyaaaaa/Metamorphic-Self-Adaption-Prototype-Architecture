# Phase status — MSA prototype experiments

Status of every deliverable in the brief. "Not run" means no result file
exists and no number for it appears anywhere in `results/`; nothing has been
estimated or placeholder-filled.

Hardware for everything below: CPU only (no CUDA device), torch 2.14.0+cpu,
transformers 5.17.0, seed 42.

| Deliverable | Status | Notes |
|---|---|---|
| `results/phase0_summary.md` | **done** | Verdict PARTIALLY: retraining shrinks the Full-MSA-vs-static prompt-loss gap but does not close it; exact match still 0.000 vs 0.133. See file. |
| `results/baselines_retrained.csv` | **done** | Same schema and methodology as `baselines.csv` (short_qa, 60 prompts, generation on). |
| `results/ablations_retrained.csv` | **done** | Same schema and methodology as `ablations.csv`. |
| `results/phase0_retraining/` | **done** | Adapters (`*_adapter.pt`), training curves (`*_training_curve.csv`), `retraining_summary.json`, raw logs. |
| `results/evaluation_retrained/` | **done** | Per-sample frames (standard runner schema) for every retrained-evaluation variant; `results/evaluation/` (pre-retraining) is untouched. |
| `results/ablation_factorial.csv` (1.1) | not run | Phase 0 gate reached; not started. |
| `results/complexity_validity.json` (1.2) | not run | Not started. |
| `results/stability_precision_recall.json` (1.3) | not run | Not started. |
| `results/failure_cases.md` (1.4) | not run | Not started. |
| `results/sensitivity_sweep.csv` (1.5) | not run | Not started. Note: the brief's Eq. 2 cost weights (alpha, beta, gamma) and penalty lambda do not exist in the local paper or code; the sweepable knobs that do exist are `ComplexityScorer.weights`, the selector thresholds (0.33/0.66), the predictor tolerance tau and the policy confidence threshold. |
| `results/contemporary_comparison_note.md` (1.6) | not run | Not started. |
| `results/statistical_summary.csv` (Phase 2) | not run | Not started; only seed 42 has been run. |

## Why the rest is not run

The session was asked to complete Phase 0 first and then deliver as quickly as
possible. Phase 0 alone took ~7 min of fine-tuning plus ~10 min of evaluation
per pass on this CPU-only machine, and one full evaluation pass had to be
discarded and rerun after a bug was found (see the provenance note in
`phase0_summary.md`). Phases 1.1-1.6 and Phase 2 at the requested fidelity
(5 seeds, generation-based exact match, factorial + sensitivity grids) are an
estimated 5-6 h of unattended CPU time; a reduced-fidelity pass is ~2-2.5 h.
Neither fits the time available, so they are reported as not run rather than
approximated.

## Code added for Phase 0 (all additive; `python test_msa.py` still passes 41/41)

- `models/adapters.py` — `ConfigAdapter`: bias + LayerNorm-only fine-tuning per configuration.
- `models/adaptive_model.py` — optional `adapter` argument on the sliced path only.
- `controller/controller.py`, `evaluation/harness.py`, `evaluation/evaluate.py` — optional `adapters` pass-through.
- `calibration/finetune_adapters.py` — training script (720/79 train/val split of the calibration corpus; eval set never seen).
- `evaluation/evaluate_phase0.py` — reruns baselines + ablations with the adapters substituted in.
- `evaluation/phase0_summary.py` — writes `phase0_summary.md` from the result files only.

## Reproduction

```powershell
.venv\Scripts\python.exe -m calibration.finetune_adapters
.venv\Scripts\python.exe -m evaluation.evaluate_phase0
.venv\Scripts\python.exe -m evaluation.phase0_summary
```
