# Metamorphic Self-Adaptation (MSA) — GPT-2 Prototype

A research prototype that runs GPT-2 with a **per-input architecture**, chosen by a
learned configuration predictor, instead of always executing the full 12-layer stack
at full width. Simple prompts get a cheap configuration, harder prompts get an
expensive one.

Three adaptation mechanisms, none of which modify the pretrained weights:

| Mechanism | What changes | Real compute saving |
|-----------|--------------|---------------------|
| **Depth** | run only the first *d* of 12 transformer blocks | yes |
| **Attention** | slice QKV/output projections to a subset of heads | yes |
| **FFN** | slice the MLP to a subset of its 3072 intermediate channels | yes |

Attention and FFN adaptation **slice the weight tensors** rather than masking outputs,
so the skipped work is genuinely not performed. `test_msa.py` asserts that each
mechanism changes the computed output.

## Pipeline

```
Input
  ├─► Task Analyzer            8 heuristic features
  ├─► f_theta                  2-layer MLP → P(shallow), P(medium), P(deep)
  ├─► Configuration Policy     argmax + confidence threshold + fallback
  ├─► Stability Monitor        volatility tracking + rollback
  ├─► Dynamic Arch. Controller depth / attention / FFN knobs + telemetry
  └─► Adaptive GPT-2           ─► logits
```

Configurations (`configs/configurations.py`):

```
shallow  depth  4   attention reduced (6/12 heads)   ffn partial (2/4 chunks)
medium   depth  8   attention reduced (6/12 heads)   ffn partial (2/4 chunks)
deep     depth 12   attention full                   ffn full
```

The `depth` mechanism holds attention and FFN at full width, which is Baseline B and
the Phase 6 depth-only controller. The `full` mechanism enables all three.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Always run inside the virtual environment.** Running the system `python`
without activating `.venv` fails with `ModuleNotFoundError: No module named
'joblib'` (or `torch`), because the dependencies are installed only in `.venv`.
In a new terminal run `.venv\Scripts\Activate.ps1` first, or call
`.venv\Scripts\python.exe` directly.

Modules are run with `-m` so the package imports resolve from the repository root.
Baseline D downloads `HuggingFaceTB/SmolLM-135M` (~270 MB) on first use.

## Usage

One command per stage, or the whole pipeline at once:

```powershell
python main.py --mode simulate          # end-to-end MSA on sample prompts
python main.py --mode calibrate         # 799 prompts x 3 configs x 2 mechanisms
python main.py --mode frontier          # quality-compute tolerance sweep
python main.py --mode train-predictor   # train f_theta (+ macro F1, confusion, confidence)
python main.py --mode ablate-features   # Task Analyzer v1 vs v2
python main.py --mode threshold-sweep   # policy confidence threshold 0.40-0.80
python main.py --mode benchmark         # warmup + repeated latency runs
python main.py --mode evaluate          # baselines, ablations, datasets, stability
python main.py --mode runner            # standard per-sample evaluation schema
python main.py --mode plot              # all figures into results/figures/
python main.py --mode tables            # research tables into results/tables.md
python main.py --mode validate          # phase-gate validation
python main.py --mode ui                # rebuild ui/index.html from results
python main.py --mode all               # everything, in order
```

Individual entry points:

```powershell
python test_predictor.py                        # Phase 1 predictor validation
python test_msa.py                              # phase-gate validation (P1-P20)
python run_msa.py --text "Explain why the sky appears blue."
python run_static.py --all-depths               # Baseline A at depths 4/8/12
python -m evaluation.evaluate --suite stability
python -m evaluation.evaluate_static --dataset short_qa --generate
python -m evaluation.evaluate_depth_adaptive --dataset gsm8k --limit 40 --generate
python -m evaluation.evaluate_msa --dataset short_qa --generate
python -m evaluation.runner --methods dense_reference --dataset short_qa --generate
```

Every module has a runnable self-check:

```powershell
python -m analyzer.task_analyzer
python -m controller.controller
python -m controller.policy
python -m monitor.stability_monitor
python -m evaluation.metrics
python -m evaluation.baselines
python -m data.load_data
```

## Project structure

```
analyzer/task_analyzer.py        8 heuristic features (v1 = original 4, v2 = +4)
calibration/run_calibration.py   depth/attention/FFN sweep → calibration_results.csv
configs/configurations.py        configuration library, policy operating point, FLOP model
configs/experiment.yaml          every default in one place (asserted by test_msa.py)
controller/controller.py         Dynamic Architecture Controller + telemetry
controller/dynamic_controller.py plan-named entry point for the controller
controller/predictor.py          f_theta inference wrapper
controller/train_predictor.py    predictor training + feature ablation
controller/policy.py             calibration targets + inference-time policy
controller/policy_sweep.py       quality-compute frontier
controller/threshold_sweep.py    policy confidence-threshold sweep (Phase 3)
data/calibration_dataset.py      799 prompts, 12 categories (rebuild with --build)
data/calibration_prompts_v1_499.json  historical 499-prompt corpus
data/load_data.py                short_qa + GSM8K downstream sets
evaluation/harness.py            every baseline and ablation as one flag-set
evaluation/runner.py             standard per-sample schema -> results/evaluation/
evaluation/evaluate_static.py    Baseline A via the runner
evaluation/evaluate_depth_adaptive.py  Baseline B via the runner
evaluation/evaluate_msa.py       Baseline E via the runner
evaluation/evaluate.py           single evaluation entry point (all suites)
evaluation/baselines.py          Baseline D: pretrained SmolLM-135M + untrained reference
evaluation/benchmark.py          reproducible latency measurement
evaluation/metrics.py            standardised metric set
evaluation/generate_plots.py     16 figures, regenerated from CSVs
evaluation/tables.py             research tables + claims checklist
models/adaptive_model.py         AdaptiveGPT2: depth + attention + FFN + routing
monitor/stability_monitor.py     volatility, status, rollback
paper/msa_paper.md               research paper, all numbers read from results/
ui/index.html                    self-contained demo page (generated)
ui/export_ui_data.py             rebuilds the demo page from results/
```

## Correctness fix that invalidated earlier results

The original `AdaptiveGPT2` passed `attention_mask=torch.ones(batch, seq)` into the
transformer blocks. Under Transformers v5 with the SDPA attention path, supplying any
explicit mask **disables the causal mask** (`is_causal = q_len > 1 and attention_mask
is None`), so the model attended bidirectionally and every token could see its own
future.

The consequences were large: at depth 12 the logits differed from stock GPT-2 by up to
67.4 and the model predicted `"capital"` instead of `"Paris"` for
`"...the capital of France is"`. All calibration losses, the quality-compute frontier
and the trained predictor were derived from that broken forward pass.

The fix is to pass `attention_mask=None`. `test_msa.py` now asserts both that depth 12
is numerically identical to the reference model and that prefix logits do not change
when later tokens are removed. All results in `results/` were regenerated afterwards.

## Findings

Calibration: 799 prompts in 12 categories × 3 configurations × 2 mechanisms =
**4,794 experiments** (`results/calibration_results.csv`). The earlier 499-prompt
version is preserved in `results/calibration/`.

Depth-only mechanism, mean over 799 prompts:

| Depth | Prompt LM loss | Relative FLOPs |
|-------|----------------|----------------|
| 4     | 8.99           | 0.333          |
| 8     | 7.17           | 0.667          |
| 12    | 3.91           | 1.000          |

Quality-compute frontier (`results/quality_compute_frontier.csv`) — the tolerance is
the maximum relative loss increase the calibration policy accepts:

| Tolerance | Avg depth | Compute reduction | Relative loss increase | shallow/medium/deep |
|-----------|-----------|-------------------|------------------------|---------------------|
| 0.10 | 12.00 |  0.0% |  0.0% | 0 / 1 / 798 |
| 0.50 | 11.44 |  4.6% |  5.6% | 24 / 63 / 712 |
| 0.75 | 10.26 | 14.5% | 23.6% | 66 / 216 / 517 |
| 1.00 |  8.51 | 29.1% | 56.0% | 148 / 402 / 249 |
| 1.50 |  5.62 | 53.2% | 105.0% | 499 / 277 / 23 |
| 2.00 |  4.41 | 63.3% | 123.3% | 718 / 81 / 0 |

Tolerance **1.00** trains the shipped predictor: it is the setting that produces a
non-degenerate three-class target distribution.

Predictor (`f_theta`, 8 features): held-out **accuracy 0.713, macro F1 0.678**
(majority class: 0.500 / 0.222); 5-fold CV accuracy 0.731 ± 0.020, macro F1
0.693 ± 0.030 (`results/calibration/predictor_evaluation.json`). The v2 features
beat v1's four by 0.081 CV macro F1, more than twice the standard deviation.

Policy (`results/policy_threshold_sweep.csv`): thresholds 0.40–0.80 swept on the
160 held-out prompts. **0.40** was selected by a rule fixed in advance (best
agreement with the calibration target, ties toward lower FLOPs). Raising it to 0.80
moves relative loss increase from 56.9% to 10.1% and FLOPs from 0.733 to 0.931.

Baselines on 60 short-QA prompts (`results/baselines.csv`):

| Method | Prompt loss | Exact match | Avg depth | Rel. FLOPs |
|--------|-------------|-------------|-----------|------------|
| A static GPT-2 | 3.89 | 0.133 | 12.00 | 1.000 |
| B depth adaptive | 5.97 | 0.017 | 7.73 | 0.644 |
| C routing / MoE-style | 6.37 | 0.000 | 12.00 | 0.668 |
| D SmolLM-135M (pretrained dense) | 1.98* | **0.750** | 30 layers | — |
| E full MSA | 9.10 | 0.000 | 7.73 | 0.322 |

\* different tokenizer, not comparable per token.

Ablation ladder (`results/ablations.csv`): A1 3.89 → A2 rule 8.07 → A3/A4
predictor 5.97 → A5 +attention 8.40, A6 +FFN 7.74 → A7/A8 full 9.10 prompt loss.
The learned predictor is the clearest win (8.07 → 5.97 while using *more* depth).

Stability (`results/stability/stability_levels.csv`): the monitor is inert at low
and medium switching. At high switching it cuts executed switches 50 → 34 and
volatility 0.830 → 0.566 with 13 rollbacks, but the rolled-back steps lose quality
(loss 8.76 vs 5.65 without rollback). It buys stability, not quality.

Downstream (`results/dataset_evaluation.csv`): **every GPT-2 variant scores 0.000 on
GSM8K**; Baseline D reaches 0.050. On GSM8K the predictor routes nearly everything to
`deep` (average depth 11.5–11.6), so the compute saving seen on short QA largely
disappears — no compute-reduction figure should be quoted without naming the dataset.

All tables (1–8) are in `results/tables.md`, regenerated by
`python main.py --mode tables`.

## Limitations

These are the honest boundaries of what the experiments establish:

- **Calibration signal is not task quality.** Targets come from next-token loss over
  the prompt. Downstream metrics are reported separately in
  `results/dataset_evaluation.csv`.
- **GPT-2 small cannot solve GSM8K.** Accuracy is near zero for every variant
  including the static baseline, so that dataset separates compute behaviour, not
  task quality.
- **Attention and FFN adaptation degrade quality substantially.** They are applied to
  pretrained weights with no retraining or distillation. The measured loss increase is
  reported rather than explained away.
- **FLOP reduction is not always latency reduction.** Depth and attention/FFN
  slicing do cut wall-clock time, but the routing baseline uses less arithmetic
  while running slower than static at sequence lengths 16 and 64. Both measures are
  reported; neither substitutes for the other.
- **The routing baseline uses a fixed seeded gate**, not a learned router.
- **Baseline D uses a different tokenizer.** Its loss/perplexity are not comparable
  with GPT-2's; exact match and latency are.
- **Rollback is not quality-aware.** It reverts to the window's most common
  configuration, which lowered quality on the rolled-back steps.
- **Summarization and code evaluation were not run** (optional in the plan); GPT-2
  small's task scores would be floored.

No claim is made that MSA always improves performance, guarantees lower latency, or
preserves quality. See the claims checklist at the end of `results/tables.md`.

## Reproducibility

Seeds are fixed at 42 throughout. `configs/experiment.yaml` records every default, and
`test_msa.py` fails if it drifts from the code. `results/environment.json` records the
Python, PyTorch, Transformers, scikit-learn, CUDA and device details. Calibration
prompts are frozen in `data/calibration_prompts.json` (rebuilding with `--build`
fetches GSM8K). A clean `python main.py --mode all` reproduced calibration losses,
the predictor and the selected threshold exactly (`results/reproduction_log.txt`).

## Benchmark

Controlled measurement (`results/benchmark.csv`): 6 pinned torch threads, batch 1,
every cell warmed 10× before any timing, 50 runs split over 5 shuffled interleaved
rounds. Medians:

| Mechanism | Config | seq=16 | seq=64 | seq=256 |
|-----------|--------|--------|--------|---------|
| depth only | shallow | 14.3 ms | 32.7 ms | 91.0 ms |
| depth only | medium | 20.2 ms | 45.9 ms | 132.4 ms |
| depth only | deep | 28.1 ms | 59.8 ms | 175.8 ms |
| depth+attn+ffn | shallow | 12.6 ms | 28.4 ms | 73.5 ms |
| depth+attn+ffn | medium | 16.8 ms | 36.1 ms | 93.9 ms |
| routing | deep | 36.2 ms | 65.1 ms | 155.9 ms |

Depth cuts latency 23–45% at seq=64; attention/FFN slicing cuts a further 12–29%.
Routing is slower than static below 256 tokens and 11% faster at 256. Baseline D
(SmolLM-135M, float32) takes 71.8 ms at seq=64.

## Deliverables

| Artefact | Path |
|----------|------|
| Research paper | `paper/msa_paper.md` |
| Demonstration page | `ui/index.html` |
| Research tables (1–8) + claims checklist | `results/tables.md` |
| Figures | `results/figures/` (16) |
| Calibration data | `results/calibration_results.csv` (4,794 rows) |
| Per-sample evaluation | `results/evaluation/` |
| Stability experiments | `results/stability/` |

### The demonstration page

`ui/index.html` is self-contained — open it directly, no server needed. Rebuild it
from current results with `python -m ui.export_ui_data`.

It splits cleanly into what is real and what is replayed:

- **Live in the browser**: the Task Analyzer's eight heuristics, the `f_theta`
  forward pass (real exported weights, 8→32→16→3), the confidence policy (threshold
  0.40), and the stability monitor with V(t) and S(t). Typing previews a decision;
  Analyze commits it to the monitor and the history panel.
- **Replayed from `results/`**: every GPT-2 loss, perplexity, latency and accuracy.

The page does not run GPT-2 and does not claim to.

## Status

**All 20 phases of `MSA-GPT2-plan-phase-1-to-20.md` are complete**, and
`python test_msa.py` passes all 41 phase-gate checks.
