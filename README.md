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

Modules are run with `-m` so the package imports resolve from the repository root.

## Usage

One command per stage, or the whole pipeline at once:

```powershell
python main.py --mode simulate          # end-to-end MSA on sample prompts
python main.py --mode calibrate         # 499 prompts x 3 configs x 2 mechanisms
python main.py --mode frontier          # quality-compute tolerance sweep
python main.py --mode train-predictor   # train f_theta
python main.py --mode ablate-features   # Task Analyzer v1 vs v2
python main.py --mode benchmark         # warmup + repeated latency runs
python main.py --mode evaluate          # baselines, ablations, datasets, stability
python main.py --mode plot              # all figures into results/figures/
python main.py --mode tables            # research tables into results/tables.md
python main.py --mode validate          # Phase 24 checklist
python main.py --mode ui                # rebuild ui/index.html from results
python main.py --mode all               # everything, in order
```

Individual entry points:

```powershell
python test_predictor.py                        # Phase 1 predictor validation
python test_msa.py                              # Phase 24 final validation
python run_msa.py --text "Explain why the sky appears blue."
python run_static.py --all-depths               # Baseline A at depths 4/8/12
python -m evaluation.evaluate --suite stability
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
configs/configurations.py        configuration library + analytical FLOP model
configs/experiment.yaml          every default in one place (Phase 18)
controller/controller.py         Dynamic Architecture Controller + telemetry
controller/predictor.py          f_theta inference wrapper
controller/train_predictor.py    predictor training + feature ablation
controller/policy.py             calibration targets + inference-time policy
controller/policy_sweep.py       quality-compute frontier
data/calibration_dataset.py      499 prompts, 10 categories (rebuild with --build)
data/load_data.py                short_qa + GSM8K downstream sets
evaluation/harness.py            every baseline and ablation as one flag-set
evaluation/evaluate.py           single evaluation entry point
evaluation/baselines.py          Baseline D dense reference (RMSNorm/RoPE/SwiGLU)
evaluation/benchmark.py          reproducible latency measurement
evaluation/metrics.py            standardised metric set
evaluation/generate_plots.py     12 figures, regenerated from CSVs
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

Calibration: 499 prompts × 3 configurations × 2 mechanisms = **2,994 experiments**
(`results/calibration_results.csv`).

Depth-only mechanism, mean over 499 prompts:

| Depth | Prompt LM loss | Relative FLOPs | Latency |
|-------|----------------|----------------|---------|
| 4     | 9.03           | 0.333          | 60 ms   |
| 8     | 7.21           | 0.667          | 94 ms   |
| 12    | 3.86           | 1.000          | 126 ms  |

Quality-compute frontier (`results/quality_compute_frontier.csv`) — the tolerance is
the maximum relative loss increase the calibration policy accepts:

| Tolerance | Avg depth | Compute reduction | Relative loss increase | shallow/medium/deep |
|-----------|-----------|-------------------|------------------------|---------------------|
| 0.10 | 12.00 |  0.0% |  0.0% | 0 / 0 / 499 |
| 0.50 | 11.48 |  4.3% |  5.5% | 15 / 35 / 449 |
| 0.75 | 10.39 | 13.4% | 22.1% | 38 / 125 / 336 |
| 1.00 |  8.87 | 26.1% | 51.5% | 79 / 233 / 187 |
| 1.50 |  5.82 | 51.5% | 105.8% | 291 / 189 / 19 |
| 2.00 |  4.46 | 62.8% | 126.3% | 441 / 58 / 0 |

Tolerance **1.00** is the operating point used to train the shipped predictor: it is
the setting that produces a non-degenerate three-class target distribution.

Predictor (`f_theta`, 8 features, tolerance 1.00): **0.72 held-out accuracy**,
0.67 ± 0.19 under 5-fold cross-validation. Task Analyzer ablation
(`results/feature_ablation.csv`): v2's eight features score 0.673 CV against v1's four
at 0.629 — an improvement, but well inside one standard deviation, so it is not a
statistically established gain.

Ablation ladder on 60 short-QA prompts (`results/ablations.csv`):

| Variant | Loss | Avg depth | Rel. FLOPs |
|---------|------|-----------|------------|
| A1 static | 3.89 | 12.00 | 1.000 |
| A2 analyzer + rule | 8.07 | 4.00 | 0.333 |
| A3/A4 predictor, depth only | 5.97 | 7.73 | 0.644 |
| A5 depth + attention | 8.40 | 7.73 | 0.536 |
| A6 depth + FFN | 7.74 | 7.73 | 0.430 |
| A8 full MSA | 9.10 | 7.73 | 0.322 |

Stability (`results/stability_experiment.csv`), on a sequence built to force
switching: with the monitor disabled, 58 switches at volatility 0.97 and no
rollbacks; with it active, volatility drops to 0.64 across 20 rollbacks and loss
does not degrade (6.74 → 6.51).

Downstream (`results/dataset_evaluation.csv`): **every variant, including static
12-layer GPT-2, scores 0.000 exact-match on GSM8K.** That dataset separates
compute behaviour, not task quality. Short QA does carry signal, and the ordering
is unflattering to adaptation: static 7.5% exact-match, depth-adaptive 2.5%, full
MSA 0.0%. On GSM8K the predictor also routes almost
everything to `deep` (average depth 11.9), so the compute saving seen on short QA
largely disappears — no single compute-reduction figure should be quoted without
naming the dataset.

Remaining measured results are in `results/tables.md`, regenerated by
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
  slicing do cut wall-clock time (17-30%), but the routing baseline uses less
  arithmetic while running ~1.9x *slower* than static on CPU. Both measures are
  reported; neither substitutes for the other.
- **The routing baseline uses a fixed seeded gate**, not a learned router.
- **Baseline D is randomly initialised.** Only its parameter count and latency are
  meaningful; its quality is reported as N/A.

No claim is made that MSA always improves performance, guarantees lower latency, or
preserves quality. See the claims checklist at the end of `results/tables.md`.

## Reproducibility

Seeds are fixed at 42 throughout. `configs/experiment.yaml` records every default;
`results/environment.json` records the Python, PyTorch, Transformers, scikit-learn and
device versions used for the benchmark run. Calibration prompts are frozen in
`data/calibration_prompts.json` so no network access is needed to reproduce a run
(rebuilding the prompt set with `--build` does fetch GSM8K).

## Benchmark

Controlled measurement (`results/benchmark.csv`): warmup 10, 50 runs, 10% trimmed
mean, CPU. The harness flags any cell whose mean exceeds 1.5x its median, so a
stray OS stall cannot silently set a headline number.

| Mechanism | Config | seq=16 | seq=64 | seq=256 |
|-----------|--------|--------|--------|---------|
| depth only | shallow | 34.8 ms | 94.3 ms | 267.6 ms |
| depth only | medium | 55.8 ms | 141.7 ms | 409.6 ms |
| depth only | deep | 77.1 ms | 183.4 ms | 518.5 ms |
| depth+attn+ffn | shallow | 28.8 ms | 79.6 ms | 193.2 ms |
| depth+attn+ffn | medium | 44.6 ms | 112.7 ms | 288.8 ms |
| routing | deep | 146.4 ms | 261.7 ms | 558.9 ms |

Attention and FFN slicing cut latency a further 17-20% at seq=16, rising to
28-30% at seq=256, on top of the depth saving. The routing baseline is slower
than the static model at every length.

## Deliverables

| Artefact | Path |
|----------|------|
| Research paper, 14 sections | `paper/msa_paper.md` |
| Demonstration page | `ui/index.html` |
| Research tables + claims checklist | `results/tables.md` |
| Figures | `results/figures/` (12) |
| Calibration data | `results/calibration_results.csv` (2,994 rows) |

### The demonstration page

`ui/index.html` is self-contained — open it directly, no server needed. Rebuild it
from current results with `python -m ui.export_ui_data`.

It splits cleanly into what is real and what is replayed:

- **Live in the browser**: the Task Analyzer's eight heuristics, the `f_theta`
  forward pass (real exported weights, 8→32→16→3), the confidence policy, and the
  stability monitor. Typing a prompt runs the genuine selection pipeline.
- **Replayed from `results/`**: every GPT-2 loss, perplexity, latency and accuracy.

The page does not run GPT-2 and does not claim to.

## Status

**Complete. All 24 phases of `plan.md` are done**, and Phase 24 validation passes
(`python test_msa.py`).
