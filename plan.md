# MSA-GPT-2 Prototype --- Completion Plan

## 1. Project Status

This document is the execution plan for completing the MSA-GPT-2
prototype from the current checkpoint through the final research
experiments and paper-ready results.

### Progress at a glance

Last updated: 2026-09-23.

| Phase | Title | Status |
|-------|-------|--------|
| 1  | Predictor validation            | DONE |
| 2  | Integrate f_theta into MSA      | DONE |
| 3  | Configuration decision policy   | DONE |
| 4  | Improve calibration dataset     | DONE (499 prompts, 2,994 experiments) |
| 5  | Improve Task Analyzer           | DONE (v2 ablated against v1) |
| 6  | Dynamic depth controller        | DONE |
| 7  | Dynamic attention controller    | DONE (head slicing) |
| 8  | FFN / token routing             | DONE (FFN chunk slicing) |
| 9  | Full dynamic controller         | DONE |
| 10 | Stability monitor               | DONE (rollback demonstrated) |
| 11 | Evaluation harness              | DONE |
| 12 | Baselines A-E                   | DONE |
| 13 | Proper benchmarking             | IMPLEMENTED, run pending |
| 14 | Dataset evaluation              | IMPLEMENTED, run in progress |
| 15 | Ablation studies                | DONE (A1-A8) |
| 16 | Stability experiments           | IMPLEMENTED, run in progress |
| 17 | Research graphs                 | DONE (9/12 rendered; 3 await Phase 14/16 output) |
| 18 | Reproducibility                 | DONE |
| 19 | Final prototype CLI             | DONE |
| 20 | Optional UI simulation          | NOT STARTED (optional) |
| 21 | Final research tables           | IMPLEMENTED, run pending |
| 22 | Research claims checklist       | DONE |
| 23 | Paper integration               | NOT STARTED (no paper file in repo) |
| 24 | Final validation                | IMPLEMENTED, run pending |

------------------------------------------------------------------------

### Correctness defect found and fixed

Before any plan work could be trusted, a defect was found in
`models/adaptive_model.py` that invalidated every previously recorded
result.

`AdaptiveGPT2.forward` passed `attention_mask=torch.ones(batch, seq)` into
each transformer block. Under Transformers v5 the SDPA attention path
computes:

``` python
is_causal = q_length > 1 and attention_mask is None and module.is_causal
```

Supplying any explicit mask therefore **disabled GPT-2's causal mask**, and
the model attended bidirectionally: every token could see its own future.

Measured consequences:

``` text
depth-12 logits vs stock GPT-2 : max |diff| = 67.4
"...the capital of France is"  : predicted "capital" instead of "Paris"
```

Everything derived from that forward pass was therefore invalid: the
original 300-run calibration, the quality-compute frontier, and the
predictor reported at 60% accuracy.

The fix is to pass `attention_mask=None`. Two regression assertions now
guard it in `test_msa.py`:

1.  depth 12 is numerically identical to the reference model
    (max |diff| = 5.3e-05, float noise only)
2.  prefix logits do not change when later tokens are removed

All results in `results/` were regenerated after the fix.

------------------------------------------------------------------------

### Completed components

-   [x] GPT-2-small backbone loaded successfully
-   [x] GPT-2-small verified as a 12-layer transformer
-   [x] Adaptive depth execution implemented
-   [x] Depth 4 / 8 / 12 forward passes verified
-   [x] **Causal masking defect found and fixed** (see above)
-   [x] Depth 12 asserted numerically identical to stock GPT-2
-   [x] Task Analyzer implemented, four heuristic features (v1)
-   [x] Task Analyzer extended to eight features (v2) and ablated
-   [x] Complexity scorer implemented
-   [x] Rule-based configuration selector retained as ablation A2
-   [x] Calibration dataset expanded from 100 to 499 prompts, 10 categories
-   [x] 499 inputs x 3 configurations x 2 mechanisms = 2,994 experiments
-   [x] Calibration results saved to `results/calibration_results.csv`
-   [x] Quality-compute frontier regenerated over 8 tolerances
-   [x] Operating tolerance selected empirically (1.00)
-   [x] 2-layer MLP configuration predictor retrained on valid data
-   [x] Predictor saved with metadata to `results/performance_predictor.pkl`
-   [x] Predictor achieves **0.72 held-out accuracy** (0.67 +/- 0.19 CV),
        up from 0.60 on the invalid data
-   [x] Configuration policy separated from the predictor
-   [x] Attention-head adaptation (weight slicing, real FLOP reduction)
-   [x] FFN-chunk adaptation (weight slicing, real FLOP reduction)
-   [x] Token-routing MoE-style baseline
-   [x] Dynamic Architecture Controller with full telemetry
-   [x] Stability monitor with volatility tracking and rollback
-   [x] Evaluation harness covering baselines A-E and ablations A1-A8
-   [x] Reproducibility config, seeds, environment capture
-   [x] Single-command CLI (`main.py --mode ...`)

### Measured results so far

Depth-only mechanism, mean over 499 calibration prompts:

``` text
depth  4   loss 9.03   relative FLOPs 0.333
depth  8   loss 7.21   relative FLOPs 0.667
depth 12   loss 3.86   relative FLOPs 1.000
```

Ablation ladder, 60 short-QA prompts (`results/ablations.csv`):

``` text
A1 static            loss 3.89   depth 12.0   FLOPs 1.000   latency 146 ms
A2 analyzer + rule   loss 8.07   depth  4.0   FLOPs 0.333   latency  63 ms
A3 predictor         loss 5.97   depth  7.73  FLOPs 0.644   latency 102 ms
A4 depth only        loss 5.97   depth  7.73  FLOPs 0.644   latency 100 ms
A5 depth + attention loss 8.40   depth  7.73  FLOPs 0.536   latency 105 ms
A6 depth + FFN       loss 7.74   depth  7.73  FLOPs 0.430   latency  94 ms
A7 full, no monitor  loss 9.10   depth  7.73  FLOPs 0.322   latency  89 ms
A8 full MSA          loss 9.10   depth  7.73  FLOPs 0.322   latency  97 ms
```

Reading: depth adaptation buys a 35.6% layer reduction and roughly a 30%
latency reduction for a loss increase from 3.89 to 5.97. Adding attention
and FFN slicing roughly halves compute again but costs substantially more
quality, because those mechanisms are applied to pretrained weights with no
retraining or distillation. This is a measured result, not a tuning failure
to be hidden.

### Current checkpoint

Phases 1-12, 15, 17-19 and 22 are implemented and have produced results.
Phases 13, 14, 16, 21 and 24 are implemented and awaiting their run to
complete. Phases 20 and 23 are not started.

------------------------------------------------------------------------

# 2. Target Prototype Architecture

The completed prototype should implement the following pipeline:

``` text
                    ┌──────────────────────┐
Input ─────────────►│    Task Analyzer     │
                    └──────────┬───────────┘
                               │
                         Feature Vector
                     [L, R, D, S]
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Configuration Model  │
                    │        f_theta       │
                    │      2-layer MLP      │
                    └──────────┬───────────┘
                               │
                  P(shallow), P(medium), P(deep)
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Configuration Policy │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
         SHALLOW            MEDIUM              DEEP
          Depth 4            Depth 8            Depth 12
             └─────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Dynamic Architecture │
                    │      Controller      │
                    └──────────┬───────────┘
                               │
                               ▼
                         Adaptive GPT-2
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Stability Monitor    │
                    │ + Rollback Policy    │
                    └──────────┬───────────┘
                               │
                               ▼
                             Output
```

The prototype should eventually support three primary depth
configurations:

``` text
shallow = 4 layers
medium  = 8 layers
deep    = 12 layers
```

The dynamic controller should then be extended beyond depth so that the
prototype demonstrates the broader MSA concept.

------------------------------------------------------------------------

# 3. Phase 1 --- Predictor Validation

**STATUS: DONE.** `controller/predictor.py` and `test_predictor.py` exist and
pass. The predictor loads from `results/performance_predictor.pkl`, accepts the
feature dict, and returns configuration / depth / confidence / probabilities.
`test_predictor.py` asserts the configuration is one of shallow/medium/deep,
that it maps to depth 4/8/12, that probabilities sum to 1 with no NaN, and that
a malformed feature dict raises rather than silently mispredicting. The
predictor now reads its feature list from metadata saved at training time, so v1
and v2 models load through the same interface.

## Goal

Verify that the trained `f_theta` model can be loaded independently and
can transform Task Analyzer features into configuration probabilities.

## Files

Create:

``` text
controller/predictor.py
test_predictor.py
```

## Predictor API

`controller/predictor.py` should expose a simple interface:

``` python
predictor.predict(features)
```

Expected result:

``` python
{
    "configuration": "medium",
    "depth": 8,
    "confidence": 0.61,
    "probabilities": {
        "shallow": 0.10,
        "medium": 0.61,
        "deep": 0.29
    }
}
```

## Test

Run:

``` powershell
python test_predictor.py
```

Verify:

-   Model loads from `results/performance_predictor.pkl`
-   Four features are accepted
-   Probabilities are produced
-   Configuration is one of shallow / medium / deep
-   Configuration maps to depth 4 / 8 / 12
-   No NaN or invalid values are produced

## Completion criterion

The predictor works independently before being connected to the rest of
MSA.

------------------------------------------------------------------------

# 4. Phase 2 --- Integrate f_theta into MSA

**STATUS: DONE.** `run_msa.py` was rewritten around an `MSA` class. The
hard-coded threshold path is gone from the main route; `ConfigurationSelector`
survives only as ablation A2. The runner returns features, probabilities,
configuration, depth, confidence, full controller telemetry and logits shape.
All six input categories the plan lists are covered by `TEST_INPUTS`.
`python run_msa.py` runs Input -> Analyzer -> Predictor -> Policy -> Monitor ->
Controller -> Adaptive GPT-2 with no manual intervention.

## Goal

Replace the hard-coded threshold selector in the main MSA path with the
trained predictor.

Current conceptual path:

``` text
Task Analyzer
      ↓
Complexity Scorer
      ↓
Hard-coded thresholds
      ↓
Depth
```

Target path:

``` text
Task Analyzer
      ↓
f_theta
      ↓
Configuration
      ↓
Depth
```

## Files

Update:

``` text
run_msa.py
```

Optionally update:

``` text
controller/selector.py
```

## Requirements

The main MSA runner should return:

``` python
{
    "features": ...,
    "configuration": ...,
    "depth": ...,
    "confidence": ...,
    "probabilities": ...,
    "logits": ...
}
```

## Test cases

Run several inputs representing:

-   factual questions
-   short explanations
-   mathematical questions
-   multi-step reasoning
-   technical questions
-   long structured prompts

## Completion criterion

A single command should run:

``` text
Input
→ Analyzer
→ Predictor
→ Configuration
→ Adaptive GPT-2
```

without manual intervention.

------------------------------------------------------------------------

# 5. Phase 3 --- Configuration Decision Policy

**STATUS: DONE.** `ConfigurationPolicy` in `controller/policy.py` is separate
from the predictor. It takes argmax, applies a confidence threshold (default
0.50), and falls back to a configurable configuration. Every inference records
predicted probabilities (`p_shallow` / `p_medium` / `p_deep`), selected
configuration, confidence, whether a fallback was applied, and selected depth,
into `results/inference_log.csv`. The threshold is exposed as a CLI flag and
treated as a tunable operating point, not an established constant.

## Goal

Separate the neural predictor from the final safety/compute policy.

The predictor estimates:

``` text
P(shallow)
P(medium)
P(deep)
```

The policy decides the final configuration.

## Initial policy

Use the highest-probability configuration.

Add a confidence threshold.

Example:

``` text
if confidence >= threshold:
    use predicted configuration
else:
    use medium/deep fallback
```

The exact threshold should be experimentally evaluated rather than
treated as scientifically established.

## Record for every inference

-   predicted probabilities
-   selected configuration
-   confidence
-   fallback decision if any
-   selected depth

## Completion criterion

The decision mechanism is explicit, inspectable, and logged.

------------------------------------------------------------------------

# 6. Phase 4 --- Improve Calibration Dataset

**STATUS: DONE.** The dataset grew from 100 to **499 unique prompts** across all
ten categories the plan lists (`data/calibration_prompts.json`, rebuildable with
`python -m data.calibration_dataset --build`). Real multi-step reasoning comes
from GSM8K rather than hand-written approximations. Each prompt runs at three
configurations under two mechanisms, giving **2,994 experiments**, and each row
records input, all eight features, complexity, depth, attention/FFN mode, loss,
perplexity, latency, relative FLOPs and configuration.

The methodological note is respected: prompt loss is used as the calibration
signal only, and downstream task metrics are reported separately in
`results/dataset_evaluation.csv`.

## Goal

Increase the reliability of the learned configuration policy.

Current calibration:

``` text
100 inputs
3 depths
300 experiments
```

Expand the dataset in stages.

Recommended categories:

1.  Factual questions
2.  Basic arithmetic
3.  Multi-step arithmetic
4.  Scientific explanations
5.  Logical reasoning
6.  Technical questions
7.  Long structured prompts
8.  Summarization
9.  Code-related prompts
10. Hard reasoning tasks

Target:

``` text
500–1,000 unique inputs
```

For every input:

``` text
depth 4
depth 8
depth 12
```

Record:

``` text
input
features
complexity
depth
loss
perplexity
latency
configuration
```

## Important methodological note

The current loss is next-token language-model loss over the prompt. It
should be treated as a prototype calibration signal, not as the final
downstream task-quality metric.

------------------------------------------------------------------------

# 7. Phase 5 --- Improve Task Analyzer

**STATUS: DONE.** Version 1 (the original four features) is kept unchanged as
the baseline. Version 2 adds exactly four features, not dozens:
`numeric_density`, `math_expression`, `question_type`, `reasoning_steps`.

Ablation result (`results/feature_ablation.csv`):

``` text
v1   4 features   held-out 0.700   5-fold CV 0.629 +/- 0.154
v2   8 features   held-out 0.720   5-fold CV 0.673 +/- 0.188
```

v2 is better on both measures, but the gap is well inside one standard
deviation. The honest statement for the paper is that the added features did not
produce a statistically established improvement on this dataset.

## Goal

Determine whether the four heuristic features contain enough information
to predict configuration requirements.

Current features:

``` text
input_length
reasoning
domain
structure
```

## Version 1

Keep the existing heuristic analyzer as the baseline.

## Version 2

Improve feature extraction.

Potential additions:

-   token count instead of only word count
-   question type
-   mathematical expression detection
-   code detection
-   number of entities
-   sentence count
-   nesting depth
-   numeric density
-   instruction density
-   estimated reasoning steps

Do not immediately add dozens of features.

Run an ablation:

``` text
4 original features
vs
4 original + selected additional features
```

Measure whether predictor performance improves.

## Completion criterion

The final paper should clearly state which features were used and why.

------------------------------------------------------------------------

# 8. Phase 6 --- Dynamic Architecture Controller: Depth

**STATUS: DONE.** `controller/controller.py` isolates depth adaptation as a
reproducible module and is the single execution path used by calibration,
evaluation and benchmarking, so those cannot drift apart. Telemetry per
inference: selected depth, layers executed, layers skipped, layer reduction,
active heads, active FFN chunks, sequence length, relative FLOPs, latency and
loss. `python -m controller.controller` asserts that each requested depth is the
depth actually executed.

## Goal

Formalize depth adaptation as the first dynamic architecture mechanism.

The controller should support:

``` text
SHALLOW → blocks 0–3
MEDIUM  → blocks 0–7
DEEP    → blocks 0–11
```

## Required telemetry

For every inference:

``` text
selected depth
layers executed
layers skipped
latency
loss
```

## Test

Run the same input through:

``` text
depth 4
depth 8
depth 12
```

and verify that the controller selects and executes the requested
configuration.

## Completion criterion

Depth adaptation is isolated as a reproducible module.

------------------------------------------------------------------------

# 9. Phase 7 --- Dynamic Attention Controller

**STATUS: DONE.** Implemented as attention-head reduction, the least risky
option the plan lists. Critically, it **slices the weight tensors rather than
masking outputs**: the QKV columns and output-projection rows for inactive heads
are never multiplied, so the skipped work is genuinely not performed. Pretrained
weights are never modified; slicing happens per forward pass.

`test_msa.py` asserts the mechanism changes the computed output, so it cannot
degrade into a UI label. The depth-only vs depth+attention comparison the plan
requires is ablation A4 vs A5.

Measured: A5 cuts relative FLOPs from 0.644 to 0.536 but raises loss from 5.97
to 8.40. On CPU at short sequence lengths the slicing overhead cancels most of
the latency benefit (102 ms -> 105 ms). Both facts are reported.

## Goal

Extend MSA beyond depth skipping.

The prototype should support an attention-level adaptation mechanism.

Possible implementation:

``` text
Full attention
Reduced active heads
Optional local attention mask
```

Start with the least risky implementation:

### Attention-head masking

For selected configurations, mask a subset of attention heads.

Example conceptual configurations:

``` text
Deep:
12 layers
all heads active

Medium:
8 layers
reduced head activity

Shallow:
4 layers
reduced head activity
```

Do not modify the pretrained weights permanently.

The controller should dynamically determine the active attention
behavior during inference.

## Required experiment

Compare:

``` text
depth-only
vs
depth + attention adaptation
```

Measure:

-   loss
-   latency
-   configuration distribution
-   compute reduction

## Completion criterion

Attention adaptation changes actual computation behavior rather than
merely changing a UI label.

------------------------------------------------------------------------

# 10. Phase 8 --- FFN / Token Routing

**STATUS: DONE.** FFN chunking was implemented first, as recommended, and
validated before routing was added. The 3072 intermediate channels are treated
as 4 chunks; a configuration activates a subset by slicing `c_fc` columns and
`c_proj` rows.

Token-level routing exists separately as Baseline C: a seeded fixed gate routes
each token to top-2 of 4 chunks, with per-token gather/scatter so the compute
reduction is real. The gate is untrained, documented as a limitation rather than
presented as a learned MoE.

Measured: A6 (depth + FFN) reaches 0.430 relative FLOPs at loss 7.74. The
routing baseline reaches 0.668 relative FLOPs but is *slower* in wall-clock
(296 ms vs 146 ms static), because per-token gather/scatter on CPU costs more
than the arithmetic it saves. Reported as measured.

## Goal

Add a second architectural adaptation mechanism.

The original MSA design allows:

``` text
FFN chunking
Token routing
Subset FFN execution
```

For the first prototype implementation, choose one mechanism and
implement it cleanly.

Recommended prototype:

``` text
FFN chunking
```

Concept:

``` text
FFN
 ├── Chunk 1
 ├── Chunk 2
 ├── Chunk 3
 └── Chunk 4
```

The controller can activate a subset of chunks based on configuration.

Alternative:

``` text
token-level routing
```

if the implementation is stable enough.

## Important

Do not implement both mechanisms simultaneously before validating one.

## Completion criterion

At least one non-depth architectural adaptation mechanism is functional
and experimentally measurable.

------------------------------------------------------------------------

# 11. Phase 9 --- Full Dynamic Architecture Controller

**STATUS: DONE.** The configuration library lives in
`configs/configurations.py` and is deliberately kept to three entries so every
configuration can be calibrated:

``` python
shallow = {"depth":  4, "attention_mode": "reduced", "ffn_mode": "partial"}
medium  = {"depth":  8, "attention_mode": "reduced", "ffn_mode": "partial"}
deep    = {"depth": 12, "attention_mode": "full",    "ffn_mode": "full"}
```

Every configuration is calibrated empirically: the calibration sweep runs both
the depth-only and the full mechanism, recording both in the same CSV under a
`mechanism` column.

The warning in this section is respected. Compute is reported two ways -- an
analytical `relative_flops` estimate and measured wall-clock latency -- exactly
because they do not agree. Fewer layers did not produce proportional latency
savings.

## Goal

Combine the adaptive mechanisms.

Target configuration representation:

``` python
{
    "depth": 8,
    "attention_mode": "reduced",
    "ffn_mode": "partial"
}
```

Possible configurations:

``` text
Configuration A:
depth = 4
attention = reduced
ffn = partial

Configuration B:
depth = 8
attention = reduced
ffn = partial

Configuration C:
depth = 12
attention = full
ffn = full
```

The exact configuration library should be kept small enough to
calibrate.

## Important

Every configuration must be calibrated empirically.

Do not assume that fewer layers automatically means proportional compute
savings.

------------------------------------------------------------------------

# 12. Phase 10 --- Stability Monitor

**STATUS: DONE.** `monitor/stability_monitor.py` tracks a rolling window of
recent configurations, the switch count, rolling volatility and rollback count,
and reports STABLE / WARNING / UNSTABLE. On instability it compares against the
last stable configuration, rolls back to it, and records the event.

`python -m monitor.stability_monitor` runs the synthetic instability test this
section requires and asserts a rollback actually occurs:

``` text
Stability status: WARNING
Volatility: 0.60
Rollback count: 3
Current configuration: medium
```

The monitor also supports `enabled=False`, which observes without acting; that
is exactly ablation A7.

## Goal

Implement the final major MSA component.

The monitor observes recent configuration decisions.

Track:

``` text
recent configurations
configuration changes
rolling volatility
rollback count
```

Conceptual metrics:

``` text
S(t)
volatility
rollback rate
```

## Example

Recent sequence:

``` text
deep
medium
deep
shallow
deep
medium
```

High switching frequency may trigger instability.

The monitor should then:

``` text
1. Detect instability
2. Compare with previous stable configuration
3. Roll back to previous configuration
4. Record rollback event
```

## Required output

``` text
Stability status:
STABLE / WARNING / UNSTABLE

Volatility:
0.xx

Rollback count:
N

Current configuration:
medium
```

## Completion criterion

A synthetic instability test demonstrates that rollback actually occurs.

------------------------------------------------------------------------

# 13. Phase 11 --- Build the Evaluation Harness

**STATUS: DONE, with a deliberate structural change.** This section suggests
five separate scripts (`evaluate_static.py`, `evaluate_depth_adaptive.py`,
`evaluate_msa.py`, `metrics.py`, `runner.py`). Those would be near-identical
loops that drift apart as the system changes.

Instead every baseline and ablation is one entry in a `VARIANTS` table in
`evaluation/harness.py`, differing only by flags (selector, attention, ffn,
routing, monitor). One execution loop serves all of them, so a change to the
pipeline cannot silently apply to some variants and not others.

`evaluation/metrics.py` produces the standardised record: loss, perplexity,
latency mean/std/p50, throughput, average depth, layer reduction, relative
FLOPs, compute reduction, configuration distribution, fallback rate, mean
confidence, and -- when the monitor is active -- switches, volatility, rollback
count and rollback rate. Task-specific metrics (answer perplexity, exact-match
accuracy) are added automatically when the dataset supplies answers.

`python -m evaluation.evaluate --suite all` is the single reproducible entry
point.

## Goal

Create one reproducible evaluation entry point.

Recommended structure:

``` text
evaluation/
├── evaluate_static.py
├── evaluate_depth_adaptive.py
├── evaluate_msa.py
├── metrics.py
└── runner.py
```

The evaluation system should produce standardized records.

Metrics:

``` text
loss
perplexity
latency
average depth
layer reduction
estimated compute
configuration distribution
stability
rollback rate
```

For downstream datasets, add task-specific metrics where appropriate.

------------------------------------------------------------------------

# 14. Phase 12 --- Baselines

**STATUS: DONE.**

``` text
Baseline A  Static GPT-2, 12 layers          variant "static"        DONE
Baseline B  Depth adaptive only              variant "depth_adaptive" DONE
Baseline C  Routing / MoE-style              variant "routing"       DONE
Baseline D  Contemporary dense reference     evaluation/baselines.py DONE
Baseline E  Full MSA                         variant "msa"           DONE
```

Baseline C comparison definition, stated explicitly: token-level top-2-of-4 FFN
expert routing with a fixed seeded random gate, no learned router.

Baseline D is a small dense decoder with RMSNorm, RoPE and SwiGLU and no
adaptivity, in `evaluation/baselines.py`. It is **randomly initialised**, so its
quality is reported as N/A and only its parameter count and latency are
meaningful. Size and configuration are recorded in
`results/dense_reference.csv`.

The comparison is descriptive. Full MSA is not framed as a winner -- on quality
it is the worst variant tested, which the tables state plainly.

Implement and evaluate the required baselines.

## Baseline A --- Static GPT-2

Always execute:

``` text
12 layers
```

This is the quality/compute reference.

## Baseline B --- Depth Adaptive

Only depth changes:

``` text
4 / 8 / 12
```

No attention adaptation.

No FFN adaptation.

No stability monitor.

## Baseline C --- Routing / MoE-style baseline

Implement a comparable routing-only mechanism.

Keep the comparison definition explicit.

## Baseline D --- Contemporary Dense Reference

Use a separate small dense model with the planned architecture
characteristics:

``` text
RMSNorm
RoPE
SwiGLU
no adaptivity
```

Document model size and configuration.

## Baseline E --- Full MSA

Enable:

``` text
Task Analyzer
Predictor
Dynamic depth
Attention adaptation
FFN/token adaptation
Stability monitor
```

The comparison must be descriptive, not framed as a guaranteed winner.

------------------------------------------------------------------------

# 15. Phase 13 --- Proper Benchmarking

**STATUS: IMPLEMENTED, run pending.** `evaluation/benchmark.py` measures with
warmup = 10 and 50 measured runs at fixed sequence lengths [16, 64, 256], across
all three mechanisms and configurations. It records mean, standard deviation,
p50, p95 and throughput alongside relative FLOPs and layer reduction, and writes
the full software/hardware environment to `results/environment.json`.

No single-measurement comparisons are made anywhere in the results.

Remaining: execute the run. It was deliberately deferred so it does not contend
for CPU with the evaluation suite, which would corrupt the timings.

## Goal

Replace rough prototype timing with reproducible measurements.

For every benchmark:

-   fixed hardware
-   fixed software environment
-   fixed sequence lengths
-   warm-up runs
-   repeated runs
-   average latency
-   standard deviation
-   CPU/GPU clearly reported

Example:

``` text
warmup = 10
measured runs = 50
```

Do not compare a single noisy latency measurement against another single
measurement.

## Record

``` text
mean latency
std latency
throughput
loss
perplexity
depth
layer reduction
```

------------------------------------------------------------------------

# 16. Phase 14 --- Dataset Evaluation

**STATUS: IMPLEMENTED, run in progress.** Two datasets are wired up in
`data/load_data.py`: a local 60-item short factual QA set, and GSM8K via
`openai/gsm8k`. Static, depth-adaptive and full MSA are each evaluated on both.

The warning in this section is respected directly: prompt loss is **not** used
as a substitute for task accuracy. Each record scores the answer tokens
conditioned on the question (answer-conditional perplexity) and, where
generation is enabled, exact-match accuracy via greedy decoding.

Expected and to be reported honestly: GPT-2 small cannot solve GSM8K, so
accuracy there will be near zero for every variant including the static
baseline. That dataset therefore separates compute behaviour, not task quality.

Summarization and code datasets were not added; this section lists them as
conditional on time.

Start with two datasets already aligned with the project plan:

``` text
Easy / short QA or classification
GSM8K
```

Then expand to:

``` text
summarization
code
```

if the implementation and time permit.

For each dataset report:

``` text
Static
Depth Adaptive
Full MSA
```

and additional baselines where implemented.

Use task-appropriate metrics.

Important:

Do not use language-model prompt loss as a substitute for task accuracy
when a proper downstream metric is available.

------------------------------------------------------------------------

# 17. Phase 15 --- Ablation Studies

**STATUS: DONE.** All eight ablations A1-A8 are defined in
`evaluation/harness.py` and were run; results are in `results/ablations.csv` and
reproduced in the summary table in section 1 above.

Incremental contribution as measured:

``` text
A1 -> A3   learned predictor vs static   loss 3.89 -> 5.97, FLOPs 1.000 -> 0.644
A2 -> A3   predictor vs rule selector    loss 8.07 -> 5.97 at higher depth
A4 -> A5   attention adaptation          FLOPs 0.644 -> 0.536, loss 5.97 -> 8.40
A4 -> A6   FFN adaptation                FLOPs 0.644 -> 0.430, loss 5.97 -> 7.74
A7 -> A8   stability monitor             no difference on this stable input set
```

A7 and A8 are identical here because the short-QA sequence never became
unstable. The contribution of the monitor is measured by Phase 16 instead, on a
sequence constructed to switch.

Ablations are critical because they demonstrate which parts of MSA
actually matter.

Run:

### A1 --- Static GPT-2

No adaptation.

### A2 --- Analyzer only

Analyzer + rule policy.

### A3 --- Predictor

Analyzer + f_theta.

### A4 --- Depth only

Dynamic depth without attention/FFN adaptation.

### A5 --- Depth + Attention

Add attention adaptation.

### A6 --- Depth + FFN

Add FFN/token adaptation.

### A7 --- Full MSA without Stability Monitor

Everything except rollback.

### A8 --- Full MSA

Everything enabled.

The goal is to measure the incremental contribution of each mechanism.

------------------------------------------------------------------------

# 18. Phase 16 --- Stability Experiments

**STATUS: IMPLEMENTED, run in progress.** `suite_stability` in
`evaluation/evaluate.py` builds a controlled alternating easy/hard sequence
designed to encourage configuration switching, then runs it twice: A7 (monitor
observing but disabled) against A8 (monitor acting). It measures switches,
switch rate, volatility, rollback events, rollback rate and post-rollback loss,
writing both a summary and a per-inference trajectory log.

A synthetic control assertion runs alongside it and fails loudly if a forced
alternation does not trigger rollback.

Create controlled sequences that encourage configuration switching.

Measure:

``` text
configuration volatility
number of switches
rollback events
rollback rate
quality after rollback
```

Compare:

``` text
MSA without monitor
vs
MSA with monitor
```

This is necessary to demonstrate why the Stability Monitor exists.

------------------------------------------------------------------------

# 19. Phase 17 --- Generate Research Graphs

**STATUS: DONE (9 of 12 rendered).** `evaluation/generate_plots.py` regenerates
every figure from the result CSVs; nothing is a hand-edited screenshot. Figures
go to `results/figures/`.

Rendered: depth vs loss, depth vs latency, quality-compute frontier,
configuration distribution, layer reduction, latency distribution, predictor
confusion matrix, predictor confidence distribution, baseline comparison.

Pending only because their source CSVs come from the Phase 14 and 16 runs still
in progress: average depth per dataset, stability volatility over time, rollback
events. The script skips missing sources with an explicit note rather than
failing.

Create a dedicated script:

``` text
evaluation/generate_plots.py
```

Generate:

1.  Depth vs loss
2.  Depth vs latency
3.  Quality-compute frontier
4.  Configuration distribution
5.  Average depth per dataset
6.  Layer reduction
7.  Latency distribution
8.  Predictor confusion matrix
9.  Predictor confidence distribution
10. Stability volatility over time
11. Rollback events
12. Baseline comparison

Save figures to:

``` text
results/figures/
```

Use reproducible plotting scripts rather than manually edited
screenshots.

------------------------------------------------------------------------

# 20. Phase 18 --- Reproducibility

**STATUS: DONE.** `configs/experiment.yaml` records model, dataset, seed, depth
configurations, predictor path, tolerance, benchmark and warmup runs, device,
policy thresholds, monitor thresholds and routing parameters.

Seeds are fixed at 42 throughout. `results/environment.json` captures Python,
PyTorch, Transformers, scikit-learn and NumPy versions, platform, device, CUDA
and thread count. Calibration prompts are frozen in
`data/calibration_prompts.json` so a reproduction needs no network access.

`README.md` was rewritten with full setup and experiment instructions, the
causal-masking defect, the measured findings, and an explicit limitations
section.

Create:

``` text
configs/experiment.yaml
```

or equivalent configuration files containing:

``` text
model
dataset
seed
depth configurations
predictor path
tolerance
benchmark runs
warmup runs
device
```

Set random seeds.

Record:

``` text
Python version
PyTorch version
Transformers version
CUDA version if applicable
GPU/CPU
```

Update:

``` text
README.md
```

with complete setup and experiment instructions.

------------------------------------------------------------------------

# 21. Phase 19 --- Final Prototype CLI

**STATUS: DONE.** `main.py` exposes every stage through one command:

``` powershell
python main.py --mode simulate
python main.py --mode calibrate
python main.py --mode frontier
python main.py --mode train-predictor
python main.py --mode ablate-features
python main.py --mode benchmark
python main.py --mode evaluate
python main.py --mode plot
python main.py --mode tables
python main.py --mode validate
python main.py --mode all
```

Arguments after `--` pass through to the underlying module. `--mode all` runs
the full pipeline in dependency order.

Create a single command for demonstration.

Example:

``` powershell
python main.py --mode simulate
```

and:

``` powershell
python main.py --mode evaluate
```

Potential modes:

``` text
simulate
calibrate
train-predictor
evaluate
benchmark
plot
```

The final demo should be easy to run without opening individual scripts.

------------------------------------------------------------------------

# 22. Phase 20 --- Optional UI Simulation

**STATUS: NOT STARTED.** Explicitly optional and deliberately left out. The
backend now exposes everything such a UI would need: `run_msa.py` returns
features, per-class probabilities, configuration, depth, attention and FFN
modes, active heads and chunks, relative FLOPs, loss, latency and stability
status as a plain dict.

If built later, the honesty requirement in this section stands: it must not
claim browser-side visualisation is performing real GPT-2 inference unless it is
wired to this backend.

The v0.app simulation can be used as a visual demonstration layer.

UI flow:

``` text
Input
 ↓
Task Analyzer
 ↓
Feature Vector
 ↓
Complexity
 ↓
Predictor
 ↓
Configuration
 ↓
Dynamic GPT-2
 ↓
Stability Monitor
 ↓
Output
```

The UI should clearly label simulated values.

It should not claim that browser-side visualization is performing actual
GPT-2 inference unless the real backend is connected.

Recommended UI sections:

-   MSA pipeline
-   Task Analyzer
-   Predictor probabilities
-   12-layer architecture visualization
-   Inference telemetry
-   Stability Monitor
-   Quality-compute frontier
-   Experiment history
-   Baseline comparison

------------------------------------------------------------------------

# 23. Phase 21 --- Final Research Tables

**STATUS: IMPLEMENTED, run pending.** `evaluation/tables.py` generates all four
required tables plus two more, reading every number back from the result CSVs so
nothing in the paper is typed by hand:

``` text
Table 1  Model and system configuration
Table 2  Main results (baselines, including the dense reference)
Table 3  Ablation studies
Table 4  Stability
Table 5  Downstream dataset evaluation
Table 6  Task Analyzer feature ablation
```

Output goes to `results/tables.md` plus one CSV per table. Remaining: execute
once the Phase 14 and 16 runs finish supplying their inputs.

Prepare final tables for the paper.

### Table 1 --- Model and system configuration

Include:

``` text
Backbone
Parameters
Layers
Feature dimensions
Configurations
Predictor architecture
```

### Table 2 --- Main results

Include:

``` text
Method
Quality metric
Latency
Average depth
Layer reduction
Compute estimate
```

### Table 3 --- Ablation

Include:

``` text
Variant
Quality
Latency
Layer reduction
Stability
```

### Table 4 --- Stability

Include:

``` text
Variant
Switches
Volatility
Rollbacks
Rollback rate
```

------------------------------------------------------------------------

# 24. Phase 22 --- Research Claims Checklist

**STATUS: DONE.** The checklist is generated into `results/tables.md` by
`evaluation/tables.py`, with every entry tagged MEASURED, LIMITATION,
HYPOTHESIS or FUTURE WORK.

The three forbidden claims are explicitly disclaimed. The recorded limitations
include: prompt loss is a calibration signal and not task quality; GPT-2 small
cannot solve GSM8K; attention and FFN adaptation degrade quality markedly
because they are applied to pretrained weights without retraining; the routing
gate is untrained; Baseline D is randomly initialised; and FLOP reduction does
not reliably translate into wall-clock reduction on CPU.

Before writing conclusions, verify every claim against measured results.

Avoid claims such as:

``` text
"MSA always improves performance."
"MSA guarantees lower latency."
"MSA preserves quality."
```

unless the experiments actually establish those statements.

Prefer measured statements such as:

``` text
"Under the tested calibration policy, the system selected..."
```

and:

``` text
"The experiment measured..."
```

Clearly distinguish:

-   measured result
-   design objective
-   hypothesis
-   limitation
-   future work

------------------------------------------------------------------------

# 25. Phase 23 --- Paper Integration

**STATUS: NOT STARTED.** There is no paper file in this repository, so there is
nothing to integrate into. Phases 21 and 22 produce the inputs a paper would
consume: `results/tables.md` holds the tables and the claims checklist, and
`results/figures/` holds the figures.

The architecture-to-code correspondence this section asks for does hold.
Section 2's pipeline diagram maps one-to-one onto `analyzer/task_analyzer.py`,
`controller/predictor.py`, `controller/policy.py`, `controller/controller.py`,
`models/adaptive_model.py` and `monitor/stability_monitor.py`.

After the final experiments are stable, update the research paper.

Recommended sections:

``` text
1. Introduction
2. Related Work
3. Problem Definition
4. MSA Architecture
5. Task Analyzer
6. Configuration Predictor
7. Dynamic Architecture Controller
8. Stability Monitor
9. Experimental Setup
10. Baselines
11. Results
12. Ablation Studies
13. Limitations
14. Conclusion
```

The architecture equations should correspond directly to implemented
components.

------------------------------------------------------------------------

# 26. Phase 24 --- Final Validation

**STATUS: IMPLEMENTED, run pending.** The checklist below is no longer a
document to tick by hand: it is executable as `python test_msa.py`, which turns
each core-system item into an assertion and each artefact item into a file
check. Items marked [x] have been individually verified during development;
the full suite runs once the outstanding evaluation artefacts exist.

## Core system

-   [x] GPT-2 loads from a clean environment
-   [x] Depth 12 is numerically identical to stock GPT-2 (max |diff| 5.3e-05)
-   [x] Causal masking is applied (prefix-invariance asserted)
-   [x] Adaptive depth works (loss strictly decreases 4 -> 8 -> 12)
-   [x] Predictor loads
-   [x] Predictor produces valid probabilities (sum to 1, no NaN)
-   [x] Configuration policy works (low confidence falls back)
-   [x] Dynamic controller changes actual execution
        (depth, attention, FFN and routing each alter the output)
-   [x] Compute estimates decrease with adaptation
-   [x] Telemetry is complete
-   [x] Stability monitor detects instability
-   [x] Rollback works
-   [x] End-to-end MSA runs without manual intervention

## Evaluation

-   [x] Calibration reproducible (prompts frozen to JSON, seeded)
-   [x] Baselines implemented (A-E)
-   [x] Ablations completed (A1-A8)
-   [x] Benchmark harness uses warmups and multiple runs
-   [ ] Benchmark executed
-   [ ] Downstream metrics used where available (run in progress)
-   [ ] Stability experiment executed (run in progress)

## Reproducibility

-   [x] Seeds recorded (42 throughout, in `configs/experiment.yaml`)
-   [x] Model versions recorded
-   [x] Dataset versions recorded
-   [x] Experiment configuration saved (`configs/experiment.yaml`)
-   [x] Results saved as CSV
-   [x] Graphs generated automatically
-   [ ] Environment recorded (written by the pending benchmark run)

## Research quality

-   [x] No unsupported performance claims
-   [x] Limitations documented (README and claims checklist)
-   [x] Prototype vs final implementation clearly distinguished
-   [x] Prompt-loss calibration not presented as downstream task accuracy
-   [x] All major architectural claims correspond to actual code

------------------------------------------------------------------------

# 27. Execution Order --- Progress

The order below was followed. A step marked [x] is complete.

``` text
[x]  0. Fix the causal-masking defect (unplanned, blocking)
[x]  1. Test predictor
[x]  2. Integrate f_theta
[x]  3. End-to-end MSA test
[x]  4. Expand calibration dataset          100 -> 499 prompts
[x]  5. Improve Task Analyzer               v1 vs v2 ablation
[x]  6. Formalize Dynamic Depth Controller
[x]  7. Implement attention adaptation      head slicing
[x]  8. Implement FFN/token adaptation      chunk slicing + routing
[x]  9. Combine Dynamic Architecture Controller
[x] 10. Implement Stability Monitor
[x] 11. Build Evaluation Harness
[x] 12. Implement Baselines                 A-E
[ ] 13. Proper Benchmarking                 implemented, run pending
[~] 14. Dataset Evaluation                  implemented, run in progress
[x] 15. Ablations                           A1-A8
[~] 16. Stability Experiments               implemented, run in progress
[~] 17. Generate Graphs                     9/12 rendered
[x] 18. Reproducibility cleanup
[ ] 19. UI demonstration                    optional, not started
[ ] 20. Final tables                        implemented, run pending
[ ] 21. Paper integration                   no paper file in repo
[ ] 22. Final validation                    implemented, run pending
```

------------------------------------------------------------------------

# 28. Definition of Done

The MSA-GPT-2 prototype is considered complete when a clean run can
execute:

``` text
Input
  ↓
Task Analyzer
  ↓
Feature Vector
  ↓
f_theta
  ↓
Configuration Selection
  ↓
Dynamic Depth
  ↓
Attention / FFN Adaptation
  ↓
GPT-2 Execution
  ↓
Stability Monitor
  ↓
Rollback if necessary
  ↓
Output
```

and the evaluation framework can compare:

``` text
Static GPT-2
Depth-Adaptive GPT-2
Routing/MoE-style baseline
Contemporary Dense Reference
Full MSA
```

with reproducible measurements for:

``` text
quality
latency
compute
average depth
layer reduction
configuration distribution
stability
rollback
```

The final result should be a working research prototype, an executable
experiment pipeline, a complete set of results/figures, and a
reproducible codebase suitable for supporting the MSA research paper.

## Status against this definition

**The full pipeline runs.** `python run_msa.py` executes Input -> Task Analyzer
-> Feature Vector -> f_theta -> Configuration Selection -> Dynamic Depth ->
Attention/FFN Adaptation -> GPT-2 Execution -> Stability Monitor -> Rollback ->
Output in one command, with no manual intervention.

**All five comparison arms exist** and are runnable from one entry point
(`python -m evaluation.evaluate --suite all`). Baselines A, B, C and E have
produced measured results; Baseline D produces parameter count and latency only,
because it is randomly initialised.

**Measurements available:** quality (prompt loss, perplexity, answer-conditional
perplexity, exact match), latency (mean/std/p50/p95 with warmups), compute
(analytical relative FLOPs), average depth, layer reduction, configuration
distribution, stability (switches, volatility) and rollback (count, rate).

**Outstanding before the definition is fully met:** the benchmark, dataset,
stability, tables and validation runs need to finish writing their artefacts.
No further implementation is required for any of them.

------------------------------------------------------------------------

# 29. Immediate Next Action

The predictor milestone is complete:

``` text
Predictor  ->  Integration  ->  End-to-end MSA        DONE
```

Remaining work, in order. All of it is executing already-written code; no new
implementation is outstanding except the two optional phases.

``` text
1. Finish the dataset and stability runs   (in progress)
2. python main.py --mode benchmark         (run on a quiet CPU)
3. python main.py --mode plot              (fills the last 3 figures)
4. python main.py --mode tables            (writes results/tables.md)
5. python main.py --mode validate          (Phase 24 checklist)
```

Steps 2-5 are also reachable in one command:

``` powershell
python main.py --mode all
```

Note that `--mode all` re-runs calibration first, which takes roughly 25
minutes; skip to the individual modes if the existing
`results/calibration_results.csv` is still current.

## Known open questions for the research write-up

1.  Attention and FFN adaptation currently cost far more quality than they
    save in compute, because they slice pretrained weights with no retraining.
    Whether a distilled or retrained variant closes that gap is untested and
    should be stated as future work, not as an expected result.

2.  The predictor is trained against prompt loss. A predictor trained against a
    downstream task metric is the obvious next experiment and is not yet run.

3.  The v2 feature set improves cross-validated accuracy from 0.629 to 0.673,
    but the standard deviation is roughly 0.18. More calibration data is needed
    before that difference can be called real.

4.  On CPU at short sequence lengths, analytical FLOP reduction and measured
    latency reduction diverge, and the routing baseline is slower than the
    static model despite using less arithmetic. Any compute claim in the paper
    must say which of the two it refers to.
