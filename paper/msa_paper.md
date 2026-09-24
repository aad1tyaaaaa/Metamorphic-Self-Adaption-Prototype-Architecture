# Metamorphic Self-Adaptation: Per-Input Architecture Selection in a Pretrained Transformer

**A prototype study on GPT-2 small**

---

## Abstract

We describe Metamorphic Self-Adaptation (MSA), a system that selects a
transformer's execution architecture per input rather than executing a fixed
stack for every prompt. A heuristic Task Analyzer extracts eight features from
the input; a small learned classifier `f_theta` maps them to one of three
configurations; a policy layer applies a confidence threshold; a stability
monitor suppresses configuration thrash; and a controller executes the selected
configuration by restricting depth, attention heads and feed-forward width.

We implement all five components on GPT-2 small and calibrate them over 4,794
controlled experiments on 799 prompts. We report what we measured, including
results that do not favour the method. Depth adaptation reduces executed layers
by 35.6% on short QA; under a controlled benchmark the 4- and 8-layer
configurations run 45% and 23% faster than 12 layers at sequence length 64. The
prompt-loss cost is 3.89 to 5.97. Adding attention and feed-forward adaptation
reduces estimated compute to 0.322 of the static model and latency by a further
13–29%, but raises loss to 9.10, because those mechanisms slice pretrained
weights with no retraining. On short factual QA, exact-match accuracy falls as
adaptation is added: 13.3% static, 1.7% depth-adaptive, 0.0% full MSA. A
pretrained dense model of similar size (SmolLM-135M) reaches 75.0%. On GSM8K
every GPT-2 variant scores 0.000. The stability monitor reduces volatility only
under high switching (0.83 to 0.57), and the rolled-back steps lose quality.

We do not claim that MSA improves the quality-compute trade-off. We claim that
its mechanisms are implementable, that their costs are measurable, and that on
this backbone depth adaptation is the only one of the three that is currently
worth its quality cost. The backbone matters far more to task quality than any
adaptation mechanism tested.

---

## 1. Introduction

Transformer inference conventionally executes an identical computation for every
input. A three-token factual lookup and a multi-step arithmetic word problem
both traverse all layers, all attention heads and the full feed-forward width.
If the difficulty of inputs varies, and the compute required to answer them
varies with it, then a fixed architecture is either over-provisioned for easy
inputs or under-provisioned for hard ones.

Metamorphic Self-Adaptation asks whether a system can choose its own execution
architecture per input, from a small calibrated library of configurations, using
a cheap predictor that runs before the expensive model.

This paper reports a prototype. The contributions are:

1. **A complete implemented pipeline** — Task Analyzer, learned configuration
   predictor, decision policy, stability monitor and dynamic controller — in
   which every architectural claim corresponds to running code (Section 4).

2. **Three adaptation mechanisms that reduce real computation.** Attention and
   feed-forward adaptation slice weight tensors rather than masking outputs, so
   the skipped arithmetic is genuinely not performed (Section 7).

3. **A calibration and evaluation protocol** covering 799 prompts across
   twelve categories (4,794 controlled runs), a measured policy-threshold sweep,
   five baselines including a pretrained contemporary dense model, eight
   ablations, and stability experiments at three switching levels
   (Sections 9–12).

4. **Negative results reported as results.** Full MSA is the worst-quality
   variant we tested. We report that, quantify it, and explain the cause rather
   than tuning around it (Sections 11–13).

5. **A reproducibility finding.** A single incorrect argument in the adaptive
   forward pass silently disabled causal masking, invalidating an entire earlier
   round of results while still producing plausible-looking loss curves
   (Section 9.1). We treat this as a methodological contribution.

---

## 2. Related Work

This prototype sits alongside several families of work. We position rather than
survey, since the prototype does not yet compete with any of them empirically.

**Early-exit and layer-skipping.** Depth-adaptive transformers attach exit
classifiers to intermediate layers and halt when a confidence criterion is met.
MSA differs in deciding depth *before* execution from input features alone,
rather than during execution from intermediate states. The advantage is that the
decision cost is paid once and is independent of depth; the disadvantage is that
the decision cannot use evidence the forward pass would reveal.

**Mixture-of-experts and conditional computation.** MoE routes tokens to a
subset of feed-forward experts via a learned gate. Our Baseline C is a
deliberately simplified analogue: top-2-of-4 routing over feed-forward chunks
with a fixed seeded gate and no training. It is a compute-reduction reference
point, not a reproduction of trained MoE.

**Structured pruning and slicing.** Removing heads or feed-forward channels
from a pretrained model normally requires retraining or distillation to recover
quality. Our attention and feed-forward mechanisms perform the slicing at
inference time with no recovery step, and Section 11 shows the resulting quality
cost is severe. This is consistent with the pruning literature and is the
clearest reason the non-depth mechanisms underperform here.

**Adaptive computation time.** ACT-style methods learn a halting policy jointly
with the model. MSA's predictor is trained post hoc against a calibration
objective on a frozen backbone, which is cheaper but cannot shape the backbone
to be adaptation-friendly.

---

## 3. Problem Definition

Let `M` be a pretrained autoregressive transformer with `L` layers, `H`
attention heads per layer and feed-forward inner dimension `F`. Let `x` be an
input sequence.

A **configuration** `c` is a triple restricting the executed architecture:

```
c = (d, alpha, phi)

d      in {1..L}      number of transformer blocks executed
alpha  in (0, 1]      fraction of attention heads executed per block
phi    in (0, 1]      fraction of feed-forward channels executed per block
```

Let `C` be a small library of configurations, and `M_c(x)` the output of
executing `M` on `x` under `c`. Let `Q(M_c(x))` be a quality measure (lower is
better) and `R(c, x)` a cost measure.

The system must learn a selector

```
f_theta : features(x)  ->  probability distribution over C
```

and a policy `pi` mapping that distribution to a single configuration, such that
over a distribution of inputs the expected cost falls substantially while the
expected quality degrades acceptably.

Two quantities are reported separately throughout, because they do not agree
(Section 11.4):

```
analytical cost      relative FLOPs of the transformer stack under c
empirical cost       measured wall-clock latency under c
```

**Calibration objective.** For a tolerance `tau`, the target configuration for
input `x` is the cheapest configuration whose loss stays within `tau` of the
full model:

```
c*(x) = argmin   R(c)   subject to   Q(M_c(x)) <= (1 + tau) * Q(M_full(x))
         c in C

with fallback c*(x) = c_full when the constraint set is empty.
```

`tau` is an operating point, not a constant of nature. Section 11.1 reports the
full sweep.

---

## 4. MSA Architecture

The implemented pipeline, with the file realising each stage:

```
      x
      |
      v
+---------------------+
|   Task Analyzer     |   analyzer/task_analyzer.py
+---------------------+
      |  v(x) in R^8
      v
+---------------------+
|   f_theta           |   controller/predictor.py
|   MLP(8, 32, 16, 3) |
+---------------------+
      |  P(shallow), P(medium), P(deep)
      v
+---------------------+
| Configuration Policy|   controller/policy.py
|  argmax + threshold |
+---------------------+
      |  c_proposed
      v
+---------------------+
| Stability Monitor   |   monitor/stability_monitor.py
|  volatility, rollback|
+---------------------+
      |  c_final
      v
+---------------------+
| Dynamic Controller  |   controller/controller.py
|  (d, alpha, phi)    |
+---------------------+
      |
      v
+---------------------+
| Adaptive GPT-2      |   models/adaptive_model.py
+---------------------+
      |
      v
   logits + telemetry
```

The configuration library is deliberately small, so that every entry can be
calibrated empirically rather than assumed:

| Name | `d` | `alpha` | `phi` | Relative FLOPs (seq 64) |
|------|-----|---------|-------|-------------------------|
| shallow | 4 | 0.5 | 0.5 | 0.167 |
| medium | 8 | 0.5 | 0.5 | 0.333 |
| deep | 12 | 1.0 | 1.0 | 1.000 |

A **depth-only** variant holds `alpha = phi = 1.0` at the same depths, giving
relative FLOPs 0.333 / 0.667 / 1.000. Both variants are calibrated, and the
`mechanism` column of the calibration data distinguishes them.

---

## 5. Task Analyzer

The analyzer maps text to a feature vector using only cheap string operations,
so its cost is negligible against a transformer forward pass.

**Version 1** (four features, retained unchanged as a baseline):

| Feature | Definition |
|---------|-----------|
| `input_length` | `min(words / 100, 1)` |
| `reasoning` | density of ten reasoning cue words, capped |
| `domain` | density of nine mathematical cue words, capped |
| `structure` | `min((sentences + commas + parens) / 10, 1)` |

**Version 2** adds four features targeting signals version 1 misses:

| Feature | Definition |
|---------|-----------|
| `numeric_density` | share of whitespace tokens containing a digit |
| `math_expression` | density of arithmetic and relational operators |
| `question_type` | question stem mapped to 0.25 / 0.50 / 0.75 / 1.00 by implied work |
| `reasoning_steps` | sequential markers plus clause count, normalised |

All features are bounded to `[0, 1]`. The empty string maps to the zero vector.

Section 12.1 reports the ablation between the two versions. The result does not
justify strong claims about the added features.

---

## 6. Configuration Predictor

`f_theta` is a scikit-learn pipeline: `StandardScaler` followed by
`MLPClassifier` with hidden layers `(32, 16)`, ReLU activation, Adam, early
stopping on a 15% validation split, seed 42.

**Training data.** One example per calibration prompt: the eight features as
input, and the target configuration from the calibration objective of Section 3
at `tau = 1.00`. Target distribution over 799 prompts:

```
medium   402
deep     249
shallow  148
```

`tau = 1.00` was kept because it is still the operating point producing a
non-degenerate three-class distribution; at `tau <= 0.50` over 89% of prompts
map to `deep` and the classification problem collapses.

**Performance.** 639 training / 160 held-out, stratified. Because the classes
are imbalanced, macro F1 is reported alongside accuracy, against a
majority-class reference (`results/calibration/predictor_evaluation.json`).

| Metric | f_theta | Majority class |
|--------|---------|----------------|
| Held-out accuracy | **0.713** | 0.500 |
| Held-out macro F1 | **0.678** | 0.222 |
| 5-fold CV accuracy | **0.731 ± 0.020** | — |
| 5-fold CV macro F1 | **0.693 ± 0.030** | — |
| Mean confidence (correct / wrong) | 0.725 / 0.641 | — |

Per class (precision / recall): shallow 0.93 / 0.43, medium 0.67 / 0.86,
deep 0.74 / 0.64.

Confusion matrix on the held-out split (rows actual, columns predicted):

|  | shallow | medium | deep |
|--|---------|--------|------|
| **shallow** | 13 | 17 | 0 |
| **medium** | 0 | 69 | 11 |
| **deep** | 1 | 17 | 32 |

Expanding the corpus from 499 to 799 prompts cut the cross-validation standard
deviation from ±0.188 to ±0.020, so the estimate is now stable. The weakest
class is still `shallow` (recall 0.43): cheap prompts are routed to `medium`.
That direction costs compute rather than quality, but it caps the saving.
Confidence is informative but weakly so: wrong predictions average 0.64
against 0.73 for correct ones.

**Decision policy.** The policy is deliberately separate from the predictor:

```
if confidence >= threshold:  c_final = argmax P
else:                        c_final = fallback        (deep)
```

The threshold was selected by a sweep on the 160 held-out prompts
(`controller/threshold_sweep.py`, Table 7). Loss, latency and FLOPs for each
decision are read from that prompt's measured calibration runs. The selection
rule was fixed in advance: maximise agreement with `c*(x)`, break ties toward
lower FLOPs.

| Threshold | Accepted | Fallbacks | Avg depth | Rel. FLOPs | Rel. loss increase | Agreement |
|-----------|----------|-----------|-----------|------------|--------------------|-----------|
| 0.00 | 160 | 0 | 8.73 | 0.727 | 58.4% | 0.713 |
| **0.40** | **158** | **2** | **8.80** | **0.733** | **56.9%** | **0.725** |
| 0.50 | 143 | 17 | 9.05 | 0.754 | 52.1% | 0.725 |
| 0.60 | 121 | 39 | 9.45 | 0.788 | 43.9% | 0.669 |
| 0.70 | 86 | 74 | 10.13 | 0.844 | 31.0% | 0.575 |
| 0.80 | 40 | 120 | 11.18 | 0.931 | 10.1% | 0.450 |

0.40 and 0.50 tie on agreement; the rule selects 0.40. Raising the threshold
moves the system smoothly back toward the static model, so the threshold is a
second, explicit quality-compute knob on top of `tau`. Every inference logs
the full probability vector, the predicted configuration, the confidence,
whether the fallback fired, and the executed depth.

---

## 7. Dynamic Architecture Controller

The controller converts a configuration name into an executed architecture and
emits telemetry. Three mechanisms, none of which modify stored weights.

**Depth.** Execute blocks `0 .. d-1`, then the final layer norm and the language
model head. Blocks `d .. L-1` are not executed.

**Attention.** For head fraction `alpha`, let `h = round(H * alpha)` and
`k = h * d_head`. GPT-2 packs `[Q|K|V]` in a single `Conv1D` weight of shape
`(D, 3D)`. We gather the first `k` columns of each third and the first `k` rows
of the output projection:

```
qkv = x @ W_attn[:, idx]  +  b_attn[idx]        idx = [0..k) u [D..D+k) u [2D..2D+k)
o   = SDPA(q, k, v, causal=True, scale=d_head^-0.5)
y   = o @ W_proj[:k, :]  +  b_proj
```

**Feed-forward.** For channel fraction `phi`, let `n = round(F * phi)`:

```
y = act(x @ W_fc[:, :n] + b_fc[:n]) @ W_proj[:n, :] + b_proj
```

Both are **slicing, not masking**. The inactive parameters are never multiplied,
so the arithmetic is genuinely skipped. `test_msa.py` asserts that each mechanism
changes the computed output, which prevents a mechanism silently degrading into
a label.

**Correctness constraint.** At `alpha = phi = 1.0` the hand-written path must
reproduce the reference implementation. Measured maximum absolute logit
difference against HuggingFace `GPT2LMHeadModel`: **5.3e-05** (floating-point
noise). The `deep` configuration routes through the stock block implementation
and matches at **0.0e+00**.

**Analytical cost model.** Per layer, counting a multiply-accumulate as 2 FLOPs,
with sequence length `s`, hidden size `D`, head dimension `d_head`:

```
QKV          2 * s * D * 3k
attention    4 * h * s^2 * d_head
output proj  2 * s * k * D
feed-forward 4 * s * D * n
```

Embeddings and the language model head are excluded, being identical across
configurations.

**Telemetry.** Every inference records selected depth, layers executed, layers
skipped, layer reduction, active heads, active feed-forward chunks, sequence
length, relative FLOPs, latency and loss.

---

## 8. Stability Monitor

If configuration selection is volatile across a request stream, the system
thrashes. The monitor observes the decision sequence over a sliding window of
`w = 6`.

**Volatility** is the fraction of adjacent pairs in the window that differ, and
the **stability score** is its complement:

```
V(t) = |{ i : c_{i} != c_{i+1} }| / (|W_t| - 1)        W_t = last w decisions
S(t) = 1 - V(t)
rollback rate = rollbacks / observations
```

**Status:** `STABLE` if `V <= 0.4`, `WARNING` if `0.4 < V <= 0.7`, `UNSTABLE`
otherwise.

**Rollback.** On `UNSTABLE`, if a previously stable configuration exists and
differs from the proposal, the monitor reverts to it and records the event. The
stable configuration is the window mode, updated whenever status is `STABLE`.

The monitor also runs in a passive mode that observes and measures without
acting, which is exactly ablation A7 and makes the with/without comparison
fair. The implementation (`monitor/stability_monitor.py`) computes exactly the
three quantities above; `test_msa.py` asserts that the synthetic sequence
`deep, medium, deep, shallow, deep, medium, deep` triggers a rollback that
changes the executed configuration.

**Reporting note.** An early version reported `V` for the final window only,
which produced the contradiction of 22 recorded switches alongside a reported
volatility of 0.00, because the last six decisions happened to agree. The
summary now reports the mean and maximum over the run.

---

## 9. Experimental Setup

**Backbone.** GPT-2 small, HuggingFace `gpt2`, 124,439,808 parameters, 12
layers, 12 heads, hidden size 768, feed-forward inner size 3072. Weights frozen
throughout; no training of the backbone at any point.

**Environment.**

| Component | Version |
|-----------|---------|
| Python | 3.14.6 |
| PyTorch | 2.14.0+cpu |
| Transformers | 5.17.0 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.3 |
| Platform | Windows 11 (10.0.26200) |
| Processor | Intel64 Family 6 Model 186 |
| Device | **CPU**, 14 torch threads |

All seeds fixed at 42. Configuration recorded in `configs/experiment.yaml`;
environment captured programmatically to `results/environment.json`.

**Calibration corpus.** 799 unique prompts across twelve categories, covering
every category the plan requires. The original 499-prompt corpus and its
results are preserved unchanged as a historical version
(`data/calibration_prompts_v1_499.json`, `results/calibration/`).

| Category | n | Category | n |
|----------|---|----------|---|
| hard_reasoning (GSM8K train) | 229 | technical | 49 |
| seed_handwritten | 100 | multi_step_arithmetic | 47 |
| factual | 70 | long_structured | 45 |
| basic_arithmetic | 53 | summarization | 40 |
| scientific_explanation | 50 | logical_reasoning | 36 |
| code | 50 | long_explanation | 30 |

Factual prompts are disjoint from the `short_qa` evaluation set. Each prompt is
executed at three configurations under two mechanisms: **4,794 experiments**,
each recording the eight features, complexity, depth, configuration, loss,
perplexity, latency and relative FLOPs. Prompts are frozen to
`data/calibration_prompts.json` so reproduction requires no network access.

**Benchmarking protocol.** Warm-up 10, measured runs 50, at fixed sequence
lengths 16, 64 and 256, for every mechanism and configuration. We report a 10%
trimmed mean alongside mean, standard deviation, median and p95, and flag any
cell whose mean exceeds 1.5× its median.

**Downstream datasets.** `short_qa`, 60 local short factual questions with
single-token answers; `gsm8k`, 40 items from `openai/gsm8k` test. Metrics are
answer-conditional perplexity (answer tokens scored given the question, prompt
tokens masked) and exact-match accuracy under greedy decoding. Prompt loss is
**not** used as a downstream metric.

### 9.1 A defect that invalidated an earlier round of results

We record this because it was not detectable from the loss curves alone.

The adaptive forward pass originally supplied `attention_mask = ones(B, S)` to
each transformer block. Under Transformers v5, the SDPA attention path computes:

```python
is_causal = q_length > 1 and attention_mask is None and module.is_causal
```

Supplying *any* explicit mask therefore disables the causal mask. The model
attended bidirectionally: every position could see its own continuation.

The symptoms were plausible rather than obviously broken. Loss still decreased
monotonically with depth, and the quality-compute frontier still had the
expected shape. The defect surfaced only on direct comparison against the
reference model:

```
max |logits_adaptive(d=12) - logits_reference|  =  67.4
"...the capital of France is"  ->  "capital"   (reference: "Paris")
```

Every result derived from that forward pass was invalid, including a calibration
sweep, a tolerance frontier, and a predictor reported at 0.60 accuracy. All were
regenerated after the one-argument fix.

Two regression assertions now guard it: depth 12 must equal the reference
model, and prefix logits must not change when trailing tokens are removed.

**Methodological implication.** Any adaptive-inference implementation that
re-enters a framework's internals should assert numerical equivalence to the
unmodified model at its identity configuration. Monotone loss curves are not
evidence of correctness.

---

## 10. Baselines

| | Baseline | Definition |
|--|----------|-----------|
| A | Static GPT-2 | All 12 layers, full width. Reference point. |
| B | Depth adaptive | `f_theta` selects depth; `alpha = phi = 1.0`; no monitor. |
| C | Routing / MoE-style | Full depth; per-token top-2-of-4 feed-forward chunk routing via a **fixed seeded random gate, untrained**. |
| D | Contemporary dense reference | `HuggingFaceTB/SmolLM-135M`: pretrained Llama-architecture decoder with RMSNorm, RoPE, SwiGLU; no adaptivity. A second, **untrained** RMSNorm/RoPE/SwiGLU decoder at GPT-2 dimensions is kept for latency reference only. |
| E | Full MSA | `f_theta` + policy + depth + attention + feed-forward adaptation + stability monitor. |

Baseline D uses a different tokenizer from GPT-2, so its per-token loss and
perplexity are **not** directly comparable to the GPT-2 variants; its
exact-match accuracy and latency are. Parameter counts are in Table 8 of
`results/tables.md`.

Baseline C's gate is untrained by design; it is a compute-reduction reference,
not a claim about trained mixture-of-experts.

All methods run through one runner (`evaluation/runner.py`) that writes a
fixed per-sample schema (method, dataset, sample id, loss, perplexity, latency,
depth, layer reduction, configuration, attention mode, FFN mode, confidence,
rollback) to `results/evaluation/<method>__<dataset>.csv`.

---

## 11. Results

### 11.1 Calibration and the quality-compute frontier

Mean over 799 prompts, depth-only mechanism:

| Depth | Prompt loss | Mean perplexity | Relative FLOPs |
|-------|-------------|-----------------|----------------|
| 4 | 8.993 | 18,195 | 0.333 |
| 8 | 7.165 | 3,077 | 0.667 |
| 12 | 3.911 | 86.5 | 1.000 |

Quality degrades sharply and non-linearly below full depth. The step from 8 to
12 layers is worth far more than the step from 4 to 8.

Tolerance sweep, showing the operating points available:

| `tau` | Avg depth | Compute reduction | Relative loss increase | shallow / medium / deep |
|-------|-----------|-------------------|------------------------|--------------------------|
| 0.05 | 12.00 | 0.0% | 0.0% | 0 / 0 / 799 |
| 0.25 | 11.94 | 0.5% | 0.2% | 4 / 3 / 792 |
| 0.50 | 11.44 | 4.6% | 5.6% | 24 / 63 / 712 |
| 0.75 | 10.26 | 14.5% | 23.6% | 66 / 216 / 517 |
| **1.00** | **8.51** | **29.1%** | **56.0%** | **148 / 402 / 249** |
| 1.50 | 5.62 | 53.2% | 105.0% | 499 / 277 / 23 |
| 2.00 | 4.41 | 63.3% | 123.3% | 718 / 81 / 0 |

The frontier is monotone, as it must be. It also shows the central difficulty:
meaningful compute reduction on this backbone requires accepting a large
relative loss increase. There is no operating point offering, say, 25% compute
reduction at under 10% loss increase.

### 11.2 Main results (60 short-QA prompts)

Exact match uses greedy decoding of up to 16 tokens. Latency here is a single
forward pass per prompt inside the evaluation loop and is noisy; Section 11.4
is the controlled latency measurement.

| Method | Prompt loss | Exact match | Avg depth | Layer red. | Rel. FLOPs | Median latency (ms) |
|--------|-------------|-------------|-----------|------------|------------|---------------------|
| A: Static GPT-2 | 3.889 | 0.133 | 12.00 | 0.0% | 1.000 | 33.9 |
| B: Depth adaptive | 5.974 | 0.017 | 7.73 | 35.6% | 0.644 | 22.0 |
| C: Routing / MoE-style | 6.366 | 0.000 | 12.00 | 0.0% | 0.668 | 40.7 |
| D: SmolLM-135M (dense, pretrained) | 1.979* | **0.767** | 30 layers | — | — | 104.3 |
| E: Full MSA | 9.097 | 0.000 | 7.73 | 35.6% | **0.322** | **15.8** |

\* Different tokenizer; not comparable per token with the GPT-2 rows.

Among GPT-2 variants, static is the best on quality. Full MSA is the cheapest
on both estimated compute and latency, and the worst on quality. Depth
adaptation occupies the middle.

Baseline D changes the frame. A contemporary dense model of similar size
answers 77% of the short questions against GPT-2's 13%, while no GPT-2
configuration, adaptive or not, exceeds 13%. On this evidence **the choice of
backbone matters far more to task quality than any adaptation mechanism tested
here**. D is about 3× slower per forward pass on CPU (30 layers vs 12).

Baseline C uses less arithmetic than static (0.668 relative FLOPs) yet runs
slower, because per-token gather/scatter on CPU costs more than the
multiplications it avoids.

### 11.3 Downstream task evaluation

| Dataset | Method | Answer PPL | Exact match | Avg depth | Rel. FLOPs |
|---------|--------|-----------|-------------|-----------|------------|
| short_qa | Static | 284.9 | 0.075 | 12.0 | 1.000 |
| short_qa | Depth adaptive | 2.31e6 | 0.025 | 7.6 | 0.633 |
| short_qa | Full MSA | 9.39e5 | 0.000 | 7.6 | 0.317 |
| short_qa | D: SmolLM-135M | 12.7* | **0.675** | 30 layers | — |
| gsm8k | Static | 1613.6 | 0.000 | 12.0 | 1.000 |
| gsm8k | Depth adaptive | 3701.6 | 0.000 | 11.5 | 0.958 |
| gsm8k | Full MSA | 1857.8 | 0.000 | 11.6 | 0.933 |
| gsm8k | D: SmolLM-135M | 8.5* | **0.050** | 30 layers | — |

\* Different tokenizer; not comparable with the GPT-2 rows. The dataset suite
uses the first 40 short-QA items; Section 11.2 uses all 60.

Four observations, all stated as measured:

1. **On short QA, task accuracy falls as adaptation is added**
   (7.5% → 2.5% → 0.0%). The compute saving is real and so is the quality loss.

2. **On GSM8K every GPT-2 variant scores 0.000, including the static baseline.**
   GPT-2 small cannot solve GSM8K. That dataset therefore measures compute
   behaviour, not task quality, for GPT-2. Even Baseline D reaches only 5%.

3. **Compute saving is dataset-dependent.** On GSM8K the predictor routes
   almost everything to `deep` (average depth 11.5–11.6), and the saving nearly
   vanishes. This is arguably correct behaviour, but it means **no single
   compute-reduction figure should be quoted without naming the dataset.**

4. **The dense pretrained reference dominates on quality.** 67.5% against 7.5%
   on the same 40 questions.

### 11.4 Controlled latency benchmark

Median of 50 runs, milliseconds, CPU, 6 pinned torch threads, batch size 1.
Every cell is warmed 10 times before any timing starts, and the 50 runs are
split over 5 shuffled interleaved rounds so background drift is spread across
all cells:

| Mechanism | Config | seq 16 | seq 64 | seq 256 |
|-----------|--------|--------|--------|---------|
| depth only | shallow | 14.3 | 32.7 | 91.0 |
| depth only | medium | 20.2 | 45.9 | 132.4 |
| depth only | deep | 28.1 | 59.8 | 175.8 |
| depth+attn+ffn | shallow | **12.6** | **28.4** | **73.5** |
| depth+attn+ffn | medium | 16.8 | 36.1 | 93.9 |
| depth+attn+ffn | deep | 28.7 | 59.5 | 175.0 |
| routing | shallow | 18.0 | 36.5 | 84.1 |
| routing | medium | 27.5 | 50.5 | 121.5 |
| routing | deep | 36.2 | 65.1 | 155.9 |

The two `deep` rows for depth-only and depth+attn+ffn execute identical
computation and agree within 1%, which is the check that the protocol is
stable. No cell is flagged as an outlier; the largest standard deviation is
19 ms.

- **Depth** cuts latency by 45% (shallow) and 23% (medium) at seq 64.
- **Attention and FFN slicing** reduce latency by a further 12–19% (shallow)
  and 17–29% (medium) at the same depth, growing with sequence length. They
  therefore deliver genuine wall-clock savings; their problem is quality, not
  speed.
- **Routing** is 29% slower than static depth-only at seq 16 and 9% slower at
  seq 64, but 11% faster at seq 256, where the saved feed-forward arithmetic
  finally outweighs per-token gather/scatter.

**Baseline D at seq 64:** SmolLM-135M (float32) 71.8 ms, untrained GPT-2-sized
RMSNorm/RoPE/SwiGLU reference 52.3 ms, against 59.8 ms for static GPT-2.

**Analytical and empirical cost disagree.** For the adaptive mechanisms they
agree in direction. For routing the relationship depends on sequence length.
Any compute claim must state which measure it refers to.

*Reproducibility note.* Two earlier benchmark runs were discarded. The first
overlapped with another job. The second used all 14 threads on a machine with
15–20% background load: cells that execute identical computation differed by
up to 36 ms, and seq 16 timed slower than seq 64. Pinning to 6 threads with
global warm-up and interleaved rounds removed both artefacts. The harness also
flags any cell whose mean exceeds 1.5× its median. A separate issue: Baseline D
first loaded in its checkpoint dtype, bfloat16, and was about 15× slower than
float32 on this CPU. It is now loaded in float32 to match GPT-2.

### 11.5 Stability

Three 60-input sequences drawn over the policy's decision buckets with switch
probability 0.1, 0.4 and 0.8 (Section 12.3). Each is run with the monitor
disabled (A7, passive) and active (A8). Volatility is the mean `V(t)` of the
*executed* configurations; `S = 1 - V`.

| Switching | Variant | Proposed sw. | Executed sw. | V | S | Rollbacks | Rollback rate | Prompt loss |
|-----------|---------|--------------|--------------|---|---|-----------|---------------|-------------|
| low | monitor off | 5 | 5 | 0.086 | 0.914 | 0 | 0.000 | 5.891 |
| low | monitor on | 5 | 5 | 0.086 | 0.914 | 0 | 0.000 | 5.891 |
| medium | monitor off | 21 | 21 | 0.323 | 0.677 | 0 | 0.000 | 7.116 |
| medium | monitor on | 21 | 21 | 0.323 | 0.677 | 0 | 0.000 | 7.116 |
| high | monitor off | 50 | 50 | 0.830 | 0.170 | 0 | 0.000 | **7.035** |
| high | monitor on | 50 | **34** | **0.566** | **0.434** | 13 | 0.217 | 7.707 |

**Quality after rollback.** On the 13 rolled-back steps, the executed
(rolled-back) configuration averaged loss **8.758**; the unmonitored run on
the same inputs at the same steps averaged **5.654**.

What this shows, as measured:

1. The monitor is inert at low and medium switching, as designed: volatility
   never crosses the 0.7 rollback threshold, so the two runs are identical.
2. At high switching it cuts executed switches from 50 to 34 and volatility
   from 0.830 to 0.566.
3. **Rollback costs quality.** The stable configuration it reverts to is the
   window mode, which on this stream is cheaper than what the predictor
   proposed, so the rolled-back steps lose 3.1 nats of loss and overall loss
   rises from 7.03 to 7.71. The monitor buys stability, not quality.

An earlier, smaller experiment on the 499-prompt corpus showed a slight loss
*improvement* under the monitor. That did not replicate, and we do not claim
it.

---

## 12. Ablation Studies

### 12.1 Task Analyzer feature set

| Feature set | n features | Held-out acc. | Held-out macro F1 | 5-fold CV acc. | 5-fold CV macro F1 |
|-------------|-----------|---------------|-------------------|----------------|--------------------|
| v1 | 4 | 0.650 | 0.554 | 0.697 ± 0.014 | 0.612 ± 0.011 |
| v2 | 8 | **0.713** | **0.678** | **0.731 ± 0.020** | **0.693 ± 0.030** |

On the 499-prompt corpus the v2 advantage sat inside one standard deviation.
On 799 prompts the cross-validated macro-F1 gap (0.081) is more than twice the
larger standard deviation, so **v2's added features now measurably help**.
This is still a single corpus and a single seed; it is evidence, not proof.

### 12.2 Mechanism ablation (60 short-QA prompts)

| Variant | Prompt loss | Exact match | Avg depth | Rel. FLOPs |
|---------|-------------|-------------|-----------|------------|
| A1: Static | **3.889** | **0.133** | 12.00 | 1.000 |
| A2: Analyzer + rule policy | 8.067 | 0.017 | 4.00 | 0.333 |
| A3: Analyzer + `f_theta` | 5.974 | 0.017 | 7.73 | 0.644 |
| A4: Depth only | 5.974 | 0.017 | 7.73 | 0.644 |
| A5: Depth + attention | 8.403 | 0.033 | 7.73 | 0.536 |
| A6: Depth + FFN | 7.741 | 0.000 | 7.73 | 0.430 |
| A7: Full, no monitor | 9.097 | 0.000 | 7.73 | 0.322 |
| A8: Full MSA | 9.097 | 0.000 | 7.73 | 0.322 |

Single-pass latencies for these runs are in `results/ablations.csv`; they are
too noisy to rank variants (A3 and A4 execute identical computation yet differ
by 26 ms), so compute is compared on FLOPs and Section 11.4.

Incremental contributions:

- **A2 → A3, the learned predictor is the clearest win.** Replacing the
  hand-tuned threshold rule with `f_theta` improves loss from 8.067 to 5.974
  while *increasing* average depth from 4.0 to 7.73. The rule selector was
  routing nearly everything to `shallow`; the predictor allocates compute where
  it is needed. This is the component most clearly justified by the data.

- **A4 → A5, attention adaptation** buys 0.644 → 0.536 relative FLOPs for
  5.974 → 8.403 loss. Poor trade.

- **A4 → A6, feed-forward adaptation** buys 0.644 → 0.430 for 5.974 → 7.741.
  Also poor, but less bad than attention.

- **A7 → A8, the stability monitor** shows no difference on this input set,
  because the sequence never became unstable. Its contribution is measured by
  Section 11.5 instead.

### 12.3 A negative result about experiment design

Our first stability experiment used hand-picked "easy" and "hard" prompts,
assuming they would route to different configurations. They did not: 48 of 60
routed to `medium`, the sequence barely switched, and no rollback fired. The
conclusion that the monitor "did nothing" would have been an artifact of the
stimulus, not a property of the monitor.

The corrected design buckets the calibration pool by the configuration the
policy *actually* selects, then draws a Markov sequence over the buckets with a
controlled switch probability (0.1 / 0.4 / 0.8 for low / medium / high), and
raises an error rather than silently producing a flat sequence if fewer than two
buckets are populated. The results in Section 11.5 use the corrected design.

---

## 13. Limitations

Stated plainly, because several of them bound the conclusions materially.

1. **The calibration signal is not task quality.** Targets derive from
   next-token loss over the prompt. This is a prototype signal. Downstream
   metrics are reported separately and are not substituted for it.

2. **Attention and feed-forward adaptation are applied without retraining.**
   Slicing pretrained weights and expecting retained quality is contrary to the
   pruning literature, and our results are consistent with that literature. The
   severe quality cost is a property of this implementation choice, not
   necessarily of the mechanisms.

3. **The backbone cannot perform one of the evaluation tasks.** GPT-2 small
   scores 0.000 on GSM8K in every configuration. Only the short-QA results carry
   task-quality signal, and that set is small (60 items) and local.

4. **Single-backbone, single-device study.** GPT-2 small on a shared CPU with 6
   pinned threads. Nothing here establishes behaviour at larger scale or on
   accelerators, where the relative cost of slicing overhead versus arithmetic
   differs substantially. In-loop latencies recorded during evaluation are
   single-pass and noisy; only Section 11.4 is a controlled measurement.

5. **The routing baseline is untrained.** It bounds what an untrained gate
   achieves and says nothing about trained mixture-of-experts.

6. **Baseline D uses a different tokenizer.** Its loss and perplexity are not
   comparable with GPT-2's; only exact match and latency are. It is a single
   model, not a survey of contemporary dense models.

7. **The heuristic analyzer.** Features are keyword and punctuation counts.
   The v2 set now measurably beats v1 (Section 12.1), but on one corpus and
   one seed.

8. **Monitor thresholds were not tuned.** The confidence threshold was swept
   (Section 6); the monitor's window and 0.4 / 0.7 thresholds are defaults.

9. **Small evaluation sets.** 60 short-QA items and 40 GSM8K items. Differences
   of a few percent in exact match are not meaningful at this size.

10. **Statistical significance is not established for any quality comparison.**
    We report point estimates and, where available, standard deviations.

11. **Summarization and code evaluation were not run.** The plan lists them as
    optional; with GPT-2 small scoring 0.000 on GSM8K and 13% on one-word
    factual QA, task metrics on either would be floored and uninformative.

### Claims we explicitly do not make

- That MSA always, or generally, improves performance.
- That MSA guarantees lower latency.
- That MSA preserves quality. On our measurements it does not.

---

## 14. Conclusion

We implemented Metamorphic Self-Adaptation end to end on GPT-2 small: a
heuristic analyzer, a learned configuration predictor, an explicit decision
policy, a stability monitor with rollback, and a controller that executes depth,
attention and feed-forward adaptation by slicing weights so that skipped work is
genuinely skipped. All components are verified by executable assertions,
including numerical equivalence to the reference model at the identity
configuration.

What the measurements support:

- **Depth adaptation is viable on this backbone.** 35.6% fewer executed layers
  on short QA, and 23–45% lower controlled latency at seq 64, for a prompt-loss
  increase from 3.89 to 5.97.

- **A learned predictor clearly beats a hand-tuned threshold rule.** Loss
  improves from 8.07 to 5.97 while allocating *more* compute, which is the
  behaviour a calibrated selector should exhibit. On 799 prompts it reaches
  0.693 cross-validated macro F1 against 0.222 for the majority class.

- **The policy threshold is a usable, measured knob.** Raising it from 0.40 to
  0.80 moves relative loss increase from 56.9% to 10.1% at the cost of FLOPs
  0.733 → 0.931.

- **The stability monitor does what it was designed to do, and no more.** It
  is inert at low and medium switching, and at high switching cuts volatility
  from 0.83 to 0.57, but the rolled-back steps lose quality.

What the measurements do not support:

- **Attention and feed-forward adaptation are not currently worth their cost.**
  They deliver real compute and latency savings but degrade quality far more
  than depth adaptation does, because they slice pretrained weights with no
  recovery step. Full MSA is the worst-quality GPT-2 variant we tested.

- **No general compute-reduction claim is warranted.** The saving is strongly
  dataset-dependent: 36.7% layer reduction on short QA, 3.3–4.2% on GSM8K.

- **Adaptation is not the lever that matters most for quality here.** A
  pretrained dense model of similar size answers 75% of the short questions,
  against at most 13% for any GPT-2 configuration.

The most useful next experiments follow directly from these negatives: retrain
or distil the sliced configurations so the non-depth mechanisms become
competitive; train the predictor against a downstream task metric rather than
prompt loss; make rollback quality-aware rather than reverting to the window
mode; and repeat on a backbone capable of the evaluation tasks, such as the
Baseline D model, so that quality differences are measurable rather than
floored at zero.

We also record a methodological result of independent interest: a single
argument in an adaptive forward pass silently disabled causal masking while
leaving loss curves qualitatively plausible. Adaptive-inference implementations
should assert numerical equivalence against the unmodified model at their
identity configuration.

---

## Reproducibility

```powershell
.venv\Scripts\Activate.ps1
python main.py --mode all        # full pipeline (calibration + generation dominate)
python test_msa.py               # 41-check phase-gate validation
```

The full pipeline was re-run from a clean state. Calibration losses, the
frontier, the trained predictor and the selected threshold reproduced
exactly; only wall-clock latencies vary between runs.

| Artefact | Location |
|----------|----------|
| Calibration data (4,794 rows) | `results/calibration_results.csv` |
| Frozen prompt set (799) | `data/calibration_prompts.json` |
| Historical 499-prompt version | `data/calibration_prompts_v1_499.json`, `results/calibration/` |
| Trained predictor + metadata | `results/performance_predictor.pkl` |
| Predictor evaluation | `results/calibration/predictor_evaluation.json` |
| Policy threshold sweep | `results/policy_threshold_sweep.csv` |
| Per-sample evaluation (standard schema) | `results/evaluation/` |
| Stability experiments | `results/stability/` |
| All result tables (Tables 1–8) | `results/tables.md` |
| All 16 figures | `results/figures/` |
| Environment capture | `results/environment.json` |
| Experiment configuration | `configs/experiment.yaml` |
| Reproduction log | `results/reproduction_log.txt` |

**Claim-to-evidence map.** Each architectural claim, with the component that
implements it, the check that verifies it and the result file behind its
numbers:

| Claim | Component | Verified by | Numbers from |
|-------|-----------|-------------|--------------|
| Eight bounded features per input | `analyzer/task_analyzer.py` | `test_msa.py` P5 | `results/feature_ablation.csv` |
| f_theta selects the executed depth | `controller/predictor.py`, `run_msa.py` | `test_msa.py` P2 | `results/calibration/predictor_evaluation.json` |
| Policy thresholding with fallback | `controller/policy.py` | `test_msa.py` policy check | `results/policy_threshold_sweep.csv` |
| 4/8/12-layer execution | `controller/controller.py`, `models/adaptive_model.py` | `test_msa.py` P6 | `results/calibration_results.csv` |
| Attention/FFN slicing changes computation | `models/adaptive_model.py` | `test_msa.py` controller check | `results/ablations.csv`, `results/benchmark.csv` |
| Rollback under instability | `monitor/stability_monitor.py` | `test_msa.py` P10 | `results/stability/stability_levels.csv` |
| Baselines A-E | `evaluation/harness.py`, `evaluation/baselines.py` | `evaluation/runner.py` schema asserts | `results/baselines.csv`, `results/dataset_evaluation.csv` |

Every number in this paper is read from those artefacts; none is transcribed by
hand. Seeds are fixed at 42 throughout.
