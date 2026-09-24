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

We implement all five components on GPT-2 small and calibrate them over 2,994
controlled experiments. We report what we measured, including results that do
not favour the method. Depth adaptation reduces executed layers by 35.6% and
median latency by roughly 28%, at a prompt-loss cost of 3.89 to 5.97. Adding
attention and feed-forward adaptation reduces estimated compute further, to
0.322 of the static model, but raises loss to 9.10, because those mechanisms
slice pretrained weights with no retraining. On short factual QA, exact-match
accuracy falls monotonically as adaptation is added: 7.5% static, 2.5%
depth-adaptive, 0.0% full MSA. On GSM8K every variant including the static
baseline scores 0.000, so that dataset separates compute behaviour rather than
task quality.

We do not claim that MSA improves the quality-compute trade-off. We claim that
its mechanisms are implementable, that their costs are measurable, and that on
this backbone depth adaptation is the only one of the three that is currently
worth its quality cost.

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

3. **A calibration and evaluation protocol** covering 499 prompts across ten
   categories, five baselines, eight ablations, and a controlled stability
   experiment (Sections 9–12).

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
at `tau = 1.00`. Target distribution over 499 prompts:

```
medium   233
deep     187
shallow   79
```

`tau = 1.00` was chosen because it is the operating point producing a
non-degenerate three-class distribution; at `tau <= 0.25` over 99% of prompts
map to `deep` and the classification problem collapses.

**Performance.** 399 training / 100 held-out, stratified.

| Metric | Value |
|--------|-------|
| Held-out accuracy | **0.720** |
| 5-fold CV accuracy | **0.673 ± 0.188** |
| Majority-class baseline | 0.467 |

Confusion matrix on the held-out split (rows actual, columns predicted):

|  | shallow | medium | deep |
|--|---------|--------|------|
| **shallow** | 6 | 9 | 1 |
| **medium** | 1 | 34 | 12 |
| **deep** | 0 | 5 | 32 |

The predictor beats the majority-class baseline by 25 points, which is the
meaningful comparison. Its weakest class is `shallow` (recall 0.375): it
systematically routes cheap prompts to `medium`. Since errors in that direction
cost compute rather than quality, this is the safer failure mode, but it caps
the achievable saving.

**Decision policy.** The policy is deliberately separate from the predictor:

```
if confidence >= threshold:  c_final = argmax P
else:                        c_final = fallback        (default: deep)
```

with `threshold = 0.50` by default. Every inference logs the full probability
vector, the predicted configuration, the confidence, whether the fallback fired,
and the executed depth.

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

**Volatility** is the fraction of adjacent pairs in the window that differ:

```
V(t) = |{ i : c_{i} != c_{i+1} }| / (w - 1)
```

**Status:** `STABLE` if `V <= 0.4`, `WARNING` if `0.4 < V <= 0.7`, `UNSTABLE`
otherwise.

**Rollback.** On `UNSTABLE`, if a previously stable configuration exists and
differs from the proposal, the monitor reverts to it and records the event. The
stable configuration is the window mode, updated whenever status is `STABLE`.

The monitor also runs in a passive mode that observes and measures without
acting, which is exactly ablation A7 and makes the with/without comparison
fair.

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

**Calibration corpus.** 499 unique prompts across ten categories:

| Category | n | Category | n |
|----------|---|----------|---|
| hard_reasoning (GSM8K) | 190 | code | 30 |
| seed_handwritten | 100 | technical | 29 |
| basic_arithmetic | 30 | long_structured | 25 |
| multi_step_arithmetic | 30 | summarization | 20 |
| scientific_explanation | 30 | logical_reasoning | 15 |

Each prompt is executed at three configurations under two mechanisms:
**2,994 experiments**. Prompts are frozen to `data/calibration_prompts.json` so
reproduction requires no network access.

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
| D | Contemporary dense reference | Separate decoder with RMSNorm, RoPE, SwiGLU; 162,148,608 parameters; **randomly initialised**. |
| E | Full MSA | `f_theta` + depth + attention + feed-forward adaptation + stability monitor. |

Baseline D is included for architecture-family latency and size reference only.
It is untrained, so its quality is reported as N/A rather than as a number that
would invite a meaningless comparison.

Baseline C's gate is untrained by design; it is a compute-reduction reference,
not a claim about trained mixture-of-experts.

---

## 11. Results

### 11.1 Calibration and the quality-compute frontier

Mean over 499 prompts, depth-only mechanism:

| Depth | Prompt loss | Perplexity | Relative FLOPs |
|-------|-------------|------------|----------------|
| 4 | 9.027 | 18,291 | 0.333 |
| 8 | 7.208 | 2,973 | 0.667 |
| 12 | 3.861 | 80.8 | 1.000 |

Quality degrades sharply and non-linearly below full depth. The step from 8 to
12 layers is worth far more than the step from 4 to 8.

Tolerance sweep, showing the operating points available:

| `tau` | Avg depth | Compute reduction | Relative loss increase | shallow / medium / deep |
|-------|-----------|-------------------|------------------------|--------------------------|
| 0.05 | 12.00 | 0.0% | 0.0% | 0 / 0 / 499 |
| 0.25 | 11.97 | 0.3% | 0.1% | 2 / 0 / 497 |
| 0.50 | 11.48 | 4.3% | 5.5% | 15 / 35 / 449 |
| 0.75 | 10.39 | 13.4% | 22.1% | 38 / 125 / 336 |
| **1.00** | **8.87** | **26.1%** | **51.5%** | **79 / 233 / 187** |
| 1.50 | 5.82 | 51.5% | 105.8% | 291 / 189 / 19 |
| 2.00 | 4.46 | 62.8% | 126.3% | 441 / 58 / 0 |

The frontier is monotone, as it must be. It also shows the central difficulty:
meaningful compute reduction on this backbone requires accepting a large
relative loss increase. There is no operating point offering, say, 25% compute
reduction at under 10% loss increase.

### 11.2 Main results (60 short-QA prompts)

| Method | Prompt loss | Avg depth | Layer red. | Rel. FLOPs | Latency (s) |
|--------|-------------|-----------|------------|------------|-------------|
| A: Static GPT-2 | **3.889** | 12.00 | 0.0% | 1.000 | 0.168 |
| B: Depth adaptive | 5.974 | 7.73 | 35.6% | 0.644 | 0.121 |
| C: Routing / MoE-style | 6.366 | 12.00 | 0.0% | 0.668 | 0.313 |
| E: Full MSA | 9.097 | 7.73 | 35.6% | **0.322** | **0.098** |
| D: Dense reference (untrained) | N/A | 12.00 | 0.0% | N/A | 0.440 |

Static is the best on quality. Full MSA is the cheapest on both estimated
compute and latency, and the worst on quality. Depth adaptation occupies the
middle and is the only variant whose quality cost is arguably proportionate to
its saving.

Baseline C is notable: it uses less arithmetic than static (0.668 relative
FLOPs) yet runs **1.9× slower**, because per-token gather/scatter on CPU costs
more than the multiplications it avoids.

### 11.3 Downstream task evaluation

| Dataset | Method | Answer PPL | Exact match | Avg depth | Rel. FLOPs |
|---------|--------|-----------|-------------|-----------|------------|
| short_qa | Static | **284.9** | **0.075** | 12.0 | 1.000 |
| short_qa | Depth adaptive | 2.31e6 | 0.025 | 7.6 | 0.633 |
| short_qa | Full MSA | 9.39e5 | 0.000 | 7.6 | 0.317 |
| gsm8k | Static | **1613.6** | 0.000 | 12.0 | 1.000 |
| gsm8k | Depth adaptive | 2138.4 | 0.000 | 11.8 | 0.983 |
| gsm8k | Full MSA | 1705.1 | 0.000 | 11.9 | 0.983 |

Three observations, all stated as measured:

1. **On short QA, task accuracy falls monotonically as adaptation is added**
   (7.5% → 2.5% → 0.0%). The compute saving is real and so is the quality loss.

2. **On GSM8K every variant scores 0.000, including the static baseline.**
   GPT-2 small cannot solve GSM8K. That dataset therefore measures compute
   behaviour, not task quality, and no quality conclusion may be drawn from it.

3. **Compute saving is dataset-dependent.** On GSM8K the predictor routes
   almost everything to `deep` (average depth 11.8–11.9), and the saving nearly
   vanishes. This is arguably correct behaviour, but it means **no single
   compute-reduction figure should be quoted without naming the dataset.**

### 11.4 Controlled latency benchmark

Trimmed mean of 50 runs after 10 warm-ups, milliseconds, CPU:

| Mechanism | Config | seq 16 | seq 64 | seq 256 |
|-----------|--------|--------|--------|---------|
| depth only | shallow | 34.8 | 94.3 | 267.6 |
| depth only | medium | 55.8 | 141.7 | 409.6 |
| depth only | deep | 77.1 | 183.4 | 518.5 |
| depth+attn+ffn | shallow | **28.8** | **79.6** | **193.2** |
| depth+attn+ffn | medium | 44.6 | 112.7 | 288.8 |
| depth+attn+ffn | deep | 76.8 | 187.5 | 504.6 |
| routing | shallow | 56.8 | 111.1 | 245.0 |
| routing | medium | 98.0 | 184.2 | 409.7 |
| routing | deep | 146.4 | 261.7 | 558.9 |

Attention and feed-forward slicing reduce latency by a further 17–20% at
sequence length 16, rising to 28–30% at 256. The mechanisms therefore deliver
genuine wall-clock savings; their problem is quality, not speed.

Routing is slower than depth-only at every length and configuration.

**Analytical and empirical cost disagree.** For the adaptive mechanisms they
agree in direction. For routing they invert: less arithmetic, more time. Any
compute claim must state which measure it refers to.

*Reproducibility note.* Our first benchmark run was discarded. It overlapped
with another job, inflating all figures roughly 3–4×, and one cell recorded
1403 ± 7362 ms from a single scheduler stall. The harness now reports a trimmed
mean and raises an `outlier_suspected` flag when the mean exceeds 1.5× the
median.

### 11.5 Stability

Controlled sequence of 60 inputs constructed to force switching (Section 12.3):

| Variant | Switches | Volatility | Rollbacks | Rollback rate | Prompt loss | Avg depth |
|---------|----------|-----------|-----------|---------------|-------------|-----------|
| A7: monitor disabled | 58 | 0.967 | 0 | 0.000 | 6.741 | 8.07 |
| A8: monitor active | 59 | **0.638** | 20 | 0.333 | **6.508** | 6.67 |

The monitor reduces volatility from 0.967 to 0.638 at a 33% rollback rate, and
loss does **not** degrade — it improves slightly, while average depth falls.
This is the evidence that justifies the component. We note the loss improvement
is small and from a single sequence; we do not claim rollback improves quality
in general.

---

## 12. Ablation Studies

### 12.1 Task Analyzer feature set

| Feature set | n features | Held-out | 5-fold CV |
|-------------|-----------|----------|-----------|
| v1 | 4 | 0.700 | 0.629 ± 0.154 |
| v2 | 8 | 0.720 | **0.673 ± 0.188** |

v2 is better on both measures. **The difference is well inside one standard
deviation and is not statistically established.** We use v2 in the shipped
predictor because it is no worse, not because the added features are
demonstrated to help. A larger calibration corpus is required to settle this.

### 12.2 Mechanism ablation (60 short-QA prompts)

| Variant | Prompt loss | Avg depth | Rel. FLOPs | Latency (s) |
|---------|-------------|-----------|------------|-------------|
| A1: Static | **3.889** | 12.00 | 1.000 | 0.212 |
| A2: Analyzer + rule policy | 8.067 | 4.00 | 0.333 | 0.111 |
| A3: Analyzer + `f_theta` | 5.974 | 7.73 | 0.644 | 0.190 |
| A4: Depth only | 5.974 | 7.73 | 0.644 | 0.205 |
| A5: Depth + attention | 8.403 | 7.73 | 0.536 | 0.186 |
| A6: Depth + FFN | 7.741 | 7.73 | 0.430 | 0.139 |
| A7: Full, no monitor | 9.097 | 7.73 | 0.322 | 0.122 |
| A8: Full MSA | 9.097 | 7.73 | 0.322 | 0.119 |

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

The corrected design buckets a 499-prompt pool by the configuration the
predictor *actually* selects, then alternates between buckets, and raises an
error rather than silently producing a flat sequence if fewer than two buckets
are populated. The results in Section 11.5 use the corrected design.

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

4. **Single-backbone, single-device study.** GPT-2 small on CPU. Nothing here
   establishes behaviour at larger scale or on accelerators, where the relative
   cost of slicing overhead versus arithmetic differs substantially.

5. **The routing baseline is untrained.** It bounds what an untrained gate
   achieves and says nothing about trained mixture-of-experts.

6. **Baseline D is randomly initialised.** Only its parameter count and latency
   are meaningful.

7. **The predictor's improvement from v2 features is not statistically
   established** (Section 12.1).

8. **The confidence threshold and monitor thresholds were not tuned.** They are
   defaults exposed as parameters; no sweep is reported.

9. **Small evaluation sets.** 60 short-QA items and 40 GSM8K items. Differences
   of a few percent in exact match are not meaningful at this size.

10. **Statistical significance is not established for any quality comparison.**
    We report point estimates and, where available, standard deviations.

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
  and roughly 28% lower latency, for a prompt-loss increase from 3.89 to 5.97.

- **A learned predictor clearly beats a hand-tuned threshold rule.** Loss
  improves from 8.07 to 5.97 while allocating *more* compute, which is the
  behaviour a calibrated selector should exhibit. This is the strongest
  component-level result in the study.

- **The stability monitor does what it was designed to do**, reducing volatility
  from 0.97 to 0.64 without degrading loss on a controlled sequence.

What the measurements do not support:

- **Attention and feed-forward adaptation are not currently worth their cost.**
  They deliver real compute and latency savings but degrade quality far more
  than depth adaptation does, because they slice pretrained weights with no
  recovery step. Full MSA is the worst-quality variant we tested.

- **No general compute-reduction claim is warranted.** The saving is strongly
  dataset-dependent: 35.6% layer reduction on short QA, 1.7% on GSM8K.

The most useful next experiments follow directly from these negatives: retrain
or distil the sliced configurations so the non-depth mechanisms become
competitive; train the predictor against a downstream task metric rather than
prompt loss; and repeat on a backbone capable of the evaluation tasks, so that
quality differences are measurable rather than floored at zero.

We also record a methodological result of independent interest: a single
argument in an adaptive forward pass silently disabled causal masking while
leaving loss curves qualitatively plausible. Adaptive-inference implementations
should assert numerical equivalence against the unmodified model at their
identity configuration.

---

## Reproducibility

```powershell
python main.py --mode all        # full pipeline (~45 min, calibration dominates)
python test_msa.py               # 27-check validation suite
```

| Artefact | Location |
|----------|----------|
| Calibration data (2,994 rows) | `results/calibration_results.csv` |
| Frozen prompt set (499) | `data/calibration_prompts.json` |
| Trained predictor + metadata | `results/performance_predictor.pkl` |
| All result tables | `results/tables.md` |
| All 12 figures | `results/figures/` |
| Environment capture | `results/environment.json` |
| Experiment configuration | `configs/experiment.yaml` |

Every number in this paper is read from those artefacts; none is transcribed by
hand. Seeds are fixed at 42 throughout.
