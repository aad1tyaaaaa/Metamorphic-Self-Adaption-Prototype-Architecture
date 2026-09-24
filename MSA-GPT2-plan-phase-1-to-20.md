# MSA-GPT-2 — Master Agent Execution Plan

This is the master execution plan for taking the current MSA-GPT-2 repository from the completed Phase 1 predictor-validation checkpoint through Phase 20, ending with a complete research prototype, reproducible experiments, UI simulation, and paper integration.

## Current status

All 20 phases complete; `python test_msa.py` passes 41/41 phase-gate checks.

- [x] Phase 1 — Predictor Validation
- [x] Phase 2 — f_theta Integration
- [x] Phase 3 — Configuration Policy
- [x] Phase 4 — Calibration Improvement
- [x] Phase 5 — Task Analyzer v2
- [x] Phase 6 — Dynamic Depth Controller
- [x] Phase 7 — Attention Adaptation
- [x] Phase 8 — FFN/Token Adaptation
- [x] Phase 9 — Full Architecture Controller
- [x] Phase 10 — Stability Monitor
- [x] Phase 11 — Evaluation Harness
- [x] Phase 12 — Baselines
- [x] Phase 13 — Proper Benchmarking
- [x] Phase 14 — Dataset Evaluation
- [x] Phase 15 — Ablations
- [x] Phase 16 — Stability Experiments
- [x] Phase 17 — Graphs
- [x] Phase 18 — Reproducibility
- [x] Phase 19 — UI Simulation
- [x] Phase 20 — Paper Integration

---

## 1. Agent rules

1. Inspect existing files before modifying them.
2. Preserve working functionality.
3. Implement one phase or tightly coupled sub-phase at a time.
4. Run tests after every meaningful change.
5. Never invent experimental results.
6. Save experiment results as CSV/JSON where practical.
7. Separate prototype calibration results from downstream task results.
8. Never describe prompt language-model loss as downstream task accuracy.
9. Keep GPT-2-small as the primary backbone.
10. Do not permanently modify pretrained weights merely to implement dynamic execution.
11. Prefer the simplest working implementation before adding complexity.
12. Record model, dataset, seed, device, software versions, and experiment parameters.
13. Do not mark a phase complete until its gate passes.
14. If the intended implementation is impossible under the available hardware/software, document the limitation and implement the smallest defensible alternative.
15. Never silently skip a failed experiment.

---

## 2. Target architecture

```text
Input
  ↓
Task Analyzer
  ↓
Feature Vector
  ↓
Performance Predictor f_theta
  ↓
Configuration Policy
  ↓
Dynamic Architecture Controller
  ├── Depth adaptation
  ├── Attention adaptation
  └── FFN/token adaptation
  ↓
Adaptive GPT-2
  ↓
Stability Monitor
  ↓
Rollback when unstable
  ↓
Output
```

Primary depth configurations:

```text
shallow = 4 layers
medium  = 8 layers
deep    = 12 layers
```

---

# PHASE 2 — f_theta INTEGRATION

## Objective

Connect the validated predictor to the actual MSA inference path.

Inspect:

```text
run_msa.py
controller/predictor.py
controller/policy.py
models/adaptive_model.py
analyzer/task_analyzer.py
```

Target flow:

```text
prompt
→ Task Analyzer v2
→ 8 features
→ f_theta
→ configuration policy
→ depth
→ AdaptiveGPT2
→ logits
```

The predictor should return:

```python
{
    "configuration": "...",
    "depth": 4 | 8 | 12,
    "confidence": ...,
    "probabilities": {...}
}
```

Create/update an integration test covering:

- factual question
- explanation
- mathematics
- multi-step reasoning
- long structured prompt

Verify that the selected depth is the depth actually executed by AdaptiveGPT2.

### Gate

Phase 2 is complete only when f_theta directly controls real GPT-2 execution.

---

# PHASE 3 — CONFIGURATION POLICY

Separate prediction from the final execution decision.

The predictor estimates:

```text
P(shallow)
P(medium)
P(deep)
```

The policy decides whether to accept the prediction.

Implement configurable confidence handling:

```text
confidence >= threshold
    → accept prediction

confidence < threshold
    → fallback configuration
```

Evaluate thresholds such as:

```text
0.40
0.50
0.60
0.70
0.80
```

Record:

```text
accepted predictions
fallback count
average depth
latency
loss
configuration distribution
```

Select an operating point from measured behavior.

### Gate

Policy behavior is explicit, measurable, and reproducible.

---

# PHASE 4 — CALIBRATION IMPROVEMENT

Current calibration:

```text
100 inputs × 3 depths = 300 rows
```

Expand toward:

```text
500–1,000 unique inputs
```

Include:

```text
factual
basic math
multi-step math
science
technology
logical reasoning
long explanations
structured prompts
summarization
code
hard reasoning
```

For every input execute:

```text
depth 4
depth 8
depth 12
```

Record:

```text
sample_id
text
all analyzer features
complexity
depth
configuration
loss
perplexity
latency
```

Preserve the existing calibration dataset as a historical version.

Retrain f_theta and evaluate:

```text
accuracy
macro F1
precision
recall
confusion matrix
confidence
```

Because classes are imbalanced, do not use accuracy alone.

### Gate

Expanded calibration exists, the predictor is retrained, and evaluation artifacts are saved.

---

# PHASE 5 — TASK ANALYZER V2

Already implemented.

Verify the eight-feature interface:

```text
input_length
reasoning
domain
structure
numeric_density
math_expression
question_type
reasoning_steps
```

Test:

```text
factual
math
code
reasoning
long structured
```

Freeze the feature interface for the main experiment unless an ablation justifies changing it.

### Gate

All eight features are consistently produced without invalid values.

---

# PHASE 6 — DYNAMIC DEPTH CONTROLLER

Formalize depth adaptation as a first-class controller.

```text
SHALLOW → 4 blocks
MEDIUM  → 8 blocks
DEEP    → 12 blocks
```

Create/refactor:

```text
controller/dynamic_controller.py
```

Prefer a configuration object:

```python
{
    "depth": 8,
    "attention_mode": "full",
    "ffn_mode": "full"
}
```

Initially only depth needs to be active.

Log:

```text
requested depth
executed depth
skipped layers
latency
```

Test that each configuration actually executes the requested number of blocks.

### Gate

4/8/12 layer execution is dynamically controlled and tested.

---

# PHASE 7 — ATTENTION ADAPTATION

Add a real attention-level adaptation mechanism.

Start with a conservative implementation such as attention-head masking.

Support at least:

```text
full attention
reduced attention
```

Do not merely store an attention mode label. The actual forward computation must change.

Test:

- output shape
- no NaN
- full mode
- reduced mode
- latency
- loss

Compare:

```text
depth only
vs
depth + attention
```

Measure:

```text
loss
perplexity
latency
average depth
configuration distribution
```

### Gate

Attention adaptation demonstrably changes actual model behavior.

---

# PHASE 8 — FFN / TOKEN ADAPTATION

Implement one additional architectural adaptation mechanism.

Preferred first implementation:

```text
FFN chunking
```

Concept:

```text
FFN
 ├── chunk 1
 ├── chunk 2
 ├── chunk 3
 └── chunk 4
```

Allow configurations to execute subsets.

Token routing is an alternative if FFN chunking is impractical.

Do not implement both before one is stable.

Measure:

```text
loss
latency
memory if available
```

### Gate

The mechanism changes actual computation and produces measurable outputs.

---

# PHASE 9 — FULL ARCHITECTURE CONTROLLER

Combine:

```text
depth
attention
FFN/token adaptation
```

Example:

```python
{
    "name": "medium",
    "depth": 8,
    "attention_mode": "reduced",
    "ffn_mode": "partial"
}
```

Keep the configuration library small enough to calibrate.

Example:

```text
C1:
depth 4
reduced attention
partial FFN

C2:
depth 8
reduced attention
partial FFN

C3:
depth 12
full attention
full FFN
```

Calibrate every configuration used in final experiments.

### Gate

One controller coordinates all active architecture dimensions and each dimension is actually applied.

---

# PHASE 10 — STABILITY MONITOR

Create:

```text
monitor/stability_monitor.py
```

Track a rolling history of:

```text
configuration
switches
volatility
rollback count
```

Implement the stability concepts used by the research design:

```text
S(t)
volatility
rollback rate
```

The implementation must match the equations used in the final paper.

Conceptual behavior:

```text
stable
    → continue

warning
    → observe

unstable
    → rollback to previous stable configuration
```

Create a synthetic sequence such as:

```text
deep
medium
deep
shallow
deep
medium
deep
```

Verify that instability is detected and rollback occurs.

### Gate

Rollback affects execution and is measurable.

---

# PHASE 11 — EVALUATION HARNESS

Create:

```text
evaluation/
├── runner.py
├── metrics.py
├── evaluate_static.py
├── evaluate_depth_adaptive.py
└── evaluate_msa.py
```

Standard result schema:

```text
method
dataset
sample_id
loss
perplexity
latency
depth
layer_reduction
configuration
attention_mode
ffn_mode
confidence
rollback
```

Add task-specific metrics where applicable.

Save:

```text
results/evaluation/*.csv
```

### Gate

All major methods can be evaluated through one standardized framework.

---

# PHASE 12 — BASELINES

Implement:

## Baseline A — Static GPT-2

Always execute 12 layers.

## Baseline B — Depth Adaptive

Only depth changes:

```text
4 / 8 / 12
```

No attention, FFN, or stability mechanisms.

## Baseline C — Routing/MoE-style

Implement a comparable routing-only baseline where feasible and document exactly what it does.

## Baseline D — Contemporary Dense Reference

Use a separate small dense model matching the planned characteristics:

```text
RMSNorm
RoPE
SwiGLU
no adaptivity
```

Record exact model and parameter count.

## Baseline E — Full MSA

Enable:

```text
Task Analyzer
f_theta
Configuration Policy
Dynamic Depth
Attention Adaptation
FFN/Token Adaptation
Stability Monitor
```

### Gate

Every baseline has reproducible code and a documented definition.

---

# PHASE 13 — PROPER BENCHMARKING

Replace rough single-pass timing with repeated measurements.

Recommended:

```text
warmup runs: 10+
measured runs: 30+
```

If hardware constraints require fewer, document the deviation.

Record:

```text
mean latency
standard deviation
median
throughput
device
sequence length
batch size
```

Keep hardware/software conditions fixed.

### Gate

Latency measurements are repeatable and reproducible.

---

# PHASE 14 — DATASET EVALUATION

Minimum planned evaluation:

```text
easy/short QA or classification
GSM8K
```

Then add if feasible:

```text
summarization
code
```

Use task-appropriate metrics:

```text
classification → accuracy / macro F1
math → exact match or appropriate metric
summarization → ROUGE or appropriate metric
code → execution/pass rate where feasible
```

Also collect:

```text
latency
average depth
layer reduction
configuration distribution
```

Do not substitute prompt loss for downstream task accuracy.

### Gate

At least the minimum planned datasets have task-level evaluation.

---

# PHASE 15 — ABLATIONS

Run:

```text
A1 — Static GPT-2

A2 — Analyzer + rule-based policy + depth

A3 — Analyzer + f_theta + policy + depth

A4 — Dynamic depth only

A5 — Depth + attention

A6 — Depth + FFN/token

A7 — Full architecture without stability monitor

A8 — Full MSA
```

For each record:

```text
quality
latency
compute/depth
stability
```

The purpose is to identify the contribution of each mechanism.

### Gate

All defined ablations have saved results.

---

# PHASE 16 — STABILITY EXPERIMENTS

Create controlled sequences with:

```text
low switching
medium switching
high switching
```

Compare:

```text
without stability monitor
vs
with stability monitor
```

Measure:

```text
configuration switches
volatility
rollback count
rollback rate
quality after rollback
latency
```

Save:

```text
results/stability/*.csv
```

### Gate

The stability experiment demonstrates measurable monitor/rollback behavior.

---

# PHASE 17 — GRAPHS

Create:

```text
evaluation/generate_plots.py
```

Save under:

```text
results/figures/
```

Generate at minimum:

1. depth vs loss
2. depth vs latency
3. quality-compute frontier
4. predictor confusion matrix
5. predictor confidence distribution
6. configuration distribution
7. average depth by dataset
8. layer reduction by method
9. latency by method
10. quality vs latency
11. stability volatility
12. rollback events
13. ablation comparison
14. baseline comparison

All plots must be generated from saved result files.

### Gate

Required plots can be regenerated automatically.

---

# PHASE 18 — REPRODUCIBILITY

Create:

```text
configs/experiment.yaml
```

Include:

```text
model
dataset
seed
configuration set
policy threshold
predictor path
warmup runs
benchmark runs
device
```

Record:

```text
Python version
PyTorch version
Transformers version
scikit-learn version
CUDA version if applicable
GPU/CPU
```

Update:

```text
requirements.txt
README.md
```

Use fixed seeds where practical.

Run a clean reproduction of the core experiment.

### Gate

Another environment can reproduce the core pipeline and regenerate results.

---

# PHASE 19 — UI SIMULATION

Build a visual demonstration of the complete MSA pipeline.

If using v0.app, use:

```text
Next.js
TypeScript
Tailwind CSS
shadcn/ui
Recharts
```

The UI must clearly distinguish simulated values from real backend inference unless actually connected.

Required panels:

### Input

Prompt entry.

### Task Analyzer

Show:

```text
input length
reasoning
domain
structure
numeric density
math expression
question type
reasoning steps
```

### Predictor

Show:

```text
P(shallow)
P(medium)
P(deep)
confidence
```

### Architecture

Show all 12 GPT-2 blocks and highlight:

```text
executed
skipped
```

Also show:

```text
attention mode
FFN mode
```

### Stability

Show:

```text
current configuration
stability state
volatility
rollback count
```

### Metrics

Show:

```text
latency
average depth
layer reduction
quality estimate
```

### History

Show recent decisions.

### Gate

The UI demonstrates the complete MSA flow and does not misrepresent simulation data as real inference.

---

# PHASE 20 — PAPER INTEGRATION

Turn the implementation and experiments into the research artifact.

Recommended structure:

```text
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
13. Stability Results
14. Limitations
15. Conclusion
```

The paper must distinguish:

```text
design objective
measured result
hypothesis
limitation
future work
```

Document limitations including:

```text
small backbone
heuristic analyzer
calibration dataset size
proxy loss where applicable
hardware constraints
prototype attention/FFN limitations
```

### Gate

Every major architectural claim maps to an implemented component, and every numerical claim maps to a saved experiment result.

---

# FINAL REPOSITORY TARGET

Approximately:

```text
MSA-gpt-2/
├── analyzer/
├── calibration/
├── configs/
├── controller/
├── data/
├── evaluation/
├── models/
├── monitor/
├── results/
│   ├── calibration/
│   ├── evaluation/
│   ├── stability/
│   ├── figures/
│   ├── calibration_results.csv
│   ├── quality_compute_frontier.csv
│   └── performance_predictor.pkl
├── tests/
├── main.py
├── run_msa.py
├── requirements.txt
├── README.md
└── plan.md
```

Do not force this exact tree if the existing project organization is cleaner.

---

# MASTER EXECUTION ORDER

Execute in this dependency-aware order:

```text
P1  Predictor validation
 ↓
P2  f_theta integration
 ↓
P3  Configuration policy
 ↓
P4  Calibration improvement
 ↓
P5  Task Analyzer v2 verification
 ↓
P6  Dynamic depth controller
 ↓
P7  Attention adaptation
 ↓
P8  FFN/token adaptation
 ↓
P9  Full architecture controller
 ↓
P10 Stability monitor
 ↓
P11 Evaluation harness
 ↓
P12 Baselines
 ↓
P13 Proper benchmarking
 ↓
P14 Dataset evaluation
 ↓
P15 Ablations
 ↓
P16 Stability experiments
 ↓
P17 Graphs
 ↓
P18 Reproducibility
 ↓
P19 UI simulation
 ↓
P20 Paper integration
```

Do not skip dependency phases.

---

# PHASE COMPLETION GATES

A phase is complete only when:

```text
P2  f_theta controls real GPT-2 execution
P3  policy has measurable confidence/fallback behavior
P4  expanded calibration exists and predictor is retrained
P5  eight-feature analyzer is verified
P6  4/8/12 depth is dynamically controlled
P7  attention adaptation changes actual behavior
P8  FFN/token adaptation changes actual behavior
P9  one controller coordinates all architecture dimensions
P10 instability causes measurable rollback
P11 standardized evaluation harness works
P12 baselines are reproducible
P13 benchmark methodology is repeatable
P14 minimum datasets have task metrics
P15 ablations are complete
P16 stability experiments are complete
P17 plots regenerate from result files
P18 clean reproduction succeeds
P19 UI demonstrates the complete pipeline
P20 paper sections map to implementation and results
```

---

# FAILURE HANDLING

If a task fails:

1. Capture the exact error.
2. Identify the cause.
3. Fix the smallest relevant component.
4. Re-run the phase test.
5. Preserve previous working versions where appropriate.
6. Document unresolved limitations.
7. Never fabricate output.

If hardware prevents an experiment:

```text
document limitation
run the valid subset
label the result appropriately
```

If a complex mechanism is unstable:

```text
implement a simpler defensible mechanism
test it
document the simplification
```

---

# RESEARCH QUALITY RULES

Distinguish:

### Software fact

Example:

```text
The controller executes four transformer blocks.
```

### Experimental result

Example:

```text
The four-layer configuration produced the measured latency under the tested benchmark conditions.
```

### Research claim

Example:

```text
The adaptive mechanism improves efficiency while preserving task quality.
```

The third statement requires appropriate evidence.

Do not turn implementation behavior into a broad research conclusion without measurement.

---

# FINAL CHECKLIST

```text
[x] Phase 1 — Predictor Validation
[x] Phase 2 — f_theta Integration
[x] Phase 3 — Configuration Policy
[x] Phase 4 — Calibration Improvement
[x] Phase 5 — Task Analyzer v2
[x] Phase 6 — Dynamic Depth Controller
[x] Phase 7 — Attention Adaptation
[x] Phase 8 — FFN/Token Adaptation
[x] Phase 9 — Full Architecture Controller
[x] Phase 10 — Stability Monitor
[x] Phase 11 — Evaluation Harness
[x] Phase 12 — Baselines
[x] Phase 13 — Proper Benchmarking
[x] Phase 14 — Dataset Evaluation
[x] Phase 15 — Ablations
[x] Phase 16 — Stability Experiments
[x] Phase 17 — Graphs
[x] Phase 18 — Reproducibility
[x] Phase 19 — UI Simulation
[x] Phase 20 — Paper Integration
```

Before declaring completion, produce a final report containing:

```text
1. Implemented architecture
2. Files created/modified
3. Tests passed
4. Datasets used
5. Baselines implemented
6. Benchmark methodology
7. Main measured results
8. Ablation results
9. Stability results
10. Limitations
11. UI status
12. Paper artifacts
13. Exact reproduction commands
```

---

# IMMEDIATE STARTING POINT

The repository is currently at:

```text
Phase 1 — COMPLETE
```

Start with Phase 2.

First inspect:

```text
run_msa.py
controller/predictor.py
controller/policy.py
models/adaptive_model.py
analyzer/task_analyzer.py
results/performance_predictor.pkl
```

Then implement and test:

```text
Prompt
 ↓
Task Analyzer v2
 ↓
8 features
 ↓
f_theta
 ↓
Configuration Policy
 ↓
Depth 4/8/12
 ↓
AdaptiveGPT2
 ↓
Logits
```

Only after Phase 2 passes should the agent proceed to Phase 3.

---

# FINAL DEFINITION OF DONE

The prototype reaches the Phase 20 target only when:

```text
Input
 ↓
Task Analyzer
 ↓
f_theta
 ↓
Configuration Policy
 ↓
Dynamic Depth
 +
Attention Adaptation
 +
FFN/Token Adaptation
 ↓
Adaptive GPT-2
 ↓
Stability Monitor
 ↓
Rollback when required
 ↓
Output
```

is implemented and tested,

and:

```text
Static GPT-2
Depth Adaptive
Routing/MoE-style baseline
Contemporary Dense Reference
Full MSA
```

can be evaluated,

and:

```text
calibration
benchmarking
dataset evaluation
ablations
stability experiments
graphs
```

have saved reproducible outputs,

and:

```text
README
experiment configuration
environment information
UI simulation
paper artifacts
```

are complete.

Do not declare the project complete merely because source files exist.

**Completion means: implemented + tested + measured + reproducible + documented.**
