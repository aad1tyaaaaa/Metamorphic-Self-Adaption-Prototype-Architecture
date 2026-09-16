# Metamorphic Self-Adaptation (MSA) Prototype

A prototype that runs GPT-2 with a **variable number of transformer layers per input**, chosen by a lightweight complexity scorer, instead of always running the full 12-layer stack. The idea: cheap/simple prompts get a shallow (fast) pass, harder prompts get a deep (accurate) pass.

## Pipeline

```
text ─▶ TaskAnalyzer ─▶ ComplexityScorer ─▶ ConfigurationSelector ─▶ depth ─▶ AdaptiveGPT2
       (4 features)      (weighted sum)      (shallow/medium/deep)   (4/8/12 layers)
```

1. **`analyzer/task_analyzer.py`** — extracts 4 features from the input text: input length, reasoning-word density, math/domain-word density, structural complexity (punctuation/sentence count).
2. **`controller/complexity.py`** — combines the 4 features into a single complexity score via fixed weights.
3. **`controller/selector.py`** — thresholds the score into `shallow` / `medium` / `deep`.
4. **`models/adaptive_model.py`** — `AdaptiveGPT2` wraps a HuggingFace `GPT2LMHeadModel` and runs only the first `depth` transformer blocks (4, 8, or 12 of GPT-2's 12) instead of the full stack.
5. **`models/backbone.py`** — loads the base `gpt2` model/tokenizer from HuggingFace.

`run_msa.py` wires all of this together end to end for a few example prompts.

## Project structure

```
analyzer/       task feature extraction
calibration/    depth-vs-quality/latency sweep, writes results/calibration_results.csv
configs/        static config values
controller/     complexity scoring + shallow/medium/deep selection
data/           dataset loading
evaluation/     eval scripts + metrics
models/         GPT-2 backbone + adaptive (variable-depth) wrapper
monitor/        stability monitoring
results/        run logs and calibration output (.md/.csv)
main.py         quick manual smoke test of the backbone
run_static.py   fixed-depth baseline (WIP)
run_msa.py      full adaptive pipeline demo
train_predictor.py   depth predictor training (WIP)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```powershell
python run_msa.py                      # full adaptive pipeline on sample prompts
python calibration/run_calibration.py  # sweep depths 4/8/12, log loss/perplexity/latency
```

## Findings so far

`calibration/run_calibration.py` (13 prompts x depths 4/8/12) shows the expected depth/quality/latency tradeoff:

| Depth | Avg. Loss | Avg. Perplexity | Avg. Latency |
|-------|-----------|------------------|--------------|
| 4     | 9.40      | 21,106           | 0.014s       |
| 8     | 6.77      | 2,467            | 0.022s       |
| 12    | 3.32      | 43               | 0.030s       |

Deeper = slower but far more accurate. The current `ComplexityScorer` weights are hand-picked, not learned — the selector currently routes most test prompts to `shallow` even where a deeper pass would clearly help (see `results/`), which is the main open problem: complexity scoring needs to be calibrated against these results, or replaced with a learned predictor (`train_predictor.py`).

## Status

Prototype / research code. `run_static.py` and `train_predictor.py` are stubs.
