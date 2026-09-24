# MSA-GPT-2 -- Final Research Tables

## Table 1 -- Model and system configuration

| Item                                     | Value                                                                                                                                       |
|:-----------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| Backbone                                 | GPT-2 small (HuggingFace `gpt2`)                                                                                                            |
| Parameters                               | 124,439,808                                                                                                                                 |
| Layers                                   | 12                                                                                                                                          |
| Attention heads                          | 12                                                                                                                                          |
| Hidden size                              | 768                                                                                                                                         |
| FFN inner size                           | 3072                                                                                                                                        |
| Feature set                              | v2 (8 features)                                                                                                                             |
| Features                                 | input_length, reasoning, domain, structure, numeric_density, math_expression, question_type, reasoning_steps                                |
| Configurations                           | shallow: depth 4, attention reduced, ffn partial; medium: depth 8, attention reduced, ffn partial; deep: depth 12, attention full, ffn full |
| Configuration depths                     | shallow=4, medium=8, deep=12                                                                                                                |
| Predictor architecture                   | StandardScaler + MLPClassifier(32, 16)                                                                                                      |
| Calibration prompts                      | 799                                                                                                                                         |
| Predictor held-out accuracy              | 0.713                                                                                                                                       |
| Predictor held-out macro F1              | 0.678                                                                                                                                       |
| Majority-class baseline (acc / macro F1) | 0.500 / 0.222                                                                                                                               |
| Predictor 5-fold CV accuracy             | 0.731 +/- 0.019                                                                                                                             |
| Predictor 5-fold CV macro F1             | 0.693 +/- 0.030                                                                                                                             |
| Calibration tolerance                    | 1.0                                                                                                                                         |
| Policy confidence threshold              | 0.4                                                                                                                                         |
| Policy fallback                          | deep                                                                                                                                        |
| Seed                                     | 42                                                                                                                                          |
| Device                                   | cpu                                                                                                                                         |
| PyTorch                                  | 2.14.0+cpu                                                                                                                                  |
| Transformers                             | 5.17.0                                                                                                                                      |

## Table 2 -- Main results (Baselines A-E)

| Method                                                                      |   Prompt loss |   Perplexity |   Accuracy |     Answer PPL |   Latency (s) |   Latency std |   Avg depth |   Layer red. |   Rel. FLOPs |   Compute red. |
|:----------------------------------------------------------------------------|--------------:|-------------:|-----------:|---------------:|--------------:|--------------:|------------:|-------------:|-------------:|---------------:|
| Baseline A: Static GPT-2 (12 layers)                                        |        3.889  |      48.8617 |     0.1333 |  191.089       |        0.0228 |        0.0023 |     12      |       0      |       1      |         0      |
| Baseline B: Depth-adaptive                                                  |        5.9736 |     392.915  |     0.0167 |    1.62685e+06 |        0.0278 |        0.0168 |      7.7333 |       0.3556 |       0.6444 |         0.3556 |
| Baseline C: Routing / MoE-style FFN                                         |        6.3663 |     581.896  |     0      | 6600.92        |        0.1053 |        0.0291 |     12      |       0      |       0.6677 |         0.3323 |
| Baseline E: Full MSA                                                        |        9.0972 |    8930.09   |     0      |    2.41641e+06 |        0.0444 |        0.0155 |      7.7333 |       0.3556 |       0.3222 |         0.6778 |
| Baseline D: Dense reference (HuggingFaceTB/SmolLM-135M, 134,515,008 params) |        1.9846 |       7.276  |     0.75   |   12.5063      |        0.1099 |        0.0046 |     30      |       0      |     nan      |       nan      |

## Table 3 -- Ablation studies

| Method                                 |   Prompt loss |   Perplexity |   Accuracy |   Latency (s) |   Avg depth |   Layer red. |   Rel. FLOPs |   Compute red. |   volatility |   rollback_count |
|:---------------------------------------|--------------:|-------------:|-----------:|--------------:|------------:|-------------:|-------------:|---------------:|-------------:|-----------------:|
| A1: Static GPT-2                       |        3.889  |      48.8617 |     0.1333 |        0.07   |     12      |       0      |       1      |         0      |     nan      |              nan |
| A2: Analyzer + rule policy             |        8.0672 |    3188.11   |     0.0167 |        0.0341 |      4      |       0.6667 |       0.3333 |         0.6667 |     nan      |              nan |
| A3: Analyzer + f_theta                 |        5.9736 |     392.915  |     0.0167 |        0.0508 |      7.7333 |       0.3556 |       0.6444 |         0.3556 |     nan      |              nan |
| A4: Depth only                         |        5.9736 |     392.915  |     0.0167 |        0.0415 |      7.7333 |       0.3556 |       0.6444 |         0.3556 |     nan      |              nan |
| A5: Depth + attention                  |        8.4026 |    4458.61   |     0.0333 |        0.017  |      7.7333 |       0.3556 |       0.5364 |         0.4636 |     nan      |              nan |
| A6: Depth + FFN                        |        7.7411 |    2301.01   |     0      |        0.0267 |      7.7333 |       0.3556 |       0.4303 |         0.5697 |     nan      |              nan |
| A7: Full MSA without stability monitor |        9.0972 |    8930.09   |     0      |        0.0495 |      7.7333 |       0.3556 |       0.3222 |         0.6778 |       0.0667 |                0 |
| A8: Full MSA                           |        9.0972 |    8930.09   |     0      |        0.03   |      7.7333 |       0.3556 |       0.3222 |         0.6778 |       0.0667 |                0 |

## Table 4 -- Stability by switching level

| Switching   | Variant                                |   Proposed switches |   Executed switches |   Volatility V |   S = 1 - V |   Rollbacks |   Rollback rate |   Prompt loss |   Loss @ rollback steps |   Same steps, no monitor |   Latency (s) |   Avg depth |
|:------------|:---------------------------------------|--------------------:|--------------------:|---------------:|------------:|------------:|----------------:|--------------:|------------------------:|-------------------------:|--------------:|------------:|
| low         | A7: Full MSA without stability monitor |                   5 |                   5 |         0.0864 |      0.9136 |           0 |          0      |        5.8911 |                nan      |                 nan      |        0.0332 |      7.8    |
| low         | A8: Full MSA                           |                   5 |                   5 |         0.0864 |      0.9136 |           0 |          0      |        5.8911 |                nan      |                 nan      |        0.0352 |      7.8    |
| medium      | A7: Full MSA without stability monitor |                  21 |                  21 |         0.3231 |      0.6769 |           0 |          0      |        7.116  |                nan      |                 nan      |        0.0321 |      7.9333 |
| medium      | A8: Full MSA                           |                  21 |                  21 |         0.3231 |      0.6769 |           0 |          0      |        7.116  |                nan      |                 nan      |        0.0245 |      7.9333 |
| high        | A7: Full MSA without stability monitor |                  50 |                  50 |         0.8303 |      0.1697 |           0 |          0      |        7.0346 |                nan      |                 nan      |        0.0296 |      8.2667 |
| high        | A8: Full MSA                           |                  50 |                  34 |         0.5656 |      0.4344 |          13 |          0.2167 |        7.7071 |                  8.7581 |                   5.6542 |        0.0234 |      8      |

## Table 5 -- Downstream dataset evaluation

| Dataset   | Method                                                                      |   Accuracy |       Answer PPL |   Prompt loss |   Avg depth |   Layer red. |   Rel. FLOPs |   Latency (s) |
|:----------|:----------------------------------------------------------------------------|-----------:|-----------------:|--------------:|------------:|-------------:|-------------:|--------------:|
| short_qa  | Baseline A: Static GPT-2 (12 layers)                                        |      0.075 |    284.892       |        3.8017 |        12   |       0      |       1      |        0.0376 |
| short_qa  | Baseline B: Depth-adaptive                                                  |      0.025 |      2.30551e+06 |        5.974  |         7.6 |       0.3667 |       0.6333 |        0.0249 |
| short_qa  | Baseline E: Full MSA                                                        |      0     | 938776           |        8.989  |         7.6 |       0.3667 |       0.3167 |        0.0457 |
| short_qa  | Baseline D: Dense reference (HuggingFaceTB/SmolLM-135M, 134,515,008 params) |      0.675 |     12.7421      |        1.8313 |        30   |       0      |     nan      |        0.0532 |
| gsm8k     | Baseline A: Static GPT-2 (12 layers)                                        |      0     |   1613.64        |        3.4066 |        12   |       0      |       1      |        0.188  |
| gsm8k     | Baseline B: Depth-adaptive                                                  |      0     |   3701.57        |        3.8708 |        11.5 |       0.0417 |       0.9583 |        0.1316 |
| gsm8k     | Baseline E: Full MSA                                                        |      0     |   1857.81        |        4.0065 |        11.6 |       0.0333 |       0.9333 |        0.1318 |
| gsm8k     | Baseline D: Dense reference (HuggingFaceTB/SmolLM-135M, 134,515,008 params) |      0.05  |      8.4912      |        2.4125 |        30   |       0      |     nan      |        0.2312 |

## Table 6 -- Task Analyzer feature ablation

| feature_set   |   n_features |   accuracy |   macro_f1 |   cv_accuracy_mean |   cv_accuracy_std |   cv_macro_f1_mean |   cv_macro_f1_std |
|:--------------|-------------:|-----------:|-----------:|-------------------:|------------------:|-------------------:|------------------:|
| v1            |            4 |     0.65   |   0.55424  |           0.697107 |         0.0143302 |           0.611885 |         0.0111647 |
| v2            |            8 |     0.7125 |   0.677726 |           0.730881 |         0.0194524 |           0.692523 |         0.0302587 |

## Table 7 -- Policy confidence-threshold sweep

| mechanism   |   threshold |   accepted |   fallback_count |   average_depth |   relative_flops |   latency_mean |   loss |   relative_loss_increase |   target_agreement |   share_shallow |   share_medium |   share_deep | selected   |
|:------------|------------:|-----------:|-----------------:|----------------:|-----------------:|---------------:|-------:|-------------------------:|-------------------:|----------------:|---------------:|-------------:|:-----------|
| depth       |         0   |        160 |                0 |           8.725 |           0.7271 |         0.0519 | 6.2259 |                   0.5845 |             0.7125 |          0.0875 |         0.6438 |       0.2688 | False      |
| depth       |         0.4 |        158 |                2 |           8.8   |           0.7333 |         0.0521 | 6.1653 |                   0.5691 |             0.725  |          0.0812 |         0.6375 |       0.2812 | True       |
| depth       |         0.5 |        143 |               17 |           9.05  |           0.7542 |         0.0541 | 5.9771 |                   0.5212 |             0.725  |          0.0812 |         0.575  |       0.3438 | False      |
| depth       |         0.6 |        121 |               39 |           9.45  |           0.7875 |         0.0551 | 5.6523 |                   0.4385 |             0.6688 |          0.075  |         0.4875 |       0.4375 | False      |
| depth       |         0.7 |         86 |               74 |          10.125 |           0.8438 |         0.0568 | 5.1455 |                   0.3095 |             0.575  |          0.075  |         0.3188 |       0.6062 | False      |
| depth       |         0.8 |         40 |              120 |          11.175 |           0.9312 |         0.0599 | 4.3243 |                   0.1005 |             0.45   |          0.0688 |         0.0688 |       0.8625 | False      |
| full        |         0   |        160 |                0 |           8.725 |           0.4979 |         0.0463 | 7.735  |                   0.9685 |             0.35   |          0.0875 |         0.6438 |       0.2688 | False      |
| full        |         0.4 |        158 |                2 |           8.8   |           0.5073 |         0.0465 | 7.6563 |                   0.9485 |             0.3625 |          0.0812 |         0.6375 |       0.2812 | False      |
| full        |         0.5 |        143 |               17 |           9.05  |           0.549  |         0.048  | 7.3097 |                   0.8603 |             0.3875 |          0.0812 |         0.575  |       0.3438 | False      |
| full        |         0.6 |        121 |               39 |           9.45  |           0.6125 |         0.0495 | 6.7565 |                   0.7195 |             0.4438 |          0.075  |         0.4875 |       0.4375 | False      |
| full        |         0.7 |         86 |               74 |          10.125 |           0.725  |         0.0523 | 5.7989 |                   0.4758 |             0.55   |          0.075  |         0.3188 |       0.6062 | False      |
| full        |         0.8 |         40 |              120 |          11.175 |           0.8969 |         0.0579 | 4.4116 |                   0.1227 |             0.7062 |          0.0688 |         0.0688 |       0.8625 | False      |

## Table 8 -- Baseline D dense references

| model                                                        | trained   |   parameters |   layers |   depth |   hidden_size |   dim | norm    | position   | activation    |   sequence_length |   runs |   latency_mean |   latency_std |
|:-------------------------------------------------------------|:----------|-------------:|---------:|--------:|--------------:|------:|:--------|:-----------|:--------------|------------------:|-------:|---------------:|--------------:|
| DenseReference (RMSNorm + RoPE + SwiGLU), GPT-2-matched dims | False     |    162148608 |      nan |      12 |           nan |   768 | RMSNorm | RoPE       | SwiGLU        |                64 |     30 |         0.0582 |        0.0085 |
| HuggingFaceTB/SmolLM-135M                                    | True      |    134515008 |       30 |     nan |           576 |   nan | RMSNorm | RoPE       | SwiGLU (silu) |                64 |     30 |         0.2873 |        0.1986 |

## Research claims checklist
Every statement below is written against a measured result. Design objectives
and hypotheses are labelled as such, and are not presented as findings.

- MEASURED: attention-head and FFN-chunk adaptation change the computed output;
  they slice the weight tensors rather than masking outputs, so the skipped work
  is not performed (verified in `test_msa.py` against the reference model).
- MEASURED: the depth-only mechanism trades quality for compute monotonically
  across calibration tolerances (`results/quality_compute_frontier.csv`).
- MEASURED: the executed depth equals the depth f_theta + policy select, for
  every configuration and mechanism (`test_msa.py`, P2 and P6 checks).
- MEASURED: raising the policy confidence threshold from 0.40 to 0.80 moves the
  system monotonically toward the static model (fallbacks 2 -> 120 of 160,
  relative loss increase 56.9% -> 10.1%); 0.40 was selected by a rule fixed
  in advance (`results/policy_threshold_sweep.csv`).
- MEASURED: the stability monitor acts only when switching is high. At the high
  level it cut executed volatility from 0.830 to 0.566 with 13 rollbacks, but
  the rolled-back steps had higher loss (8.76 vs 5.65), so rollback traded
  quality for stability (`results/stability/stability_levels.csv`).
- MEASURED: under the controlled benchmark, depth and attention/FFN slicing
  reduce wall-clock latency; routing is slower than static depth-only at
  sequence lengths 16 and 64 and faster only at 256 (`results/benchmark.csv`).
- MEASURED: a pretrained dense model of similar size (SmolLM-135M) reaches 0.750
  exact match on short QA against 0.133 for static GPT-2 (`results/baselines.csv`).
- LIMITATION: calibration targets come from next-token loss over the prompt.
  That is a prototype signal, not downstream task quality. Downstream numbers
  are reported separately in `results/dataset_evaluation.csv`.
- LIMITATION: GPT-2 small cannot solve GSM8K. Reported accuracy on it is near
  zero for every variant, including the static baseline, so that dataset
  separates compute behaviour, not task quality.
- LIMITATION: attention and FFN adaptation are applied to pretrained weights
  with no retraining or distillation, so quality degrades markedly. The measured
  loss increase is reported and not explained away.
- LIMITATION: the routing baseline uses a fixed seeded gate, not a learned
  router, so it is a compute-reduction reference and not a trained MoE.
- LIMITATION: Baseline D uses a different tokenizer, so its per-token loss and
  perplexity are not comparable with GPT-2's; exact match and latency are.
- LIMITATION: latency is measured on a shared CPU with 6 pinned threads. The
  single-pass latencies recorded inside evaluation runs are noisy; only
  `results/benchmark.csv` is a controlled measurement.
- HYPOTHESIS (not established here): a predictor trained on downstream task
  quality rather than prompt loss would select configurations more usefully.
- FUTURE WORK: calibrate attention/FFN configurations with retrained or
  distilled weights so the non-depth mechanisms are competitive.

Claims deliberately NOT made: that MSA always improves performance, that it
guarantees lower latency, or that it preserves quality. The experiments in this
repository do not establish any of those.
