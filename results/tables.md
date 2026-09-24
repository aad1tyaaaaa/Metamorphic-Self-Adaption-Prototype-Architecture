# MSA-GPT-2 -- Final Research Tables

## Table 1 -- Model and system configuration

| Item                         | Value                                                                                                                                       |
|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|
| Backbone                     | GPT-2 small (HuggingFace `gpt2`)                                                                                                            |
| Parameters                   | 124,439,808                                                                                                                                 |
| Layers                       | 12                                                                                                                                          |
| Attention heads              | 12                                                                                                                                          |
| Hidden size                  | 768                                                                                                                                         |
| FFN inner size               | 3072                                                                                                                                        |
| Feature set                  | v2 (8 features)                                                                                                                             |
| Features                     | input_length, reasoning, domain, structure, numeric_density, math_expression, question_type, reasoning_steps                                |
| Configurations               | shallow: depth 4, attention reduced, ffn partial; medium: depth 8, attention reduced, ffn partial; deep: depth 12, attention full, ffn full |
| Configuration depths         | shallow=4, medium=8, deep=12                                                                                                                |
| Predictor architecture       | StandardScaler + MLPClassifier(32, 16)                                                                                                      |
| Predictor held-out accuracy  | 0.720                                                                                                                                       |
| Predictor 5-fold CV accuracy | 0.673 +/- 0.188                                                                                                                             |
| Calibration tolerance        | 1.0                                                                                                                                         |
| Seed                         | 42                                                                                                                                          |
| Device                       | cpu                                                                                                                                         |
| PyTorch                      | 2.14.0+cpu                                                                                                                                  |
| Transformers                 | 5.17.0                                                                                                                                      |

## Table 2 -- Main results

| Method                                                       |   Prompt loss |   Perplexity |   Latency (s) |   Latency std |   Avg depth |   Layer red. |   Rel. FLOPs |   Compute red. |
|:-------------------------------------------------------------|--------------:|-------------:|--------------:|--------------:|------------:|-------------:|-------------:|---------------:|
| Baseline A: Static GPT-2 (12 layers)                         |        3.889  |      48.8617 |        0.168  |        0.0259 |     12      |       0      |       1      |         0      |
| Baseline B: Depth-adaptive                                   |        5.9736 |     392.915  |        0.1205 |        0.0282 |      7.7333 |       0.3556 |       0.6444 |         0.3556 |
| Baseline C: Routing / MoE-style FFN                          |        6.3663 |     581.896  |        0.3134 |        0.0459 |     12      |       0      |       0.6677 |         0.3323 |
| Baseline E: Full MSA                                         |        9.0972 |    8930.08   |        0.0981 |        0.0244 |      7.7333 |       0.3556 |       0.3222 |         0.6778 |
| Baseline D: Dense reference (RMSNorm/RoPE/SwiGLU, untrained) |      nan      |     nan      |        0.4402 |        0.0713 |     12      |       0      |     nan      |       nan      |

## Table 3 -- Ablation studies

| Method                                 |   Prompt loss |   Perplexity |   Latency (s) |   Avg depth |   Layer red. |   Rel. FLOPs |   Compute red. |
|:---------------------------------------|--------------:|-------------:|--------------:|------------:|-------------:|-------------:|---------------:|
| A1: Static GPT-2                       |        3.889  |      48.8617 |        0.2119 |     12      |       0      |       1      |         0      |
| A2: Analyzer + rule policy             |        8.0672 |    3188.11   |        0.1106 |      4      |       0.6667 |       0.3333 |         0.6667 |
| A3: Analyzer + f_theta                 |        5.9736 |     392.915  |        0.1904 |      7.7333 |       0.3556 |       0.6444 |         0.3556 |
| A4: Depth only                         |        5.9736 |     392.915  |        0.2054 |      7.7333 |       0.3556 |       0.6444 |         0.3556 |
| A5: Depth + attention                  |        8.4026 |    4458.61   |        0.1855 |      7.7333 |       0.3556 |       0.5364 |         0.4636 |
| A6: Depth + FFN                        |        7.7411 |    2301.01   |        0.1388 |      7.7333 |       0.3556 |       0.4303 |         0.5697 |
| A7: Full MSA without stability monitor |        9.0972 |    8930.08   |        0.122  |      7.7333 |       0.3556 |       0.3222 |         0.6778 |
| A8: Full MSA                           |        9.0972 |    8930.08   |        0.1187 |      7.7333 |       0.3556 |       0.3222 |         0.6778 |

## Table 4 -- Stability

| Variant                                |   Switches |   Switch rate |   Volatility |   Rollbacks |   Rollback rate |   Prompt loss |   Avg depth |
|:---------------------------------------|-----------:|--------------:|-------------:|------------:|----------------:|--------------:|------------:|
| A7: Full MSA without stability monitor |         58 |        0.9831 |       0.9667 |           0 |          0      |        6.7406 |      8.0667 |
| A8: Full MSA                           |         59 |        1      |       0.6378 |          20 |          0.3333 |        6.5077 |      6.6667 |

## Table 5 -- Downstream dataset evaluation

| Dataset   | Method                               |       Answer PPL |   Accuracy |   Prompt loss |   Avg depth |   Layer red. |   Rel. FLOPs |   Latency (s) |
|:----------|:-------------------------------------|-----------------:|-----------:|--------------:|------------:|-------------:|-------------:|--------------:|
| short_qa  | Baseline A: Static GPT-2 (12 layers) |    284.892       |      0.075 |        3.8017 |        12   |       0      |       1      |        0.1664 |
| short_qa  | Baseline B: Depth-adaptive           |      2.30551e+06 |      0.025 |        5.974  |         7.6 |       0.3667 |       0.6333 |        0.1098 |
| short_qa  | Baseline E: Full MSA                 | 938778           |      0     |        8.989  |         7.6 |       0.3667 |       0.3167 |        0.0803 |
| gsm8k     | Baseline A: Static GPT-2 (12 layers) |   1613.64        |      0     |        3.4066 |        12   |       0      |       1      |        0.3108 |
| gsm8k     | Baseline B: Depth-adaptive           |   2138.4         |      0     |        3.58   |        11.8 |       0.0167 |       0.9833 |        0.2601 |
| gsm8k     | Baseline E: Full MSA                 |   1705.11        |      0     |        3.5517 |        11.9 |       0.0083 |       0.9833 |        0.3745 |

## Table 6 -- Task Analyzer feature ablation

| feature_set   |   n_features |   accuracy |   cv_accuracy_mean |   cv_accuracy_std |
|:--------------|-------------:|-----------:|-------------------:|------------------:|
| v1            |            4 |       0.7  |           0.628929 |          0.154101 |
| v2            |            8 |       0.72 |           0.672909 |          0.187658 |

## Research claims checklist (Phase 22)
Every statement below is written against a measured result. Design objectives
and hypotheses are labelled as such, and are not presented as findings.

- MEASURED: attention-head and FFN-chunk adaptation change the computed output;
  they slice the weight tensors rather than masking outputs, so the skipped work
  is not performed (verified in `test_msa.py` against the reference model).
- MEASURED: the depth-only mechanism trades quality for compute monotonically
  across calibration tolerances (`results/quality_compute_frontier.csv`).
- MEASURED: the stability monitor detects a synthetic alternating sequence and
  performs rollbacks (`results/stability_experiment.csv`).
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
- LIMITATION: Baseline D is randomly initialised. Only its parameter count and
  latency are meaningful; its quality is reported as N/A.
- LIMITATION: latency is measured on CPU. Slicing overhead can exceed the saved
  work at short sequence lengths, so FLOP reduction does not always translate
  into wall-clock reduction. Both are reported.
- HYPOTHESIS (not established here): a predictor trained on downstream task
  quality rather than prompt loss would select configurations more usefully.
- FUTURE WORK: calibrate attention/FFN configurations with retrained or
  distilled weights so the non-depth mechanisms are competitive.

Claims deliberately NOT made: that MSA always improves performance, that it
guarantees lower latency, or that it preserves quality. The experiments in this
repository do not establish any of those.
