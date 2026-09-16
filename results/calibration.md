# calibration/run_calibration.py

```
python calibration/run_calibration.py
```

First attempt failed with `ModuleNotFoundError: No module named 'models'` — fixed by appending the project root to `sys.path` at the top of the script (run from a subfolder, so Python couldn't resolve the sibling `models`/`analyzer`/`controller` packages).

## Successful run

13 samples x 3 depths (4, 8, 12) = 39 experiments. Full per-sample results saved to `results/calibration_results.csv`.

```
Total experiments: 39
Saved to: results/calibration_results.csv

Average results by depth:
           loss    perplexity  latency_seconds
depth
4      9.402457  21106.071271         0.014305
8      6.770293   2466.869854         0.022357
12     3.322379     43.429360         0.029921
```

Deeper models are both more accurate (lower loss/perplexity) and slower — the expected depth/quality/latency tradeoff the MSA controller is meant to exploit.
