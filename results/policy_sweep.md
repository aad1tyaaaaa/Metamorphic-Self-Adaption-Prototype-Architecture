(.venv) PS C:\Users\Aditya\Documents\Github\MSA-gpt-2> python controller/policy_sweep.py


======================================================================
MSA QUALITY-COMPUTE FRONTIER
======================================================================

 tolerance  average_depth  layer_reduction_percent  average_latency  average_loss  relative_loss_increase  shallow  medium  deep
      0.10          12.00                 0.000000         0.030914      3.152854                0.000000        0       0   100
      0.25          12.00                 0.000000         0.030914      3.152854                0.000000        0       0   100
      0.50          11.80                 1.666667         0.030272      3.227105                0.023551        1       3    96
      0.75          11.16                 7.000000         0.028896      3.568290                0.131765        4      13    83
      1.00           9.88                17.666667         0.025999      4.421713                0.402448       14      25    61
      1.50           8.16                32.000000         0.022732      5.705219                0.809542       27      42    31
      2.00           6.84                43.000000         0.020286      6.760231                1.144163       42      45    13


======================================================================
INTERPRETATION
======================================================================

Tolerance: 10%
  Depth: 12.00
  Layer reduction: 0.00%
  Shallow: 0
  Medium: 0
  Deep: 100

Tolerance: 25%
  Depth: 12.00
  Layer reduction: 0.00%
  Shallow: 0
  Medium: 0
  Deep: 100

Tolerance: 50%
  Depth: 11.80
  Layer reduction: 1.67%
  Shallow: 1
  Medium: 3
  Deep: 96

Tolerance: 75%
  Depth: 11.16
  Layer reduction: 7.00%
  Shallow: 4
  Medium: 13
  Deep: 83

Tolerance: 100%
  Depth: 9.88
  Layer reduction: 17.67%
  Shallow: 14
  Medium: 25
  Deep: 61

Tolerance: 150%
  Depth: 8.16
  Layer reduction: 32.00%
  Shallow: 27
  Medium: 42
  Deep: 31

Tolerance: 200%
  Depth: 6.84
  Layer reduction: 43.00%
  Shallow: 42
  Medium: 45
  Deep: 13

Saved:
results/quality_compute_frontier.csv