(.venv) PS C:\Users\Aditya\Documents\Github\MSA-gpt-2> python controller/train_predictor.py
======================================================================
TRAINING MSA PERFORMANCE PREDICTOR
======================================================================
Calibration samples: 300

Target distribution:
target_depth
4     14
8     25
12    61
Name: count, dtype: int64

Training 2-layer MLP...
C:\Users\Aditya\Documents\Github\MSA-gpt-2\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (2000) reached andthe optimization hasn't converged yet.
  warnings.warn(

Predictor evaluation
----------------------------------------
MAE : 1.7176
RMSE: 2.1671

Sample predictions:
 actual_depth  predicted_depth
            4        10.008143
           12        11.688852
           12        10.343798
           12        10.448802
           12        10.553807
           12        10.448802
           12        11.688852
            8         6.983594
            8        11.163830
            8        11.583847

Saved:
results/performance_predictor.pkl