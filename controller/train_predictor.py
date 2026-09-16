import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

TOLERANCE = 1.00

FEATURES = [
    "input_length",
    "reasoning",
    "domain",
    "structure"
]


# ============================================================
# LOAD CALIBRATION DATA
# ============================================================

df = pd.read_csv("results/calibration_results.csv")

print("=" * 70)
print("TRAINING MSA PERFORMANCE PREDICTOR")
print("=" * 70)

print(f"Calibration samples: {len(df)}")


# ============================================================
# CREATE TARGETS
# ============================================================

targets = []

for sample_id, group in df.groupby("sample_id"):

    group = group.sort_values("depth")

    deep_row = group[group["depth"] == 12].iloc[0]

    baseline_loss = deep_row["loss"]

    maximum_allowed_loss = baseline_loss * (1 + TOLERANCE)

    acceptable = group[
        group["loss"] <= maximum_allowed_loss
    ]

    if len(acceptable) > 0:

        selected = acceptable.sort_values(
            "depth"
        ).iloc[0]

    else:

        selected = deep_row

    targets.append({
        "sample_id": sample_id,
        "target_depth": selected["depth"]
    })


targets_df = pd.DataFrame(targets)


# ============================================================
# MERGE FEATURES + TARGET
# ============================================================

sample_features = (
    df.groupby("sample_id")[FEATURES]
    .first()
    .reset_index()
)

training_df = sample_features.merge(
    targets_df,
    on="sample_id"
)


print("\nTarget distribution:")

print(
    training_df["target_depth"]
    .value_counts()
    .sort_index()
)


# ============================================================
# PREPARE DATA
# ============================================================

X = training_df[FEATURES].values

y = training_df["target_depth"].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# TRAIN 2-LAYER MLP
# ============================================================

print("\nTraining 2-layer MLP...")

predictor = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)

predictor.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

predictions = predictor.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


print("\nPredictor evaluation")
print("-" * 40)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# SHOW PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "actual_depth": y_test,
    "predicted_depth": predictions
})

print("\nSample predictions:")
print(results.head(10).to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    predictor,
    "results/performance_predictor.pkl"
)

print("\nSaved:")
print("results/performance_predictor.pkl")