"""Train the configuration predictor f_theta.

    python -m controller.train_predictor [--feature-set v1|v2] [--tolerance 0.25]
    python -m controller.train_predictor --ablation

The Phase 5 ablation trains the same model on the original four features and on
the extended eight, so the value of the added features is measured rather than
assumed.
"""

import argparse
import json
import os
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configs.configurations import CONFIG_NAMES, FEATURE_SETS
from controller.policy import CalibrationPolicy

CALIBRATION_PATH = "results/calibration_results.csv"
MODEL_PATH = "results/performance_predictor.pkl"
SEED = 42


def build_training_set(frame, feature_set, tolerance, mechanism="depth"):
    features = FEATURE_SETS[feature_set]

    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(
            f"calibration data is missing {missing}; re-run "
            "`python -m calibration.run_calibration`"
        )

    targets = CalibrationPolicy(tolerance).create_targets(frame, mechanism=mechanism)

    sample_features = (
        frame[frame["mechanism"] == mechanism]
        .groupby("sample_id")[features].first().reset_index()
    )

    data = sample_features.merge(targets, on="sample_id")

    X = data[features].to_numpy(dtype=np.float64)
    y = data["target_configuration"].to_numpy(dtype=str)

    return X, y, data


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=100,
            random_state=SEED,
        )),
    ])


def train(X, y, verbose=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    model = make_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Cross-validation is the more trustworthy number on a dataset this size.
    cv_scores = cross_val_score(make_model(), X, y, cv=5)

    if verbose:
        print(f"\nHeld-out accuracy: {accuracy:.4f}")
        print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions, labels=CONFIG_NAMES,
                                    zero_division=0))
        print("Confusion matrix (rows = actual):")
        print(pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=CONFIG_NAMES),
            index=CONFIG_NAMES, columns=CONFIG_NAMES,
        ))

    return model, {
        "accuracy": float(accuracy),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=CONFIG_NAMES
        ).tolist(),
        "confusion_labels": CONFIG_NAMES,
    }


def run(feature_set="v2", tolerance=1.00, model_path=MODEL_PATH, verbose=True):
    frame = pd.read_csv(CALIBRATION_PATH)

    X, y, data = build_training_set(frame, feature_set, tolerance)

    if verbose:
        print("=" * 70)
        print(f"TRAINING PREDICTOR  feature_set={feature_set}  tolerance={tolerance}")
        print("=" * 70)
        print(f"Calibration rows: {len(frame)}  samples: {len(X)}")
        print("\nTarget distribution:")
        print(data["target_configuration"].value_counts().to_string())

    model, metrics = train(X, y, verbose=verbose)

    payload = {
        "model": model,
        "features": FEATURE_SETS[feature_set],
        "feature_set": feature_set,
        "tolerance": tolerance,
        "seed": SEED,
        "classes": list(model.classes_),
        "architecture": "StandardScaler + MLPClassifier(32, 16)",
        "python": platform.python_version(),
        "target_distribution": data["target_configuration"].value_counts().to_dict(),
        **metrics,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(payload, model_path)

    if verbose:
        print(f"\nSaved: {model_path}")

    return payload


def ablation(tolerance=1.00):
    """Phase 5: do the four extra features earn their place?"""
    frame = pd.read_csv(CALIBRATION_PATH)
    rows = []

    for feature_set in FEATURE_SETS:
        X, y, _ = build_training_set(frame, feature_set, tolerance)
        _, metrics = train(X, y, verbose=False)

        rows.append({
            "feature_set": feature_set,
            "n_features": len(FEATURE_SETS[feature_set]),
            "accuracy": metrics["accuracy"],
            "cv_accuracy_mean": metrics["cv_accuracy_mean"],
            "cv_accuracy_std": metrics["cv_accuracy_std"],
        })

    table = pd.DataFrame(rows)

    print("=" * 70)
    print("PHASE 5 ABLATION: TASK ANALYZER FEATURE SETS")
    print("=" * 70)
    print(table.round(4).to_string(index=False))

    table.to_csv("results/feature_ablation.csv", index=False)
    print("\nSaved: results/feature_ablation.csv")

    best = table.sort_values("cv_accuracy_mean").iloc[-1]
    print(f"\nBest by cross-validated accuracy: {best['feature_set']} "
          f"({best['cv_accuracy_mean']:.4f})")

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="v2", choices=list(FEATURE_SETS))
    parser.add_argument("--tolerance", type=float, default=1.00)
    parser.add_argument("--output", default=MODEL_PATH)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.ablation:
        ablation(tolerance=args.tolerance)
    else:
        payload = run(args.feature_set, args.tolerance, args.output)
        print("\n" + json.dumps(
            {k: v for k, v in payload.items() if k not in ("model", "confusion_matrix")},
            indent=2, default=str,
        ))
