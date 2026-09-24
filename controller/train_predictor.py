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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configs.configurations import CONFIG_NAMES, FEATURE_SETS
from controller.policy import CalibrationPolicy

CALIBRATION_PATH = "results/calibration_results.csv"
MODEL_PATH = "results/performance_predictor.pkl"
EVALUATION_DIR = "results/calibration"
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


def train(X, y, verbose=True, sample_ids=None):
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = make_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, labels=CONFIG_NAMES, average="macro",
                        zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=CONFIG_NAMES, zero_division=0
    )

    # Classes are imbalanced, so accuracy alone would flatter a majority-class model.
    majority = pd.Series(y_train).value_counts().idxmax()
    majority_accuracy = float(np.mean(y_test == majority))
    majority_macro_f1 = f1_score(y_test, np.full(len(y_test), majority),
                                 labels=CONFIG_NAMES, average="macro", zero_division=0)

    # Cross-validation is the more trustworthy number on a dataset this size.
    cv = cross_validate(
        make_model(), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring=["accuracy", "f1_macro"],
    )
    cv_scores = cv["test_accuracy"]
    cv_f1 = cv["test_f1_macro"]
    correct = predictions == y_test

    if verbose:
        print(f"\nHeld-out accuracy: {accuracy:.4f}  macro F1: {macro_f1:.4f}")
        print(f"Majority-class baseline: accuracy {majority_accuracy:.4f}  "
              f"macro F1 {majority_macro_f1:.4f}")
        print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"5-fold CV macro F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
        print(f"Mean confidence: {confidence.mean():.4f} "
              f"(correct {confidence[correct].mean():.4f}, "
              f"wrong {confidence[~correct].mean() if (~correct).any() else float('nan'):.4f})")
        print("\nClassification report:")
        print(classification_report(y_test, predictions, labels=CONFIG_NAMES,
                                    zero_division=0))
        print("Confusion matrix (rows = actual):")
        print(pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=CONFIG_NAMES),
            index=CONFIG_NAMES, columns=CONFIG_NAMES,
        ))

    held_out = pd.DataFrame({
        "sample_id": (np.asarray(sample_ids)[test_idx] if sample_ids is not None
                      else test_idx),
        "target": y_test,
        "predicted": predictions,
        "confidence": confidence,
        "correct": correct,
        **{f"p_{label}": probabilities[:, i] for i, label in enumerate(model.classes_)},
    })

    return model, {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f),
                    "support": int(s)}
            for label, p, r, f, s in zip(CONFIG_NAMES, precision, recall, f1, support)
        },
        "majority_class": str(majority),
        "majority_accuracy": majority_accuracy,
        "majority_macro_f1": float(majority_macro_f1),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_macro_f1_mean": float(cv_f1.mean()),
        "cv_macro_f1_std": float(cv_f1.std()),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_wrong": (float(confidence[~correct].mean())
                                  if (~correct).any() else float("nan")),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_sample_ids": held_out["sample_id"].astype(int).tolist(),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=CONFIG_NAMES
        ).tolist(),
        "confusion_labels": CONFIG_NAMES,
    }, held_out


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

    model, metrics, held_out = train(X, y, verbose=verbose,
                                     sample_ids=data["sample_id"].to_numpy())

    payload = {
        "model": model,
        "features": FEATURE_SETS[feature_set],
        "feature_set": feature_set,
        "tolerance": tolerance,
        "seed": SEED,
        "classes": list(model.classes_),
        "architecture": "StandardScaler + MLPClassifier(32, 16)",
        "python": platform.python_version(),
        "calibration_prompts": int(len(X)),
        "target_distribution": data["target_configuration"].value_counts().to_dict(),
        **metrics,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(payload, model_path)

    os.makedirs(EVALUATION_DIR, exist_ok=True)
    held_out.to_csv(f"{EVALUATION_DIR}/predictor_held_out.csv", index=False)
    with open(f"{EVALUATION_DIR}/predictor_evaluation.json", "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in payload.items() if k != "model"}, handle,
                  indent=2, default=str)

    if verbose:
        print(f"\nSaved: {model_path}")
        print(f"Saved: {EVALUATION_DIR}/predictor_evaluation.json, "
              f"{EVALUATION_DIR}/predictor_held_out.csv")

    return payload


def ablation(tolerance=1.00):
    """Phase 5: do the four extra features earn their place?"""
    frame = pd.read_csv(CALIBRATION_PATH)
    rows = []

    for feature_set in FEATURE_SETS:
        X, y, _ = build_training_set(frame, feature_set, tolerance)
        _, metrics, _ = train(X, y, verbose=False)

        rows.append({
            "feature_set": feature_set,
            "n_features": len(FEATURE_SETS[feature_set]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "cv_accuracy_mean": metrics["cv_accuracy_mean"],
            "cv_accuracy_std": metrics["cv_accuracy_std"],
            "cv_macro_f1_mean": metrics["cv_macro_f1_mean"],
            "cv_macro_f1_std": metrics["cv_macro_f1_std"],
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
            {k: v for k, v in payload.items()
             if k not in ("model", "confusion_matrix", "test_sample_ids")},
            indent=2, default=str,
        ))
