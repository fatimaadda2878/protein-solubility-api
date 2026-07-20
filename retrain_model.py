"""Entraînement reproductible LightGBM avec suivi MLflow.

Ce script :
- conserve les splits train / valid / test ;
- entraîne uniquement sur train ;
- choisit le seuil sur valid ;
- évalue une seule fois sur test ;
- enregistre le modèle et les métriques dans MLflow ;
- exporte le modèle réel dans model/lgbm_model.joblib ;
- génère model/model_meta.json.
"""

from __future__ import annotations

import json
import os
import pickle
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_URL = (
    "https://raw.githubusercontent.com/sameerkhurana10/"
    "DSOL_rv0.2/master/data/protein_with_bio.data"
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///C:/Users/adda-/mlflow.db",
)
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "protein-solubility-ecoli",
)

RANDOM_STATE = 42
FN_COST_EUR = 1200
FP_COST_EUR = 200

BASE_FEATURES = [
    "pI",
    "log_mw",
    "gravy_norm",
    "log_instability",
    "aromaticity",
    "pct_helix",
    "pct_turn",
    "pct_sheet",
]

FEATURES = BASE_FEATURES + [
    "pI_distance",
    "pct_coil",
    "helix_x_gravy",
    "stability_score",
    "helix_sheet_ratio",
]


def load_dataset() -> dict:
    print("Téléchargement du dataset DeepSol...")
    raw = urllib.request.urlopen(DATA_URL, timeout=120).read()
    return pickle.loads(raw)


def make_features(source_data) -> pd.DataFrame:
    frame = pd.DataFrame(
        [[float(values[index]) for index in range(8)] for values in source_data],
        columns=BASE_FEATURES,
    )

    frame["pI_distance"] = (frame["pI"] - 7.0).abs()
    frame["pct_coil"] = (
        1.0
        - frame["pct_helix"]
        - frame["pct_turn"]
        - frame["pct_sheet"]
    ).clip(0.0, 1.0)
    frame["helix_x_gravy"] = frame["pct_helix"] * frame["gravy_norm"]
    frame["stability_score"] = (
        frame["log_instability"] * frame["gravy_norm"]
    )
    frame["helix_sheet_ratio"] = (
        frame["pct_helix"] / (frame["pct_sheet"] + 1e-6)
    )

    return frame[FEATURES]


def get_split(dataset: dict, split: str):
    features = make_features(dataset[split]["src_bio"])
    target = np.asarray(dataset[split]["tgt"], dtype=int)
    return features, target


def choose_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        cost = fn * FN_COST_EUR + fp * FP_COST_EUR

        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)

    return best_threshold, best_cost


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost_eur": int(
            fn * FN_COST_EUR + fp * FP_COST_EUR
        ),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = load_dataset()

    x_train, y_train = get_split(dataset, "train")
    x_valid, y_valid = get_split(dataset, "valid")
    x_test, y_test = get_split(dataset, "test")

    params = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    with mlflow.start_run(run_name="lightgbm-independent-test") as run:
        model = lgb.LGBMClassifier(**params)

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=40,
                    verbose=False,
                )
            ],
        )

        validation_probabilities = model.predict_proba(x_valid)[:, 1]
        threshold, validation_cost = choose_threshold(
            y_valid,
            validation_probabilities,
        )

        validation_metrics = evaluate(
            y_valid,
            validation_probabilities,
            threshold,
        )

        test_probabilities = model.predict_proba(x_test)[:, 1]
        test_metrics = evaluate(
            y_test,
            test_probabilities,
            threshold,
        )

        mlflow.log_params(params)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("threshold_selection_split", "validation")
        mlflow.log_param("false_negative_cost_eur", FN_COST_EUR)
        mlflow.log_param("false_positive_cost_eur", FP_COST_EUR)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_validation", len(y_valid))
        mlflow.log_param("n_test", len(y_test))

        mlflow.log_metric("validation_threshold", threshold)
        mlflow.log_metric("validation_cost_eur", validation_cost)

        for name, value in validation_metrics.items():
            mlflow.log_metric(f"validation_{name}", value)

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        output_dir = Path("model")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "lgbm_model.joblib"
        joblib.dump(model, model_path)

        metadata = {
            "run_id": run.info.run_id,
            "source": "mlflow",
            "tracking_uri": TRACKING_URI,
            "experiment_name": EXPERIMENT_NAME,
            "artifact_path": "model",
            "model_version": datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            ),
            "algorithm": "LightGBM",
            "dataset": "DeepSol",
            "features": FEATURES,
            "random_state": RANDOM_STATE,
            "split_sizes": {
                "train": len(y_train),
                "validation": len(y_valid),
                "test": len(y_test),
            },
            "threshold": round(threshold, 4),
            "threshold_selection": {
                "split": "validation",
                "false_negative_cost_eur": FN_COST_EUR,
                "false_positive_cost_eur": FP_COST_EUR,
                "validation_cost_eur": int(validation_cost),
            },
            "validation_metrics": {
                key: round(value, 6)
                for key, value in validation_metrics.items()
            },
            "test_metrics": {
                key: round(value, 6)
                for key, value in test_metrics.items()
            },
        }

        meta_path = output_dir / "model_meta.json"
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("Entraînement terminé.")
        print(f"Run ID MLflow : {run.info.run_id}")
        print(f"Seuil validation : {threshold:.2f}")
        print(f"AUC validation : {validation_metrics['roc_auc']:.4f}")
        print(f"AUC test indépendante : {test_metrics['roc_auc']:.4f}")
        print(f"Modèle local : {model_path}")
        print(f"Métadonnées : {meta_path}")
        print("Artefact MLflow : model")


if __name__ == "__main__":
    main()
