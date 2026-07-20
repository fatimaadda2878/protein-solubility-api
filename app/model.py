"""
Chargement et inférence du modèle LightGBM.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


_model = None
_scaler = None

FEATURES = [
    "pI",
    "log_mw",
    "gravy_norm",
    "log_instability",
    "aromaticity",
    "pct_helix",
    "pct_turn",
    "pct_sheet",
    "pI_distance",
    "pct_coil",
    "helix_x_gravy",
    "stability_score",
    "helix_sheet_ratio",
]

DEFAULT_THRESHOLD = 0.30


def get_model_metadata() -> Dict[str, Any]:
    """Charge les métadonnées du modèle depuis model/model_meta.json."""
    metadata_path = Path(
        os.getenv("MODEL_META_PATH", "model/model_meta.json")
    )

    defaults = {
        "model_type": "LightGBM",
        "dataset": "DeepSol",
        "run_id": None,
        "auc_validation": None,
        "auc_test": None,
        "threshold": DEFAULT_THRESHOLD,
    }

    if not metadata_path.exists():
        return defaults

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if not isinstance(metadata, dict):
            return defaults

        result = {**defaults, **metadata}

        try:
            threshold = float(result.get("threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            threshold = DEFAULT_THRESHOLD

        if not 0.0 <= threshold <= 1.0:
            threshold = DEFAULT_THRESHOLD

        result["threshold"] = threshold
        return result

    except (OSError, json.JSONDecodeError):
        return defaults


def is_model_loaded() -> bool:
    """Indique si le modèle est déjà chargé en mémoire."""
    return _model is not None


def load_model():
    """Charge le modèle et, s'il existe, le scaler."""
    global _model, _scaler

    if _model is not None:
        return _model

    model_path = Path(
        os.getenv("MODEL_PATH", "model/lgbm_model.joblib")
    )
    scaler_path = Path(
        os.getenv("SCALER_PATH", "model/scaler.joblib")
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. "
            "Exécutez d'abord python retrain_model.py."
        )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return _model


def _compute_derived_features(data: dict) -> np.ndarray:
    """Construit les 13 variables utilisées lors de l'entraînement."""
    p_i = data["pI"]
    log_mw = data["log_mw"]
    gravy_norm = data["gravy_norm"]
    log_instability = data["log_instability"]
    aromaticity = data["aromaticity"]
    pct_helix = data["pct_helix"]
    pct_turn = data["pct_turn"]
    pct_sheet = data["pct_sheet"]

    p_i_distance = abs(p_i - 7.0)
    pct_coil = max(
        0.0,
        1.0 - pct_helix - pct_turn - pct_sheet,
    )
    helix_x_gravy = pct_helix * gravy_norm
    stability_score = log_instability * gravy_norm
    helix_sheet_ratio = pct_helix / (pct_sheet + 1e-6)

    return np.array(
        [
            p_i,
            log_mw,
            gravy_norm,
            log_instability,
            aromaticity,
            pct_helix,
            pct_turn,
            pct_sheet,
            p_i_distance,
            pct_coil,
            helix_x_gravy,
            stability_score,
            helix_sheet_ratio,
        ],
        dtype=float,
    ).reshape(1, -1)


def predict(protein_input) -> Dict[str, Any]:
    """Effectue une prédiction de solubilité."""
    global _model, _scaler

    if _model is None:
        load_model()

    if hasattr(protein_input, "model_dump"):
        data = protein_input.model_dump()
    else:
        data = protein_input.dict()

    features = _compute_derived_features(data)

    if _scaler is not None:
        features = _scaler.transform(features)

    probabilities = _model.predict_proba(features)[0]
    probability_insoluble = float(probabilities[0])
    probability_soluble = float(probabilities[1])

    threshold = float(
        get_model_metadata().get(
            "threshold",
            DEFAULT_THRESHOLD,
        )
    )
    prediction = int(probability_soluble >= threshold)

    maximum_probability = max(
        probability_soluble,
        probability_insoluble,
    )

    if maximum_probability >= 0.80:
        confidence = "Élevé"
    elif maximum_probability >= 0.60:
        confidence = "Modéré"
    else:
        confidence = "Faible"

    if prediction == 1 and probability_soluble >= 0.70:
        recommendation = (
            "Protéine probablement soluble — expression standard recommandée."
        )
    elif prediction == 1:
        recommendation = (
            "Solubilité modérée — envisager une induction à basse température "
            "et un tag de solubilisation."
        )
    else:
        recommendation = (
            "Risque élevé de corps d'inclusion — réduire la température "
            "d'induction ou utiliser un tag de solubilisation."
        )

    return {
        "soluble": prediction,
        "probability_soluble": round(probability_soluble, 4),
        "probability_insoluble": round(probability_insoluble, 4),
        "confidence": confidence,
        "recommendation": recommendation,
    }
