"""
Chargement et inférence du modèle LightGBM
Le modèle est chargé une seule fois au démarrage pour optimiser les performances.
"""

import numpy as np
import joblib
import os
from pathlib import Path
from typing import Dict, Any

# ── Variable globale pour stocker le modèle en mémoire ───────
_model = None
_scaler = None

# ── Ordre des features (doit correspondre à l'entraînement) ──
FEATURES = [
    "pI",
    "log_mw",
    "gravy_norm",
    "log_instability",
    "aromaticity",
    "pct_helix",
    "pct_turn",
    "pct_sheet",
    # Features dérivées calculées automatiquement
    "pI_distance",
    "pct_coil",
    "helix_x_gravy",
    "stability_score",
    "helix_sheet_ratio",
]

# ── Seuil de décision optimal (issu de l'optimisation du coût métier) ──
OPTIMAL_THRESHOLD = 0.05


def load_model():
    """
    Je charge le modèle LightGBM depuis le fichier joblib.
    Cette fonction est appelée une seule fois au démarrage de l'API.
    """
    global _model, _scaler

    model_path = Path(os.getenv("MODEL_PATH", "model/lgbm_model.joblib"))
    scaler_path = Path(os.getenv("SCALER_PATH", "model/scaler.joblib"))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. "
            "Veuillez exporter le modèle depuis MLflow avec export_model.py"
        )

    _model  = joblib.load(model_path)

    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)

    return _model


def _compute_derived_features(data: dict) -> np.ndarray:
    """
    Je calcule les features dérivées à partir des features de base.
    Ces features sont identiques à celles créées lors de l'entraînement.
    """
    pI            = data["pI"]
    log_mw        = data["log_mw"]
    gravy_norm    = data["gravy_norm"]
    log_instab    = data["log_instability"]
    pct_helix     = data["pct_helix"]
    pct_turn      = data["pct_turn"]
    pct_sheet     = data["pct_sheet"]

    pI_distance       = abs(pI - 7.0)
    pct_coil          = max(0, 1 - pct_helix - pct_turn - pct_sheet)
    helix_x_gravy     = pct_helix * gravy_norm
    stability_score   = log_instab * gravy_norm
    helix_sheet_ratio = pct_helix / (pct_sheet + 1e-6)

    return np.array([
        pI,
        log_mw,
        gravy_norm,
        log_instab,
        data["aromaticity"],
        pct_helix,
        pct_turn,
        pct_sheet,
        pI_distance,
        pct_coil,
        helix_x_gravy,
        stability_score,
        helix_sheet_ratio,
    ]).reshape(1, -1)


def predict(protein_input) -> Dict[str, Any]:
    """
    Je réalise la prédiction de solubilité pour une protéine.

    Args:
        protein_input : objet ProteinInput avec les features biochimiques

    Returns:
        dict avec prediction, probabilités, confiance et recommandation
    """
    global _model, _scaler

    if _model is None:
        load_model()

    # Construction du vecteur de features
    data = protein_input.dict()
    X = _compute_derived_features(data)

    # Normalisation si scaler disponible
    if _scaler is not None:
        X = _scaler.transform(X)

    # Prédiction
    proba = _model.predict_proba(X)[0]
    prob_insoluble = float(proba[0])
    prob_soluble   = float(proba[1])

    # Application du seuil optimal (issu de l'optimisation du coût métier)
    prediction = 1 if prob_soluble >= OPTIMAL_THRESHOLD else 0

    # Niveau de confiance
    max_prob = max(prob_soluble, prob_insoluble)
    if max_prob >= 0.80:
        confidence = "Élevé"
    elif max_prob >= 0.60:
        confidence = "Modéré"
    else:
        confidence = "Faible"

    # Recommandation pratique
    if prediction == 1 and prob_soluble >= 0.70:
        recommendation = (
            "Protéine probablement soluble — expression standard recommandée. "
            "Température d'induction 25-37°C."
        )
    elif prediction == 1:
        recommendation = (
            "Solubilité modérée — envisager une induction à 16-20°C "
            "avec un tag de solubilisation (MBP ou SUMO)."
        )
    else:
        recommendation = (
            "Risque élevé de corps d'inclusion — réduire la T° d'induction, "
            "utiliser un tag MBP/SUMO, ou co-exprimer avec des chaperones."
        )

    return {
        "soluble":              prediction,
        "probability_soluble":  round(prob_soluble,   4),
        "probability_insoluble":round(prob_insoluble, 4),
        "confidence":           confidence,
        "recommendation":       recommendation,
    }
