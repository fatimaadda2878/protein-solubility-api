"""
Chargement et inférence du modèle — ONNX Runtime (avec fallback joblib)
Le modèle est chargé une seule fois au démarrage pour optimiser les performances.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ── Chemins des modèles ───────────────────────────────────────
ONNX_PATH       = Path(os.getenv("ONNX_PATH",       "model/lgbm_model.onnx"))
MODEL_PATH      = Path(os.getenv("MODEL_PATH",      "model/lgbm_model.joblib"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", "model/model_meta.json"))

# ── Variables globales ────────────────────────────────────────
_session   = None
_meta: Dict[str, Any] = {}
_threshold = 0.25
_use_onnx  = False


def load_model() -> None:
    """
    Je charge la session ONNX Runtime (ou joblib en fallback).
    Appelé une seule fois au démarrage de l'API.
    """
    global _session, _meta, _threshold, _use_onnx

    # Lecture des métadonnées
    if MODEL_META_PATH.exists():
        _meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        _threshold = float(_meta.get("threshold", 0.25))

    # Priorité ONNX
    if ONNX_PATH.exists():
        import onnxruntime as rt
        _session = rt.InferenceSession(str(ONNX_PATH))
        _use_onnx = True
        return

    # Fallback joblib
    if MODEL_PATH.exists():
        import joblib
        _session = joblib.load(MODEL_PATH)
        _use_onnx = False
        return

    raise FileNotFoundError(
        f"Modèle introuvable : {ONNX_PATH} ni {MODEL_PATH}. "
        "Exécutez d'abord python retrain_model.py puis "
        "python optimization/onnx_optimization.py"
    )


def is_model_loaded() -> bool:
    """Retourne True si le modèle est chargé en mémoire."""
    return _session is not None


def get_model_metadata() -> Optional[Dict[str, Any]]:
    """Retourne les métadonnées du modèle déployé."""
    if not _meta:
        if MODEL_META_PATH.exists():
            return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        return None
    return _meta


def _compute_derived_features(data: dict) -> np.ndarray:
    """
    Je calcule les 5 features dérivées à partir des 8 features de base.
    """
    pI          = data["pI"]
    gravy_norm  = data["gravy_norm"]
    log_instab  = data["log_instability"]
    pct_helix   = data["pct_helix"]
    pct_turn    = data["pct_turn"]
    pct_sheet   = data["pct_sheet"]

    pI_distance       = abs(pI - 7.0)
    pct_coil          = max(0.0, 1.0 - pct_helix - pct_turn - pct_sheet)
    helix_x_gravy     = pct_helix * gravy_norm
    stability_score   = log_instab * gravy_norm
    helix_sheet_ratio = pct_helix / (pct_sheet + 1e-6)

    return np.array([
        data["pI"], data["log_mw"], gravy_norm, log_instab,
        data["aromaticity"], pct_helix, pct_turn, pct_sheet,
        pI_distance, pct_coil, helix_x_gravy,
        stability_score, helix_sheet_ratio,
    ], dtype=np.float32).reshape(1, -1)


def predict(protein_input) -> Dict[str, Any]:
    """
    Je réalise la prédiction de solubilité pour une protéine.
    Utilise ONNX Runtime si disponible, sinon joblib.
    """
    global _session

    if _session is None:
        load_model()

    # Construction du vecteur de features
    if hasattr(protein_input, "model_dump"):
        data = protein_input.model_dump()
    else:
        data = protein_input.dict()

    X = _compute_derived_features(data)

    # Inférence
    if _use_onnx:
        import onnxruntime as rt
        input_name = _session.get_inputs()[0].name
        results = _session.run(None, {input_name: X})
        proba_dict     = results[1][0]
        prob_insoluble = float(proba_dict[0])
        prob_soluble   = float(proba_dict[1])
    else:
        proba          = _session.predict_proba(X)[0]
        prob_insoluble = float(proba[0])
        prob_soluble   = float(proba[1])

    # Application du seuil optimal
    prediction = 1 if prob_soluble >= _threshold else 0

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
        "soluble":               prediction,
        "probability_soluble":   round(prob_soluble,   4),
        "probability_insoluble": round(prob_insoluble, 4),
        "confidence":            confidence,
        "recommendation":        recommendation,
    }
