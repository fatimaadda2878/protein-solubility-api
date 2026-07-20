"""
API FastAPI — Prédiction de la Solubilité des Protéines Recombinantes
Entreprise : Prêt à Dépenser (adapté pour protéines)
Auteur : Fatima Adda-Rezig
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ProteinInput, PredictionOutput
from app.model import load_model, predict
import time
import logging
import json
import os
from datetime import datetime
from pathlib import Path

# ── Logging structuré (JSON) ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Dossier logs ─────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Initialisation de l'application ──────────────────────────
app = FastAPI(
    title="Protein Solubility Prediction API",
    description=(
        "API de prédiction de la solubilité des protéines recombinantes "
        "lors de l'expression dans E. coli. "
        "Retourne la probabilité qu'une protéine soit soluble (1) "
        "ou forme des corps d'inclusion (0)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement du modèle au démarrage (une seule fois) ────────
@app.on_event("startup")
async def startup_event():
    """Je charge le modèle une seule fois au démarrage de l'API."""
    logger.info("Chargement du modèle LightGBM...")
    load_model()
    logger.info("Modèle chargé avec succès.")


# ── Endpoints ────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Endpoint racine — vérification que l'API est en ligne."""
    return {
        "message": "Protein Solubility Prediction API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint de santé — vérifie que le modèle est chargé."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_solubility(protein: ProteinInput):
    """
    Prédit la probabilité de solubilité d'une protéine recombinante.

    - **pI** : Point isoélectrique (2.5 – 12.0)
    - **log_mw** : log(Masse moléculaire en Da) (7.0 – 13.0)
    - **gravy_norm** : Score GRAVY normalisé (0.0 – 1.0)
    - **log_instability** : log(Indice d'instabilité) (-4.0 – 2.0)
    - **aromaticity** : Aromaticité (0.0 – 0.3)
    - **pct_helix** : % hélice alpha (0.0 – 1.0)
    - **pct_turn** : % turn (0.0 – 1.0)
    - **pct_sheet** : % feuillet beta (0.0 – 1.0)
    """
    start_time = time.time()

    try:
        result = predict(protein)
        inference_time = round(time.time() - start_time, 4)

        # Log structuré JSON
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": protein.dict(),
            "prediction": result["soluble"],
            "probability": result["probability_soluble"],
            "inference_time_s": inference_time,
        }
        with open(LOG_DIR / "predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"Prediction: {result['soluble']} | P={result['probability_soluble']:.4f} | t={inference_time}s")

        return PredictionOutput(
            soluble=result["soluble"],
            probability_soluble=result["probability_soluble"],
            probability_insoluble=result["probability_insoluble"],
            confidence=result["confidence"],
            inference_time_s=inference_time,
            recommendation=result["recommendation"],
        )

    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary", tags=["Monitoring"])
def logs_summary():
    """Retourne un résumé des prédictions loggées."""
    log_file = LOG_DIR / "predictions.jsonl"
    if not log_file.exists():
        return {"message": "Aucun log disponible", "n_predictions": 0}

    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line))

    n = len(entries)
    if n == 0:
        return {"n_predictions": 0}

    n_soluble   = sum(1 for e in entries if e["prediction"] == 1)
    avg_proba   = sum(e["probability"] for e in entries) / n
    avg_latency = sum(e["inference_time_s"] for e in entries) / n

    return {
        "n_predictions":   n,
        "n_soluble":       n_soluble,
        "n_insoluble":     n - n_soluble,
        "pct_soluble":     round(n_soluble / n * 100, 1),
        "avg_probability": round(avg_proba, 4),
        "avg_latency_s":   round(avg_latency, 4),
    }
