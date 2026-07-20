"""
API FastAPI de prédiction de la solubilité des protéines recombinantes.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import (
    get_model_metadata,
    is_model_loaded,
    load_model,
    predict,
)
from app.schemas import PredictionOutput, ProteinInput


API_VERSION = "1.0.0"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Charge le modèle au démarrage sans empêcher l'API de démarrer."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Chargement du modèle LightGBM...")
        load_model()
        logger.info("Modèle chargé avec succès.")
    except Exception as exc:
        logger.exception(
            "Le modèle n'a pas pu être chargé au démarrage : %s",
            exc,
        )

    yield


app = FastAPI(
    title="Protein Solubility Prediction API",
    description=(
        "Prédiction de la solubilité des protéines recombinantes "
        "lors de leur expression dans E. coli."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utc_timestamp() -> str:
    """Retourne un horodatage UTC au format ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _input_to_dict(protein: ProteinInput) -> Dict[str, Any]:
    """Compatibilité Pydantic v1 et v2."""
    if hasattr(protein, "model_dump"):
        return protein.model_dump()
    return protein.dict()


def _append_prediction_log(entry: Dict[str, Any]) -> None:
    """Ajoute une prédiction au fichier JSON Lines."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )


@app.get("/", tags=["Health"])
def root() -> Dict[str, str]:
    """Informations générales sur l'API."""
    return {
        "message": "Protein Solubility Prediction API",
        "version": API_VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Vérifie que l'API peut accéder au modèle."""
    try:
        if not is_model_loaded():
            load_model()

        return {
            "status": "healthy",
            "model_loaded": True,
            "timestamp": _utc_timestamp(),
            "version": API_VERSION,
            "model_metadata": get_model_metadata(),
        }

    except Exception as exc:
        logger.exception("Échec du contrôle de santé : %s", exc)

        return {
            "status": "unhealthy",
            "model_loaded": False,
            "timestamp": _utc_timestamp(),
            "version": API_VERSION,
            "error": str(exc),
        }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
)
def predict_solubility(protein: ProteinInput) -> PredictionOutput:
    """Prédit la probabilité de solubilité d'une protéine."""
    start_time = time.perf_counter()

    try:
        result = predict(protein)

        inference_time = max(
            time.perf_counter() - start_time,
            1e-6,
        )

        log_entry = {
            "timestamp": _utc_timestamp(),
            "input": _input_to_dict(protein),
            "prediction": result["soluble"],
            "probability": result["probability_soluble"],
            "inference_time_s": inference_time,
            "data_source": "real_api_request",
        }

        try:
            _append_prediction_log(log_entry)
        except OSError as log_error:
            logger.warning(
                "La prédiction a réussi, mais son journal n'a pas "
                "pu être écrit : %s",
                log_error,
            )

        logger.info(
            "Prediction=%s | P_soluble=%.4f | temps=%.6fs",
            result["soluble"],
            result["probability_soluble"],
            inference_time,
        )

        return PredictionOutput(
            soluble=result["soluble"],
            probability_soluble=result["probability_soluble"],
            probability_insoluble=result["probability_insoluble"],
            confidence=result["confidence"],
            inference_time_s=round(inference_time, 6),
            recommendation=result["recommendation"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur de prédiction : %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la prédiction.",
        ) from exc


@app.get("/logs/summary", tags=["Monitoring"])
def logs_summary() -> Dict[str, Any]:
    """Retourne un résumé des véritables requêtes enregistrées par l'API."""
    if not LOG_FILE.exists():
        return {
            "n_predictions": 0,
            "message": "Aucun log de prédiction disponible.",
            "data_source": "real_api_logs",
        }

    entries = []

    try:
        with LOG_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Une ligne de log invalide a été ignorée."
                    )
                    continue

                if isinstance(entry, dict):
                    entries.append(entry)

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de lire les logs : {exc}",
        ) from exc

    n_predictions = len(entries)

    if n_predictions == 0:
        return {
            "n_predictions": 0,
            "message": "Aucun log de prédiction valide.",
            "data_source": "real_api_logs",
        }

    n_soluble = sum(
        1
        for entry in entries
        if entry.get("prediction") == 1
    )

    probabilities = [
        float(entry["probability"])
        for entry in entries
        if isinstance(entry.get("probability"), (int, float))
    ]

    latencies = [
        float(entry["inference_time_s"])
        for entry in entries
        if isinstance(entry.get("inference_time_s"), (int, float))
    ]

    average_probability = (
        sum(probabilities) / len(probabilities)
        if probabilities
        else 0.0
    )
    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    return {
        "n_predictions": n_predictions,
        "n_soluble": n_soluble,
        "n_insoluble": n_predictions - n_soluble,
        "pct_soluble": round(
            n_soluble / n_predictions * 100,
            1,
        ),
        "avg_probability": round(
            average_probability,
            4,
        ),
        "avg_latency_s": round(
            average_latency,
            6,
        ),
        "data_source": "real_api_logs",
    }
