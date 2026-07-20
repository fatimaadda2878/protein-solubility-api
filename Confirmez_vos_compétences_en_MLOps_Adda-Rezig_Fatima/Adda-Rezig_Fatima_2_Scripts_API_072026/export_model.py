"""
Script d'export du modèle depuis MLflow vers joblib
A exécuter une fois pour préparer le modèle pour l'API.

Usage :
    python app/export_model.py
"""

import mlflow
import joblib
import os
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
MLFLOW_DB   = r"C:\Users\adda-\mlflow.db"
EXPERIMENT  = "protein-solubility-ecoli"
MODEL_DIR   = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

def export_best_model():
    """J'exporte le meilleur modèle MLflow vers un fichier joblib."""

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT}' introuvable dans MLflow")

    # Je cherche le meilleur run (AUC le plus élevé)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.test_auc > 0",
        order_by=["metrics.test_auc DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("Aucun run avec test_auc trouvé")

    best_run = runs[0]
    print(f"Meilleur run : {best_run.info.run_id}")
    print(f"AUC : {best_run.data.metrics.get('test_auc', 'N/A')}")

    # Chargement du modèle depuis MLflow
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)

    # Sauvegarde en joblib
    joblib.dump(model, MODEL_DIR / "lgbm_model.joblib")
    print(f"Modèle sauvegardé : {MODEL_DIR / 'lgbm_model.joblib'}")

    # Sauvegarde des métadonnées
    meta = {
        "run_id":   best_run.info.run_id,
        "auc":      best_run.data.metrics.get("test_auc"),
        "threshold":best_run.data.metrics.get("seuil_decision", 0.05),
    }
    import json
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Métadonnées sauvegardées : {MODEL_DIR / 'model_meta.json'}")

if __name__ == "__main__":
    export_best_model()
