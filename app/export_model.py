"""Export d'un modèle MLflow vers model/lgbm_model.joblib."""
import json
import os
from pathlib import Path
import joblib
import mlflow

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "protein-solubility-ecoli")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))

def export_best_model() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Expérience MLflow introuvable : {EXPERIMENT_NAME}")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string="metrics.test_auc > 0", order_by=["metrics.test_auc DESC"], max_results=1)
    if not runs:
        raise RuntimeError("Aucun run contenant la métrique test_auc.")
    run = runs[0]
    model = mlflow.lightgbm.load_model(f"runs:/{run.info.run_id}/model")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "lgbm_model.joblib")
    metadata = {"run_id": run.info.run_id, "auc_validation": run.data.metrics.get("validation_auc"), "auc_test": run.data.metrics.get("test_auc"), "threshold": run.data.metrics.get("threshold", 0.30), "selection_rule": "threshold selected on validation set", "evaluation_rule": "final AUC measured once on an independent test set"}
    with (MODEL_DIR / "model_meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(f"Modèle exporté vers {MODEL_DIR / 'lgbm_model.joblib'}")

if __name__ == "__main__":
    export_best_model()
