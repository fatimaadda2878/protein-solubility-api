"""Crée un modèle factice réservé aux tests automatisés."""

from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

output_dir = Path("tests/artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
features = rng.random((100, 13))
labels = np.array([0, 1] * 50)

model = DummyClassifier(strategy="prior")
model.fit(features, labels)
joblib.dump(model, output_dir / "test_model.joblib")

metadata = {
    "model_version": "test-only",
    "threshold": 0.5,
    "features": [
        "pI", "log_mw", "gravy_norm", "log_instability", "aromaticity",
        "pct_helix", "pct_turn", "pct_sheet", "pI_distance", "pct_coil",
        "helix_x_gravy", "stability_score", "helix_sheet_ratio"
    ],
    "test_metrics": {"roc_auc": None},
}
with (output_dir / "test_model_meta.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)
