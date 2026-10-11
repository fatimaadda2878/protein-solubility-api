"""
Créer les artifacts de test pour CI/CD.
À exécuter une fois, puis committer les fichiers générés.

    python tests/create_test_artifacts.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES = 13  # Nombre de features P7

# ── 1. DummyClassifier joblib ─────────────────────────────────────────────
dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
dummy = DummyClassifier(strategy="prior")
X_dummy = np.random.rand(100, N_FEATURES)
y_dummy = np.array([0] * 60 + [1] * 40)
dummy.fit(X_dummy, y_dummy)
dummy.n_features_in_ = N_FEATURES
joblib.dump(dummy, dummy_path)
print(f"[OK] {dummy_path}")

# ── 2. ONNX du DummyClassifier ────────────────────────────────────────────
onnx_path = ARTIFACTS_DIR / "model_dummy.onnx"
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(dummy, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[OK] {onnx_path}")
except ImportError:
    print(f"[SKIP] skl2onnx non installé — ONNX non généré")
except Exception as e:
    print(f"[WARN] ONNX génération échouée : {e}")

# ── 3. model_meta.json factice ────────────────────────────────────────────
import json

meta_path = ARTIFACTS_DIR / "model_meta.json"
meta = {
    "model_type": "DummyClassifier",
    "n_features": N_FEATURES,
    "optimal_threshold": 0.23,
    "auc_roc": 0.5,
    "cost_fn": 1200,
    "cost_fp": 200,
    "features": [
        "pI", "log_mw", "gravy_norm", "log_instability",
        "aromaticity", "pct_helix", "pct_turn", "pct_sheet",
        "pI_distance", "pct_coil", "helix_x_gravy",
        "stability_score", "helix_sheet_ratio",
    ],
    "_note": "Dummy model for CI/CD testing — not for production",
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[OK] {meta_path}")

print("\n✓ Artifacts de test créés dans", ARTIFACTS_DIR)
print("  → git add tests/artifacts/ && git commit -m 'test: add CI test artifacts'")
