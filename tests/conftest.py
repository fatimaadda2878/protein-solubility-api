"""
conftest.py — Configuration pytest pour P7 Protein Solubility
==============================================================
Fournit des fixtures partagées et configure le chemin du modèle de test.

Le modèle de test est un DummyClassifier dans tests/artifacts/
pour éviter de dépendre du modèle de production dans model/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
TESTS_DIR = ROOT_DIR / "tests"
ARTIFACTS_DIR = TESTS_DIR / "artifacts"

# Force pytest à utiliser le package app/ (pas un éventuel app.py)
# Cette ligne garantit que sys.path pointe bien vers la racine du projet
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def use_test_model(monkeypatch, tmp_path):
    """
    Redirige MODEL_PATH vers le DummyClassifier de test.
    Évite FileNotFoundError sur model/lgbm_model.joblib en CI.
    """
    dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
    if not dummy_path.exists():
        _create_dummy_model(dummy_path)

    monkeypatch.setenv("MODEL_PATH", str(dummy_path))
    monkeypatch.setenv("ONNX_PATH", str(ARTIFACTS_DIR / "model_dummy.onnx"))
    yield


def _create_dummy_model(path: Path):
    """Crée un DummyClassifier léger pour les tests."""
    import joblib
    from sklearn.dummy import DummyClassifier

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((10, 13)), [0] * 5 + [1] * 5)
    dummy.n_features_in_ = 13
    joblib.dump(dummy, path)
    print(f"[conftest] DummyClassifier créé : {path}")


@pytest.fixture
def sample_protein_input():
    """Données d'entrée valides pour un appel /predict."""
    return {
        "pI": 7.2,
        "log_mw": 11.5,
        "gravy_norm": -0.3,
        "log_instability": 3.8,
        "aromaticity": 0.065,
        "pct_helix": 0.42,
        "pct_turn": 0.12,
        "pct_sheet": 0.25,
    }


@pytest.fixture
def sample_fasta():
    """Séquence FASTA minimale pour tests /predict/sequence."""
    return ">test_protein\nMKLLPTINYGFLKGGGTSTKSATLRRRLWFKKNNKQLYNSTLSEINHSRFLSS"
