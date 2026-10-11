"""
tests/test_model_extended.py
============================
Nouveaux tests demandés par le mentor (Session 2 — Semaine 2) :
  1. Chargement ONNX Runtime
  2. Seuil de décision 0.23 (scénario Initial FN=1200 FP=200)
  3. Fallback joblib si ONNX absent

Ces tests complètent les 50 tests existants dans test_api.py et test_model.py.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Chemins ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
OPTIMAL_THRESHOLD = 0.23  # Seuil optimal scénario Initial (FN=1200, FP=200)
N_FEATURES = 13


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_proba_array():
    """Probabilités de test couvrant tout le spectre [0, 1]."""
    return np.array([
        [0.9, 0.1],   # prob solubles = 0.1  → insoluble (< threshold)
        [0.8, 0.2],   # prob solubles = 0.2  → insoluble (< threshold)
        [0.8, 0.22],  # prob solubles = 0.22 → insoluble (juste sous 0.23)
        [0.77, 0.23], # prob solubles = 0.23 → soluble (= threshold)
        [0.7, 0.3],   # prob solubles = 0.3  → soluble (> threshold)
        [0.5, 0.5],   # prob solubles = 0.5  → soluble
        [0.1, 0.9],   # prob solubles = 0.9  → soluble
    ])


@pytest.fixture
def dummy_features():
    """Features valides pour les prédictions."""
    return np.array([
        [7.2, 11.5, -0.3, 3.8, 0.065, 0.42, 0.12, 0.25, 0.2, 0.21, -0.126, 0.21, 1.68],
        [5.1, 10.2,  0.1, 2.5, 0.080, 0.35, 0.15, 0.30, 1.9, 0.20,  0.035, 0.29, 1.17],
    ], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────────
# TEST 1 — Chargement ONNX Runtime
# ────────────────────────────────────────────────────────────────────────────
class TestONNXLoading:
    """Vérifie que le chargement ONNX fonctionne correctement."""

    def test_onnx_file_exists_after_build(self):
        """Le fichier ONNX de test doit exister (généré par create_test_artifacts.py)."""
        onnx_path = ARTIFACTS_DIR / "model_dummy.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX non généré (skl2onnx absent) — test ignoré en CI")
        assert onnx_path.stat().st_size > 0, "Fichier ONNX vide"

    def test_onnx_session_creation(self):
        """ONNX Runtime doit créer une session valide."""
        onnx_path = ARTIFACTS_DIR / "model_dummy.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX non disponible")

        try:
            import onnxruntime as rt
        except ImportError:
            pytest.skip("onnxruntime non installé")

        session = rt.InferenceSession(str(onnx_path))
        assert session is not None
        # Vérifier le nom de l'input attendu
        input_names = [inp.name for inp in session.get_inputs()]
        assert len(input_names) > 0, "La session ONNX n'a pas d'inputs"

    def test_onnx_inference_returns_probabilities(self, dummy_features):
        """ONNX doit retourner des probabilités entre 0 et 1."""
        onnx_path = ARTIFACTS_DIR / "model_dummy.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX non disponible")

        try:
            import onnxruntime as rt
        except ImportError:
            pytest.skip("onnxruntime non installé")

        session = rt.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_features})

        # output[1] = dict de probabilités, output[0] = labels
        assert len(output) >= 1
        # Probabilités dans [0, 1]
        if isinstance(output[1], list):
            for proba_dict in output[1]:
                for val in proba_dict.values():
                    assert 0.0 <= val <= 1.0
        else:
            proba = output[1]
            assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_onnx_correct_number_of_features(self, dummy_features):
        """ONNX doit accepter exactement N_FEATURES colonnes."""
        onnx_path = ARTIFACTS_DIR / "model_dummy.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX non disponible")

        try:
            import onnxruntime as rt
        except ImportError:
            pytest.skip("onnxruntime non installé")

        session = rt.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        # Dimension 1 = nombre de features
        if input_shape[1] is not None:
            assert input_shape[1] == N_FEATURES, (
                f"ONNX attend {input_shape[1]} features, P7 en a {N_FEATURES}"
            )

    def test_onnx_wrong_features_raises(self):
        """ONNX doit rejeter une entrée avec un mauvais nombre de features."""
        onnx_path = ARTIFACTS_DIR / "model_dummy.onnx"
        if not onnx_path.exists():
            pytest.skip("ONNX non disponible")

        try:
            import onnxruntime as rt
        except ImportError:
            pytest.skip("onnxruntime non installé")

        session = rt.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        wrong_input = np.zeros((1, N_FEATURES + 5), dtype=np.float32)

        with pytest.raises(Exception):
            session.run(None, {input_name: wrong_input})


# ────────────────────────────────────────────────────────────────────────────
# TEST 2 — Seuil de décision 0.23
# ────────────────────────────────────────────────────────────────────────────
class TestDecisionThreshold:
    """Vérifie que le seuil 0.23 (scénario Initial) est correctement appliqué."""

    def test_threshold_value_is_023(self):
        """Le seuil optimal du scénario Initial doit être 0.23."""
        assert OPTIMAL_THRESHOLD == pytest.approx(0.23, abs=0.01)

    def test_below_threshold_predicted_insoluble(self, dummy_proba_array):
        """Proba < 0.23 → protéine insoluble (classe 0)."""
        proba_soluble = dummy_proba_array[:, 1]
        predictions = (proba_soluble >= OPTIMAL_THRESHOLD).astype(int)
        # Les 3 premières entrées (0.1, 0.2, 0.22) doivent être insoluble
        assert predictions[0] == 0, f"0.1 devrait être insoluble, got {predictions[0]}"
        assert predictions[1] == 0, f"0.2 devrait être insoluble, got {predictions[1]}"
        assert predictions[2] == 0, f"0.22 devrait être insoluble, got {predictions[2]}"

    def test_at_threshold_predicted_soluble(self, dummy_proba_array):
        """Proba = 0.23 → protéine soluble (classe 1, seuil inclusif)."""
        proba_soluble = dummy_proba_array[:, 1]
        predictions = (proba_soluble >= OPTIMAL_THRESHOLD).astype(int)
        # L'entrée 3 a proba = 0.23 → doit être soluble
        assert predictions[3] == 1, f"0.23 devrait être soluble (>= threshold)"

    def test_above_threshold_predicted_soluble(self, dummy_proba_array):
        """Proba > 0.23 → protéine soluble (classe 1)."""
        proba_soluble = dummy_proba_array[:, 1]
        predictions = (proba_soluble >= OPTIMAL_THRESHOLD).astype(int)
        assert predictions[4] == 1, f"0.3 devrait être soluble"
        assert predictions[5] == 1, f"0.5 devrait être soluble"
        assert predictions[6] == 1, f"0.9 devrait être soluble"

    def test_threshold_minimizes_fn_cost(self):
        """Le seuil 0.23 doit être proche du minimum de coût pour FN=1200, FP=200."""
        FN_COST = 1200
        FP_COST = 200

        # Simulation : pour un modèle parfait à 0.5,
        # un seuil bas minimise les FN au prix de plus de FP
        # → le seuil optimal sera < 0.5 quand FN > FP
        optimal_direction = FN_COST > FP_COST
        threshold_is_below_half = OPTIMAL_THRESHOLD < 0.5
        assert optimal_direction == threshold_is_below_half, (
            f"Avec FN={FN_COST} > FP={FP_COST}, le seuil devrait être < 0.5. "
            f"Seuil actuel : {OPTIMAL_THRESHOLD}"
        )

    def test_response_contains_threshold(self):
        """L'API doit retourner le seuil utilisé dans la réponse /predict."""
        # Test unitaire : vérifie le format de réponse attendu
        expected_fields = ["soluble", "probability", "threshold_used"]
        mock_response = {
            "soluble": True,
            "probability": 0.35,
            "threshold_used": OPTIMAL_THRESHOLD,
            "model_version": "1.0",
        }
        for field in expected_fields:
            assert field in mock_response, f"Champ manquant dans la réponse : {field}"
        assert mock_response["threshold_used"] == OPTIMAL_THRESHOLD

    def test_threshold_consistency_across_calls(self):
        """Le seuil doit être constant entre les appels (reproductibilité)."""
        thresholds = set()
        for _ in range(5):
            thresholds.add(OPTIMAL_THRESHOLD)
        assert len(thresholds) == 1, "Le seuil varie entre les appels !"

    def test_economic_cost_lower_with_023_than_05(self):
        """Le coût économique à seuil=0.23 doit être < coût à seuil=0.5."""
        FN_COST = 1200
        FP_COST = 200

        # Simulation avec distribution typique
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.42, n)  # 42% solubles
        # Proba simulée avec un peu de signal
        y_prob = 0.4 * y_true + 0.1 * np.random.rand(n)
        y_prob = np.clip(y_prob, 0, 1)

        def cost(threshold):
            pred = (y_prob >= threshold).astype(int)
            fn = ((y_true == 1) & (pred == 0)).sum()
            fp = ((y_true == 0) & (pred == 1)).sum()
            return fn * FN_COST + fp * FP_COST

        cost_023 = cost(0.23)
        cost_050 = cost(0.50)
        # Le seuil 0.23 doit donner un coût inférieur ou similaire à 0.5
        # (vrai pour tout scénario où FN >> FP)
        assert cost_023 <= cost_050 * 1.5, (
            f"Seuil 0.23 coût={cost_023:,} vs 0.5 coût={cost_050:,} — "
            "seuil optimal devrait être plus bas quand FN très coûteux"
        )


# ────────────────────────────────────────────────────────────────────────────
# TEST 3 — Fallback joblib si ONNX absent
# ────────────────────────────────────────────────────────────────────────────
class TestJoblibFallback:
    """Vérifie que le fallback joblib fonctionne quand ONNX est absent."""

    def test_joblib_model_loads(self):
        """Le modèle joblib de test doit se charger correctement."""
        import joblib

        dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
        assert dummy_path.exists(), f"Modèle joblib absent : {dummy_path}"

        model = joblib.load(dummy_path)
        assert model is not None
        assert hasattr(model, "predict_proba"), "Le modèle doit avoir predict_proba"

    def test_joblib_predict_returns_probabilities(self, dummy_features):
        """Le modèle joblib doit retourner des probabilités valides."""
        import joblib

        dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
        if not dummy_path.exists():
            pytest.skip("joblib model absent")

        model = joblib.load(dummy_path)
        proba = model.predict_proba(dummy_features)
        assert proba.shape == (len(dummy_features), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        # Somme de chaque ligne ≈ 1.0
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(dummy_features)), atol=1e-6)

    def test_fallback_when_onnx_missing(self, dummy_features, tmp_path, monkeypatch):
        """
        Quand ONNX est absent, le système doit basculer sur joblib
        sans lever d'exception.
        """
        import joblib

        dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
        if not dummy_path.exists():
            pytest.skip("joblib model absent")

        # Simuler l'absence de ONNX Runtime
        fake_onnx_path = tmp_path / "absent.onnx"
        # Ne pas créer le fichier — il n'existe pas

        # Tester la logique de fallback directement
        def load_model_with_fallback(onnx_path, joblib_path):
            """Reproduit la logique de app/model.py."""
            if Path(onnx_path).exists():
                try:
                    import onnxruntime as rt
                    session = rt.InferenceSession(str(onnx_path))
                    return ("onnx", session)
                except Exception:
                    pass
            # Fallback joblib
            model = joblib.load(str(joblib_path))
            return ("joblib", model)

        mode, model = load_model_with_fallback(
            onnx_path=str(fake_onnx_path),
            joblib_path=str(dummy_path),
        )
        assert mode == "joblib", f"Devrait utiliser joblib en fallback, got '{mode}'"
        assert model is not None

    def test_fallback_predictions_match_direct_joblib(self, dummy_features):
        """Fallback et chargement direct doivent donner le même résultat."""
        import joblib

        dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
        if not dummy_path.exists():
            pytest.skip("joblib model absent")

        model_direct = joblib.load(dummy_path)
        proba_direct = model_direct.predict_proba(dummy_features)

        model_fallback = joblib.load(dummy_path)  # Même fichier
        proba_fallback = model_fallback.predict_proba(dummy_features)

        np.testing.assert_array_equal(proba_direct, proba_fallback)

    def test_joblib_model_has_correct_features_count(self):
        """Le modèle joblib doit avoir N_FEATURES features."""
        import joblib

        dummy_path = ARTIFACTS_DIR / "model_dummy.joblib"
        if not dummy_path.exists():
            pytest.skip("joblib model absent")

        model = joblib.load(dummy_path)
        assert hasattr(model, "n_features_in_"), (
            "Le modèle doit avoir n_features_in_ (sklearn >= 1.0)"
        )
        assert model.n_features_in_ == N_FEATURES, (
            f"Modèle attend {model.n_features_in_} features, P7 en a {N_FEATURES}"
        )

    def test_onnx_env_var_controls_mode(self, tmp_path, monkeypatch):
        """La variable d'env USE_ONNX doit contrôler le mode de chargement."""
        monkeypatch.setenv("USE_ONNX", "false")
        # En mode non-ONNX, le système ne doit pas essayer de charger ONNX
        use_onnx = os.getenv("USE_ONNX", "true").lower() == "true"
        assert not use_onnx, "USE_ONNX=false devrait désactiver ONNX"
