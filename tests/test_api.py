"""
Tests unitaires — API FastAPI Protein Solubility
Tests des endpoints et de la validation des données d'entrée.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# ── Fixture : client de test ──────────────────────────────────
@pytest.fixture
def client():
    """Je crée un client de test avec le modèle mocké."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    with patch("app.model._model", mock_model), \
         patch("app.model._scaler", None), \
         patch("app.main.load_model", return_value=mock_model):
        from app.main import app
        return TestClient(app)


# ── Données valides de référence ──────────────────────────────
VALID_PROTEIN = {
    "pI": 6.2,
    "log_mw": 10.5,
    "gravy_norm": 0.21,
    "log_instability": 0.5,
    "aromaticity": 0.08,
    "pct_helix": 0.38,
    "pct_turn": 0.30,
    "pct_sheet": 0.18,
}


# ── Tests des endpoints de santé ─────────────────────────────
class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        """Je vérifie que l'endpoint racine répond correctement."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "online"

    def test_health_returns_healthy(self, client):
        """Je vérifie que l'endpoint /health retourne healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True


# ── Tests de prédiction valide ────────────────────────────────
class TestPredictionValid:

    def test_valid_input_returns_200(self, client):
        """Je vérifie qu'une entrée valide retourne 200."""
        response = client.post("/predict", json=VALID_PROTEIN)
        assert response.status_code == 200

    def test_valid_input_returns_prediction(self, client):
        """Je vérifie que la réponse contient tous les champs attendus."""
        response = client.post("/predict", json=VALID_PROTEIN)
        data = response.json()
        assert "soluble" in data
        assert "probability_soluble" in data
        assert "probability_insoluble" in data
        assert "confidence" in data
        assert "inference_time_s" in data
        assert "recommendation" in data

    def test_prediction_is_binary(self, client):
        """Je vérifie que la prédiction est 0 ou 1."""
        response = client.post("/predict", json=VALID_PROTEIN)
        assert response.json()["soluble"] in [0, 1]

    def test_probabilities_sum_to_one(self, client):
        """Je vérifie que les probabilités somment à 1."""
        response = client.post("/predict", json=VALID_PROTEIN)
        data = response.json()
        total = data["probability_soluble"] + data["probability_insoluble"]
        assert abs(total - 1.0) < 0.01

    def test_inference_time_is_positive(self, client):
        """Je vérifie que le temps d'inférence est positif."""
        response = client.post("/predict", json=VALID_PROTEIN)
        assert response.json()["inference_time_s"] > 0


# ── Tests de validation des entrées (cas limites) ────────────
class TestInputValidation:

    def test_missing_field_returns_422(self, client):
        """Je vérifie qu'un champ manquant retourne 422."""
        incomplete = {k: v for k, v in VALID_PROTEIN.items() if k != "pI"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_pi_too_low_returns_422(self, client):
        """Je vérifie qu'un pI trop bas (< 2.5) est rejeté."""
        bad_input = {**VALID_PROTEIN, "pI": -1.0}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_pi_too_high_returns_422(self, client):
        """Je vérifie qu'un pI trop élevé (> 12.0) est rejeté."""
        bad_input = {**VALID_PROTEIN, "pI": 15.0}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_gravy_norm_out_of_range_returns_422(self, client):
        """Je vérifie qu'un GRAVY normalisé hors plage est rejeté."""
        bad_input = {**VALID_PROTEIN, "gravy_norm": 2.5}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_negative_aromaticity_returns_422(self, client):
        """Je vérifie qu'une aromaticité négative est rejetée."""
        bad_input = {**VALID_PROTEIN, "aromaticity": -0.1}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_string_instead_of_float_returns_422(self, client):
        """Je vérifie qu'un type incorrect (string) est rejeté."""
        bad_input = {**VALID_PROTEIN, "pI": "not_a_number"}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        """Je vérifie qu'un corps vide est rejeté."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_pct_helix_above_one_returns_422(self, client):
        """Je vérifie qu'une fraction > 1.0 est rejetée."""
        bad_input = {**VALID_PROTEIN, "pct_helix": 1.5}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422

    def test_log_mw_below_range_returns_422(self, client):
        """Je vérifie qu'un log_mw trop bas est rejeté."""
        bad_input = {**VALID_PROTEIN, "log_mw": 5.0}
        response = client.post("/predict", json=bad_input)
        assert response.status_code == 422


# ── Tests des cas limites valides ────────────────────────────
class TestEdgeCases:

    def test_minimum_valid_values(self, client):
        """Je teste avec les valeurs minimales valides."""
        min_input = {
            "pI": 2.5,
            "log_mw": 7.0,
            "gravy_norm": 0.0,
            "log_instability": -4.0,
            "aromaticity": 0.0,
            "pct_helix": 0.0,
            "pct_turn": 0.0,
            "pct_sheet": 0.0,
        }
        response = client.post("/predict", json=min_input)
        assert response.status_code == 200

    def test_maximum_valid_values(self, client):
        """Je teste avec les valeurs maximales valides."""
        max_input = {
            "pI": 12.0,
            "log_mw": 13.0,
            "gravy_norm": 1.0,
            "log_instability": 2.0,
            "aromaticity": 0.3,
            "pct_helix": 0.33,
            "pct_turn": 0.33,
            "pct_sheet": 0.33,
        }
        response = client.post("/predict", json=max_input)
        assert response.status_code == 200
