"""
Tests unitaires — API FastAPI Protein Solubility (version enrichie)
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    with patch("app.model._model", mock_model), \
         patch("app.model._scaler", None), \
         patch("app.main.load_model", return_value=mock_model):
        from app.main import app
        return TestClient(app)


VALID_PROTEIN = {
    "pI": 6.2, "log_mw": 10.5, "gravy_norm": 0.21,
    "log_instability": 0.5, "aromaticity": 0.08,
    "pct_helix": 0.38, "pct_turn": 0.30, "pct_sheet": 0.18,
}


class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_contains_version(self, client):
        assert client.get("/").json()["version"] == "1.0.0"

    def test_root_contains_docs_link(self, client):
        assert "docs" in client.get("/").json()

    def test_health_returns_healthy(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health_contains_timestamp(self, client):
        assert "timestamp" in client.get("/health").json()


class TestPredictionValid:
    def test_valid_input_returns_200(self, client):
        assert client.post("/predict", json=VALID_PROTEIN).status_code == 200

    def test_all_fields_present(self, client):
        data = client.post("/predict", json=VALID_PROTEIN).json()
        for field in ["soluble","probability_soluble","probability_insoluble",
                      "confidence","inference_time_s","recommendation"]:
            assert field in data

    def test_prediction_is_binary(self, client):
        assert client.post("/predict", json=VALID_PROTEIN).json()["soluble"] in [0, 1]

    def test_probabilities_sum_to_one(self, client):
        data = client.post("/predict", json=VALID_PROTEIN).json()
        assert abs(data["probability_soluble"] + data["probability_insoluble"] - 1.0) < 0.01

    def test_inference_time_positive(self, client):
        assert client.post("/predict", json=VALID_PROTEIN).json()["inference_time_s"] > 0

    def test_probability_between_0_and_1(self, client):
        prob = client.post("/predict", json=VALID_PROTEIN).json()["probability_soluble"]
        assert 0.0 <= prob <= 1.0

    def test_confidence_valid_value(self, client):
        conf = client.post("/predict", json=VALID_PROTEIN).json()["confidence"]
        assert conf in ["Élevé", "Modéré", "Faible"]

    def test_recommendation_non_empty(self, client):
        rec = client.post("/predict", json=VALID_PROTEIN).json()["recommendation"]
        assert isinstance(rec, str) and len(rec) > 0


class TestMissingValues:
    def test_missing_pI(self, client):
        data = {k: v for k, v in VALID_PROTEIN.items() if k != "pI"}
        assert client.post("/predict", json=data).status_code == 422

    def test_missing_log_mw(self, client):
        data = {k: v for k, v in VALID_PROTEIN.items() if k != "log_mw"}
        assert client.post("/predict", json=data).status_code == 422

    def test_missing_gravy_norm(self, client):
        data = {k: v for k, v in VALID_PROTEIN.items() if k != "gravy_norm"}
        assert client.post("/predict", json=data).status_code == 422

    def test_missing_aromaticity(self, client):
        data = {k: v for k, v in VALID_PROTEIN.items() if k != "aromaticity"}
        assert client.post("/predict", json=data).status_code == 422

    def test_missing_pct_helix(self, client):
        data = {k: v for k, v in VALID_PROTEIN.items() if k != "pct_helix"}
        assert client.post("/predict", json=data).status_code == 422

    def test_empty_body(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_null_pI(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": None}).status_code == 422


class TestAberrantValues:
    def test_pi_too_low(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": -1.0}).status_code == 422

    def test_pi_too_high(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": 15.0}).status_code == 422

    def test_pi_zero(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": 0.0}).status_code == 422

    def test_gravy_above_one(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "gravy_norm": 2.5}).status_code == 422

    def test_gravy_negative(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "gravy_norm": -0.1}).status_code == 422

    def test_aromaticity_negative(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "aromaticity": -0.1}).status_code == 422

    def test_aromaticity_too_high(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "aromaticity": 0.5}).status_code == 422

    def test_pct_helix_above_one(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pct_helix": 1.5}).status_code == 422

    def test_pct_sheet_negative(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pct_sheet": -0.1}).status_code == 422

    def test_log_mw_below_range(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "log_mw": 5.0}).status_code == 422

    def test_log_mw_above_range(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "log_mw": 15.0}).status_code == 422

    def test_log_instability_below_range(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "log_instability": -5.0}).status_code == 422

    def test_log_instability_above_range(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "log_instability": 3.0}).status_code == 422


class TestIncorrectTypes:
    def test_string_pI(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": "six"}).status_code == 422

    def test_string_log_mw(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "log_mw": "dix"}).status_code == 422

    def test_list_as_pI(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": [6.2]}).status_code == 422

    def test_dict_as_pI(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": {"val": 6.2}}).status_code == 422

    def test_false_as_pI_out_of_range(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": False}).status_code == 422


class TestEdgeCases:
    def test_minimum_valid_values(self, client):
        data = {"pI": 2.5, "log_mw": 7.0, "gravy_norm": 0.0,
                "log_instability": -4.0, "aromaticity": 0.0,
                "pct_helix": 0.0, "pct_turn": 0.0, "pct_sheet": 0.0}
        assert client.post("/predict", json=data).status_code == 200

    def test_maximum_valid_values(self, client):
        data = {"pI": 12.0, "log_mw": 13.0, "gravy_norm": 1.0,
                "log_instability": 2.0, "aromaticity": 0.3,
                "pct_helix": 0.33, "pct_turn": 0.33, "pct_sheet": 0.33}
        assert client.post("/predict", json=data).status_code == 200

    def test_pi_exactly_neutral(self, client):
        assert client.post("/predict", json={**VALID_PROTEIN, "pI": 7.0}).status_code == 200

    def test_all_secondary_structure_zero(self, client):
        data = {**VALID_PROTEIN, "pct_helix": 0.0, "pct_turn": 0.0, "pct_sheet": 0.0}
        assert client.post("/predict", json=data).status_code == 200


class TestLogsSummary:
    def test_logs_summary_returns_200(self, client):
        assert client.get("/logs/summary").status_code == 200

    def test_logs_summary_no_predictions(self, client):
        r = client.get("/logs/summary")
        data = r.json()
        assert "n_predictions" in data
