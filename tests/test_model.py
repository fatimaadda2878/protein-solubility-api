"""
Tests unitaires — Logique du modèle
Tests des fonctions de chargement et de prédiction.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestComputeFeatures:

    def test_derived_features_shape(self):
        """Je vérifie que les features dérivées ont la bonne dimension."""
        from app.model import _compute_derived_features
        data = {
            "pI": 6.2, "log_mw": 10.5, "gravy_norm": 0.21,
            "log_instability": 0.5, "aromaticity": 0.08,
            "pct_helix": 0.38, "pct_turn": 0.30, "pct_sheet": 0.18,
        }
        X = _compute_derived_features(data)
        assert X.shape == (1, 13)

    def test_pi_distance_is_positive(self):
        """Je vérifie que la distance au pI neutre est positive."""
        from app.model import _compute_derived_features
        data = {
            "pI": 4.0, "log_mw": 10.5, "gravy_norm": 0.21,
            "log_instability": 0.5, "aromaticity": 0.08,
            "pct_helix": 0.38, "pct_turn": 0.30, "pct_sheet": 0.18,
        }
        X = _compute_derived_features(data)
        pI_distance = X[0, 8]  # index de pI_distance
        assert pI_distance == abs(4.0 - 7.0)
        assert pI_distance > 0

    def test_pct_coil_is_non_negative(self):
        """Je vérifie que le % coil est toujours >= 0."""
        from app.model import _compute_derived_features
        data = {
            "pI": 6.2, "log_mw": 10.5, "gravy_norm": 0.21,
            "log_instability": 0.5, "aromaticity": 0.08,
            "pct_helix": 0.50, "pct_turn": 0.30, "pct_sheet": 0.20,
        }
        X = _compute_derived_features(data)
        pct_coil = X[0, 9]
        assert pct_coil >= 0


class TestPredict:

    def test_predict_returns_dict(self):
        """Je vérifie que predict retourne un dictionnaire."""
        from app.schemas import ProteinInput
        from app.model import predict

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        with patch("app.model._model", mock_model), \
             patch("app.model._scaler", None):
            protein = ProteinInput(
                pI=6.2, log_mw=10.5, gravy_norm=0.21,
                log_instability=0.5, aromaticity=0.08,
                pct_helix=0.38, pct_turn=0.30, pct_sheet=0.18
            )
            result = predict(protein)

        assert isinstance(result, dict)
        assert "soluble" in result
        assert "probability_soluble" in result
        assert "recommendation" in result

    def test_high_probability_gives_high_confidence(self):
        """Je vérifie que P > 0.80 donne confiance Élevée."""
        from app.schemas import ProteinInput
        from app.model import predict

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])

        with patch("app.model._model", mock_model), \
             patch("app.model._scaler", None):
            protein = ProteinInput(
                pI=6.2, log_mw=10.5, gravy_norm=0.21,
                log_instability=0.5, aromaticity=0.08,
                pct_helix=0.38, pct_turn=0.30, pct_sheet=0.18
            )
            result = predict(protein)

        assert result["confidence"] == "Élevé"

    def test_insoluble_prediction_gives_warning(self):
        """Je vérifie qu'une prédiction insoluble donne une recommandation adaptée."""
        from app.schemas import ProteinInput
        from app.model import predict

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

        with patch("app.model._model", mock_model), \
             patch("app.model._scaler", None):
            protein = ProteinInput(
                pI=9.8, log_mw=11.5, gravy_norm=0.80,
                log_instability=1.5, aromaticity=0.15,
                pct_helix=0.20, pct_turn=0.30, pct_sheet=0.30
            )
            result = predict(protein)

        assert result["soluble"] == 0
        assert "corps d'inclusion" in result["recommendation"].lower() or \
               "risque" in result["recommendation"].lower()
