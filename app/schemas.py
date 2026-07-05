"""
Schémas Pydantic — validation des entrées/sorties de l'API
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class ProteinInput(BaseModel):
    """
    Données d'entrée pour la prédiction de solubilité.
    Toutes les features sont issues du dataset DeepSol (Khurana et al. 2018).
    """
    pI: float = Field(
        ...,
        ge=2.5, le=12.0,
        description="Point isoélectrique de la protéine",
        example=6.2
    )
    log_mw: float = Field(
        ...,
        ge=7.0, le=13.0,
        description="Logarithme de la masse moléculaire (Da)",
        example=10.5
    )
    gravy_norm: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Score GRAVY normalisé (hydrophobicité)",
        example=0.21
    )
    log_instability: float = Field(
        ...,
        ge=-4.0, le=2.0,
        description="Logarithme de l'indice d'instabilité",
        example=0.5
    )
    aromaticity: float = Field(
        ...,
        ge=0.0, le=0.3,
        description="Aromaticité (fraction Tyr/Trp/Phe)",
        example=0.08
    )
    pct_helix: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Fraction de résidus en hélice alpha",
        example=0.38
    )
    pct_turn: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Fraction de résidus en turn",
        example=0.44
    )
    pct_sheet: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Fraction de résidus en feuillet beta",
        example=0.18
    )

    @validator('pct_helix', 'pct_turn', 'pct_sheet')
    def check_structure_sum(cls, v, values):
        """Je vérifie que la somme des fractions de structure secondaire <= 1."""
        total = v
        for key in ['pct_helix', 'pct_turn', 'pct_sheet']:
            if key in values:
                total += values[key]
        if total > 1.05:
            raise ValueError(
                "La somme des fractions de structure secondaire "
                "(helix + turn + sheet) ne peut pas dépasser 1.0"
            )
        return v

    class Config:
        schema_extra = {
            "example": {
                "pI": 6.2,
                "log_mw": 10.5,
                "gravy_norm": 0.21,
                "log_instability": 0.5,
                "aromaticity": 0.08,
                "pct_helix": 0.38,
                "pct_turn": 0.44,
                "pct_sheet": 0.18
            }
        }


class PredictionOutput(BaseModel):
    """Résultat de la prédiction de solubilité."""
    soluble: int = Field(..., description="1 = soluble, 0 = corps d'inclusion")
    probability_soluble: float = Field(..., description="Probabilité de solubilité (0-1)")
    probability_insoluble: float = Field(..., description="Probabilité d'insolubilité (0-1)")
    confidence: str = Field(..., description="Niveau de confiance : Élevé / Modéré / Faible")
    inference_time_s: float = Field(..., description="Temps d'inférence en secondes")
    recommendation: str = Field(..., description="Recommandation pratique")

    class Config:
        schema_extra = {
            "example": {
                "soluble": 1,
                "probability_soluble": 0.823,
                "probability_insoluble": 0.177,
                "confidence": "Élevé",
                "inference_time_s": 0.012,
                "recommendation": "Protéine probablement soluble — expression standard recommandée."
            }
        }
