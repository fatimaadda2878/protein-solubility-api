"""Schémas Pydantic de l'API."""

from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProteinInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pI": 6.2,
                "log_mw": 10.5,
                "gravy_norm": 0.21,
                "log_instability": 0.5,
                "aromaticity": 0.08,
                "pct_helix": 0.38,
                "pct_turn": 0.30,
                "pct_sheet": 0.18,
            }
        }
    )

    pI: float = Field(..., ge=2.5, le=12.0)
    log_mw: float = Field(..., ge=7.0, le=13.0)
    gravy_norm: float = Field(..., ge=0.0, le=1.0)
    log_instability: float = Field(..., ge=-4.0, le=2.0)
    aromaticity: float = Field(..., ge=0.0, le=0.3)
    pct_helix: float = Field(..., ge=0.0, le=1.0)
    pct_turn: float = Field(..., ge=0.0, le=1.0)
    pct_sheet: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_secondary_structure(self) -> Self:
        total = self.pct_helix + self.pct_turn + self.pct_sheet

        if total > 1.05:
            raise ValueError(
                "La somme pct_helix + pct_turn + pct_sheet "
                "doit être inférieure ou égale à 1.05."
            )

        return self


class PredictionOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "soluble": 1,
                "probability_soluble": 0.7183,
                "probability_insoluble": 0.2817,
                "confidence": "Modéré",
                "inference_time_s": 0.03355,
                "recommendation": (
                    "Protéine probablement soluble — "
                    "expression standard recommandée."
                ),
            }
        }
    )

    soluble: int = Field(..., ge=0, le=1)
    probability_soluble: float = Field(..., ge=0.0, le=1.0)
    probability_insoluble: float = Field(..., ge=0.0, le=1.0)
    confidence: str
    inference_time_s: float = Field(..., ge=0.0)
    recommendation: str