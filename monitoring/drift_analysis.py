"""Dashboard de monitoring avec distinction données réelles/simulées.

La méthode appliquée est un contrôle simple par z-score.
Ce fichier n'utilise pas Evidently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

FEATURES = [
    "pI", "log_mw", "gravy_norm", "log_instability",
    "aromaticity", "pct_helix", "pct_turn", "pct_sheet",
]
MIN_PREDICTIONS = 100
ZSCORE_THRESHOLD = 2.0


def load_production():
    path = Path("logs/predictions.jsonl")
    if not path.is_file():
        return pd.DataFrame(), "absentes"

    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                entry = json.loads(line)
                row = dict(entry["input"])
                row["prediction"] = entry["prediction"]
                row["probability"] = entry["probability"]
                row["inference_time_s"] = entry["inference_time_s"]
                rows.append(row)
            except (json.JSONDecodeError, KeyError):
                continue

    return pd.DataFrame(rows), "réelles"


def load_reference():
    path = Path("data/reference_data.csv")
    if path.is_file():
        return pd.read_csv(path), "réelles"

    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        feature: rng.normal(0.5, 0.1, 500)
        for feature in FEATURES
    })
    return frame, "simulées"


st.set_page_config(page_title="Monitoring Protein Solubility", layout="wide")
st.title("🧬 Monitoring — Protein Solubility Prediction API")

production, production_status = load_production()
reference, reference_status = load_reference()

fully_real = production_status == "réelles" and reference_status == "réelles"

if fully_real:
    st.success("MODE : DONNÉES RÉELLES")
else:
    st.warning(
        f"MODE NON PRODUCTIF — production : {production_status}, "
        f"référence : {reference_status}."
    )

st.caption(
    "Méthode : écart standardisé des moyennes (z-score). "
    "Il ne s'agit pas d'une analyse Evidently."
)

if production.empty:
    st.info("Aucune prédiction réelle disponible dans logs/predictions.jsonl.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prédictions totales", len(production))
col2.metric("% solubles prédits", f"{production['prediction'].mean() * 100:.1f}%")
col3.metric("Latence moyenne", f"{production['inference_time_s'].mean() * 1000:.1f} ms")
col4.metric("Score moyen", f"{production['probability'].mean():.3f}")

results = []
for feature in FEATURES:
    if feature not in production or feature not in reference:
        continue

    ref_mean = reference[feature].mean()
    ref_std = reference[feature].std(ddof=1)
    prod_mean = production[feature].mean()
    z_score = (
        abs(prod_mean - ref_mean) / ref_std
        if pd.notna(ref_std) and ref_std > 0
        else np.nan
    )

    conclusive = fully_real and len(production) >= MIN_PREDICTIONS
    detected = bool(conclusive and pd.notna(z_score) and z_score > ZSCORE_THRESHOLD)

    results.append({
        "Feature": feature,
        "Moy. Référence": round(ref_mean, 4),
        "Moy. Production": round(prod_mean, 4),
        "Δ (z-score)": round(z_score, 3) if pd.notna(z_score) else None,
        "Drift détecté": "⚠️ OUI" if detected else "✅ NON" if conclusive else "ℹ️ NON CONCLUANT",
    })

drift = pd.DataFrame(results)
st.subheader("🔍 Détection du Data Drift")
st.dataframe(drift, use_container_width=True)

if not fully_real:
    st.warning("Les données ne permettent pas de produire une alerte réelle.")
elif len(production) < MIN_PREDICTIONS:
    st.info(
        f"Échantillon insuffisant : {len(production)} prédictions. "
        f"Au moins {MIN_PREDICTIONS} sont requises avant de conclure."
    )
elif drift["Drift détecté"].str.contains("OUI").any():
    st.error("Dérive potentielle détectée. Une analyse statistique complémentaire est nécessaire.")
else:
    st.success("Aucune dérive potentielle détectée avec cette méthode simple.")
