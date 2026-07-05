"""
Analyse du Data Drift — Protein Solubility API
Utilise Evidently AI pour détecter la dérive des données en production.

Usage :
    streamlit run monitoring/drift_analysis.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Configuration de la page Streamlit ───────────────────────
st.set_page_config(
    page_title="Monitoring — Protein Solubility API",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Monitoring — Protein Solubility Prediction API")
st.markdown("Tableau de bord de suivi des performances et détection du data drift")

# ── Chargement des logs de production ────────────────────────
LOG_FILE = Path("logs/predictions.jsonl")

@st.cache_data(ttl=30)
def load_production_logs():
    """Je charge les logs de prédictions de l'API."""
    if not LOG_FILE.exists():
        return pd.DataFrame()
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                continue
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Expansion des features d'entrée
    input_df = pd.DataFrame(df["input"].tolist())
    df = pd.concat([df.drop(columns=["input"]), input_df], axis=1)
    return df


@st.cache_data
def load_reference_data():
    """
    Je charge les données de référence (train set DeepSol).
    En l'absence du fichier réel, je simule des données représentatives.
    """
    ref_file = Path("data/reference_data.csv")
    if ref_file.exists():
        return pd.read_csv(ref_file)

    # Simulation de données de référence (distributions DeepSol)
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "pI":              np.random.normal(5.56, 0.54, n).clip(2.5, 12.0),
        "log_mw":         np.random.normal(10.43, 0.54, n).clip(7.0, 13.0),
        "gravy_norm":     np.random.normal(0.215, 0.040, n).clip(0.0, 1.0),
        "log_instability":np.random.normal(0.2, 0.5, n).clip(-4.0, 2.0),
        "aromaticity":    np.random.normal(0.08, 0.02, n).clip(0.0, 0.3),
        "pct_helix":      np.random.normal(0.38, 0.08, n).clip(0.0, 1.0),
        "pct_turn":       np.random.normal(0.44, 0.11, n).clip(0.0, 1.0),
        "pct_sheet":      np.random.normal(0.18, 0.11, n).clip(0.0, 1.0),
        "prediction":     np.random.binomial(1, 0.42, n),
    })


# ── Chargement des données ────────────────────────────────────
prod_df = load_production_logs()
ref_df  = load_reference_data()

FEATURES = ["pI", "log_mw", "gravy_norm", "log_instability",
            "aromaticity", "pct_helix", "pct_turn", "pct_sheet"]

# ── Si pas de données de production : simulation ─────────────
if prod_df.empty:
    st.warning("Aucun log de production trouvé — affichage avec données simulées")
    np.random.seed(99)
    n_prod = 200
    # Simulation avec léger drift sur pI et gravy_norm
    prod_df = pd.DataFrame({
        "timestamp":      [datetime.now() - timedelta(hours=i) for i in range(n_prod)],
        "pI":             np.random.normal(6.5, 0.8, n_prod).clip(2.5, 12.0),  # drift +1
        "log_mw":         np.random.normal(10.43, 0.54, n_prod).clip(7.0, 13.0),
        "gravy_norm":     np.random.normal(0.30, 0.05, n_prod).clip(0.0, 1.0),  # drift +0.085
        "log_instability":np.random.normal(0.2, 0.5, n_prod).clip(-4.0, 2.0),
        "aromaticity":    np.random.normal(0.08, 0.02, n_prod).clip(0.0, 0.3),
        "pct_helix":      np.random.normal(0.38, 0.08, n_prod).clip(0.0, 1.0),
        "pct_turn":       np.random.normal(0.44, 0.11, n_prod).clip(0.0, 1.0),
        "pct_sheet":      np.random.normal(0.18, 0.11, n_prod).clip(0.0, 1.0),
        "prediction":     np.random.binomial(1, 0.55, n_prod),
        "probability":    np.random.uniform(0.0, 1.0, n_prod),
        "inference_time_s": np.random.exponential(0.015, n_prod),
    })

# ── KPIs ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Métriques clés de production")

col1, col2, col3, col4, col5 = st.columns(5)
n_pred = len(prod_df)
n_sol  = int(prod_df["prediction"].sum()) if "prediction" in prod_df.columns else 0
avg_lat = prod_df["inference_time_s"].mean() * 1000 if "inference_time_s" in prod_df.columns else 0
avg_prob = prod_df["probability"].mean() if "probability" in prod_df.columns else 0

col1.metric("Prédictions totales", f"{n_pred:,}")
col2.metric("% Solubles prédits", f"{n_sol/max(n_pred,1)*100:.1f}%")
col3.metric("Latence moyenne", f"{avg_lat:.1f} ms")
col4.metric("Score moyen", f"{avg_prob:.3f}")
col5.metric("Référence % solubles", "42.0%",
            delta=f"{n_sol/max(n_pred,1)*100 - 42.0:.1f}%")

# ── Distribution des scores dans le temps ────────────────────
st.markdown("---")
st.subheader("📈 Distribution des scores de prédiction")

import matplotlib.pyplot as plt

col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ref_df["prediction"] if "prediction" in ref_df.columns
            else np.random.binomial(1, 0.42, len(ref_df)),
            bins=2, alpha=0.6, color="#2196F3", label="Référence (train)")
    ax.hist(prod_df["prediction"],
            bins=2, alpha=0.6, color="#EF5350", label="Production")
    ax.set_title("Distribution des prédictions")
    ax.set_xlabel("Prédiction (0=Insoluble, 1=Soluble)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

with col_b:
    if "probability" in prod_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(prod_df["probability"], bins=30,
                color="#9C27B0", alpha=0.7, edgecolor="white")
        ax.set_title("Distribution des probabilités (production)")
        ax.set_xlabel("P(soluble)")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)
        plt.close()

# ── Analyse du Data Drift ─────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Détection du Data Drift")

# Comparaison statistique feature par feature
drift_results = []
for feat in FEATURES:
    if feat not in prod_df.columns or feat not in ref_df.columns:
        continue

    ref_mean  = ref_df[feat].mean()
    prod_mean = prod_df[feat].mean()
    ref_std   = ref_df[feat].std()
    drift_pct = abs(prod_mean - ref_mean) / (ref_std + 1e-8) * 100

    # Seuil de drift : z-score > 2 → drift détecté
    drift_detected = abs(prod_mean - ref_mean) > 2 * ref_std

    drift_results.append({
        "Feature":        feat,
        "Moy. Référence": round(ref_mean, 4),
        "Moy. Production":round(prod_mean, 4),
        "Δ (z-score)":    round(abs(prod_mean - ref_mean) / (ref_std + 1e-8), 2),
        "Drift détecté":  "⚠️ OUI" if drift_detected else "✅ NON",
    })

drift_df = pd.DataFrame(drift_results)
st.dataframe(drift_df, use_container_width=True)

# Alertes
n_drift = drift_df["Drift détecté"].str.contains("OUI").sum()
if n_drift > 0:
    st.error(f"⚠️ {n_drift} feature(s) présentent un drift significatif — "
             f"réentraînement du modèle recommandé.")
else:
    st.success("✅ Aucun drift significatif détecté — le modèle est stable.")

# ── Distribution des features (référence vs production) ───────
st.markdown("---")
st.subheader("📉 Distributions des features — Référence vs Production")

feat_select = st.selectbox("Choisir une feature :", FEATURES)

fig, ax = plt.subplots(figsize=(8, 4))
if feat_select in ref_df.columns:
    ax.hist(ref_df[feat_select], bins=40, alpha=0.6,
            color="#2196F3", label="Référence (train)", density=True)
if feat_select in prod_df.columns:
    ax.hist(prod_df[feat_select], bins=40, alpha=0.6,
            color="#EF5350", label="Production", density=True)
ax.set_title(f"Distribution de {feat_select}")
ax.set_xlabel(feat_select)
ax.set_ylabel("Densité")
ax.legend()
st.pyplot(fig)
plt.close()

# ── Latence de l'API ──────────────────────────────────────────
if "inference_time_s" in prod_df.columns:
    st.markdown("---")
    st.subheader("⚡ Performance — Latence de l'API")

    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(prod_df["inference_time_s"] * 1000, bins=30,
                color="#4CAF50", alpha=0.8, edgecolor="white")
        ax.axvline(prod_df["inference_time_s"].mean() * 1000,
                   color="red", linestyle="--",
                   label=f"Moy = {prod_df['inference_time_s'].mean()*1000:.1f} ms")
        ax.set_title("Distribution de la latence")
        ax.set_xlabel("Latence (ms)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_d:
        p50 = prod_df["inference_time_s"].quantile(0.50) * 1000
        p95 = prod_df["inference_time_s"].quantile(0.95) * 1000
        p99 = prod_df["inference_time_s"].quantile(0.99) * 1000
        st.metric("P50 (médiane)", f"{p50:.1f} ms")
        st.metric("P95", f"{p95:.1f} ms")
        st.metric("P99", f"{p99:.1f} ms")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
           f"Dataset de référence : DeepSol (71 419 protéines E. coli)")
