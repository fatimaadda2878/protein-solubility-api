"""
app/monitoring.py — Data Drift Monitoring avec Evidently
=========================================================
Compare les données reçues en production (logs de prédictions)
avec le dataset de référence DeepSol pour détecter un drift.

Mentionnée par la mentor comme alternative plus robuste à la validation
Pydantic seule, qui est une validation de schéma, pas un test de qualité
de données au sens MLOps.

Usage :
    from app.monitoring import DriftMonitor
    monitor = DriftMonitor()
    monitor.check_drift(current_data_df)     # retourne rapport Evidently
    report_html = monitor.generate_html_report(current_data_df)
"""
import json
import logging
import pickle
import urllib.request
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
REFERENCE_CACHE = Path("model/deepsol_reference_sample.parquet")
DEEPSOL_URL = (
    "https://raw.githubusercontent.com/sameerkhurana10/DSOL_rv0.2"
    "/master/data/protein_with_bio.data"
)
REFERENCE_SAMPLE_SIZE = 1000  # Taille de l'échantillon de référence
DRIFT_Z_THRESHOLD = 2.0       # |z| > 2 = alerte (95% confiance)

FEATURE_NAMES = [
    "pI", "log_mw", "gravy_norm", "log_instability",
    "aromaticity", "pct_helix", "pct_turn", "pct_sheet",
    "pI_distance", "pct_coil", "helix_x_gravy",
    "stability_score", "helix_sheet_ratio",
]

# Mapping features P7 → indices dans src_bio DeepSol
BIO_FEATURE_MAP = {
    "pI": 0, "log_mw": 1, "gravy_norm": 2, "log_instability": 3,
    "aromaticity": 4, "pct_helix": 5, "pct_turn": 6, "pct_sheet": 7,
}


class DriftMonitor:
    """
    Détecteur de drift basé sur Evidently (avec fallback z-score).

    Charge un échantillon du dataset DeepSol comme référence,
    puis compare les nouvelles entrées à cette distribution.
    """

    def __init__(self, reference_path: Optional[Path] = None):
        self._reference: Optional[pd.DataFrame] = None
        self._reference_path = reference_path or REFERENCE_CACHE
        self._use_evidently = self._check_evidently()

    def _check_evidently(self) -> bool:
        try:
            import evidently  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "Evidently non installé — fallback sur z-score. "
                "pip install evidently"
            )
            return False

    # ── Chargement de la référence ────────────────────────────────────────────
    def load_reference(self, force_reload: bool = False) -> pd.DataFrame:
        """Charge l'échantillon de référence DeepSol (avec cache)."""
        if self._reference is not None and not force_reload:
            return self._reference

        if self._reference_path.exists() and not force_reload:
            logger.info(f"Référence DeepSol chargée depuis cache : {self._reference_path}")
            self._reference = pd.read_parquet(self._reference_path)
            return self._reference

        logger.info("Téléchargement dataset DeepSol pour référence…")
        try:
            self._reference = self._build_reference_from_deepsol()
            self._reference_path.parent.mkdir(parents=True, exist_ok=True)
            self._reference.to_parquet(self._reference_path, index=False)
            logger.info(f"Référence sauvegardée : {self._reference_path}")
        except Exception as e:
            logger.error(f"Impossible de charger DeepSol : {e}")
            self._reference = self._build_synthetic_reference()

        return self._reference

    def _build_reference_from_deepsol(self) -> pd.DataFrame:
        """Télécharge DeepSol et extrait les features P7."""
        with urllib.request.urlopen(DEEPSOL_URL, timeout=30) as resp:
            data = pickle.load(resp)

        rows = []
        for split_name in ["train", "valid", "test"]:
            if split_name not in data:
                continue
            split = data[split_name]
            for label, bio in zip(split["tgt"], split["src_bio"]):
                row = {"label": label}
                for feat, idx in BIO_FEATURE_MAP.items():
                    row[feat] = float(bio[idx])
                # Features dérivées
                row["pI_distance"] = abs(row["pI"] - 7.0)
                row["pct_coil"] = max(
                    0, 1 - row["pct_helix"] - row["pct_turn"] - row["pct_sheet"]
                )
                row["helix_x_gravy"] = row["pct_helix"] * row["gravy_norm"]
                row["stability_score"] = 1 / (1 + abs(row["log_instability"]))
                row["helix_sheet_ratio"] = row["pct_helix"] / (abs(row["pct_sheet"]) + 1e-9)
                rows.append(row)

        df = pd.DataFrame(rows)
        # Échantillon représentatif stratifié
        n_per_class = min(REFERENCE_SAMPLE_SIZE // 2, (df["label"] == 0).sum(),
                          (df["label"] == 1).sum())
        ref = pd.concat([
            df[df["label"] == 0].sample(n_per_class, random_state=42),
            df[df["label"] == 1].sample(n_per_class, random_state=42),
        ]).reset_index(drop=True)
        logger.info(f"Référence DeepSol : {len(ref)} protéines")
        return ref[FEATURE_NAMES + ["label"]]

    def _build_synthetic_reference(self) -> pd.DataFrame:
        """Référence synthétique si DeepSol inaccessible."""
        np.random.seed(42)
        n = REFERENCE_SAMPLE_SIZE
        return pd.DataFrame({
            "pI": np.random.normal(6.5, 1.5, n),
            "log_mw": np.random.normal(11.0, 1.0, n),
            "gravy_norm": np.random.normal(-0.2, 0.8, n),
            "log_instability": np.random.normal(3.7, 0.5, n),
            "aromaticity": np.random.uniform(0.03, 0.15, n),
            "pct_helix": np.random.uniform(0.2, 0.6, n),
            "pct_turn": np.random.uniform(0.05, 0.2, n),
            "pct_sheet": np.random.uniform(0.1, 0.4, n),
            "pI_distance": np.abs(np.random.normal(0.5, 1.5, n)),
            "pct_coil": np.random.uniform(0.1, 0.4, n),
            "helix_x_gravy": np.random.normal(-0.08, 0.3, n),
            "stability_score": np.random.uniform(0.1, 0.5, n),
            "helix_sheet_ratio": np.random.uniform(0.5, 3.0, n),
            "label": np.random.binomial(1, 0.42, n),
        })

    # ── Détection de drift ────────────────────────────────────────────────────
    def check_drift(self, current: pd.DataFrame) -> dict:
        """
        Compare current avec la référence DeepSol.

        Returns:
            dict avec 'drift_detected', 'feature_drifts', 'n_drifted_features'
        """
        reference = self.load_reference()

        if self._use_evidently:
            return self._evidently_drift(reference, current)
        else:
            return self._zscore_drift(reference, current)

    def _evidently_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        """Rapport de drift Evidently."""
        from evidently import ColumnMapping
        from evidently.metrics import DataDriftTable
        from evidently.report import Report

        features_only = [f for f in FEATURE_NAMES if f in current.columns]
        report = Report(metrics=[DataDriftTable()])
        report.run(
            reference_data=reference[features_only],
            current_data=current[features_only],
            column_mapping=ColumnMapping(numerical_features=features_only),
        )
        result = report.as_dict()
        metrics = result.get("metrics", [{}])[0].get("result", {})
        feature_drifts = metrics.get("drift_by_columns", {})

        drifted = {
            feat: info
            for feat, info in feature_drifts.items()
            if info.get("drift_detected", False)
        }
        return {
            "drift_detected": bool(drifted),
            "n_drifted_features": len(drifted),
            "total_features": len(features_only),
            "drifted_features": list(drifted.keys()),
            "feature_drifts": feature_drifts,
            "method": "evidently",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _zscore_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        """Fallback z-score (|z| > 2 = drift à 95% confiance)."""
        features = [f for f in FEATURE_NAMES if f in current.columns and f in reference.columns]
        drifted = []
        details = {}

        for feat in features:
            ref_mean = reference[feat].mean()
            ref_std = reference[feat].std()
            cur_mean = current[feat].mean()

            if ref_std < 1e-9:
                continue
            z = abs((cur_mean - ref_mean) / ref_std)
            drift = z > DRIFT_Z_THRESHOLD
            details[feat] = {"z_score": round(z, 3), "drift_detected": drift}
            if drift:
                drifted.append(feat)

        return {
            "drift_detected": bool(drifted),
            "n_drifted_features": len(drifted),
            "total_features": len(features),
            "drifted_features": drifted,
            "feature_drifts": details,
            "method": "zscore",
            "z_threshold": DRIFT_Z_THRESHOLD,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ── Rapport HTML Evidently ────────────────────────────────────────────────
    def generate_html_report(
        self, current: pd.DataFrame, output_path: Optional[Path] = None
    ) -> str:
        """Génère un rapport HTML Evidently complet."""
        if not self._use_evidently:
            return self._simple_html_report(current)

        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        reference = self.load_reference()
        features_only = [f for f in FEATURE_NAMES if f in current.columns]

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference[features_only],
            current_data=current[features_only],
            column_mapping=ColumnMapping(numerical_features=features_only),
        )

        if output_path:
            report.save_html(str(output_path))
            logger.info(f"Rapport Evidently sauvegardé : {output_path}")

        return report.get_html()

    def _simple_html_report(self, current: pd.DataFrame) -> str:
        """Rapport HTML minimal (fallback sans Evidently)."""
        result = self.check_drift(current)
        color = "#e74c3c" if result["drift_detected"] else "#27ae60"
        status = "⚠️ DRIFT DÉTECTÉ" if result["drift_detected"] else "✅ Pas de drift"
        rows = "".join(
            f"<tr><td>{feat}</td>"
            f"<td>{'⚠️' if info.get('drift_detected') else '✅'}</td>"
            f"<td>{info.get('z_score', 'N/A'):.3f}</td></tr>"
            for feat, info in result["feature_drifts"].items()
        )
        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Drift Report</title></head>
<body style="font-family: sans-serif; padding: 20px;">
<h1 style="color:{color}">{status}</h1>
<p>Features en drift : {result['n_drifted_features']} / {result['total_features']}</p>
<p>Méthode : {result['method']} (z-threshold={DRIFT_Z_THRESHOLD})</p>
<table border="1" cellpadding="8">
<thead><tr><th>Feature</th><th>Drift</th><th>Z-score</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p><small>Généré le {result['timestamp']} — Référence : DeepSol</small></p>
</body>
</html>"""
