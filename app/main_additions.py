"""
Ajouts à app/main.py — À intégrer dans votre fichier existant
==============================================================

Ce fichier contient :
1. L'endpoint /monitoring/drift (Evidently)
2. La modification de /predict pour inclure threshold_used
3. L'endpoint /predict/sequence (Point 4 — Semaine 2)

Instructions :
- Copier les imports dans votre app/main.py
- Copier les endpoints dans app/main.py après /logs/summary
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTS à ajouter en haut de app/main.py
# ──────────────────────────────────────────────────────────────────
IMPORTS_TO_ADD = """
from app.monitoring import DriftMonitor
from app.schemas import ProteinInputSequence  # nouveau schéma FASTA (Point 4)
import pandas as pd

# Singleton drift monitor (chargé une fois au démarrage)
_drift_monitor: DriftMonitor = None
"""

# ──────────────────────────────────────────────────────────────────
# Modification de /predict : ajouter threshold_used à la réponse
# ──────────────────────────────────────────────────────────────────
PREDICT_THRESHOLD_PATCH = '''
# Dans l'endpoint /predict, modifier la réponse pour inclure threshold_used :
# Avant (réponse actuelle) :
return {
    "soluble": bool(prediction),
    "probability": float(probability),
}

# Après (ajouter threshold_used) :
return {
    "soluble": bool(prediction),
    "probability": float(probability),
    "threshold_used": float(model_meta.get("optimal_threshold", 0.5)),
    "model_version": model_meta.get("version", "1.0"),
}
'''

# ──────────────────────────────────────────────────────────────────
# ENDPOINT /monitoring/drift — à copier dans app/main.py
# ──────────────────────────────────────────────────────────────────
DRIFT_ENDPOINT = '''
@app.get("/monitoring/drift", tags=["Monitoring"])
async def check_data_drift(last_n: int = 100):
    """
    Vérifie le drift entre les N dernières prédictions
    et le dataset de référence DeepSol.

    Utilise Evidently si installé, sinon fallback z-score.
    Retourne un résumé JSON du drift détecté.
    """
    global _drift_monitor

    # Charger les logs de prédiction
    log_file = Path("logs/predictions.jsonl")
    if not log_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Aucun log de prédiction trouvé. Faites d\'abord des appels /predict."
        )

    # Lire les N dernières prédictions
    records = []
    with open(log_file) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if len(records) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Pas assez de données : {len(records)} logs, minimum 10 requis."
        )

    current_df = pd.DataFrame(records[-last_n:])
    feature_cols = [c for c in [
        "pI", "log_mw", "gravy_norm", "log_instability",
        "aromaticity", "pct_helix", "pct_turn", "pct_sheet",
        "pI_distance", "pct_coil", "helix_x_gravy",
        "stability_score", "helix_sheet_ratio",
    ] if c in current_df.columns]

    if not feature_cols:
        raise HTTPException(
            status_code=422,
            detail="Les logs ne contiennent pas les features attendues."
        )

    # Initialiser le monitor si nécessaire
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()

    result = _drift_monitor.check_drift(current_df[feature_cols])

    return {
        "status": "drift_detected" if result["drift_detected"] else "ok",
        "n_samples_analyzed": len(current_df),
        **result,
    }


@app.get("/monitoring/drift/report", tags=["Monitoring"])
async def drift_html_report(last_n: int = 100):
    """
    Génère un rapport HTML Evidently complet sur le drift.
    Retourne le HTML comme réponse directe (affichable dans un navigateur).
    """
    from fastapi.responses import HTMLResponse
    import json

    global _drift_monitor

    log_file = Path("logs/predictions.jsonl")
    if not log_file.exists():
        return HTMLResponse("<h1>Aucun log disponible</h1>")

    records = []
    with open(log_file) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    current_df = pd.DataFrame(records[-last_n:])

    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()

    html = _drift_monitor.generate_html_report(current_df)
    return HTMLResponse(content=html)
'''

# ──────────────────────────────────────────────────────────────────
# ENDPOINT /predict/sequence — Point 4 mentor (Semaine 2)
# Accepte directement une séquence FASTA
# ──────────────────────────────────────────────────────────────────
FASTA_ENDPOINT = '''
@app.post("/predict/sequence", tags=["Prediction"])
async def predict_from_sequence(request: ProteinInputSequence):
    """
    Prédit la solubilité à partir d\'une séquence FASTA ou séquence brute.

    Exemple :
        {"sequence": "MKLLLPTINYGFLKGGG..."}
    ou FASTA :
        {"sequence": ">prot1\\nMKLLLPTIN..."}

    Calcule automatiquement les propriétés physico-chimiques via Biopython.
    """
    from app.features import compute_features_from_sequence  # Point 4

    try:
        features = compute_features_from_sequence(request.sequence)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    probability, prediction = predict(features)

    return {
        "soluble": bool(prediction),
        "probability": float(probability),
        "threshold_used": float(model_meta.get("optimal_threshold", 0.5)),
        "sequence_length": len(request.clean_sequence),
        "features_computed": features,
    }
'''

if __name__ == "__main__":
    print("Ce fichier contient des snippets à intégrer dans app/main.py")
    print("Lisez les commentaires dans chaque section pour les instructions.")
