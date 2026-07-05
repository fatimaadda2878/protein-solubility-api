"""
Script de réentraînement du modèle sur les 13 features physicochimiques.
A exécuter depuis le dossier racine du projet P7.
"""
import pickle
import numpy as np
import pandas as pd
import joblib
import urllib.request
from pathlib import Path
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

print("Chargement DeepSol...")
url = (
    "https://raw.githubusercontent.com/sameerkhurana10/"
    "DSOL_rv0.2/master/data/protein_with_bio.data"
)
data = urllib.request.urlopen(url, timeout=60).read()
obj  = pickle.loads(data)

all_bio, all_tgt = [], []
for split in ["train", "valid", "test"]:
    all_bio.extend(obj[split]["src_bio"])
    all_tgt.extend(obj[split]["tgt"])

print(f"Protéines chargées : {len(all_bio)}")

# Construction du DataFrame avec les 13 features
df = pd.DataFrame(
    [[float(b[i]) for i in range(8)] for b in all_bio],
    columns=[
        "pI", "log_mw", "gravy_norm", "log_instability",
        "aromaticity", "pct_helix", "pct_turn", "pct_sheet"
    ]
)
df["pI_distance"]      = abs(df["pI"] - 7.0)
df["pct_coil"]         = (1 - df["pct_helix"] - df["pct_turn"] - df["pct_sheet"]).clip(0, 1)
df["helix_x_gravy"]    = df["pct_helix"] * df["gravy_norm"]
df["stability_score"]  = df["log_instability"] * df["gravy_norm"]
df["helix_sheet_ratio"]= df["pct_helix"] / (df["pct_sheet"] + 1e-6)

y = [int(t) for t in all_tgt]
print(f"Features : {df.shape[1]} | Labels : {len(y)}")

# Entraînement LightGBM
print("Entraînement LightGBM sur 13 features...")
model = lgb.LGBMClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)
model.fit(df.values, y)

# Evaluation
auc = roc_auc_score(y, model.predict_proba(df.values)[:, 1])
print(f"AUC : {round(auc, 4)}")

# Sauvegarde
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/lgbm_model.joblib")
print("Modele sauvegarde : model/lgbm_model.joblib")
print("Termine !")
