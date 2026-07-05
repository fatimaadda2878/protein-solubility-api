"""
Script d'optimisation ONNX — Protein Solubility API
Compare les performances LightGBM joblib vs ONNX Runtime

Usage :
    python optimization/onnx_optimization.py
"""

import numpy as np
import joblib
import time
import json
from pathlib import Path

print("=" * 55)
print("  OPTIMISATION ONNX — Protein Solubility Model")
print("=" * 55)

MODEL_PATH = Path("model/lgbm_model.joblib")
ONNX_PATH  = Path("model/lgbm_model.onnx")
Path("model").mkdir(exist_ok=True)
Path("optimization").mkdir(exist_ok=True)

# ── Chargement du modèle ──────────────────────────────────────
print("\n1. Chargement du modèle LightGBM...")
model = joblib.load(MODEL_PATH)
print(f"   Modèle chargé : {type(model).__name__}")

# ── Données de test ───────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 1000
N_RUNS    = 10

FEATURE_NAMES = [
    "pI", "log_mw", "gravy_norm", "log_instability",
    "aromaticity", "pct_helix", "pct_turn", "pct_sheet",
    "pI_distance", "pct_coil", "helix_x_gravy",
    "stability_score", "helix_sheet_ratio"
]

import pandas as pd
X_test_df = pd.DataFrame(
    np.random.rand(N_SAMPLES, 13).astype(np.float64),
    columns=FEATURE_NAMES
)
X_test_np = X_test_df.values.astype(np.float32)

# ── Benchmark modèle original ─────────────────────────────────
print("\n2. Benchmark modèle original (LightGBM joblib)...")
times_original = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    _ = model.predict_proba(X_test_df)
    times_original.append(time.perf_counter() - t0)

mean_original    = np.mean(times_original) * 1000
std_original     = np.std(times_original) * 1000
per_sample_orig  = mean_original / N_SAMPLES

print(f"   Temps moyen ({N_SAMPLES} samples) : {mean_original:.2f} ms +/- {std_original:.2f} ms")
print(f"   Temps par sample                 : {per_sample_orig:.4f} ms")

# ── Conversion ONNX via onnxmltools ───────────────────────────
print("\n3. Conversion du modèle en ONNX...")
try:
    import onnxmltools
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, 13]))]
    onnx_model = convert_lightgbm(
        model.booster_,
        initial_types=initial_type,
        target_opset=12
    )

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"   Modèle ONNX sauvegardé : {ONNX_PATH}")
    print(f"   Taille joblib : {MODEL_PATH.stat().st_size / 1024:.1f} KB")
    print(f"   Taille ONNX   : {ONNX_PATH.stat().st_size / 1024:.1f} KB")

    # ── Benchmark ONNX Runtime ────────────────────────────────
    print("\n4. Benchmark ONNX Runtime...")
    import onnxruntime as rt

    sess       = rt.InferenceSession(str(ONNX_PATH))
    input_name = sess.get_inputs()[0].name

    times_onnx = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: X_test_np})
        times_onnx.append(time.perf_counter() - t0)

    mean_onnx       = np.mean(times_onnx) * 1000
    std_onnx        = np.std(times_onnx) * 1000
    per_sample_onnx = mean_onnx / N_SAMPLES
    speedup         = mean_original / mean_onnx
    gain_pct        = (1 - mean_onnx / mean_original) * 100

    print(f"   Temps moyen ({N_SAMPLES} samples) : {mean_onnx:.2f} ms +/- {std_onnx:.2f} ms")
    print(f"   Temps par sample                 : {per_sample_onnx:.4f} ms")

    print("\n" + "=" * 55)
    print("  RESULTATS COMPARATIFS")
    print("=" * 55)
    print(f"  LightGBM joblib : {mean_original:.2f} ms")
    print(f"  ONNX Runtime    : {mean_onnx:.2f} ms")
    print(f"  Acceleration    : {speedup:.2f}x plus rapide")
    print(f"  Gain            : {gain_pct:.1f}%")

    # ── Vérification cohérence ────────────────────────────────
    print("\n5. Verification coherence des predictions...")
    pred_orig = model.predict_proba(X_test_df[:10])[:, 1]
    pred_onnx = sess.run(None, {input_name: X_test_np[:10]})[1][:, 1]
    max_diff  = float(np.max(np.abs(pred_orig - pred_onnx)))
    print(f"   Difference max : {max_diff:.6f}")
    print(f"   {'OK - Predictions coherentes' if max_diff < 0.01 else 'Verifier les predictions'}")

    # ── Sauvegarde JSON ───────────────────────────────────────
    results = {
        "original_lgbm": {
            "mean_ms":       round(mean_original, 4),
            "std_ms":        round(std_original, 4),
            "per_sample_ms": round(per_sample_orig, 6),
            "size_kb":       round(MODEL_PATH.stat().st_size / 1024, 1),
        },
        "onnx_runtime": {
            "mean_ms":       round(mean_onnx, 4),
            "std_ms":        round(std_onnx, 4),
            "per_sample_ms": round(per_sample_onnx, 6),
            "size_kb":       round(ONNX_PATH.stat().st_size / 1024, 1),
        },
        "speedup_x":     round(speedup, 2),
        "gain_pct":      round(gain_pct, 1),
        "max_pred_diff": round(max_diff, 6),
        "n_samples":     N_SAMPLES,
        "n_runs":        N_RUNS,
    }
    with open("optimization/onnx_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n   Resultats sauvegardes : optimization/onnx_results.json")

except Exception as e:
    print(f"\n   Erreur : {e}")
    print("\n   Benchmark joblib uniquement sauvegarde.")
    results = {
        "original_lgbm": {
            "mean_ms":       round(mean_original, 4),
            "std_ms":        round(std_original, 4),
            "per_sample_ms": round(per_sample_orig, 6),
        },
        "onnx_runtime": None,
        "note": str(e)
    }
    with open("optimization/onnx_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("\nScript termine !")
