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

# ── Chargement du modèle original ────────────────────────────
print("=" * 55)
print("  OPTIMISATION ONNX — Protein Solubility Model")
print("=" * 55)

MODEL_PATH = Path("model/lgbm_model.joblib")
ONNX_PATH  = Path("model/lgbm_model.onnx")
Path("model").mkdir(exist_ok=True)
Path("optimization").mkdir(exist_ok=True)

print("\n1. Chargement du modèle LightGBM...")
model = joblib.load(MODEL_PATH)
print(f"   Modèle chargé : {type(model).__name__}")

# ── Données de test ───────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 1000
X_test = np.random.rand(N_SAMPLES, 13).astype(np.float32)

# ── Benchmark modèle original ─────────────────────────────────
print("\n2. Benchmark modèle original (LightGBM joblib)...")
N_RUNS = 10
times_original = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    _ = model.predict_proba(X_test)
    times_original.append(time.perf_counter() - t0)

mean_original = np.mean(times_original) * 1000
std_original  = np.std(times_original) * 1000
per_sample_original = mean_original / N_SAMPLES

print(f"   Temps moyen ({N_SAMPLES} samples) : {mean_original:.2f} ms ± {std_original:.2f} ms")
print(f"   Temps par sample                 : {per_sample_original:.4f} ms")

# ── Conversion ONNX ───────────────────────────────────────────
print("\n3. Conversion du modèle en ONNX...")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, 13]))]
    onnx_model = convert_sklearn(
        model,
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

    sess = rt.InferenceSession(str(ONNX_PATH))
    input_name = sess.get_inputs()[0].name

    times_onnx = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: X_test})
        times_onnx.append(time.perf_counter() - t0)

    mean_onnx = np.mean(times_onnx) * 1000
    std_onnx  = np.std(times_onnx) * 1000
    per_sample_onnx = mean_onnx / N_SAMPLES

    print(f"   Temps moyen ({N_SAMPLES} samples) : {mean_onnx:.2f} ms ± {std_onnx:.2f} ms")
    print(f"   Temps par sample                 : {per_sample_onnx:.4f} ms")

    # ── Résultats comparatifs ─────────────────────────────────
    speedup = mean_original / mean_onnx
    gain_pct = (1 - mean_onnx / mean_original) * 100

    print("\n" + "=" * 55)
    print("  RÉSULTATS COMPARATIFS")
    print("=" * 55)
    print(f"  LightGBM joblib : {mean_original:.2f} ms")
    print(f"  ONNX Runtime    : {mean_onnx:.2f} ms")
    print(f"  Accélération    : {speedup:.2f}x plus rapide")
    print(f"  Gain            : {gain_pct:.1f}%")

    # ── Vérification cohérence des prédictions ────────────────
    print("\n5. Vérification cohérence des prédictions...")
    pred_orig = model.predict_proba(X_test[:10])[:, 1]
    pred_onnx = sess.run(None, {input_name: X_test[:10]})[1][:, 1]
    max_diff = np.max(np.abs(pred_orig - pred_onnx))
    print(f"   Différence max entre joblib et ONNX : {max_diff:.6f}")
    if max_diff < 0.01:
        print("   ✅ Prédictions cohérentes — pas de régression")
    else:
        print("   ⚠️  Différence détectée — vérification nécessaire")

    # ── Sauvegarde des résultats ──────────────────────────────
    results = {
        "original_lgbm": {
            "mean_ms":        round(mean_original, 4),
            "std_ms":         round(std_original, 4),
            "per_sample_ms":  round(per_sample_original, 6),
            "size_kb":        round(MODEL_PATH.stat().st_size / 1024, 1),
        },
        "onnx_runtime": {
            "mean_ms":        round(mean_onnx, 4),
            "std_ms":         round(std_onnx, 4),
            "per_sample_ms":  round(per_sample_onnx, 6),
            "size_kb":        round(ONNX_PATH.stat().st_size / 1024, 1),
        },
        "speedup_x":    round(speedup, 2),
        "gain_pct":     round(gain_pct, 1),
        "max_pred_diff":round(float(max_diff), 6),
        "n_samples":    N_SAMPLES,
        "n_runs":       N_RUNS,
    }

    with open("optimization/onnx_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n   Résultats sauvegardés : optimization/onnx_results.json")

except Exception as e:
    print(f"\n   Erreur conversion ONNX : {e}")
    print("   Le modèle sklearn DummyClassifier n'est pas toujours")
    print("   compatible avec skl2onnx — utilisez le vrai modèle LightGBM")
    print("\n   Solution : relancez avec le vrai modèle LightGBM :")
    print("   python app/export_model.py")
    print("   python optimization/onnx_optimization.py")

print("\n✅ Script terminé")
