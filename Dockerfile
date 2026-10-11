FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour LightGBM / ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p model logs

# Générer le modèle ONNX depuis joblib (Linux pour compatibilité)
# Si le modèle joblib existe, on génère l'ONNX ; sinon on crée un DummyClassifier
RUN python - <<'EOF'
import os
import sys
from pathlib import Path

joblib_path = Path("model/lgbm_model.joblib")
onnx_path = Path("model/lgbm_model.onnx")

if joblib_path.exists():
    try:
        import joblib, lightgbm as lgb
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        model = joblib.load(joblib_path)
        initial_type = [("float_input", FloatTensorType([None, model.n_features_in_]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[OK] ONNX généré : {onnx_path}")
    except Exception as e:
        print(f"[WARN] Impossible de générer ONNX : {e}")
else:
    print(f"[INFO] Modèle joblib absent, création d'un DummyClassifier…")
    import joblib
    from sklearn.dummy import DummyClassifier
    import numpy as np
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((10, 13)), [0]*5 + [1]*5)
    dummy.n_features_in_ = 13
    joblib_path.parent.mkdir(exist_ok=True)
    joblib.dump(dummy, joblib_path)
    print(f"[OK] DummyClassifier créé : {joblib_path}")
EOF

# Exposer le port HuggingFace Spaces
EXPOSE 7860

# Entry point : server.py (évite le conflit avec le package app/)
CMD ["python", "server.py"]
