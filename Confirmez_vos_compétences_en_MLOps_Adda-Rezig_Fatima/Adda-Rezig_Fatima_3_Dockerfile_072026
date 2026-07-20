# ── Image de base Python légère ──────────────────────────────
FROM python:3.10-slim

# ── Métadonnées ───────────────────────────────────────────────
LABEL maintainer="Fatima Adda-Rezig"
LABEL description="API FastAPI - Prédiction Solubilité Protéines Recombinantes"
LABEL version="1.0.0"

# ── Variables d'environnement ─────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=model/lgbm_model.joblib
ENV SCALER_PATH=model/scaler.joblib

# ── Répertoire de travail ─────────────────────────────────────
WORKDIR /app

# ── Installation des dépendances système ─────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Copie et installation des dépendances Python ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copie du code source ──────────────────────────────────────
COPY app/ ./app/
COPY model/ ./model/

# ── Création du dossier logs ──────────────────────────────────
RUN mkdir -p logs

# ── Exposition du port ────────────────────────────────────────
EXPOSE 8000

# ── Commande de démarrage ─────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
