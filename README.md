---
title: Protein Solubility Prediction API
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Protein Solubility Prediction API

API FastAPI de prédiction de la solubilité de protéines recombinantes exprimées dans *E. coli*.

## Résultats reproductibles

| Mesure | Valeur |
|---|---:|
| AUC validation | 0.6291 |
| AUC test indépendante | 0.5895 |
| Seuil de décision | 0.25 |
| Run MLflow | `d87e39a5657745a3aab16b3a53e1fa6f` |

Le seuil est choisi uniquement sur le jeu de validation. Le jeu de test reste séparé et n'est utilisé que pour l'évaluation finale. Ces valeurs remplacent les anciennes valeurs non reproductibles `0.759`, `1.0` et le seuil fixe `0.05`.

## Installation

```bash
pip install -r requirements.txt
```
## Démarrage

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API : `http://127.0.0.1:8000`
- Swagger : `http://127.0.0.1:8000/docs`
- Santé : `http://127.0.0.1:8000/health`

## Exemple d'entrée

```json
{
  "pI": 6.2,
  "log_mw": 10.5,
  "gravy_norm": 0.21,
  "log_instability": 0.5,
  "aromaticity": 0.08,
  "pct_helix": 0.38,
  "pct_turn": 0.30,
  "pct_sheet": 0.18
}
```

## MLflow

Sous Windows :

```bat
set MLFLOW_TRACKING_URI=sqlite:///C:/Users/adda-/mlflow.db
```

Le chemin MLflow n'est plus codé en dur dans le code.

## Monitoring

```bash
streamlit run monitoring/drift_analysis.py
```

Le dashboard actuel applique une comparaison statistique par z-score. Il ne doit pas être présenté comme un rapport Evidently tant qu'Evidently n'est pas réellement exécuté dans le code. Les données simulées servent uniquement à démontrer le dashboard lorsqu'aucun log réel n'est disponible.

## Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Limites

```bash
streamlit run monitoring/drift_analysis.py
```

Le dashboard affiche :
- Distribution des scores prédits vs référence
- Détection automatique du drift par feature (z-score)
- Métriques de latence (P50, P95, P99)
- Alertes en cas de drift significatif

## Pipeline CI/CD (GitHub Actions)

Le pipeline se déclenche à chaque push sur `main` :

1. **Test** : exécution des tests unitaires avec couverture de code
2. **Build** : construction et validation de l'image Docker
3. **Deploy** : déploiement sur Hugging Face Spaces (si `HF_TOKEN` configuré)

### Configuration des secrets GitHub

Dans Settings → Secrets → Actions :
- `HF_TOKEN` : token Hugging Face pour le déploiement

## Modèle

- **Algorithme** : LightGBM (optimisé via Optuna, 50 trials)
- **Dataset** : DeepSol — 71 419 protéines *E. coli* (Khurana et al. 2018)
- **AUC Validation** : 0.6291 sur le jeu de validation
- **AUC de test Indépendante** : 0.5895
- **Seuil de décision** : 0.25 (optimisé selon coût métier : FN = 1 200€, FP = 200€)
- **Tracking** : MLflow (experiment `protein-solubility-ecoli`)

## Auteur

Fatima Adda-Rezig — Projet MLOps Partie 2/2
Les performances restent modestes sur le test indépendant. La prédiction ne remplace pas une validation expérimentale.
