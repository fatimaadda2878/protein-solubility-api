# 🧬 Protein Solubility Prediction API

API de prédiction de la solubilité des protéines recombinantes lors de l'expression dans *E. coli*, développée dans le cadre du projet MLOps (Partie 2/2).

## Architecture

```
protein-solubility-api/
├── app/
│   ├── main.py          # Application FastAPI
│   ├── model.py         # Chargement et inférence du modèle
│   ├── schemas.py       # Schémas Pydantic (validation entrées/sorties)
│   └── export_model.py  # Export du modèle depuis MLflow
├── tests/
│   ├── test_api.py      # Tests des endpoints
│   └── test_model.py    # Tests de la logique du modèle
├── monitoring/
│   └── drift_analysis.py # Dashboard Streamlit de monitoring
├── model/
│   └── lgbm_model.joblib # Modèle LightGBM sérialisé
├── .github/workflows/
│   └── ci_cd.yml        # Pipeline GitHub Actions
├── Dockerfile
├── requirements.txt
└── README.md
```

## Lancement de l'API

### Prérequis

```bash
pip install -r requirements.txt
```

### Export du modèle depuis MLflow

```bash
python app/export_model.py
```

### Démarrage local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'API est disponible sur : http://localhost:8000  
Documentation Swagger : http://localhost:8000/docs

### Avec Docker

```bash
# Build
docker build -t protein-solubility-api .

# Run
docker run -p 8000:8000 protein-solubility-api
```

## Utilisation de l'API

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pI": 6.2,
    "log_mw": 10.5,
    "gravy_norm": 0.21,
    "log_instability": 0.5,
    "aromaticity": 0.08,
    "pct_helix": 0.38,
    "pct_turn": 0.30,
    "pct_sheet": 0.18
  }'
```

### Exemple de réponse

```json
{
  "soluble": 1,
  "probability_soluble": 0.823,
  "probability_insoluble": 0.177,
  "confidence": "Élevé",
  "inference_time_s": 0.012,
  "recommendation": "Protéine probablement soluble — expression standard recommandée."
}
```

## Description des features d'entrée

| Feature | Description | Plage |
|---------|-------------|-------|
| `pI` | Point isoélectrique | 2.5 – 12.0 |
| `log_mw` | log(Masse moléculaire) | 7.0 – 13.0 |
| `gravy_norm` | Score GRAVY normalisé (hydrophobicité) | 0.0 – 1.0 |
| `log_instability` | log(Indice d'instabilité) | -4.0 – 2.0 |
| `aromaticity` | Fraction Tyr/Trp/Phe | 0.0 – 0.3 |
| `pct_helix` | Fraction hélice alpha | 0.0 – 1.0 |
| `pct_turn` | Fraction turn | 0.0 – 1.0 |
| `pct_sheet` | Fraction feuillet beta | 0.0 – 1.0 |

## Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Monitoring du Data Drift

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
- **AUC** : 0.759 sur le jeu de test
- **Seuil de décision** : 0.05 (optimisé selon coût métier : FN = 1 200€, FP = 200€)
- **Tracking** : MLflow (experiment `protein-solubility-ecoli`)

## Auteur

Fatima Adda-Rezig — Projet MLOps Partie 2/2
