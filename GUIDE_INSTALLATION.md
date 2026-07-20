# Installation des corrections

Copier les fichiers aux emplacements suivants :

```text
app/export_model.py
app/model.py
app/main.py
monitoring/drift_analysis.py
.github/workflows/ci_cd.yml
README.md
.env.example
.gitattributes
```

## 1. Exporter ton vrai modèle MLflow

Dans Anaconda Prompt :

```powershell
cd "C:\Users\adda-\Desktop\OpenClass Room\Projets\P7 - Confirmez vos compétences en MLOps (Partie 2-2)"
conda activate protein-mlops
$env:MLFLOW_TRACKING_URI = "sqlite:///C:/Users/adda-/mlflow.db"
$env:MLFLOW_RUN_ID = "TON_RUN_ID"
python app/export_model.py
```

## 2. Vérifier les artefacts

```powershell
dir model
```

Les fichiers suivants doivent exister :

```text
model\lgbm_model.joblib
model\model_meta.json
```

## 3. Tester localement

```powershell
pytest tests/ -v
uvicorn app.main:app --reload --port 8000
```

Ouvrir :

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 4. Ajouter le modèle avec Git LFS

```powershell
git lfs install
git add .gitattributes
git add model/lgbm_model.joblib model/model_meta.json
git add app README.md monitoring .github .env.example
git commit -m "fix: deploy real reproducible model and clarify monitoring"
git push origin main
```

## 5. Paramètres GitHub

Dans `Settings > Secrets and variables > Actions` :

- secret : `HF_TOKEN`
- variable : `HF_SPACE_ID`
