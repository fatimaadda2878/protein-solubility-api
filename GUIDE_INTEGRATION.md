# Guide d'intégration

1. Copie chaque fichier de ce dossier à la même localisation dans ton dépôt.
2. Depuis la racine du dépôt, installe les dépendances :
   `pip install -r requirements.txt`
3. Génère le vrai modèle :
   `python retrain_model.py`
4. Vérifie la présence de :
   - `model/lgbm_model.joblib`
   - `model/model_meta.json`
5. Active Git LFS puis ajoute l'artefact :
   `git lfs install`
   `git add .gitattributes model/lgbm_model.joblib model/model_meta.json`
6. Lance les tests :
   `pytest tests/ -v`
7. Lance l'API :
   `uvicorn app.main:app --reload --port 8000`
8. Vérifie :
   - `http://localhost:8000/health`
   - `http://localhost:8000/docs`
   - une requête `POST /predict`
9. Dans GitHub :
   - secret Actions : `HF_TOKEN`
   - variable Actions : `HF_SPACE_ID` avec la forme `utilisateur/nom-du-space`
10. Commit et push uniquement après les vérifications locales.
