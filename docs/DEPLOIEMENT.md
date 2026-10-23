# Déploiement de la V7 (Hugging Face Spaces Gradio + ZeroGPU)

**Space actuel** : https://huggingface.co/spaces/fatimaadda2878/protein-solubility-esm150-ft

## Prérequis

- Compte Hugging Face autorisé à utiliser ZeroGPU ; un Space Gradio déjà créé.
- Dépôt modèle privé avec `finetune_best.pt`, `platt_calibrator.joblib` et `finalisation_figee.json` (non contenus dans ce dépôt).
- Dans **Settings → Variables and secrets** : variable `HF_MODEL_REPO`, secret `HF_TOKEN` (lecture du dépôt modèle privé). Ne jamais révéler ce secret.

## Mise à jour du Space depuis Colab

```python
from huggingface_hub import HfApi, get_token, notebook_login
if not get_token():
    notebook_login()
HfApi().upload_folder(
    repo_id="fatimaadda2878/protein-solubility-esm150-ft",
    repo_type="space",
    folder_path="/content/protein_solubility_gradio_conseils_v7",
    ignore_patterns=["__pycache__/**", "*.pyc", "*.zip", "*.pdf", ".env*", ".git/**", "docs/**", ".github/**"],
    commit_message="V7 - reference deployee"
)
```

Cela suppose que `/content/protein_solubility_gradio_conseils_v7` contient les sources V7. Ne pas téléverser les poids ni les datasets. Si vous téléversez depuis ce paquet GitHub plutôt que les sources V7 originales, utilisez les chemins du projet extrait, en excluant les fichiers GitHub optionnels si souhaité.

## Procédure de vérification

1. Vérifier **Running on ZERO**.
2. Dans le Space, tester une séquence courte de référence (GFP par exemple) ; confirmer que la sortie numérique est proche de la référence déjà observée pour les mêmes entrées et options.
3. Vérifier identification PDB/UniProt, conseils hiérarchisés, export PDF et lien entre score et séquence.
4. Tester les trois branches de longueur : 238 AA, 600 AA, 1 200 AA ; vérifier les mentions « mode expérimental » et AA analysés/ignorés.
5. Si l'inférence 1 022 AA échoue par manque de mémoire/temps de ZeroGPU, le signaler comme une limite et ne pas remplacer silencieusement ce résultat par une sortie 512 AA.
6. Vérifier les logs du Space, notamment les erreurs d'accès au modèle privé.

## Sécurité et confidentialité

- Le token est configuré **en secret de Space**, jamais en dur dans le code ou Git.
- La recherche en ligne transmet la séquence saisie à des bases publiques. L'application avertit l'utilisateur avant analyse.
- `finalisation_figee.json` valide l'intégrité SHA du checkpoint et du calibrateur ; l'accès au modèle privé reste requis.
- Ne pas désactiver `weights_only=False` sans examiner le format du checkpoint historique ; **ne charger des poids que depuis votre dépôt privé maîtrisé**.

## Limitation scientifique

Le protocole historique 512 AA et la politique exploratoire 1 022 AA n'ont pas le même statut de calibration. Voir `docs/RESULTATS_ET_LIMITES.md`.
