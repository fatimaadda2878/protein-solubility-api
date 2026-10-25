# Solubilité protéique — prédiction et conception expérimentale

**Version conservée : V7 expérimentale** · Interface Gradio · ESM-2 150M fine-tuné · identification PDB/UniProt · aide à la conception

[**Essayer l’application sur Hugging Face**](https://huggingface.co/spaces/fatimaadda2878/protein-solubility-esm150-ft)

## Présentation

Cette application recherche une correspondance de séquence dans les références PDB/UniProt, estime la solubilité d'une protéine avec un **ESM-2 150M partiellement fine-tuné** et propose un plan expérimental **heuristique et explicable** de construction, expression, purification et contrôles fonctionnels. Elle part d'une **seule séquence** et d'un **seul bouton**. Un compte rendu PDF est proposé.

Les recommandations expérimentales **ne sont pas des prédictions ML de l'efficacité d'un tag, d'un hôte ou d'un protocole**. Elles reposent sur des connaissances métier et des annotations de référence lorsqu'une concordance de séquence est suffisante.

## Fonctionnalités de la V7

- **Prédiction :** ESM-2 `esm2_t30_150M_UR50D` (30 couches), deux dernières couches adaptées à la tâche et tête neuronale associée à 13 descripteurs biochimiques (**Bio13**).
- **Score et décision :** calibrateur de type Platt et seuil historique **0,296491** fixé d'après les coûts simulés faux négatif = **1 200 €**, faux positif = **600 €**, avec contrainte de rappel de la classe soluble ≥ 40 %.
- **Longueur adaptative :** entrée ESM-2 jusqu'à 512 AA dans la configuration historique, jusqu'à 1 022 AA en mode expérimental pour les séquences de 513–1 022 AA, retour aux 512 premiers AA pour les séquences plus longues. Bio13 utilise toujours la séquence complète.
- **Identification automatique :** recherche de similarité RCSB PDB et récupération possible d'annotations UniProt. Identité et couvertures sont vérifiées ; une ressemblance incertaine n'est pas présentée comme une identification fonctionnelle.
- **Aide à la conception :** suggestions hiérarchisées de tags, constructions, hôtes, conditions exploratoires et mesures de contrôle selon annotations, score et caractéristiques de séquence. Le score non recalibré du mode 1 022 AA **n'entre pas dans la hiérarchisation de ces conseils**.
- **Traçabilité :** export PDF et détails techniques, sans exposer les poids du modèle.

## Résultats expérimentaux

Les métriques suivantes proviennent du développement du projet : **elles ne constituent pas une validation externe**.

| Expérience | AUC ROC sur DeepSol Tune (4 052 protéines) |
|---|---:|
| ESM-2 150M figé + Bio13 + LightGBM | 0,756177 |
| ESM-2 150M + optimisation Optuna de LightGBM | 0,761432 |
| ESM-2 650M figé + Bio13 + LightGBM | 0,766446 |
| **ESM-2 150M fine-tuné partiellement + Bio13** | **0,832857** |

La comparaison des entrées à 512 et à 1 022 AA avec le **même checkpoint**, sur Tune :

| Cohorte | n | AUC 512 AA | AUC 1 022 AA | Δ AUC |
|---|---:|---:|---:|---:|
| Toutes les protéines | 4 052 | 0,832857 | 0,834727 | +0,001869 |
| Jusqu'à 512 AA | 3 672 | 0,834522 | 0,834522 | 0 |
| 513–1 022 AA | 360 | 0,813852 | 0,848630 | +0,034779 |
| Plus de 1 022 AA | 20 | 0,854167 | 0,802083 | −0,052083 |

La règle de longueur V7 a été retenue **après exploration de Tune** ; elle n'a pas été confirmée sur une nouvelle cohorte indépendante. Le calibrateur et le seuil proviennent du protocole historique **512 AA** : sur 513–1 022 AA, le score et la classe de la V7 sont donc **indicatifs**, non des probabilités et décisions recalibrées/validées.

Voir [le protocole, les expériences et les limites](docs/RESULTATS_ET_LIMITES.md).

## Architecture

```text
Séquence protéique brute
    │
    ├── Validation + Bio13 (séquence complète)
    ├── ESM-2 150M fine-tuné (fenêtre 512 ou 1 022 AA)
    │      └── Logit → calibrateur historique → score + statut de validation
    └── Recherche PDB → annotation UniProt si correspondance fiable
               │
               └── Moteur de règles métier (tags/hôtes/contrôles)
                            └── Interface Gradio + fiche PDF
```

Détails dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Déploiement et configuration

L'application publique tourne sur **Hugging Face Spaces Gradio + ZeroGPU**. Le checkpoint et les fichiers de calibration sont **dans un dépôt Model privé distinct**, et non dans le dépôt de code.

1. Créer / disposer d'un Space **Gradio**, configuré pour **ZeroGPU**.
2. Définir la variable `HF_MODEL_REPO` avec l'identifiant du dépôt Model (format `compte/depot`).
3. Définir le secret `HF_TOKEN` avec un token ayant les droits **lecture** sur ce dépôt privé. Ne jamais committer de token.
4. Le dépôt Model privé doit contenir `finetune_best.pt`, `platt_calibrator.joblib`, `finalisation_figee.json` ; leurs empreintes sont vérifiées au démarrage.
5. Publier les fichiers de ce dépôt à la racine du Space. `requirements.txt` est installé par le service.

Le `README.md` de GitHub ne contient pas le front matter YAML du Space : la configuration Gradio est conservée dans le dépôt Hugging Face, qui reste indépendant.

[Guide détaillé d'installation / déploiement](docs/DEPLOIEMENT.md) · [Architecture](docs/ARCHITECTURE.md).

## Exécuter les tests hors GPU

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. python tests/v7/test_conseils_v3.py
PYTHONPATH=. python tests/v7/test_identification_v4.py
PYTHONPATH=. python tests/v7/test_workflow_v5.py
PYTHONPATH=. python tests/v7/test_personnalisation_v6.py
PYTHONPATH=. python tests/v7/test_longueur_v7.py
```

Ces tests utilisent notamment des réponses de services simulées. Ils ne remplacent pas un test de bout en bout sur le Space ZeroGPU avec le checkpoint privé.

## Organisation du dépôt

- `app.py` : Gradio / modèle / pipeline de prédiction / sécurité des poids.
- `length_policy_v7.py` : fenêtre d'inférence adaptative et avertissements.
- `workflow_v5.py` : liaison du résultat ML à l'identification et aux recommandations.
- `protein_identifier.py` : PDB, annotation UniProt, vérification d'identité et couverture.
- `expert_rules_v6.py`, `protein_advice.py` : recommandations biologiques, justifications et PDF.
- `finaliser_finetune_v3_securise.py` : fonctions Bio13 et architecture nécessaires au chargement du checkpoint ; utilitaire de finalisation historique.
- `tests/v7/test_*.py` : contrôles de non-régression sans GPU.
- `docs/` : architecture, protocole scientifique, déploiement et étapes Git Bash.

## Confidentialité et limites

- Cliquer sur **Analyser la séquence** envoie aussi la séquence aux services **publics** RCSB PDB/UniProt. Ne jamais utiliser de séquence confidentielle sans autorisation.
- Une identification par similarité ne prouve pas la fonction de la cible. L'absence de correspondance PDB ne prouve pas l'absence de fiche UniProt.
- Le modèle prédit un score lié aux annotations de solubilité des données d'apprentissage ; il ne prédit **ni un rendement soluble mesuré, ni l'efficacité d'un tag, ni des paramètres d'induction optimaux**.
- La méthode comporte une troncature des longues séquences ; l'AUC annoncée (0,832857) porte sur **DeepSol Tune**, déjà exploité pendant le développement.
- La calibration et la sélection du seuil ont été effectuées sur deux sous-ensembles de l'ancienne partition `threshold` ; ces résultats ne sont pas une validation externe.
- **Aucune décision expérimentale coûteuse ne doit reposer uniquement sur ces scores ou conseils.**

Le rapport métier et le rapport scientifique détaillés pourront être actualisés séparément. Ce dépôt documente l'**état V7 figé** de l'application, sans réentraîner ni remplacer le modèle.

## Sources / outils

- [RCSB PDB Search API](https://search.rcsb.org/)
- [UniProt REST API](https://www.uniprot.org/help/api)
- [ESM / FAIR Protein Language Models](https://github.com/facebookresearch/esm)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)

## Historique des architectures

La branche `archive/avant-nettoyage-v7` conserve les anciens composants FastAPI, ONNX, Docker et monitoring. La branche principale presente exclusivement le demonstrateur Gradio V7. Les poids et les donnees restent hors GitHub.
