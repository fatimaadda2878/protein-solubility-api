# Résultats, protocole et limites — état V7

Document de traçabilité pour préparer la mise à jour ultérieure du rapport métier. **Ne pas interpréter les résultats exploratoires comme une validation clinique, réglementaire ou externe.**

## Jeu de données et partitions

- DeepSol : 62 478 protéines dans les partitions du projet.
- **Fit** : 46 270 protéines pour entraîner les modèles lors de la campagne V3.
- **Calibration** : 4 052 protéines, utilisée comme validation/arrêt anticipé de la campagne V3 (pas vierge pour calibrer ce checkpoint).
- **Tune** : 4 052 protéines, réutilisée pour comparer de nombreux candidats et, plus tard, les longueurs d'inférence.
- **Threshold** : 4 052 protéines, divisées en 2 × 2 026 dans le protocole de finalisation pour calibrer et sélectionner un seuil économique.
- **Holdout** : 4 052 protéines non consultées pour le modèle fine-tuné dans les campagnes décrites ici ; un holdout DeepSol a déjà servi à l'évaluation finale d'un modèle précédent. Il ne serait donc pas honnête de le présenter comme une nouvelle validation externe indépendante pour la V7.

Le contrôle des homologues MMseqs2 entre jeux réduit certains risques de recouvrement ; il ne prouve pas l'absence exhaustive d'homologie. Les comparaisons Tune demeurent optimistes du fait des décisions successives fondées sur cette partition.

## Expériences ML

| Expérience | AUC ROC Tune | PR-AUC Tune | Commentaire |
|---|---:|---:|---|
| ESM-2 150M figé + Bio13 + LightGBM | 0,756177 | 0,686373 | Référence de développement |
| ESM-2 150M + Optuna LightGBM | 0,761432 | 0,693242 | Optimisation exploratoire |
| ESM-2 650M figé + Bio13 | 0,766446 | 0,701039 | Coût de calcul plus élevé |
| **ESM-2 150M fine-tuné partiellement + Bio13** | **0,832857** | **0,786022** | Candidat retenu |

Le fine-tuning adapte les deux dernières couches d'ESM-2 et une tête de classification, avec une troncature à 512 AA pendant l'entraînement. La comparaison change aussi l'architecture aval : on ne peut pas attribuer tout le gain uniquement au fine-tuning des couches.

## Calibration et décision économique historiques

- Checkpoint fine-tuné : `finetune_best.pt` ; empreinte vérifiée via manifeste.
- Deux sous-ensembles de la partition Threshold ont servi aux opérations distinctes : **2 026** pour ajuster un calibrateur Platt et **2 026** pour fixer le seuil.
- **Seuil figé ≈ 0,296491**.
- Coûts hypothétiques : **faux négatif 1 200 €**, **faux positif 600 €**, rappel soluble contraint à ≥40 %.
- Sur le sous-ensemble de 2 026 ayant servi à choisir le seuil, le rappel était **0,8575**, coût simulé **387 600 €** ; ce coût a été **optimisé sur les mêmes données** et ne constitue donc pas une estimation de coût indépendante.
- Le même seuil et le même calibrateur sont conservés sans modification dans V7 ; pour les séquences de **513–1 022 AA**, ils ne sont **pas validés** avec la nouvelle longueur.

## Test de fenêtre 512 → 1 022 AA

Même checkpoint, Bio13 complets, relecture de Tune :

| Groupe | n | AUC 512 AA | AUC 1 022 AA | Delta |
|---|---:|---:|---:|---:|
| Toutes | 4 052 | 0,832857 | 0,834727 | +0,001869 |
| ≤512 AA | 3 672 | 0,834522 | 0,834522 | 0,000000 |
| 513–1 022 AA | 360 | 0,813852 | 0,848630 | +0,034779 |
| >1 022 AA | 20 | 0,854167 | 0,802083 | −0,052083 |
| >512 AA, ensemble | 380 | 0,814772 | 0,845574 | +0,030802 |

*Attention :* sur le sous-groupe >1 022 AA (n=20), l'effectif est insuffisant pour conclure ; aucune incertitude indépendante n'a été estimée. La V7 utilise une politique de longueur choisie après consultation de Tune, ce qui peut majorer la performance espérée sur une nouvelle cohorte. Les résultats ne justifient pas de revendiquer une AUC finale V7 = 0,834727.

## UESolDS 2024 (expérience écartée)

- 70 031 entrées fournies, 70 029 séquences valides et uniques dans l'audit.
- 10 994 exactes DeepSol et 25 386 proches détectées via MMseqs2 écartées ; **33 649** entrées UESolDS conservées.
- Entraînement DeepSol Fit seul : **0,756177** d'AUC Tune.
- DeepSol Fit + UESolDS filtré (79 919 exemples) : **0,733012** d'AUC Tune.
- La fusion n'a donc pas amélioré la généralisation vers Tune DeepSol. Causes possibles : différences d'étiquetage, de domaine expérimental ou de pondération ; non attribuées causalement.

## Recommandations et déploiement

L'identification RCSB PDB/UniProt utilise l'identité et la couverture des alignements pour décider de l'affichage d'annotations. Ces homologies ne constituent pas une validation de la fonction et la recherche ne couvre pas nécessairement toutes les protéines UniProt.

Les recommandations sur les tags (His6, MBP, SUMO...), hôtes (E. coli, Sf9, HEK293) et conditions expérimentales sont des **heuristiques**, pas un second modèle prédictif mesurant des gains de rendement. Elles n'ont pas été validées expérimentalement sur chaque cible. Les contrôles proposés (fluorescence, redox, etc.) dépendent des annotations reconnues et des contraintes renseignées.

Le fonctionnement sur Hugging Face ZeroGPU a été démontré pour une courte séquence ; le mode 1 022 AA peut rencontrer une limite de ressources. **Aucun jeu externe indépendant récent n'a confirmé l'ensemble du pipeline V7.**

## À faire lors de la rédaction du rapport métier

Ajouter les objectifs biologiques et coûts métiers avec leurs hypothèses, méthodologie de séparation et contrôle des homologues, essais comparatifs, matrice de confusion du modèle historique si utile et sa non-comparabilité directe, bilan MLOps, limites de transposition, valeur des règles explicables et démonstration Gradio. Documenter séparément toute future validation externe au lieu de réétiqueter Tune en « test final ».
