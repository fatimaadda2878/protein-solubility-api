# Évaluation du modèle ESM-2 150M fine-tuné

## Protocole

- Fit : apprentissage du modèle.
- Calibration historique : sélection des époques du fine-tuning.
- Tune : comparaison exploratoire des architectures.
- Threshold : séparation en deux sous-ensembles pour la calibration Platt et le choix du seuil.

## Résultats historiques sur Tune

- Nombre de protéines : 4 052
- AUC ROC : 0,832857
- PR-AUC : 0,786022
- Modèle : ESM-2 150M partiellement fine-tuné + Bio13

Ces résultats correspondent au développement, et non à
une validation externe indépendante.

## Export rétrospectif

Les métriques complémentaires sont calculées à partir
de prédictions historiques et d'une reconstruction
numérique des logits avant application du calibrateur.

Cet export ne constitue pas une nouvelle inférence exacte
sur le checkpoint.

## Limites et perspectives

Une future évaluation sur une cohorte indépendante devra
inclure un contrôle des homologies et une vérification de
la calibration, notamment pour les séquences de 513 à
1 022 acides aminés.

Aucune validation externe de cette branche n'est revendiquée.
