# Protein Solubility — protocole d’évaluation gelée et traçabilité

**Date : 28 octobre 2026.** Architecture conservée : ESM-2 150M partiellement fine-tuné (deux dernières couches) + Bio13. La comparaison sur DeepSol **Tune** est une mesure **exploratoire de développement**, et non une performance finale indépendante.

## 1. Partitions, sans confusion des rôles

| Partition | Effectif | Utilisation correcte |
| --- | ---: | --- |
| Fit | 46 270 | Apprentissage des poids et paramètres du modèle. |
| Calibration | 4 052 | Validation interne pendant le fine-tuning, choix de l’époque optimale et arrêt anticipé ; **déjà consommée**. |
| Tune | 4 052 | Comparaison des architectures, choix et mises au point répétées : **pas un jeu de test indépendant**. |
| Threshold | 4 052 | Deux sous-ensembles distincts issus du regroupement approximatif par homologie : l’un pour **Platt**, l’autre pour le **seuil économique** ; chacun participe au réglage. Les tailles exactes doivent être récupérées dans `bilan_calibration_seuil.json`. |
| Holdout historique | 4 052 | Déjà consulté dans d'autres expériences ; ne pas l’annoncer comme test final de cette version. |
| Nouvelle cohorte | À déterminer | Collecte future, non encore réalisée. |

La logique est directement cohérente avec les scripts originaux `derniere_campagne_esm2_v3.py` (early stopping) et `finaliser_finetune_v3_securise.py`, aujourd'hui `solubility/finetune.py` (split de Threshold, Platt, seuil économique).

## 2. État des résultats documentés

Le fichier `resultats/resultats_developpement_documentes.csv` reproduit les métriques historiques du rapport sans les présenter comme une nouvelle exécution. Pour le modèle final de développement sur **Tune** (n = 4 052), AUC ROC = **0,832857** et PR-AUC = **0,786022**. Le seuil historique rapporté est **0,296491** ; le rappel **0,8575** et le coût hypothétique **387 600 €** proviennent du sous-ensemble utilisé pour fixer le seuil : ils **ne constituent pas** une validation indépendante. Les SHA-256 exacts du checkpoint et du calibrateur ne figurent pas dans les fichiers de rapport disponibles : ne pas les inventer.

## 3. Comment produire un export de mesures reproductibles

1. Geler les versions des fichiers : checkpoint `.pt`, calibrateur `.joblib`, `finalisation_figee.json`, seuil et règles de longueur. Conserver leurs vrais SHA-256.
2. Exporter par inférence **sans ré-entraînement ni nouveau réglage** un CSV des prédictions réellement produites, au minimum `protein_id,label,score,partition,longueur_aa`; ajouter `group_id` et la provenance si disponibles. La colonne `score` doit être la probabilité après **application du calibrateur figé**, pas un logit.
3. Exécuter (en adaptant chemins et SHA-256 réels) :

```bash
python scripts/evaluer_predictions_figees.py \
  --predictions predictions_reelles.csv \
  --sortie resultats/mesure_v1 \
  --version-modele esm2_t30_150M_UR50D_finetune_last2_bio13 \
  --checkpoint-sha256 VRAIE_EMPREINTE_SHA256_SUR_64_CARACTERES \
  --calibrateur-sha256 VRAIE_EMPREINTE_SHA256_CALIBRATEUR \
  --version-donnees deepSol_partition_manifest_version_X \
  --seuil 0.296491 \
  --type-evaluation developpement
```

4. Le script produit `metriques_par_partition.csv`, `predictions_auditees.csv` et `evaluation_manifest.json` (versions, partitions, effectifs, ROC-AUC, PR-AUC / average precision, Brier, ECE, rappel, précision, spécificité, matrice de confusion, coût simulé, empreintes). Le script **ne génère pas** de scores et **ne valide pas** les homologies.
5. Pour une véritable cohorte nouvelle, utiliser `--type-evaluation cohorte_externe` et fournir deux documents réels : `--audit-homologie audit_homologie.json --protocole-gele protocole_gele.md`. Le script refuse de qualifier une partition historique d’externe. Un rapport d’audit doit démontrer l'absence des proximités problématiques **entre la cohorte externe et toutes les partitions utilisées pendant le développement**, pas seulement à l'intérieur de la cohorte.

## 4. Perspectives scientifiques et limites

- Constituer une cohorte **inédite** dont les définitions expérimentales de solubilité sont harmonisées. Geler a priori la version du modèle, les seuils, les filtres d’homologie, les métriques et le plan d’analyse.
- Contrôler doublons exacts et homologies (ex. MMseqs2), documenter les seuils d’identité/couverture et les cas ambigus ; préserver une évaluation suffisamment indépendante.
- Calculer les métriques pour l'ensemble et par tranche de longueur : **≤ 512 AA**, **513–1 022 AA**, **> 1 022 AA**. La calibration des protéines **513–1 022 AA** doit être contrôlée spécifiquement : la différence d’AUC sur Tune ne la valide pas.
- Comparer les taux de réussite biologique dans un pilote pré-déclaré à des contrôles, recueillir coûts complets et activités. **Aucun de ces essais futurs n’a été réalisé** au 28 octobre 2026.
- Les recommandations de tags, hôtes, conditions et contrôles reposent sur des **règles métier** et non sur une prédiction validée de leur efficacité par ESM-2.

## 5. Hypothèses budgétaires, pas des coûts réels

Trois scénarios **cumulatifs**, montants HT illustratifs :

| Poste | Quantité x unité | Estimation € |
| --- | --- | ---: |
| A — audit data | 8 J x 450 € | 3 600 |
| A — expertise biochimie | 4 J x 500 € | 2 000 |
| A — qualité/test pipeline | 2 J x 450 € | 900 |
| A — stockage/outils | 1 forfait x 500 € | 500 |
| **Total A** | **14 J + forfait** | **7 000** |
| B — évaluation ML | 6 J x 550 € | + 3 300 |
| B — expertise biochimie | 3 J x 500 € | + 1 500 |
| B — GPU | 1 forfait x 700 € | + 700 |
| **Total B (cumul)** | **23 J + forfaits** | **12 500** |
| C — conduite labo | 5 J x 500 € | + 2 500 |
| C — criblage de 10 constructions (consommables/plateforme, hors travail déjà facturé) | 10 x 600 € | + 6 000 |
| C — bilan pilote | 2 J x 500 € | + 1 000 |
| C — aléas non redondants | 1 forfait x 1 000 € | + 1 000 |
| **Total C (cumul)** | **30 J, 10 essais + forfaits** | **23 000** |

La valorisation illustrative d'un **cycle expérimental complet évité à 2 000 €** est distincte du **criblage ponctuel à 600 €** et des **pénalités théoriques** de faux positifs/faux négatifs dans le réglage du seuil. Les dépenses constatées, devis et économies réelles sont à collecter ; ne pas revendiquer de ROI observé.
