# Journal des évolutions

## V7 — version conservée (2026-10)

- Inférence adaptative : ≤512 AA sous le protocole historique, 513–1 022 AA sous une extension expérimentale, >1 022 AA retour aux 512 premiers AA.
- Signalement explicite de la calibration non validée pour l'extension 1 022 AA ; taille analysée/ignorée indiquée dans les résultats et PDF.
- Conservation intégrale des poids, du calibrateur et du seuil antérieurs ; aucune nouvelle AUC finale indépendante annoncée.

## V6 — recommandations enrichies

- Exploitation encadrée des annotations fonctionnelles UniProt et des propriétés de séquence.
- Conseils mieux contextualisés pour thioredoxine, ubiquitine, GFP, lysozyme, protéines membranaires, etc.

## V5 — workflow unifié

- Un seul bouton « Analyser la séquence » : inférence, identification par similarité et plan expérimental.
- Lisibilité de l'interface renforcée et visuel de ruban protéique illustratif.

## V4 — identification automatisée

- Recherche de similarité RCSB PDB, puis liens/annotations UniProt sous conditions d'identité et de couverture.

## V3 — contextualisation des recommandations

- Mise en relation du résultat ESM-2 et des recommandations lorsque la séquence est identique.
- Classement des tags/constructions, contrôles fonctionnels selon la famille déclarée.

## V2 — conseils biochimiques / PDF

- Ajout du plan d'expression/purification et de l'export PDF.

## Modèle et expérimentation (pré-V2)

- Comparaison d'ESM-2 figé, LightGBM/Optuna, modèle ESM-2 650M figé, fine-tuning partiel du 150M.
- Checkpoint fine-tuné figé ; calibration historique et optimisation du seuil économique avec coûts hypothétiques.
