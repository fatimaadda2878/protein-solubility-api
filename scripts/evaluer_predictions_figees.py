#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation en lecture seule de prédictions déjà calculées (aucun ré-entraînement).

Fichier d'entrée CSV : protein_id,label,score,partition,longueur_aa
Champs facultatifs : group_id (pour audit de groupes), source_dataset, protocole.

Ce programme ne génère pas de scores ESM-2 : l'inférence doit être réalisée avec le
checkpoint et le calibrateur figés du projet, puis exportée en CSV.

Par défaut aucune évaluation n'est dite indépendante. Le mot « indépendant »
nécessite une cohorte externe réellement inédite et un audit d'homologie.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, roc_auc_score, precision_score, recall_score, f1_score
)

HISTORICAL = {'fit','tune','calibration','threshold','threshold_calibration','threshold_seuil','holdout'}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def calc_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = 0.
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1], np.linspace(0,1,bins+1)[1:]):
        mask=(p >= lo)&((p <= hi) if hi==1.0 else (p<hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(total)

def compute_metrics(sub: pd.DataFrame, threshold: float) -> dict:
    y=sub.label.to_numpy(dtype=int)
    prob=sub.score.to_numpy(dtype=float)
    pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    both=len(np.unique(y))==2
    return {
      'n':int(len(y)), 'positifs':int(y.sum()), 'negatifs':int(len(y)-y.sum()),
      'auc_roc':float(roc_auc_score(y,prob)) if both else None,
      'pr_auc_average_precision':float(average_precision_score(y,prob)) if both else None,
      'brier':float(brier_score_loss(y,prob)),
      'log_loss':float(log_loss(y,prob,labels=[0,1])),
      'ece_10_bins':calc_ece(y,prob),
      'seuil':float(threshold), 'rappel_soluble':float(recall_score(y,pred,zero_division=0)),
      'precision_soluble':float(precision_score(y,pred,zero_division=0)),
      'f1_soluble':float(f1_score(y,pred,zero_division=0)),
      'specificite':float(tn/(tn+fp)) if tn+fp else None,
      'TN':tn,'FP':fp,'FN':fn,'TP':tp,
      'cout_simule_eur':int(fn*1200+fp*600),
    }

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,type=Path)
    ap.add_argument('--sortie',required=True,type=Path)
    ap.add_argument('--version-modele',required=True)
    ap.add_argument('--checkpoint-sha256',required=True,help='SHA-256 réel (64 hex) du checkpoint figé')
    ap.add_argument('--calibrateur-sha256',default='',help='SHA-256 réel du calibrateur appliqué à la sortie')
    ap.add_argument('--version-donnees',required=True)
    ap.add_argument('--seuil',required=True,type=float,help='Seuil déjà figé, jamais optimisé ici')
    ap.add_argument('--type-evaluation',choices=['developpement','reglage_interne','cohorte_externe'],required=True)
    ap.add_argument('--audit-homologie',type=Path,help='Rapport réel de contrôle des homologies (obligatoire en externe)')
    ap.add_argument('--protocole-gele',type=Path,help='Protocole daté avant l’accès à la cohorte externe')
    args=ap.parse_args()
    for name,sha in [('checkpoint',args.checkpoint_sha256),('calibrateur',args.calibrateur_sha256)]:
        if sha and (len(sha)!=64 or any(ch not in '0123456789abcdefABCDEF' for ch in sha)):
            ap.error(f'Empreinte {name} invalide : utiliser la valeur SHA-256 authentique')
    if not 0 <= args.seuil <= 1:
        ap.error('Le seuil doit être compris entre 0 et 1')
    df=pd.read_csv(args.predictions,dtype={'protein_id':str,'partition':str,'group_id':str})
    cols={'protein_id','label','score','partition','longueur_aa'}
    if not cols.issubset(df.columns):
        ap.error('Colonnes requises absentes : '+', '.join(sorted(cols-set(df.columns))))
    if len(df)==0 or df.protein_id.isna().any() or df.protein_id.duplicated().any():
        ap.error('IDs manquants ou dupliqués, ou jeu vide')
    if df[list(cols)].isna().any().any():
        ap.error('Valeurs manquantes dans les colonnes requises')
    if not df.label.isin([0,1]).all() or not (pd.to_numeric(df.label,errors='coerce').isin([0,1])).all():
        ap.error('Labels non binaires')
    df['label']=df.label.astype(int)
    df['score']=pd.to_numeric(df.score,errors='raise').astype(float)
    df['longueur_aa']=pd.to_numeric(df.longueur_aa,errors='raise').astype(int)
    if not np.isfinite(df.score.to_numpy()).all() or not df.score.between(0,1).all():
        ap.error('Scores non finis ou hors [0, 1]')
    if not df.longueur_aa.ge(1).all():
        ap.error('Longueurs non valides')
    df['partition']=df.partition.astype(str).str.strip().str.lower()
    if args.type_evaluation=='cohorte_externe':
        if any(p in HISTORICAL for p in df.partition.unique()):
            ap.error('Une partition historique ne peut pas être déclarée externe/indépendante')
        if not (args.audit_homologie and args.protocole_gele and args.audit_homologie.is_file() and args.protocole_gele.is_file()):
            ap.error('Cohorte externe : audit d’homologie ET protocole gelé obligatoires')
    elif args.type_evaluation=='developpement' and (df.partition=='holdout').any():
        ap.error('Holdout historique déjà consulté : ne pas le réévaluer comme développement')
    df['tranche_longueur']=pd.cut(df.longueur_aa, bins=[0,512,1022,float('inf')],
                              labels=['1-512','513-1022','>1022']).astype(str)
    warnings=[]
    if (df.longueur_aa>512).any():
        warnings.append('Probabilités 513–1022 AA non recalibrées/validées indépendamment ; ne pas transposer le seuil historique sans vérification.')
    if (df.longueur_aa>1022).any():
        warnings.append('Au-delà de 1022 AA : entrée tronquée dans le démonstrateur ; validité à vérifier.')
    if args.type_evaluation!='cohorte_externe':
        warnings.append('Mesures internes, non indépendantes ; Tune a guidé le développement, Threshold a servi au réglage.')
    if 'group_id' not in df.columns:
        warnings.append('Aucune colonne group_id : impossible de vérifier les homologies dans cet export.')
    # Un audit réel reste nécessaire, même si des group_id sont présents.
    rows=[]
    for partition,sub in df.groupby('partition',sort=True):
        for tranche,ss in [('ensemble',sub),*[(k,sub[sub.tranche_longueur==k]) for k in ['1-512','513-1022','>1022']]]:
            if len(ss)==0:continue
            rows.append({'partition':partition,'tranche_longueur':tranche,
                         'version_modele':args.version_modele,
                         'version_donnees':args.version_donnees,
                         'statut':args.type_evaluation,**compute_metrics(ss,args.seuil)})
    out=args.sortie
    out.mkdir(parents=True,exist_ok=True)
    metrics_path=out/'metriques_par_partition.csv'
    pd.DataFrame(rows).to_csv(metrics_path,index=False,encoding='utf-8-sig')
    # Le csv détaillé est un EXPORT des prédictions fournies, pas une inférence créée par le script.
    df.to_csv(out/'predictions_auditees.csv',index=False,encoding='utf-8-sig')
    manifest={
      'date_utc':datetime.now(timezone.utc).isoformat(),
      'modele_version':args.version_modele,
      'checkpoint_sha256':args.checkpoint_sha256.lower(),
      'calibrateur_sha256':args.calibrateur_sha256.lower() or None,
      'seuil_preexistant':args.seuil,'type_evaluation':args.type_evaluation,
      'dataset_version':args.version_donnees,
      'predictions_entree_sha256':sha256_file(args.predictions),
      'n':int(len(df)), 'partitions':df.partition.value_counts().sort_index().to_dict(),
      'metriques':rows,
      'audit_homologie_sha256':sha256_file(args.audit_homologie) if args.audit_homologie else None,
      'protocole_gele_sha256':sha256_file(args.protocole_gele) if args.protocole_gele else None,
      'alertes':warnings,
      'note':'Les scores et labels proviennent du CSV d’entrée. Ce script ne réalise ni inférence, ni apprentissage, ni recherche d’homologies.'
    }
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Exports créés : {metrics_path}, predictions_auditees.csv, evaluation_manifest.json')

if __name__=='__main__': main()
