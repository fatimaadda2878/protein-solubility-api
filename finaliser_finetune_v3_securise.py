#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finaliser le checkpoint de la campagne DeepSol V3 sans toucher au Holdout.

IMPORTANT : V3 utilisait Calibration pour choisir l'epoque de fine-tuning.
Le jeu Threshold est divise en deux ensembles disjoints groupes par homologie
approximative MMseqs2 : calibration des logits et selection du seuil FN/FP.
Ce protocole utilise deux moities de Threshold : impossible de pretendre avoir
une evaluation finale independante sur Threshold entier. Pas de Holdout ici.

Colab:
 python -u finaliser_finetune_v3_securise.py --zip /content/P7_ESM2_colab_inputs.zip \
  --campaign /content/drive/MyDrive/P7_ESM2/campagne_finale_v3 \
  --out /content/drive/MyDrive/P7_ESM2/finalisation_finetune_v3
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
AA = set('ACDEFGHIKLMNPQRSTVWY')
HYDRO=set('AILMFWVY'); AROMA=set('FWY'); BASIC=set('KRH'); ACIDIC=set('DE'); POLAR=set('STNQ')
KD={'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
BIONAMES=['bio_log_longueur','bio_frac_hydrophobe','bio_frac_aromatique','bio_frac_basique','bio_frac_acide','bio_charge_nette_proxy','bio_frac_polaire','bio_frac_glycine','bio_frac_proline','bio_frac_cysteine','bio_frac_asn_gln','bio_plus_long_segment_hydrophobe','bio_hydropathie_moyenne']


def bio13(seq):
    n=len(seq);c=Counter(seq)
    if not n: raise ValueError('Sequence vide')
    f=lambda chars:sum(c[a] for a in chars)/n
    streak=longest=0
    for a in seq:
        streak=streak+1 if a in HYDRO else 0
        longest=max(longest,streak)
    return [np.log1p(n),f(HYDRO),f(AROMA),f(BASIC),f(ACIDIC),(c['K']+c['R']-c['D']-c['E'])/n,f(POLAR),c['G']/n,c['P']/n,c['C']/n,(c['N']+c['Q'])/n,longest/n,sum(KD.get(a,0)*v for a,v in c.items())/n]


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**22),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)


def load_threshold(zip_path):
    """Accede aux etiquettes du seul Threshold; ne lit pas celles du Holdout."""
    with zipfile.ZipFile(zip_path) as z:
        def member(name):
            matches=[x for x in z.namelist() if Path(x).name.lower()==name]
            if len(matches)!=1: raise ValueError(f'Fichier {name} absent/ambigu dans le ZIP : {matches}')
            return z.read(matches[0])
        parts=pd.read_csv(BytesIO(member('partitions.csv')),dtype={'protein_id':str})
        labels=pd.read_csv(BytesIO(member('labels.csv')),dtype={'protein_id':str})
        required={'protein_id','partition'}
        if not required.issubset(parts.columns) or not {'protein_id','label'}.issubset(labels.columns):
            raise ValueError('Colonnes partitions/labels incompletes')
        if parts.protein_id.duplicated().any() or labels.protein_id.duplicated().any():
            raise ValueError('Protein_id non uniques')
        if parts.partition.value_counts().to_dict()!={'fit':46270,'tune':4052,'calibration':4052,'threshold':4052,'holdout':4052}:
            raise ValueError('Partitions inattendues dans le ZIP')
        selected=parts.loc[parts.partition=='threshold',['protein_id']].copy()
        selected=selected.merge(labels[['protein_id','label']],on='protein_id',validate='one_to_one',how='left',sort=False)
        if selected.label.isna().any():raise ValueError('Labels Threshold manquants')
        selected.label=pd.to_numeric(selected.label,errors='raise').astype(int)
        if not selected.label.isin([0,1]).all():raise ValueError('Labels non binaires')
        needed=set(selected.protein_id)
        seq={}; key=None; chain=[]
        for line in member('sequences.fasta').decode('utf-8-sig').splitlines():
            if line.startswith('>'):
                if key in needed:seq[key]=''.join(chain).upper()
                key=line[1:].split()[0];chain=[]
            else:chain.append(line.strip())
        if key in needed:seq[key]=''.join(chain).upper()
        if set(seq)!=needed or any(not x or (set(x)-AA) for x in seq.values()):
            raise ValueError('FASTA incoherent ou residus non standards')
    return selected.reset_index(drop=True),seq


class UnionFind:
    def __init__(self,n):self.p=list(range(n));self.s=[1]*n
    def find(self,a):
        while a!=self.p[a]:
            self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def union(self,a,b):
        x=self.find(a);y=self.find(b)
        if x==y:return
        if self.s[x]<self.s[y]:x,y=y,x
        self.p[y]=x;self.s[x]+=self.s[y]


def homology_split(ids,seqs,output,identity=.25,coverage=.7,threads=4):
    """Build connected components from MMseqs2 approximate self-search.
    Split by components only, no labels involved. Refuses a giant component.
    """
    if shutil.which('mmseqs') is None:
        raise RuntimeError('Installer MMseqs2: !apt-get -qq install -y mmseqs2')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fasta=output/'threshold_only.fasta'; hits=output/'threshold_internal_hits.tsv'
    tag=[f't{i:05d}' for i in range(len(ids))]
    with fasta.open('w',encoding='utf-8') as f:
        for ident,pid in zip(tag,ids):f.write(f'>{ident}\n{seqs[pid]}\n')
    # Cache only if FASTA and parameters exactly match.
    signature=hashlib.sha256(fasta.read_bytes()+f'|{identity}|{coverage}'.encode()).hexdigest()
    manifest=output/'mmseqs_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['signature']!=signature:
            raise RuntimeError('Signature homologie changee: utiliser un nouveau --out')
        if not hits.is_file():raise RuntimeError('Manifest present mais TSV absent; verifier audit')
    else:
        with tempfile.TemporaryDirectory(prefix='mmseqs_thr_',dir='/content') as tmp:
            command=['mmseqs','easy-search',str(fasta),str(fasta),str(hits),str(Path(tmp)/'work'),
                     '--min-seq-id',str(identity),'-c',str(coverage),'--cov-mode','0',
                     '--format-output','query,target,fident,qcov,tcov,evalue','-s','7.5',
                     '--threads',str(threads),'--max-seqs','300']
            print('MMseqs Threshold interne: verification d homologie...',flush=True)
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        atomic_json(manifest,{'signature':signature,'min_seq_id':identity,'coverage':coverage,
                              'limitation':'Recherche approximative; maximum 300 resultats par query.'})
    uf=UnionFind(len(ids));count=0
    with hits.open(encoding='utf-8') as f:
        for line in f:
            fields=line.rstrip('\n').split('\t')
            if len(fields)<2:continue
            a,b=fields[:2]
            if not(a.startswith('t') and b.startswith('t')):raise ValueError('IDs MMseqs inconnus')
            uf.union(int(a[1:]),int(b[1:]));count+=1
    components={}
    for i in range(len(ids)):components.setdefault(uf.find(i),[]).append(i)
    groups=list(components.values())
    if max(map(len,groups))>len(ids)*.60:
        raise RuntimeError(f'Une composante homologue relie plus de 60% de Threshold ({max(map(len,groups))}/{len(ids)}). '
                           'Une separation independante par homologie est impossible avec ce jeu. '
                           'Obtenir une nouvelle cohorte de calibration avant de figer le seuil.')
    rng=np.random.default_rng(SEED)
    order=rng.permutation(len(groups))
    groups=[groups[i] for i in order]
    groups.sort(key=len,reverse=True)
    cal=[];thr=[]
    for g in groups:
        (cal if len(cal)<=len(thr) else thr).extend(g)
    cal=np.array(sorted(cal),dtype=int);thr=np.array(sorted(thr),dtype=int)
    if min(len(cal),len(thr))<900:raise RuntimeError('Separation par homologues trop desequilibree')
    if set(cal)&set(thr) or len(cal)+len(thr)!=len(ids):raise RuntimeError('Split invalide')
    meta={'components':len(groups),'hits':count,'max_component':max(map(len,groups)),
          'n_calibration_nouvelle':len(cal),'n_seuil_nouveau':len(thr),
          'homologie':'MMseqs recherche approchee, id>=0.25, couverture>=0.7, max 300 hits; pas de preuve exhaustive'}
    atomic_json(output/'audit_split_threshold.json',meta)
    return cal,thr,meta


def create_net(encoder,abc,torch):
    import torch.nn as nn
    pad=abc.padding_idx;clsidx=abc.cls_idx;eosidx=abc.eos_idx
    class Classifier(nn.Module):
        def __init__(self,enc):
            super().__init__();self.encoder=enc
            self.head=nn.Sequential(nn.Linear(640+13,128),nn.GELU(),nn.Dropout(.20),nn.Linear(128,1))
        def forward(self,tokens,bio):
            rep=self.encoder(tokens,repr_layers=[30],return_contacts=False)['representations'][30]
            mask=((tokens!=pad)&(tokens!=clsidx)&(tokens!=eosidx)).unsqueeze(-1)
            pool=(rep*mask).sum(1)/mask.sum(1).clamp(min=1)
            return self.head(torch.cat([pool.float(),bio],dim=1)).squeeze(-1)
    return Classifier(encoder)


def predict_logits(checkpoint,ids,seqs,out,batch=1):
    import torch,esm
    out=Path(out);file=out/'threshold_logits.npz'
    checksha=sha_file(checkpoint)
    seqsha=hashlib.sha256('\n'.join(f'{p}\t{seqs[p]}' for p in ids).encode()).hexdigest()
    if file.exists():
        with np.load(file,allow_pickle=False) as z:
            if (str(z['checkpoint_sha'])!=checksha or str(z['sequence_sha'])!=seqsha
                or z['protein_id'].astype(str).tolist()!=ids or z['logits'].shape!=(len(ids),)):
                raise RuntimeError('Cache logits non conforme')
            print('Logits Threshold reutilises, aucun GPU necessaire.',flush=True)
            return z['logits'].astype(np.float64),checksha
    if not torch.cuda.is_available():raise RuntimeError('GPU Colab necessaire pour generer les logits une fois')
    print('Chargement du checkpoint fige (pas de nouvel entrainement)...',flush=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('version')!='esm2_t30_150M_UR50D' or int(state.get('ft_maxlen',0))!=512:
        raise RuntimeError('Checkpoint ESM2/ft_maxlen non conforme')
    if state.get('bio13_columns')!=BIONAMES:raise RuntimeError('Bio13 dans le checkpoint different')
    mu=np.asarray(state['mu'],dtype=np.float32);sd=np.asarray(state['sd'],dtype=np.float32)
    if mu.shape!=(13,) or sd.shape!=(13,) or not np.isfinite(mu).all() or not (sd>0).all():
        raise RuntimeError('Normalisation Bio13 invalide')
    encoder,abc=esm.pretrained.esm2_t30_150M_UR50D()
    net=create_net(encoder,abc,torch).to('cuda').eval()
    net.load_state_dict(state['model'],strict=True)
    converter=abc.get_batch_converter(truncation_seq_length=512)
    results=np.zeros(len(ids),dtype=np.float64)
    for start in range(0,len(ids),batch):
        chosen=ids[start:start+batch]
        samples=[(name,seqs[name][:512]) for name in chosen]
        _,_,tokens=converter(samples)
        bio=np.asarray([(np.asarray(bio13(seqs[i]),np.float32)-mu)/sd for i in chosen],np.float32)
        with torch.inference_mode():
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                vals=net(tokens.cuda(),torch.from_numpy(bio).cuda()).float().cpu().numpy()
        results[start:start+len(chosen)]=vals
        if (start+len(chosen))%250==0:print(f'Logits Threshold {start+len(chosen)}/{len(ids)}',flush=True)
    if not np.isfinite(results).all():raise RuntimeError('Logits invalides')
    tmp=out/'threshold_logits.partial.npz'
    np.savez_compressed(tmp,protein_id=np.asarray(ids),logits=results,checkpoint_sha=checksha,sequence_sha=seqsha)
    os.replace(tmp,file)
    del net,encoder
    torch.cuda.empty_cache()
    return results,checksha


def threshold_min_cost(y,proba,fn_cost=1200,fp_cost=600,min_recall=.4):
    y=np.asarray(y,dtype=int); p=np.asarray(proba,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not set(np.unique(y)).issubset({0,1}) or y.sum()==0 or y.sum()==len(y):
        raise ValueError('Threshold y/p invalides')
    if not np.isfinite(p).all():raise ValueError('Probabilites invalides')
    # Explore all distinct observed probabilities (threshold positive if p >= t).
    order=np.argsort(-p,kind='mergesort'); ys=y[order]; ps=p[order]
    tp=np.cumsum(ys==1);fp=np.cumsum(ys==0);n_pos=int(y.sum())
    ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(y)-1]
    candidates=[]
    for i in ends:
        recall=float(tp[i]/n_pos)
        if recall<min_recall:continue
        fn=n_pos-int(tp[i]);fp_n=int(fp[i])
        cost=fn*fn_cost+fp_n*fp_cost
        candidates.append((cost,fp_n,-float(ps[i]),float(ps[i]),recall,fn))
    if not candidates:raise RuntimeError('Aucun seuil ne respecte le rappel minimal')
    cost,fp_n,neg_t,t,recall,fn=min(candidates)
    return {'seuil':t,'cout_euros':float(cost),'rappel_soluble':float(recall),
            'fn':int(fn),'fp':int(fp_n),'n':int(len(y)), 'min_recall':float(min_recall),
            'fn_cost':float(fn_cost),'fp_cost':float(fp_cost)}


def finalize(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
    import joblib
    campaign=Path(args.campaign);ckpt=campaign/'finetune_best.pt';status=campaign/'finetune_status.json'
    if not ckpt.exists() or not status.exists():raise FileNotFoundError('Checkpoint/status V3 manquants')
    # Do not overwrite finalized results: avoids iterative tuning on threshold.
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    done=out/'finalisation_figee.json'
    if done.exists():
        print('Finalisation deja figée:',done)
        print('Ne pas optimiser une deuxieme fois en fonction de ces resultats.')
        return
    with status.open() as f:trainstatus=json.load(f)
    script=campaign/'rapport_campagne.json'
    if script.exists():
        note=json.loads(script.read_text()).get('calibration_warning','')
        if 'Calibration' not in note:raise RuntimeError('Protocole V3 de validation non reconnaissable')
    ckptsha=sha_file(ckpt)
    frozen=out/'checkpoint_frozen.json'
    meta={'checkpoint_path':str(ckpt.resolve()),'checkpoint_sha256':ckptsha,
          'training_validation':'Calibration historique utilisee pour selection des epoques en V3',
          'new_calibration':'La moitie homologie-disjointe de Threshold',
          'new_threshold':'Autre moitie homologie-disjointe de Threshold',
          'original_holdout':'INTERDIT, deja evalue dans experience historique',
          'model':'esm2_t30_150M_UR50D','truncation_aa':512,'bio13':'13 caracteristiques sequence complete, normalisees par mu/sd checkpoint',
          'training_best_epoch':trainstatus.get('best_epoch'),'training_best_val_auc':trainstatus.get('best_auc')}
    if frozen.exists() and json.loads(frozen.read_text()).get('checkpoint_sha256')!=ckptsha:
        raise RuntimeError('Checkpoint a change depuis le gel initial')
    atomic_json(frozen,meta)
    rows,seq=load_threshold(args.zip)
    ids=rows.protein_id.astype(str).tolist();y=rows.label.to_numpy(dtype=int)
    print('Threshold original',len(ids),'proteines. Calibrateur/seuil sur moities distinctes.',flush=True)
    cal_idx,thr_idx,split_report=homology_split(ids,seq,out/'homologie_threshold',threads=args.threads)
    logits,sha=predict_logits(ckpt,ids,seq,out,batch=args.batch)
    if len(np.unique(y[cal_idx]))!=2 or len(np.unique(y[thr_idx]))!=2:
        raise RuntimeError('Classes manquantes dans une des moities Threshold')
    cal=LogisticRegression(solver='lbfgs',random_state=SEED,max_iter=1000)
    cal.fit(logits[cal_idx].reshape(-1,1),y[cal_idx])
    pcal=cal.predict_proba(logits[cal_idx].reshape(-1,1))[:,1]
    pthr=cal.predict_proba(logits[thr_idx].reshape(-1,1))[:,1]
    best=threshold_min_cost(y[thr_idx],pthr,1200,600,.40)
    yp=(pthr>=best['seuil']).astype(int)
    tn,fp,fn,tp=confusion_matrix(y[thr_idx],yp,labels=[0,1]).ravel()
    report={'model':'esm150_finetune_partiel','stage':'CALIBRATION/SEUIL INTERNE pour DEMO',
            'note':'Pas de test independant: ne jamais presenter AUC obtenue ici comme AUC finale externe.',
            'logits':'Logits du checkpoint V3, non reentrainé',
            'auc_calibration_inner':float(roc_auc_score(y[cal_idx],pcal)),
            'brier_calibration_inner':float(brier_score_loss(y[cal_idx],pcal)),
            'auc_threshold_inner':float(roc_auc_score(y[thr_idx],pthr)),
            'pr_auc_threshold_inner':float(average_precision_score(y[thr_idx],pthr)),
            'brier_threshold_inner':float(brier_score_loss(y[thr_idx],pthr)),
            'calibrator_params':{'coef':cal.coef_.tolist(),'intercept':cal.intercept_.tolist()},
            'selection':best,'confusion_threshold':{'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)},
            'homology_split':split_report,
            'N_calibration_new':int(len(cal_idx)),'N_threshold_new':int(len(thr_idx)),
            'historical_tune_auc':0.832857, 'historical_holdout_ft':'Non evalue',
            'protocol_warning':'Calibration historique consommee pour early stopping du FT V3; Threshold divise en moities.',
            'no_reuse_holdout':True}
    artifact=out/'platt_calibrator.joblib'
    joblib.dump(cal,artifact)
    atomic_json(out/'bilan_calibration_seuil.json',report)
    # Freeze only when both files written.
    atomic_json(done,{'frozen_checkpoint_sha256':sha,'calibrator_path':str(artifact),
                      'calibrator_sha256':sha_file(artifact), 'threshold':best['seuil'],
                      'result_json':str(out/'bilan_calibration_seuil.json'),
                      'created_utc':datetime.now(timezone.utc).isoformat(),
                      'no_final_independent_holdout':True})
    print('\n=== FINALISATION EXPERIMENTALE ===')
    print('Checkpoint SHA256:',sha)
    print('N calibration',len(cal_idx),'N selection seuil',len(thr_idx))
    print('Seuil fixe',best['seuil'],'Rappel',best['rappel_soluble'],'Cout',best['cout_euros'],'EUR')
    print('AUC sur moitie de Threshold pour description :',report['auc_threshold_inner'])
    print('Gel sauvegarde:',done)
    print('AUC independante du fine-tuning: INCONNUE; Holdout historique interdit.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',default='/content/P7_ESM2_colab_inputs.zip')
    ap.add_argument('--campaign',default='/content/drive/MyDrive/P7_ESM2/campagne_finale_v3')
    ap.add_argument('--out',default='/content/drive/MyDrive/P7_ESM2/finalisation_finetune_v3')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--batch',type=int,default=1)
    args=ap.parse_args()
    finalize(args)
if __name__=='__main__':main()
