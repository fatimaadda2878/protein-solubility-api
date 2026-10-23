"""Identification déclenchée par l’analyse de la séquence, via RCSB PDB puis UniProtKB.

Ne jamais déclarer une famille sans concordance suffisante en identité ET couverture.
La PDB ne couvre pas toute UniProt : absence de hit != protéine inconnue.
"""
from __future__ import annotations

import hashlib
import html
import re
from typing import Any

import requests
from Bio.Align import PairwiseAligner

SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_ENTITY = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}'
UNIPROT = 'https://rest.uniprot.org/uniprotkb/{}.json'
AA = set('ACDEFGHIKLMNPQRSTVWY')
MAX_HITS = 8


def normalize(seq):
    s = re.sub(r'\s+', '', (seq or '').upper())
    if not s or len(s) < 25:
        raise ValueError('Recherche PDB : au moins 25 résidus nécessaires.')
    if len(s) > 10000 or set(s) - AA:
        raise ValueError('Séquence invalide : 20 acides aminés standards, 10 000 aa maximum.')
    return s


def alignment_stats(query: str, reference: str) -> dict:
    """Identité calculée sur l'alignement local, deux couvertures séparées."""
    query = normalize(query)
    reference = ''.join(c for c in reference.upper() if c in AA)
    if not reference:
        raise ValueError('Séquence de référence manquante')
    al = PairwiseAligner()
    al.mode = 'local'
    al.match_score = 2
    al.mismatch_score = -1
    al.open_gap_score = -9
    al.extend_gap_score = -0.5
    result = al.align(query, reference)[0]
    qa, ra = result.aligned
    matches = 0
    aligned = 0
    for (a,b),(c,d) in zip(qa,ra):
        assert b-a == d-c
        aligned += b-a
        matches += sum(x==y for x,y in zip(query[a:b],reference[c:d]))
    return {
        'identity': round(matches / aligned, 5) if aligned else 0.,
        'query_coverage': round(aligned / len(query), 5),
        'reference_coverage': round(aligned / len(reference), 5),
        'aligned_residues': aligned,
    }


def infer_family(name: str) -> str | None:
    """Priorités conservatrices; pas d'inférence fonctionnelle sans hit solide."""
    t = re.sub(r'[^a-z0-9 ]+', ' ', (name or '').lower())
    if any(x in t for x in ('green fluorescent protein','fluorescent protein','gfp',
                            'egfp','mcherry','mvenus')):
        return 'GFP / protéine fluorescente'
    if 'lysozyme' in t or 'muramidase' in t:
        return 'Lysozyme / enzyme muramidase'
    # Ne pas classer la thioredoxine réductase dans la famille des thioredoxines.
    if ('thioredoxin' in t or 'trx a' in t or 'trx1' in t) and not any(
        x in t for x in ('reductase','thioredoxin like domain','thioredoxin domain')):
        return 'Thioredoxine / système redox'
    if any(x in t for x in ('ubiquitin','ubiquitin like','ubiquitin b','polyubiquitin','sumo protein')) and not any(
            x in t for x in ('ubiquitin ligase','ubiquitin hydrolase','ubiquitin activating','ubiquitin conjugating')):
        return 'Ubiquitine / ubiquitine-like'
    if any(x in t for x in ('molecular chaperone','chaperonin','heat shock protein','chaperone protein',
                            'groel','dnak','hsp70','hsp90')):
        return 'Chaperonne / protéine de repliement'
    if any(x in t for x in ('immunoglobulin', 'antibody','single chain variable fragment','fab fragment')):
        return 'Anticorps / fragment liant un antigène'
    if any(x in t for x in ('membrane protein','transmembrane','membrane receptor','ion channel')):
        return 'Protéine membranaire (fonction connue)'
    if any(x in t for x in ('dna binding','dna binding protein','rna binding',
                            'transcription factor','transcriptional regulator','rna recognition')):
        return 'Liaison ADN / ARN'
    if any(x in t for x in ('enzyme','protease','kinase','hydrolase','oxidase','reductase',
                            'transferase','atpase','ligase','nuclease','polymerase','isomerase')):
        return 'Enzyme (autre)'
    return None


def uniprot_annotations(rec: dict, query_sequence: str) -> dict:
    """Extraction limitée des champs UniProtKB; preuve uniquement si alignement fort.

    Les coordonnées de features concernent la SEQUENCE de référence UniProt,
    pas automatiquement celles de la séquence saisie.
    """
    result={'sequence_verified': False, 'reviewed': False, 'feature_types': [],
            'feature_details': [], 'function': '', 'location': '', 'organism_class': 'Inconnue'}
    if not isinstance(rec,dict):
        return result
    refseq=(rec.get('sequence') or {}).get('value') or ''
    try:
        st=alignment_stats(query_sequence,refseq)
        strong=(st['identity']>=.90 and st['query_coverage']>=.90 and
                st['reference_coverage']>=.85)
    except (ValueError,IndexError):
        strong=False
    if not strong: return result
    result['sequence_verified']=True
    result['reviewed']=str(rec.get('entryType','')).lower().startswith('uniprotkb reviewed')
    org=rec.get('organism') or {}
    lineages=[str(x).lower() for x in org.get('lineage',[])]
    if 'eukaryota' in lineages:result['organism_class']='Eucaryote'
    elif any(x in lineages for x in ('bacteria','archaea')):result['organism_class']='Bactérienne'
    elif 'viruses' in lineages:result['organism_class']='Virale'
    for c in rec.get('comments') or []:
        typ=str(c.get('commentType') or '')
        if typ=='FUNCTION' and not result['function']:
            texts=c.get('texts') or []
            if texts:result['function']=str(texts[0].get('value') or '')[:320]
        if typ=='SUBCELLULAR LOCATION' and not result['location']:
            locs=c.get('subcellularLocations') or []
            names=[]
            for item in locs:
                entry=item.get('location') or {}
                if entry.get('value'):names.append(str(entry['value']))
            result['location']=', '.join(names)[:180]
    useful={'SIGNAL','TRANSMEM','INTRAMEM','DISULFID','CARBOHYD','LIPID',
            'ACT_SITE','BINDING','COFACTOR'}
    for f in rec.get('features') or []:
        typ=str(f.get('type') or '').upper().replace(' ','_')
        aliases={'TRANSMEMBRANE':'TRANSMEM', 'DISULFIDE_BOND':'DISULFID',
                 'SIGNAL_PEPTIDE':'SIGNAL','ACTIVE_SITE':'ACT_SITE',
                 'GLYCOSYLATION':'CARBOHYD','LIPIDATION':'LIPID'}
        typ=aliases.get(typ,typ)
        if typ in useful:
            result['feature_types'].append(typ)
            result['feature_details'].append({
                'type':typ,'description':str(f.get('description') or '')[:120],
                'location':f.get('location') or {}})
    result['feature_types']=sorted(set(result['feature_types']))
    result['feature_details']=result['feature_details'][:25]
    return result


def _get_json(url, timeout=12):
    response = requests.get(url, timeout=timeout, headers={'Accept':'application/json'})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def identify_sequence(seq: str) -> dict:
    """Recherche en ligne déclenchée par le bouton commun « Analyser la séquence »."""
    seq = normalize(seq)
    digest = hashlib.sha256(seq.encode()).hexdigest()
    query = {
        'query':{'type':'terminal','service':'sequence','parameters':{
            'evalue_cutoff':0.001,'identity_cutoff':0.75,
            'target':'pdb_protein_sequence','value':seq}},
        'return_type':'polymer_entity',
        'request_options':{'paginate':{'start':0,'rows':MAX_HITS},
                           'sort':[{'sort_by':'score','direction':'desc'}]}
    }
    try:
        r = requests.post(SEARCH_URL, json=query, timeout=24,
                          headers={'Content-Type':'application/json','Accept':'application/json'})
        if r.status_code in (204,404):
            return {'sequence_sha256':digest,'sequence':seq,'status':'aucun','hits':[],
                    'message':'Aucune correspondance PDB trouvée. Cela ne prouve pas que la protéine est inconnue dans UniProt.'}
        r.raise_for_status()
        response = r.json()
        hits = response.get('result_set', []) if isinstance(response, dict) else []
        candidates=[]
        uniprot_cache={}
        for hit in hits[:MAX_HITS]:
            identifier = str(hit.get('identifier',''))
            if not re.fullmatch(r'[a-zA-Z0-9]{4}_[0-9]+',identifier):
                continue
            pdb_id, entity_id = identifier.split('_',1)
            info = _get_json(RCSB_ENTITY.format(pdb_id,entity_id))
            if not isinstance(info,dict):
                continue
            e=info.get('entity_poly') or {}
            pdb_seq = str(e.get('pdbx_seq_one_letter_code_can') or '').replace('\n','')
            if not pdb_seq:
                continue
            stats=alignment_stats(seq,pdb_seq)
            desc=(info.get('rcsb_polymer_entity') or {}).get('pdbx_description') or ''
            crossrefs=(info.get('rcsb_polymer_entity_container_identifiers') or {}).get('reference_sequence_identifiers') or []
            if not crossrefs:
                crossrefs=(info.get('rcsb_polymer_entity') or {}).get('rcsb_uniprot') or []
            accessions=[]
            for ref in crossrefs:
                if not isinstance(ref,dict): continue
                d=str(ref.get('database_name','')).lower()
                accession=ref.get('database_accession') or ref.get('accession')
                if accession and ('uniprot' in d or ('database_name' not in ref and ref.get('accession'))):
                    accessions.append(str(accession).split('-')[0])
            protein_name = desc
            organism = ''
            uniprot_id = ''
            reviewed = False
            origin_class = "Inconnue"
            annotations = {}
            for acc in list(dict.fromkeys(accessions))[:2]:
                if not re.fullmatch(r'[A-Z0-9]{6,10}',acc): continue
                if acc not in uniprot_cache:
                    try:
                        uniprot_cache[acc] = _get_json(UNIPROT.format(acc),timeout=10)
                    except requests.RequestException:
                        uniprot_cache[acc] = None
                rec=uniprot_cache[acc]
                if not isinstance(rec,dict): continue
                refseq=(rec.get('sequence') or {}).get('value','')
                if refseq:
                    ref_stats=alignment_stats(seq,refseq)
                    # On ne fusionne PAS les statistiques PDB et UniProt :
                    # les métriques affichées décrivent toujours le match PDB.
                    annotations=uniprot_annotations(rec,seq)
                uniprot_id=rec.get('primaryAccession') or acc
                # Ne transposer une annotation fonctionnelle/organisme que
                # lorsque la chaîne UniProt concorde réellement avec la requête.
                if annotations.get('sequence_verified'):
                    rd=(rec.get('proteinDescription') or {}).get('recommendedName') or {}
                    protein_name=(rd.get('fullName') or {}).get('value') or protein_name
                    organism_data = (rec.get('organism') or {})
                    organism=organism_data.get('scientificName') or ''
                    lineage=[str(x).lower() for x in organism_data.get('lineage', [])]
                    if 'eukaryota' in lineage:
                        origin_class='Eucaryote'
                    elif 'bacteria' in lineage or 'archaea' in lineage:
                        origin_class='Bactérienne'
                    elif 'viruses' in lineage:
                        origin_class='Virale'
                    reviewed=annotations.get('reviewed',False)
                break
            family=infer_family(protein_name) or infer_family(desc)
            candidates.append({
                'pdb_id':pdb_id,'entity_id':entity_id,'name':str(protein_name)[:180],
                'pdb_description':str(desc)[:180], 'organism':str(organism)[:100],
                'uniprot_id':uniprot_id, 'reviewed':bool(reviewed),'family':family,
                'origin_class':origin_class,
                'annotation':annotations if annotations.get('sequence_verified') else {},
                'pdb_url':f'https://www.rcsb.org/structure/{pdb_id}',
                'uniprot_url':f'https://www.uniprot.org/uniprotkb/{uniprot_id}/entry' if uniprot_id else None,
                **stats
            })
        candidates.sort(key=lambda h:(min(h['query_coverage'],h['reference_coverage'])*h['identity'],h['reviewed']),reverse=True)
        if not candidates:
            return {'sequence_sha256':digest,'sequence':seq,'status':'aucun','hits':[],
                    'message':'La recherche ne fournit aucune annotation exploitable. Vérifier manuellement UniProt.'}
        best=candidates[0]
        strong=(len(seq)>=60 and best['identity']>=0.9 and best['query_coverage']>=0.9 and best['reference_coverage']>=0.85)
        exact=(best['identity']>=0.999 and best['query_coverage']>=0.999 and best['reference_coverage']>=0.999)
        similar=[h for h in candidates[1:] if h['identity']>=best['identity']-.035 and
                 h['query_coverage']>=best['query_coverage']-.08 and
                 h['reference_coverage']>=best['reference_coverage']-.08]
        conflicting=any(h['family'] and best['family'] and h['family']!=best['family'] for h in similar)
        if conflicting:
            status='ambigu'
        elif exact:
            status='exact'
        elif strong:
            status='proche'
        else:
            status='homologue'
        auto_family=best['family'] if status in ('exact','proche') else None
        return {'sequence_sha256':digest,'sequence':seq,'status':status,
                'auto_family':auto_family,'best':best,'hits':candidates[:5],
                'message':('Correspondance vérifiée par alignement local. Identité et couvertures quantifiées. '
                           'Une annotation homologue ne garantit pas une fonction identique. '+
                           'La PDB ne couvre pas l’ensemble de UniProt.')}
    except requests.RequestException as ex:
        return {'sequence_sha256':digest,'sequence':seq,'status':'erreur','hits':[],
                'message':f'Service externe indisponible ({type(ex).__name__}). Les conseils restent fondés sur la séquence et le score ML.'}


def to_markdown(result: dict) -> str:
    esc=lambda s: html.escape(str(s))
    lines=['### Identification par similarité de séquence',
           f"**Statut : {esc(result.get('status','?'))}** — {esc(result.get('message',''))}"]
    best=result.get('best')
    if best:
        lines += [f"**Meilleure correspondance : {esc(best['name'])}**" +
                  (f" — *{esc(best['organism'])}*" if best['organism'] else ''),
                  f"Identité alignée : **{best['identity']:.1%}** ; "
                  f"couverture de ta séquence : **{best['query_coverage']:.1%}** ; "
                  f"couverture de la référence : **{best['reference_coverage']:.1%}**.",
                  f"[Structure PDB {esc(best['pdb_id'])}]({best['pdb_url']})"+
                  (f" · [UniProt {esc(best['uniprot_id'])}]({best['uniprot_url']})" if best['uniprot_url'] else '')]
        if result.get('auto_family'):
            lines.append(f"Famille proposée automatiquement : **{esc(result['auto_family'])}**. Vérifie l’annotation avant un usage expérimental.")
        else:
            lines.append('**Aucune famille appliquée automatiquement** : résultat trop partiel, ambigu ou catégorie non prise en charge.')
    return '\n\n'.join(lines)
