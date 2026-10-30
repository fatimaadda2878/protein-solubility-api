"""Tests hors réseau et hors GPU : plans spécifiques, provenance des annotations, PDF."""
from pathlib import Path
from unittest.mock import patch
from solubility.protein_identifier import infer_family, uniprot_annotations, identify_sequence
from solubility.protein_advice import make_guide, generate_recommendation, to_markdown
from test_advice import GFP, STATE

TRXA = 'MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA'
UBQ = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
THR = 0.29649105

class Response:
    def __init__(self,payload): self.payload=payload;self.status_code=200
    def json(self):return self.payload
    def raise_for_status(self):pass

def record(seq,name,organism,annotations=None,entry='UniProtKB reviewed (Swiss-Prot)'):
    return {'primaryAccession':'P0AA25', 'entryType': entry,
            'proteinDescription':{'recommendedName':{'fullName':{'value':name}}},
            'organism':{'scientificName':organism,'lineage':['Bacteria'] if organism.startswith('Escherichia') else ['Eukaryota']},
            'sequence': {'value': seq},
            'comments': (annotations or {}).get('comments', []),
            'features': (annotations or {}).get('features',[])}

def match(seq, name,uniprot):
    def post(url,**k):
        return Response({'result_set':[{'identifier':'1XYZ_1'}]})
    def get(url,**k):
        if '/polymer_entity/' in url:
            return Response({'entity_poly':{'pdbx_seq_one_letter_code_can':seq},
                             'rcsb_polymer_entity':{'pdbx_description':name},
                             'rcsb_polymer_entity_container_identifiers':{
                                 'reference_sequence_identifiers':[{'database_name':'UniProt','database_accession':'P0AA25'}]}})
        return Response(uniprot)
    with patch('solubility.protein_identifier.requests.post',side_effect=post), patch('solubility.protein_identifier.requests.get',side_effect=get):
        return identify_sequence(seq)

def run():
    assert infer_family('Thioredoxin-1')=='Thioredoxine / système redox'
    assert infer_family('Thioredoxin reductase')=='Enzyme (autre)'
    assert infer_family('Ubiquitin')=='Ubiquitine / ubiquitine-like'
    assert infer_family('Ubiquitin ligase E3')=='Enzyme (autre)'
    trx = record(TRXA,'Thioredoxin-1','Escherichia coli',{
        'comments':[{'commentType':'FUNCTION','texts':[{'value':'Participates in thiol-disulfide exchange.'}]},
                    {'commentType':'SUBCELLULAR LOCATION','subcellularLocations':[{'location':{'value':'Cytoplasm'}}]}],
        'features':[{'type':'Active site','description':'Cys redox','location':{'start':{'value':32}}}]})
    ident=match(TRXA,'Thioredoxin-1',trx)
    assert ident['status']=='exact',ident
    assert ident['auto_family']=='Thioredoxine / système redox'
    assert ident['best']['annotation']['sequence_verified']
    assert ident['best']['annotation']['reviewed']
    assert 'ACT_SITE' in ident['best']['annotation']['feature_types']
    state={'sequence':TRXA,'probability_soluble':0.352,'threshold':THR}
    tr = make_guide(TRXA,prediction_state=state,identification=ident)
    assert 'thioredoxine' in tr['family'].lower()
    assert 'His6' in tr['tags'][0]['essai'] and 'native' in tr['tags'][0]['essai'].lower()
    assert 'SHuffle' not in tr['hosts'][0]['systeme']
    assert any('redox' in s.lower() for s in tr['validation'])
    assert any('proche du seuil' in s.lower() for s in tr['limits'])
    assert len(tr['induction'])==2
    assert 'ACT_SITE' in str(ident['best']['annotation'])
    assert 'Cytoplasm' in str(tr['observations'])
    pdf_md, pdf_file, indicators = generate_recommendation(
        TRXA,'Inconnue','Inconnu','Inconnu','Inconnu',
        'Solubilité et purification','Non défini',state,
        'Non précisée','',ident)
    assert Path(pdf_file).read_bytes()[:4] == b'%PDF'
    assert '35.2%' in pdf_md and 'redox' in pdf_md.lower()
    assert indicators['conseils_personnalises_par_annotations']

    ubi=record(UBQ,'Ubiquitin','Homo sapiens')
    ubq_ident=match(UBQ,'Ubiquitin',ubi)
    ug=make_guide(UBQ,prediction_state={'sequence':UBQ,'probability_soluble':0.60,'threshold':THR},identification=ubq_ident)
    assert ug['family']=='Ubiquitine / ubiquitine-like'
    assert 'C-terminale' in ' '.join(row['pourquoi'] for row in ug['tags'])
    assert 'E. coli' in ug['hosts'][0]['systeme']
    assert any('di-Gly' in x for x in ug['validation'])
    assert len(ug['induction'])==2

    # Protéine sans identité annotée : MBP seulement hypothèse conditionnelle.
    unknown='MKQLEDKVEELLSKNYHLENEVARLKKLV'
    gg=make_guide(unknown,prediction_state={'sequence':unknown,'probability_soluble':0.10,'threshold':THR})
    assert 'His6' in gg['tags'][0]['essai']
    assert any('MBP' in x['essai'] for x in gg['tags'])
    assert any('inconnue' in x.lower() or 'non déterminée' in x.lower() for x in gg['limits']+gg['decision_rationale'])
    # Annotation UniProt incompatible : NE PAS transférer annotations structurales.
    fake=record('A'*120,'Membrane protein','Escherichia coli', {'features':[{'type':'Transmembrane'}]})
    rejected=uniprot_annotations(fake, TRXA)
    assert rejected['sequence_verified'] is False and rejected['feature_types']==[]
    # Annotation fiable de domaine transmembranaire : primer sur score favorable.
    mem_rec=record(TRXA,'Membrane protein','Escherichia coli', {'features':[{'type':'Transmembrane','description':'TM'}]})
    mem=match(TRXA,'Membrane protein',mem_rec)
    mg=make_guide(TRXA,prediction_state={'sequence':TRXA,'probability_soluble':0.8,'threshold':THR},identification=mem)
    assert 'membranaire' in mg['family'].lower()
    assert 'Sf9' in mg['hosts'][0]['systeme']
    assert 'membranaire' in mg['decision_rationale'][0].lower()
    unreviewed=record(TRXA,'Thioredoxin-1','Escherichia coli',entry='UniProtKB unreviewed (TrEMBL)')
    assert uniprot_annotations(unreviewed,TRXA)['reviewed'] is False
    print('V6 OK : TrxA / ubiquitine / membrane / annot exactes / inconnu / PDF')

if __name__=='__main__': run()
