"""Tests sans réseau : simule les API RCSB/UniProt. """
from pathlib import Path
from unittest.mock import patch
from protein_identifier import identify_sequence, alignment_stats, infer_family, to_markdown
from protein_advice import make_guide, generate_recommendation
from test_conseils_v3 import GFP, STATE

class FakeResponse:
    def __init__(self, payload, code=200):
        self.payload=payload;self.status_code=code
    def raise_for_status(self):
        if self.status_code>=400: raise RuntimeError(self.status_code)
    def json(self):return self.payload

def fake_post(url, **kwargs):
    assert kwargs['json']['return_type']=='polymer_entity'
    assert kwargs['json']['query']['service']=='sequence'
    return FakeResponse({'result_set':[{'identifier':'1EMA_1','score':99.0}]})

def fake_get(url, **kwargs):
    if '/polymer_entity/' in url:
        return FakeResponse({'entity_poly':{'pdbx_seq_one_letter_code_can':GFP},
          'rcsb_polymer_entity':{'pdbx_description':'Green fluorescent protein'},
          'rcsb_polymer_entity_container_identifiers':{
            'reference_sequence_identifiers':[{'database_name':'UniProt','database_accession':'P42212'}]}})
    if '/P42212.json' in url:
        return FakeResponse({'primaryAccession':'P42212','entryType':'UniProtKB reviewed (Swiss-Prot)',
          'proteinDescription':{'recommendedName':{'fullName':{'value':'Green fluorescent protein'}}},
          'organism':{'scientificName':'Aequorea victoria'},'sequence':{'value':GFP}})
    raise AssertionError(url)

def test_identifier():
    assert infer_family('Green fluorescent protein') == 'GFP / protéine fluorescente'
    s=alignment_stats(GFP,GFP)
    assert s['identity']==1 and s['query_coverage']==1
    variant = GFP[:30] + 'A' + GFP[31:]
    st=alignment_stats(variant,GFP)
    assert st['identity']>.99 and st['query_coverage']==1
    with patch('protein_identifier.requests.post',side_effect=fake_post), \
         patch('protein_identifier.requests.get',side_effect=fake_get):
        result=identify_sequence(GFP)
    assert result['status']=='exact'
    assert result['auto_family']=='GFP / protéine fluorescente'
    assert result['best']['uniprot_id']=='P42212'
    assert 'UniProt P42212' in to_markdown(result)
    g=make_guide(GFP,'Eucaryote','Non','Non','Non','Activité biologique','E. coli BL21(DE3)',STATE,
                 'Non précisée','',result)
    assert g['family']=='GFP / protéine fluorescente'
    assert any('fluorescen' in x.lower() for x in g['validation'])
    assert any('UniProt P42212' in x[0] for x in g['references'])
    md,pdf,diag=generate_recommendation(GFP,'Eucaryote','Non','Non','Non','Activité biologique','E. coli BL21(DE3)',STATE,
                                         'Non précisée','',result)
    assert Path(pdf).read_bytes()[:4]==b'%PDF'
    assert diag['famille_declaree_ou_annotee']=='GFP / protéine fluorescente'
    # Une simple région de 70 aa correspondant à la GFP ne suffit pas à annoter toute la cible.
    frag = GFP[:70]
    with patch('protein_identifier.requests.post', side_effect=fake_post), \
         patch('protein_identifier.requests.get', side_effect=fake_get):
        partial = identify_sequence(frag)
    assert partial['status'] == 'homologue', partial
    assert partial['auto_family'] is None
    # Aucun hit : ne pas inventer une identité.
    with patch('protein_identifier.requests.post', return_value=FakeResponse({'result_set': []})):
        unknown=identify_sequence(GFP)
    assert unknown['status']=='aucun' and not unknown['hits']
    # Réutilisation d'une identification sur une AUTRE séquence bloquée.
    changed = GFP[:-1] + ('A' if GFP[-1]!='A' else 'G')
    changed_guide = make_guide(changed, prediction_state=None, identification=result)
    assert changed_guide['family'] == 'Non précisée'
    print('Test V4 : identification auto, biochimie et PDF : OK')

if __name__=='__main__': test_identifier()
