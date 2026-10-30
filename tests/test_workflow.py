"""Tests V5 hors réseau, hors checkpoint et hors GPU."""
from pathlib import Path
from unittest.mock import patch
from test_advice import GFP, STATE
from test_identification import fake_post, fake_get
from solubility.protein_advice import clean_sequence, make_guide
from solubility.workflow import identify_safe, identity_html, score_html, plan_auto


def run():
    assert clean_sequence(' '+GFP[:20]+'\n'+GFP[20:]+'.') == GFP
    for bad in [GFP[:10]+'.'+GFP[10:], GFP+'**']:
        try: clean_sequence(bad)
        except ValueError: pass
        else: raise AssertionError('Caractère invalide accepté : '+bad[:50])

    with patch('solubility.protein_identifier.requests.post', side_effect=fake_post), \
         patch('solubility.protein_identifier.requests.get', side_effect=fake_get):
        # injecter l'outil réel car valeur par défaut liée à la fonction d'origine
        from solubility.protein_identifier import identify_sequence
        ident=identify_safe(GFP, search_fn=identify_sequence)
    assert ident['auto_family']=='GFP / protéine fluorescente'
    html=identity_html(ident)
    assert 'Green fluorescent protein' in html and 'UniProt P42212' in html
    md,pdf,details=plan_auto(GFP,STATE,ident)
    assert '64.7%' in md
    assert 'Fluorescence' in md
    assert 'His6' in md
    assert 'MBP' in md and 'Secours' in md
    assert Path(pdf).read_bytes()[:4]==b'%PDF'
    assert details['famille_declaree_ou_annotee']=='GFP / protéine fluorescente'
    assert len(make_guide(GFP,prediction_state=STATE,identification=ident)['induction'])==2
    score=score_html(STATE['probability_soluble'],STATE['threshold'],len(GFP))
    assert '64.7' in score and 'Soluble' in score and '29.65%' in score
    assert 'score-card positive' in score
    score_low=score_html(0.1,0.29,600)
    assert 'Insoluble' in score_low and '512' in score_low

    with patch('solubility.protein_identifier.requests.post', side_effect=ConnectionError('réseau inaccessible')):
        from solubility.protein_identifier import identify_sequence
        ident_error=identify_safe(GFP, search_fn=identify_sequence)
    assert ident_error['status']=='erreur'
    md2,pdf2,details2=plan_auto(GFP,STATE,ident_error)
    assert '64.7%' in md2 and 'Fluorescence' not in md2
    assert details2['famille_declaree_ou_annotee']=='Non précisée'
    assert Path(pdf2).read_bytes()[:4]==b'%PDF'

    other=GFP[:-1]+'A'
    try: plan_auto(other,STATE,ident)
    except ValueError: pass
    else: raise AssertionError('Ancien score réutilisé pour autre séquence')
    short=identify_safe('MKQLAI',search_fn=lambda s: (_ for _ in ()).throw(AssertionError('appel interdit')))
    assert short['status']=='non_effectue'
    print('V5 OK : un flux, identification + biochimie/PDF, refus scores obsolètes, panne réseau, contrastes')


if __name__=='__main__': run()
