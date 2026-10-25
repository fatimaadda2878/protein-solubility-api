"""Tests rapides métier, sans charger le checkpoint ESM-2."""
from pathlib import Path
from protein_advice import make_guide, generate_recommendation, clean_sequence

GFP = ('MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK')
assert len(GFP)==238
STATE = {'sequence':GFP,'probability_soluble':0.646731,'threshold':0.29649105,'class_predicted':1}

def test_all():
    g = make_guide(GFP, 'Eucaryote','Non','Non','Non', 'Activité biologique','E. coli BL21(DE3)', STATE, 'GFP / protéine fluorescente', 'GFP P42212')
    assert g['p_pred'] == 0.646731
    assert len([x for x in g['tags'] if x['niveau']=='Priorité 1']) == 1
    assert 'His6' in g['tags'][0]['essai']
    assert not any('MBP' in x['essai'] and x['niveau']=='Priorité 1' for x in g['tags'])
    assert 'E. coli' in g['hosts'][0]['systeme']
    assert any('fluorescen' in x.lower() for x in g['validation'])
    assert len(g['induction']) == 2  # V6 : criblage GFP plus ciblé
    md, pdf, diag=generate_recommendation(GFP,'Eucaryote','Non','Non','Non','Activité biologique','E. coli BL21(DE3)',STATE,'GFP / protéine fluorescente','GFP P42212')
    assert Path(pdf).read_bytes()[:4] == b'%PDF'
    assert '64.7%' in md or '64,7' in md
    assert diag['prediction_ml_reutilisee']
    assert diag['tags_prioritaires'] == ['His6 clivable (construction simple de référence)']
    assert 'Fluorescence' in md or 'fluorescen' in md.lower()
    # Faux appariement sequence-proba : score non réutilisé.
    wrong = make_guide(GFP,'Eucaryote','Non','Non','Non','Activité biologique', 'E. coli BL21(DE3)', {'sequence':'OTHER','probability_soluble':0.999}, 'GFP / protéine fluorescente','')
    assert wrong['p_pred'] is None
    # Lysozyme : redox + essai de muramidase (jamais de fluorescence GFP)
    lysozyme = 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL'
    ly = make_guide(lysozyme,'Eucaryote','Non','Oui (requis)','Non','Activité biologique','E. coli SHuffle/oxydante', None, 'Lysozyme / enzyme muramidase','P00698')
    assert 'SHuffle' in ly['hosts'][0]['systeme']
    assert any('muramidase' in v.lower() for v in ly['validation'])
    assert not any('Fluorescen' in v for v in ly['validation'])
    # faible score sur protéine cytosolique générique : MBP comparatif conditionnel
    low = make_guide('MKQLEDKVEELLSKNYHLENEVARLKKLV', 'Bactérienne','Non','Non','Non', 'Solubilité et purification','E. coli BL21(DE3)', {'sequence':'MKQLEDKVEELLSKNYHLENEVARLKKLV','probability_soluble':0.12,'threshold':0.29649}, 'Non précisée','')
    assert any('MBP' in z['essai'] and z['niveau'].startswith('Priorité 2') for z in low['tags'])
    # mammalian PTM : pas d'IPTG et HEK293 prioritaire
    mammal = make_guide(GFP,'Eucaryote','Oui (nécessaires pour activité)','Non','Non','Activité biologique','HEK293 / cellules mammifères',STATE,'Enzyme (autre)','')
    assert mammal['induction']==[] and 'HEK293' in mammal['hosts'][0]['systeme']
    for bad in ('', '>a\nAA', 'AA*', 'AAAAZ'):
        try: clean_sequence(bad)
        except ValueError: pass
        else: raise AssertionError('Not rejected: '+bad)
    print('Tests V3 métiers + export PDF : OK',pdf)

if __name__=='__main__': test_all()
