"""Tests V7 hors réseau et hors GPU, y compris cohérence du PDF."""
from pathlib import Path
from length_policy_v7 import policy_for_length
from workflow_v5 import score_html
from protein_advice import make_guide, generate_recommendation

THRESHOLD = 0.29649105048292634
SEQ600 = 'ACDEFGHIKLMNPQRSTVWY' * 30
SEQ1200 = SEQ600 * 2


def run():
    assert len(SEQ600) == 600 and len(SEQ1200) == 1200
    for n, mode, limit in [
        (1,'reference_512',512), (512,'reference_512',512),
        (513,'experimental_1022',1022), (1022,'experimental_1022',1022),
        (1023,'fallback_512_over_1022',512), (1200,'fallback_512_over_1022',512),
    ]:
        p = policy_for_length(n)
        assert p.mode == mode and p.residue_limit == limit, (n,p)
    assert 'calibrée' in score_html(.646731, THRESHOLD, 238)
    experimental = score_html(.58, THRESHOLD, 600, 1022, True)
    assert 'indicatif' in experimental and 'sans recalibration' in experimental
    assert 'Analysés par ESM-2 <b>600 AA</b>' in experimental
    fallback = score_html(.58, THRESHOLD, 1200, 512, False)
    assert '512 premiers AA sur 1200' in fallback
    state600 = {'sequence': SEQ600, 'probability_soluble': .58,
                'threshold': THRESHOLD, 'inference_mode': 'experimental_1022',
                'esm2_aa_analyzed': 600, 'calibration_transfer_unverified': True}
    plan = make_guide(SEQ600, prediction_state=state600)
    assert plan['calibration_transfer_unverified']
    assert not any('seulement les 512' in x for x in plan['observations'])
    assert any('la totalité des 600' in x for x in plan['observations'])
    assert any('n’ont pas été validés' in x for x in plan['observations'])
    assert any('n’entre pas dans la priorisation' in x for x in plan['limits'])
    md, path, diag = generate_recommendation(SEQ600, 'Inconnue', 'Inconnu', 'Inconnu',
                                            'Inconnu', 'Solubilité et purification',
                                            'Non défini', state600)
    assert diag['mode_inference']=='experimental_1022'
    assert diag['acides_amines_analyses_esm2']==600
    assert diag['calibration_transferee_non_validee'] is True
    assert '1022' in md.replace('1 022', '1022')
    assert Path(path).read_bytes().startswith(b'%PDF')
    state1200 = {'sequence': SEQ1200, 'probability_soluble': .6,
                'threshold': THRESHOLD, 'inference_mode': 'fallback_512_over_1022',
                'esm2_aa_analyzed':512}
    plan2=make_guide(SEQ1200,prediction_state=state1200)
    assert any('512 premiers AA sur 1200' in x for x in plan2['observations'])
    assert not plan2['calibration_transfer_unverified']
    # Le changement de séquence invalide bien les probabilités mises en cache.
    invalid=make_guide(SEQ600[:-1]+'A',prediction_state=state600)
    assert invalid['p_pred'] is None
    print('V7 OK : 512 / 1022 / >1022, statut calibration, recommandations, PDF')

if __name__=='__main__': run()
