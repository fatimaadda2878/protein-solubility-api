"""Chaîne métier V5 : prédiction déjà calculée -> identification automatique -> plan/PDF.

Aucun poids PyTorch chargé dans ce module. Les erreurs PDB/UniProt sont non bloquantes
pour la prédiction. L'utilisateur déclenche l'envoi vers l'API par le bouton Analyser.
"""
from __future__ import annotations

import html
import re
import hashlib

from protein_advice import clean_sequence, generate_recommendation
from protein_identifier import identify_sequence, to_markdown


def cleaned(seq: str) -> str:
    return clean_sequence(seq)


def identification_status(ident: dict) -> str:
    status = ident.get('status')
    if status == 'exact':
        return 'Correspondance exacte de séquence avec une référence structurale'
    if status == 'proche':
        return 'Correspondance forte avec une référence structurale'
    if status == 'homologue':
        return 'Similarité partielle : famille non attribuée automatiquement'
    if status == 'ambigu':
        return 'Correspondances ambiguës : famille non attribuée automatiquement'
    if status == 'aucun':
        return 'Aucune référence structurale suffisamment informative trouvée'
    if status == 'non_effectue':
        return 'Recherche non effectuée pour une séquence de moins de 25 aa'
    return 'Identification momentanément indisponible'


def identify_safe(seq: str, search_fn=identify_sequence) -> dict:
    s = cleaned(seq)
    digest = hashlib.sha256(s.encode()).hexdigest()
    if len(s) < 25:
        return {'status': 'non_effectue', 'sequence': s, 'sequence_sha256': digest, 'hits': [],
                'auto_family': None, 'message': 'Recherche PDB exigeant au moins 25 acides aminés.'}
    try:
        result = search_fn(s)
        if not isinstance(result, dict) or result.get('sequence_sha256') != digest:
            raise ValueError('Correspondance de séquence invalide dans le résultat de recherche')
        return result
    except Exception as exc:
        # Le service externe est optionnel : ne pas casser la prédiction ML.
        return {'status': 'erreur', 'sequence': s, 'sequence_sha256': digest, 'hits': [],
                'auto_family': None, 'message': f'Recherche distante indisponible ({type(exc).__name__}).'}


def identity_html(ident: dict) -> str:
    esc = lambda x: html.escape(str(x), quote=True)
    status = ident.get('status', 'erreur')
    strong = status in ('exact', 'proche')
    best = ident.get('best') or {}
    tone = 'ok' if strong else 'muted'
    head = identification_status(ident)
    content = f'<div class="id-heading">{esc(head)}</div>'
    if best:
        name = esc(best.get('name') or 'Annotation indisponible')
        species = esc(best.get('organism') or 'Organisme non renseigné')
        content += f'<div class="id-protein">{name}</div><div class="id-organism">{species}</div>'
        if all(isinstance(best.get(k),(float,int)) for k in ('identity','query_coverage','reference_coverage')):
            content += ('<div class="id-metrics">'
                        f'<span>Identité <b>{best["identity"]:.1%}</b></span>'
                        f'<span>Couverture requête <b>{best["query_coverage"]:.1%}</b></span>'
                        f'<span>Couverture référence <b>{best["reference_coverage"]:.1%}</b></span>'
                        '</div>')
        links=[]
        pdb=best.get('pdb_id','')
        if re.fullmatch(r'[0-9A-Za-z]{4}',str(pdb)):
            links.append(f'<a href="https://www.rcsb.org/structure/{esc(pdb)}" target="_blank" rel="noopener noreferrer">Voir PDB {esc(pdb)}</a>')
        up=best.get('uniprot_id','')
        if re.fullmatch(r'[A-Z0-9]{6,10}',str(up)):
            links.append(f'<a href="https://www.uniprot.org/uniprotkb/{esc(up)}/entry" target="_blank" rel="noopener noreferrer">Voir UniProt {esc(up)}</a>')
        if links: content += '<div class="id-links">' + ' · '.join(links) + '</div>'
        fam=ident.get('auto_family') if strong else None
        if fam: content += f'<div class="id-family">Famille suggérée : <b>{esc(fam)}</b></div>'
    message=ident.get('message') or ''
    if message:
        content += f'<p class="id-note">{esc(message)}</p>'
    content += '<p class="id-note">L’annotation est fondée sur une similarité PDB/UniProt, pas sur une identification fonctionnelle expérimentale.</p>'
    return f'<section class="id-card {tone}">{content}</section>'


def score_html(probability: float, threshold: float, length: int,
               model_aa_limit: int = 512, experimental_1022: bool = False) -> str:
    """Carte de résultat : ne pas appeler « calibré » le score 1022 exploratoire.

    Les arguments supplémentaires laissent les appels V5/V6 existants compatibles.
    """
    if not (0 <= probability <= 1 and 0 < threshold < 1):
        raise ValueError('Probabilité ou seuil invalide')
    positive = probability >= threshold
    label = 'Soluble' if positive else 'Insoluble'
    theme = 'positive' if positive else 'negative'
    percent = 100 * probability
    processed = min(length, model_aa_limit)
    if experimental_1022:
        calibration_label = 'Score indicatif — calibrateur 512 AA transféré sans recalibration'
        warning = ('<div class="score-warning">Mode 1 022 AA expérimental : le modèle a été entraîné '
                   'avec 512 AA. Le score, la classe et le seuil ci-dessus sont indicatifs ; '
                   'la calibration à cette longueur n’a pas été validée indépendamment.</div>')
        class_label = f'{label} — indicatif'
        cutoff_label = 'Seuil historique 512 AA (indicatif)'
    else:
        calibration_label = 'Probabilité calibrée avec le protocole historique 512 AA'
        class_label = label
        cutoff_label = 'Seuil de décision figé (512 AA)'
        if length > processed:
            warning = (f'<div class="score-warning">Attention : ESM-2 analyse seulement les '
                       f'{processed} premiers AA sur {length} ; '
                       'Bio13 utilise la séquence entière. Résultat à interpréter avec prudence.</div>')
        else:
            warning = ''
    return f'''<section class="score-card {theme}">
      <div class="score-kicker">Prédiction de solubilité</div>
      <div class="score-status">{class_label} <span class="score-class">classe {int(positive)}</span></div>
      <div class="score-percentage">{percent:.1f}<span>%</span></div>
      <div class="score-explainer">{calibration_label}</div>
      <div class="score-progress" aria-label="Score de solubilité {percent:.1f} pour cent">
        <div class="score-progress-fill" style="width:{percent:.2f}%"></div>
      </div>
      <div class="score-meta"><span>{cutoff_label} <b>{threshold:.2%}</b></span>
        <span>Séquence <b>{length} AA</b></span>
        <span>Analysés par ESM-2 <b>{processed} AA</b></span>
      </div>
      {warning}
    </section>'''


def plan_auto(seq: str, prediction: dict, ident: dict,
              ptm='Inconnu', disulfides='Inconnu', membrane='Inconnu',
              objective='Solubilité et purification', host='Non défini'):
    s=cleaned(seq)
    if not isinstance(prediction,dict) or prediction.get('sequence') != s:
        raise ValueError('Prédiction non disponible pour cette séquence. Relancer l’analyse.')
    if ident.get('sequence_sha256') != hashlib.sha256(s.encode()).hexdigest():
        raise ValueError('Identification issue d’une séquence différente. Relancer l’analyse.')
    valid_ident = ident if ident.get('status') in ('exact','proche') else None
    best=(valid_ident or {}).get('best') or {}
    name = str(best.get('name') or '') if valid_ident else ''
    origin = best.get('origin_class', 'Inconnue') if valid_ident else 'Inconnue'
    if origin not in {'Inconnue','Eucaryote','Bactérienne','Virale'}:
        origin='Inconnue'
    result = generate_recommendation(s, origin, ptm, disulfides, membrane,
                                     objective, host, prediction, 'Non précisée',
                                     name, valid_ident)
    return result
