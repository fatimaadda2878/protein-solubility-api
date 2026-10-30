"""Heuristiques transparentes d'aide à la conception de constructions protéiques.

NE PRÉDIT PAS un rendement, une solubilité après ajout d'un tag ni une température optimale.
Cette logique est volontairement indépendante du checkpoint ESM-2.
"""
from __future__ import annotations

import hashlib
import html
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from solubility.expert_rules import decide

AA = set("ACDEFGHIKLMNPQRSTVWY")
KD = {"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,
      "I":4.5,"K":-3.9,"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,
      "R":-4.5,"S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3}
HYDRO = set("AILMFWVY")
SOURCES = [
    ("NEB — Why is the induced protein insoluble?", "https://www.neb.com/faqs/why-is-the-induced-protein-insoluble-c2530"),
    ("NEB — Fusion Protein Expression: optimisation", "https://www.neb.com/en-gb/protocols/fusion-protein-expression-e6901"),
    ("NEB — Target Protein Insolubility", "https://www.neb.com/en-fr/applications/protein-expression/expression-of-difficult-proteins/target-protein-insolubility"),
]


def clean_sequence(raw: str) -> str:
    """Accepte une séquence brute, espaces/retours ; interdit FASTA multi-entrées."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Saisir une séquence protéique.")
    if ">" in text:
        raise ValueError("Coller uniquement les acides aminés, sans en-tête FASTA (>).")
    # Tolérer un unique point final de ponctuation, mais jamais un point interne.
    if text.endswith('.'):
        text = text[:-1].rstrip()
    seq = re.sub(r"\s+", "", text).upper()
    if len(seq) > 10000:
        raise ValueError("Séquence trop longue : limite de 10 000 acides aminés.")
    if not seq:
        raise ValueError("Séquence protéique vide après nettoyage.")
    wrong = set(seq) - AA
    if wrong:
        raise ValueError("Résidus non reconnus : " + ", ".join(sorted(wrong)))
    return seq


def features(seq: str) -> dict:
    n = len(seq)
    counts = Counter(seq)
    kd_mean = sum(KD[c] for c in seq) / n
    hydrophobic_fraction = sum(counts[c] for c in HYDRO) / n
    # Détection très approximative de fenêtres hydrophobes (ne remplace pas TMHMM/SignalP).
    windows = []
    if n >= 19:
        for i in range(0, n - 19 + 1):
            win = seq[i:i+19]
            mean = sum(KD[c] for c in win) / 19
            if mean >= 1.6:
                windows.append(i + 1)
    return {
        "length": n,
        "n_cys": counts["C"],
        "cys_pct": 100 * counts["C"] / n,
        "hydrophobic_pct": 100 * hydrophobic_fraction,
        "mean_kd": kd_mean,
        "hydrophobic_windows_19aa": len(windows),
        "n_terminal_hydrophobic_window": any(i <= 35 for i in windows),
        "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
        "model_truncates_at_512": n > 512,
    }


def make_guide(seq_input: str, origin: str = "Inconnue", ptm: str = "Inconnu",
               disulfides: str = "Inconnu", membrane: str = "Inconnu",
               objective: str = "Solubilité et purification", host: str = "Non défini",
               prediction_state: dict | None = None,
               family: str = "Non précisée", protein_name: str = "", identification: dict | None = None) -> dict:
    """Plan hiérarchisé, contexte déclaré ou famille issue d’une annotation fiable PDB/UniProt."""
    seq = clean_sequence(seq_input)
    f = features(seq)
    fam = family or "Non précisée"
    # Vérifier que le résultat distant appartient exactement à la séquence en cours.
    ident = (identification if isinstance(identification, dict)
             and identification.get('sequence_sha256') == f['sequence_sha256']
             and identification.get('status') in ('exact','proche') else None)
    if fam == 'Non précisée' and ident and ident.get('auto_family'):
        fam = ident['auto_family']
    p = None
    threshold = None
    if isinstance(prediction_state, dict) and prediction_state.get("sequence") == seq:
        v = prediction_state.get("probability_soluble")
        t = prediction_state.get("threshold", 0.29649105048292634)
        if isinstance(v, (float, int)) and 0 <= float(v) <= 1 and isinstance(t, (float, int)) and 0 < float(t) < 1:
            p = float(v)
            threshold = float(t)
    # La V7 n'extrapole pas la calibration 512 sans le signaler.
    prediction_mode = (prediction_state.get('inference_mode', 'reference_512')
                       if isinstance(prediction_state, dict) and prediction_state.get('sequence') == seq
                       else 'reference_512')
    calibration_unverified = prediction_mode == 'experimental_1022'
    analysed_aa = (prediction_state.get('esm2_aa_analyzed')
                   if isinstance(prediction_state, dict) and prediction_state.get('sequence') == seq
                   else None)
    if not isinstance(analysed_aa, int) or not 1 <= analysed_aa <= len(seq):
        analysed_aa = min(len(seq), 512)
    score_low = p is not None and p < threshold
    gfp = fam == "GFP / protéine fluorescente"
    lys = fam == "Lysozyme / enzyme muramidase"
    enzyme = fam == "Enzyme (autre)"
    ab = fam == "Anticorps / fragment liant un antigène"
    fam_membrane = fam == "Protéine membranaire (fonction connue)"
    disulf = disulfides.startswith("Oui")
    membr = membrane.startswith("Oui") or fam_membrane
    ptm_required = ptm.startswith("Oui")
    # L'origine eucaryote n'impose PAS HEK293 à elle seule (ex. GFP).
    host_ecoli = host.startswith("E. coli")
    compatible_ecoli = (not ptm_required and not membr and not ab and not disulf)
    declared_ecoli_t7 = host_ecoli

    context = {"origine": origin, "ptm": ptm, "ponts_disulfure": disulfides,
               "proteine_membranaire": membrane, "objectif": objective,
               "hote_actuel": host, "famille_suggeree": fam,
               "nom_identifie": (protein_name or "Non identifié").strip()[:120],
               "origine_famille": "PDB/UniProt (correspondance forte)" if ident and ident.get('auto_family') and family == 'Non précisée' else "non déterminée automatiquement"}
    observations = [
        f"Longueur : {f['length']} aa ; Cys : {f['n_cys']} ({f['cys_pct']:.1f} %) ; résidus hydrophobes : {f['hydrophobic_pct']:.1f} %.",
        f"Indice de Kyte-Doolittle moyen : {f['mean_kd']:.2f} (descripteur seulement).",
        "Famille fonctionnelle proposée : « " + fam + " » (" + ("annotation PDB/UniProt à vérifier" if ident and ident.get('auto_family') and family == 'Non précisée' else "pas d'annotation fiable") + ").",
    ]
    if ident and ident.get('best'):
        h = ident['best']
        observations.append(f"Correspondance PDB {h['pdb_id']}" +
                            (f" / UniProt {h['uniprot_id']}" if h.get('uniprot_id') else '') +
                            f" : identité {h['identity']:.1%}, couvertures requête {h['query_coverage']:.1%} et référence {h['reference_coverage']:.1%}.")
    if f['hydrophobic_windows_19aa']:
        observations.append(f"{f['hydrophobic_windows_19aa']} fenêtre(s) hydrophobe(s) de 19 aa : confirmer avec une méthode spécialisée pour les segments transmembranaires.")
    if f['n_terminal_hydrophobic_window']:
        observations.append("Fenêtre N-terminale hydrophobe : vérifier l'existence d'un peptide signal ou d'un segment transmembranaire.")
    if f['n_cys'] >= 4 and disulfides == "Inconnu":
        observations.append("Plusieurs cystéines présentes : ne pas en déduire automatiquement des ponts disulfure fonctionnels.")
    if analysed_aa < len(seq):
        observations.append(f'ESM-2 analyse seulement les {analysed_aa} premiers AA sur {len(seq)} ; '
                            'Bio13 utilise la séquence entière.')
    else:
        observations.append(f'ESM-2 analyse la totalité des {len(seq)} AA ; Bio13 utilise aussi la séquence entière.')
    if calibration_unverified:
        observations.append('ATTENTION — inférence jusqu’à 1 022 AA exploratoire : '
                            'le calibrateur et le seuil historiques à 512 AA n’ont pas été validés pour cette fenêtre.')
    if p is None:
        observations.append("Score ESM-2 indisponible pour cette séquence.")
    else:
        descriptor = 'Score ESM-2 indicatif' if calibration_unverified else 'Score ESM-2 calibré (protocole 512 AA)'
        observations.append(f"{descriptor} pour CETTE séquence : {p:.1%}, seuil historique {threshold:.2%} ; "
                            f"classe {'insoluble' if score_low else 'soluble'}"
                            f"{' (indicative)' if calibration_unverified else ''}. "
                            "Ce score ne prédit pas l'effet du tag.")

    # V6 : recommandation fondée sur la famille, les features UniProt vérifiées,
    # les propriétés de la séquence et les contraintes réellement déclarées.
    annotation = (ident.get('best') or {}).get('annotation') if ident else None
    f['model_truncates_at_512'] = analysed_aa == 512 and len(seq) > 512
    f['model_aa_analyzed'] = analysed_aa
    f['inference_mode'] = prediction_mode
    # Ne pas utiliser une probabilité NON recalibrée à 1022 AA pour décider
    # automatiquement de renforcer les stratégies de sauvetage de solubilité.
    # Les propriétés biologiques et annotations restent disponibles.
    decision_score = None if calibration_unverified else p
    advice = decide(fam, f, decision_score, threshold, ptm, disulfides, membrane,
                    objective, host, annotation)
    tags = advice['tags']
    hosts = advice['hosts']
    context['annotation_uniprot_applicable'] = ('oui, après vérification de séquence'
                                              if advice['annotation_used'] else 'non / non disponible')
    observations.extend(advice['evidence'])
    warnings_v6 = advice['warnings']
    reasons_v6 = advice['rationale']
    # La reconnaissance de famille n'équivaut PAS à une certification biologique.
    if ident and ident.get('status') == 'proche':
        observations.append('Fonction proposée par homologie forte, à vérifier avant de la transposer à la cible.')

    induction = []
    # L'identification automatique peut suggérer un point de départ T7/IPTG,
    # sans présumer que le vecteur utilisé possède un promoteur T7.
    t7_comparaison_hypothetique = (host == 'Non défini' and
                                   not any(x in str(hosts[0]['systeme']).lower()
                                           for x in ('sf9','hek293','périplasme','shuf','membran')) and
                                   ('e. coli' in str(hosts[0]['systeme']).lower()) and
                                   ident is not None and ident.get('auto_family') is not None)
    if (declared_ecoli_t7 or t7_comparaison_hypothetique) and not ptm_required and not membr and not ('TRANSMEM' in set((annotation or {}).get('feature_types') or [])):
        if fam == 'GFP / protéine fluorescente':
            induction = [
                {'essai':'Référence GFP','temperature':'30 °C','iptg':'0,2 mM','duree':'4–6 h'},
                {'essai':'Si maturation ou agrégation difficile','temperature':'20 °C','iptg':'0,1 mM','duree':'12–18 h'},
            ]
        elif fam == 'Thioredoxine / système redox':
            induction = [
                {'essai':'Référence TrxA','temperature':'30 °C','iptg':'0,1–0,2 mM','duree':'4–6 h'},
                {'essai':'Alternative si qualité insuffisante','temperature':'20 °C','iptg':'0,1 mM','duree':'12–16 h'},
            ]
        elif fam == 'Ubiquitine / ubiquitine-like':
            induction = [
                {'essai':'Référence ubiquitine','temperature':'30 °C','iptg':'0,1–0,2 mM','duree':'4–6 h'},
                {'essai':'Contrôle alternatif','temperature':'25 °C','iptg':'0,1 mM','duree':'6–8 h'},
            ]
        elif lys or disulf:
            induction = [
                {'essai':'Essai de repliement tempéré','temperature':'20 °C','iptg':'0,1 mM','duree':'12–16 h'},
                {'essai':'Comparateur','temperature':'25 °C','iptg':'0,1 mM','duree':'6–8 h'},
            ]
        else:
            induction = [
                {'essai':'Référence modérée','temperature':'30 °C','iptg':'0,2 mM','duree':'4–6 h'},
                {'essai':'Faible induction','temperature':'18 °C','iptg':'0,1 mM','duree':'12–18 h'},
                {'essai':'Intermédiaire','temperature':'25 °C','iptg':'0,1–0,2 mM','duree':'6–8 h'},
            ]
        induction_note = ('Exemples conditionnels UNIQUEMENT si un vecteur T7/IPTG est utilisé dans E. coli ; '
                          'ce ne sont ni des paramètres optimaux calculés, ni un protocole universel. '
                          'Inclure témoin non induit et comparer rendement, activité et fraction soluble.')
    else:
        induction_note = ('Pas de conditions IPTG par défaut : choisir l’hôte et le vecteur puis adapter les '
                          'paramètres d’expression. Sf9 et HEK293 ne reposent pas sur l’induction bactérienne T7/IPTG.')

    constructs = ['Comparer d’abord une construction simple et un essai d’expression à petite échelle avant de complexifier les fusions.']
    ann_flags = set((annotation or {}).get('feature_types') or []) if advice['annotation_used'] else set()
    if membr or 'TRANSMEM' in ann_flags or f['n_terminal_hydrophobic_window']:
        constructs.append('Valider architecture de domaines / segments TM / peptide signal par des outils dédiés avant de choisir les frontières de construction.')
    if lys or disulf or 'DISULFID' in ann_flags:
        constructs.append('Vérifier le nombre et la nécessité des ponts disulfure avant d’optimiser le système redox ou les constructions.')
    if objective == 'Cristallographie / biophysique':
        constructs.append('Prioriser monodispersité, homogénéité et stabilité : le rendement soluble seul est insuffisant.')
    elif objective == 'Activité biologique':
        constructs.append('Contrôler l’activité et l’état oligomérique après purification et après clivage éventuel du tag.')

    tests = [
        'SDS-PAGE : fractions totales, solubles et insolubles à biomasse comparable ; quantifier la fraction soluble.',
        'Rendement, pureté, monodispersité/SEC et stabilité ; comparer au moins deux conditions avec témoins.',
    ]
    specific = advice['specific_validation']
    tests.extend(specific)
    if not specific:
        tests.append('Pour une protéine non identifiée, définir et valider un essai fonctionnel adapté après annotation indépendante.')

    caveats = [
        'La famille est proposée par similarité PDB/UniProt si la correspondance est forte. Une annotation similaire n’établit pas une fonction identique.',
        'Les recommandations sont des hypothèses biochimiques et ne sont pas des sorties du modèle ESM-2.',
        'La probabilité ESM-2 est obtenue sans tag ; elle ne prédit ni le rendement ni l’amélioration après fusion.',
        'L’hydropathie de la séquence seule n’établit pas la présence d’un segment TM ou d’un peptide signal.',
        'Les conditions d’induction sont des exemples de criblage ; elles ne sont pas prédites.',
    ]
    if calibration_unverified:
        caveats.append('Mode 1 022 AA exploratoire : la calibration et le seuil n’ont pas été refaits. '
                       'Ce score n’entre pas dans la priorisation des tags et des hôtes ; '
                       'une calibration et une validation externes sont nécessaires avant utilisation décisionnelle.')
    caveats.extend(warnings_v6)
    refs = list(SOURCES)
    if ident and ident.get('best'):
        h = ident['best']
        refs.append((f"RCSB PDB {h['pdb_id']} — séquence correspondante", h['pdb_url']))
        if h.get('uniprot_url'):
            refs.append((f"UniProt {h['uniprot_id']} — annotation de référence", h['uniprot_url']))
    if gfp:
        refs.append(('UniProt GFP P42212','https://www.uniprot.org/uniprotkb/P42212/entry'))
        refs.append(('GFP : maturation et fluorescence (étude)','https://pmc.ncbi.nlm.nih.gov/articles/PMC19369/'))
    if lys:
        refs.append(('UniProt lysozyme blanc d’œuf P00698','https://www.uniprot.org/uniprotkb/P00698/entry'))
    return {'sequence':seq,'features':f,'context':context,'observations':observations,
            'tags':tags,'hosts':hosts,'induction':induction,'induction_note':induction_note,
            'constructs':constructs,'validation':tests,'limits':caveats,'references':refs,
            'decision_rationale':reasons_v6,'annotation_evidence':advice['evidence'],
            'p_pred':p,'threshold':threshold,'family':fam,'identification':ident,
            'inference_mode':prediction_mode,'esm2_aa_analyzed':analysed_aa,
            'calibration_transfer_unverified':calibration_unverified,
            'generated_at':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}


def to_markdown(guide: dict) -> str:
    esc=lambda s: html.escape(str(s))
    o=['## Plan expérimental hiérarchisé',
       '**Module d’aide métier — recommandations heuristiques, distinctes des prédictions ESM-2.**','']
    if guide['p_pred'] is not None:
        o.append(f"**Score ESM-2 relié automatiquement à cette séquence : {guide['p_pred']:.1%}** "
                 f"(seuil {guide['threshold']:.2%}, prédiction {'insoluble' if guide['p_pred'] < guide['threshold'] else 'soluble'}).")
        if guide.get('calibration_transfer_unverified'):
            o.append('**Attention : mode 1 022 AA exploratoire.** Le calibrateur et le seuil proviennent du protocole 512 AA. '
                     'Le score est indicatif et **n’est pas utilisé pour prioriser tags/hôtes**.')
        else:
            o.append('Le score oriente les essais ; il ne permet pas de prédire le meilleur tag ou hôte.')
    else:
        o.append('**Score ML non disponible pour cette séquence.** Relancez l’analyse.')
    o.extend(['','### Signaux observés'])
    o.extend('- '+esc(s) for s in guide['observations'])
    o.extend(['','### Constructions et tags — par priorité'])
    for r in guide['tags']:
        o.append(f"- **{esc(r['niveau'])} — {esc(r['essai'])}** : {esc(r['pourquoi'])}")
    o.extend(['','### Pourquoi cet ordre de priorité ?'])
    o.extend('- '+esc(x) for x in guide.get('decision_rationale',[]))
    o.extend(['','### Hôtes d’expression — par priorité'])
    for r in guide['hosts']:
        o.append(f"- **{esc(r['priorite'])} — {esc(r['systeme'])}** : {esc(r['justification'])}")
    o.extend(['','### Conditions d’expression (essais exploratoires)'])
    if guide['induction']:
        o.extend(['| Essai | Température | IPTG | Durée indicative |','|---|---:|---:|---:|'])
        for r in guide['induction']:
            o.append(f"| {esc(r['essai'])} | {esc(r['temperature'])} | {esc(r['iptg'])} | {esc(r['duree'])} |")
    o.extend(['',esc(guide['induction_note']),'','### Constructions et vérifications'])
    o.extend('- '+esc(x) for x in guide['constructs'])
    o.extend(['','### Contrôles fonctionnels et qualité'])
    o.extend('- '+esc(x) for x in guide['validation'])
    o.extend(['','### Limites'])
    o.extend('- '+esc(x) for x in guide['limits'])
    o.extend(['','**Sources et bonnes pratiques :**'])
    for name,url in guide['references']:
        o.append(f'- [{esc(name)}]({url})')
    return '\n'.join(o)


def to_pdf(guide: dict) -> str:
    """PDF par ReportLab ; fichier éphémère délivré par gr.File."""
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.lib.pagesizes import A4

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17,
                              textColor=colors.HexColor("#174750"), spaceAfter=12, leading=21))
    styles.add(ParagraphStyle(name="HeadingX", parent=styles["Heading2"], textColor=colors.HexColor("#216A69"), fontSize=11,
                              spaceBefore=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallX", parent=styles["Normal"], fontSize=8.5, leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="TinyX", parent=styles["Normal"], fontSize=7.3, leading=10))
    styles.add(ParagraphStyle(name="WarningX", parent=styles["Normal"], fontSize=9, leading=13,
                              textColor=colors.HexColor("#7A4929"), spaceAfter=9))
    esc = lambda v: escape(str(v))
    story = [Paragraph("Plan d'expression et de purification", styles["CoverX"]),
             Paragraph("Fiche de conseils expérimentaux - heuristiques, pas prédictions validées", styles["WarningX"]),
             Paragraph(f"Généré : {esc(guide['generated_at'])} | Séquence : {guide['features']['length']} aa | "
                       f"SHA-256 : {guide['features']['sequence_sha256'][:18]}...", styles["SmallX"]), Spacer(1, 0.1*cm)]

    def heading(t): story.append(Paragraph(t,styles["HeadingX"]))
    def entries(values):
        for val in values:
            story.append(Paragraph("&#8226; " + esc(val), styles["SmallX"]))

    if guide['p_pred'] is not None:
        prefix = 'Score ESM-2 INDICATIF (fenêtre 1 022 AA)' if guide.get('calibration_transfer_unverified') else 'Score ESM-2 (protocole 512 AA)'
        story.append(Paragraph(f"{prefix} : {guide['p_pred']:.1%} | seuil historique : {guide['threshold']:.2%}. N'indique pas le tag optimal.", styles['WarningX']))
        if guide.get('calibration_transfer_unverified'):
            story.append(Paragraph('Calibration et seuil 512 AA NON validés pour 1 022 AA ; score NON utilisé pour prioriser tags et hôtes.', styles['WarningX']))
    else:
        story.append(Paragraph('Score ESM-2 non calculé pour cette séquence.', styles['WarningX']))
    heading("Contexte et annotations")
    for k,v in guide["context"].items():
        story.append(Paragraph(f"<b>{esc(k.replace('_',' ').capitalize())} :</b> {esc(v)}", styles["SmallX"]))
    heading("Observations sur la séquence")
    entries(guide["observations"])
    heading("Tags et constructions - ordre de priorité")
    for r in guide["tags"]:
        story.append(Paragraph(f"<b>{esc(r['niveau'])} : {esc(r['essai'])}</b> - {esc(r['pourquoi'])}", styles["SmallX"]))
    heading("Justification de la priorisation")
    entries(guide.get('decision_rationale', []))
    heading("Hôtes de production à considérer")
    for r in guide["hosts"]:
        story.append(Paragraph(f"<b>{esc(r['priorite'])} : {esc(r['systeme'])}</b> - {esc(r['justification'])}",styles["SmallX"]))
    heading("Plan de criblage / induction")
    if guide["induction"]:
        rows = [[Paragraph(h,styles["TinyX"]) for h in ("Essai", "Température", "IPTG", "Durée")]]
        for r in guide["induction"]:
            rows.append([Paragraph(esc(r[c]),styles["TinyX"]) for c in ("essai","temperature","iptg","duree")])
        t = Table(rows,colWidths=[6*cm,3.0*cm,2.1*cm,3.5*cm],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DFF1ED")),
                              ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CEDBDA")),
                              ("VALIGN",(0,0),(-1,-1),"TOP"),
                              ("LEFTPADDING",(0,0),(-1,-1),7),
                              ("BOTTOMPADDING",(0,0),(-1,-1),6),
                              ("TOPPADDING",(0,0),(-1,-1),6)]))
        story.append(t)
    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph(esc(guide["induction_note"]),styles["SmallX"]))
    heading("Choix de constructions")
    entries(guide["constructs"])
    heading("Contrôles fonctionnels et qualité")
    entries(guide["validation"])
    heading("Limites scientifiques")
    entries(guide["limits"])
    heading("Références")
    for name,url in guide["references"]:
        story.append(Paragraph(esc(name),styles["TinyX"]))
        story.append(Paragraph(esc(url),styles["TinyX"]))
    with tempfile.NamedTemporaryFile(prefix="plan_solubilite_", suffix=".pdf", delete=False) as f:
        target=f.name
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica",8)
        canvas.setFillColor(colors.HexColor("#536D70"))
        canvas.drawString(1.6*cm, 1.15*cm, "Aide exploratoire - Validation expérimentale indispensable")
        canvas.drawRightString(19.4*cm, 1.15*cm, f"Page {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(target, pagesize=A4, leftMargin=1.65*cm,rightMargin=1.65*cm,
                            topMargin=1.7*cm,bottomMargin=1.8*cm,title="Plan expérimental de solubilité")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return target


def generate_recommendation(seq, origin, ptm, disulfides, membrane, objective, host, pred_state, family="Non précisée", protein_name="", identification=None):
    """Point d'entrée de l'UI Gradio (ne nécessite pas ZeroGPU)."""
    try:
        guide = make_guide(seq, origin, ptm, disulfides, membrane, objective, host, pred_state, family, protein_name, identification)
        path = to_pdf(guide)
        digest = {
            "taille": guide["features"]["length"],
            "nombre_cysteines": guide["features"]["n_cys"],
            "pourcentage_hydrophobe": round(guide["features"]["hydrophobic_pct"],2),
            "fenetres_hydrophobes_19aa": guide["features"]["hydrophobic_windows_19aa"],
            "prediction_ml_reutilisee": guide["p_pred"] is not None,
            "mode_inference": guide.get('inference_mode'),
            "acides_amines_analyses_esm2": guide.get('esm2_aa_analyzed'),
            "calibration_transferee_non_validee": guide.get('calibration_transfer_unverified'),
            "probabilite_esm2": round(guide["p_pred"], 6) if guide["p_pred"] is not None else None,
            "famille_declaree_ou_annotee": guide["family"],
            "identification_pdb_uniprot": (guide["identification"]["best"] if guide["identification"] else None),
            "tags_prioritaires": [r["essai"] for r in guide["tags"] if r["niveau"] == "Priorité 1"],
            "conseils_personnalises_par_annotations": bool(guide.get('annotation_evidence')),
            "raisons_priorisation": guide.get('decision_rationale', []),
            "recommandations_validees_experimentalement": False,
        }
        return to_markdown(guide), path, digest
    except ValueError as exc:
        # En UI on ne cherche pas à corriger silencieusement une séquence invalide.
        raise ValueError(str(exc)) from exc
