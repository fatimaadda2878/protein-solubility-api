"""Gradio ZeroGPU : inférence ESM-2 150M fine-tuné, calibrée et figée.

Le dépôt Model (éventuellement privé) doit contenir :
  finetune_best.pt
  platt_calibrator.joblib
  finalisation_figee.json
Le Space requiert HF_MODEL_REPO, et HF_TOKEN si modèle privé.
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import gradio as gr
import spaces  # indispensable : activer la compatibilité ZeroGPU avant torch
import numpy as np
import torch
import joblib
import esm
from huggingface_hub import hf_hub_download

from finaliser_finetune_v3_securise import (
    AA, BIONAMES, bio13, create_net, sha_file
)
from protein_advice import clean_sequence
from workflow_v5 import identify_safe, identity_html, score_html, plan_auto
from length_policy_v7 import policy_for_length

CACHE_DIR = Path('/tmp/esm2_finetune_artifacts')
ARTIFACTS = ('finetune_best.pt', 'platt_calibrator.joblib', 'finalisation_figee.json')


def init_model():
    """Téléchargement + vérification SHA et chargement au démarrage du Space.

    Placer le modèle sur cuda au démarrage (recommandation ZeroGPU HF).
    Aucune étiquette du dataset n'est utilisée.
    """
    import json

    repo = os.getenv('HF_MODEL_REPO', '').strip()
    if not repo or '/' not in repo:
        raise RuntimeError('Configurer HF_MODEL_REPO dans Settings > Variables du Space.')

    token = os.getenv('HF_TOKEN') or None
    revision = os.getenv('HF_MODEL_REVISION') or None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ARTIFACTS:
        paths[name] = Path(hf_hub_download(
            repo_id=repo, filename=name, token=token,
            revision=revision,
        ))

    manifest = json.loads(paths['finalisation_figee.json'].read_text(encoding='utf-8'))
    if sha_file(paths['finetune_best.pt']) != manifest['frozen_checkpoint_sha256']:
        raise RuntimeError('SHA256 du checkpoint incorrect : arrêt de sécurité.')
    if sha_file(paths['platt_calibrator.joblib']) != manifest['calibrator_sha256']:
        raise RuntimeError('SHA256 du calibrateur incorrect : arrêt de sécurité.')
    threshold = float(manifest['threshold'])
    if not (0 < threshold < 1):
        raise RuntimeError('Seuil figé invalide.')

    # Charger uniquement un checkpoint créé par soi-même et jugé fiable.
    state = torch.load(paths['finetune_best.pt'], map_location='cpu', weights_only=False)
    if state.get('version') != 'esm2_t30_150M_UR50D' or int(state.get('ft_maxlen', 0)) != 512:
        raise RuntimeError('Checkpoint ESM-2 150M / troncature 512 incohérent.')
    if state.get('bio13_columns') != BIONAMES:
        raise RuntimeError('Bio13 du checkpoint incompatible avec le code.')

    mu = np.asarray(state['mu'], dtype=np.float32)
    sd = np.asarray(state['sd'], dtype=np.float32)
    if mu.shape != (13,) or sd.shape != (13,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or not (sd > 0).all():
        raise RuntimeError('Paramètres de normalisation Bio13 invalides.')

    # Reconstruction locale de l'architecture : aucun modèle pré-entraîné distant additionnel.
    alphabet = esm.data.Alphabet.from_architecture('ESM-1b')
    encoder = esm.ESM2(num_layers=30, embed_dim=640, attention_heads=20, alphabet=alphabet)
    net = create_net(encoder, alphabet, torch)
    net.load_state_dict(state['model'], strict=True)
    net.eval()

    # Important : modèle CUDA initialisé HORS de la fonction @spaces.GPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    converter = alphabet.get_batch_converter(truncation_seq_length=1022)
    calibrator = joblib.load(paths['platt_calibrator.joblib'])
    if not hasattr(calibrator, 'predict_proba'):
        raise RuntimeError('Calibrateur Platt incompatible.')
    print(f'Modèle ESM-2 150M prêt sur {device}, seuil={threshold:.6f}', flush=True)
    return (net, converter, calibrator, mu, sd, threshold, device)


MODEL = None
MODEL_INIT_ERROR = None
try:
    MODEL = init_model()
except Exception as exc:
    MODEL_INIT_ERROR = str(exc)
    print('Impossible de charger le modèle au démarrage :', str(exc), flush=True)
    traceback.print_exc()


@spaces.GPU(duration=120)
def prediction_solubilite(sequence: str):
    '''Même checkpoint/Bio13; fenêtre adaptative 512/1022; seuil historique inchangé.'''
    if MODEL is None:
        raise gr.Error('Modèle non initialisé. Vérifier les variables et les Logs du Space. ' + str(MODEL_INIT_ERROR))

    try:
        seq = clean_sequence(sequence)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    net, converter, calibrator, mu, sd, threshold, device = MODEL
    policy = policy_for_length(len(seq))
    _, _, tokens = converter([('proteine', seq[:policy.residue_limit])])
    b = (np.asarray(bio13(seq), dtype=np.float32) - mu) / sd

    with torch.inference_mode():
        x = tokens.to(device)
        xb = torch.from_numpy(b[None, :]).to(device)
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                logit = float(net(x, xb).float().item())
        else:
            logit = float(net(x, xb).float().item())

    probability = float(calibrator.predict_proba(np.array([[logit]], dtype=np.float64))[0, 1])
    if not np.isfinite(probability):
        raise gr.Error('Probabilité non finie : inférence refusée.')
    predicted = int(probability >= threshold)
    meta = {
        'inference_mode': policy.mode,
        'esm2_aa_analyzed': min(len(seq), policy.residue_limit),
        'esm2_aa_ignored': max(0, len(seq) - policy.residue_limit),
        'length_explanation': policy.explanation,
        'calibration_status': ('transfert_512_vers_1022_non_valide' if policy.experimental else 'calibrateur_512_historique'),
        'threshold_status': ('indicatif_non_reoptimise' if policy.experimental else 'seuil_512_historique'),
        'probability_soluble': round(probability, 6),
        'class_predicted': predicted,
        'threshold': round(threshold, 6),
        'sequence_length': len(seq),
        'truncated_at_512': policy.residue_limit == 512 and len(seq) > 512,
        'truncated_at_1022': policy.residue_limit == 1022 and len(seq) > 1022,
        'validation_status': 'exploratoire; pas de validation externe du fine-tuning',
        'model': 'esm2_t30_150M_UR50D_finetune_last2_bio13',
    }
    state = {'sequence': seq, 'probability_soluble': probability,
             'class_predicted': predicted, 'threshold': threshold,
             'inference_mode': policy.mode,
             'esm2_aa_analyzed': min(len(seq), policy.residue_limit),
             'esm2_aa_ignored': max(0, len(seq) - policy.residue_limit),
             'calibration_transfer_unverified': policy.experimental,
             'model': 'esm2_t30_150M_UR50D_finetune_last2_bio13'}
    return score_html(probability, threshold, len(seq),
                      model_aa_limit=policy.residue_limit,
                      experimental_1022=policy.experimental), meta, state


def identification_et_plan(sequence, prediction_state, ptm, disulfides, membrane, objective, host):
    # PDB externe après inférence : échec sans effet sur le calcul ML.
    try:
        seq = clean_sequence(sequence)
        ident = identify_safe(seq)
        md, pdf, details = plan_auto(seq, prediction_state, ident, ptm,
                                     disulfides, membrane, objective, host)
        return identity_html(ident), md, pdf, details, ident
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def actualiser_plan(sequence, prediction_state, identification_state,
                    ptm, disulfides, membrane, objective, host):
    '''Modifier les contraintes met à jour le plan sans réinférence GPU ni requête réseau.'''
    if not isinstance(prediction_state, dict) or not isinstance(identification_state, dict):
        return 'Analyse la séquence pour générer le plan.', None, None
    try:
        md, pdf, details = plan_auto(sequence, prediction_state, identification_state,
                                     ptm, disulfides, membrane, objective, host)
        return md, pdf, details
    except ValueError:
        return 'La séquence a changé : relance l’analyse.', None, None


def reset_ui():
    return (INIT_SCORE, INIT_IDENT, 'Le plan sera généré automatiquement après l’analyse.',
            None, None, None, None, None)


def start_ui():
    return ('<section class="pending-card">Calcul de solubilité en cours…</section>',
            '<section class="pending-card">Identification automatique après la prédiction…</section>',
            'Le plan apparaîtra après le calcul de solubilité et la recherche des références.',
            None, None, None, None, None)


# Ruban protéique illustratif : décoration (pas une structure réelle inférée).
PROTEIN_RIBBON = r'''<svg viewBox="0 0 370 220" role="img" aria-label="Illustration stylisée d’une structure tridimensionnelle protéique" xmlns="http://www.w3.org/2000/svg">
<defs>
  <linearGradient id="pRibbon" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#60DBD0"/><stop offset="50%" stop-color="#12A1A0"/><stop offset="100%" stop-color="#1761A7"/></linearGradient>
  <linearGradient id="pGold" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#F8DE83"/><stop offset="100%" stop-color="#D89B25"/></linearGradient>
  <filter id="pShadow" x="-25%" y="-25%" width="160%" height="160%"><feDropShadow dx="0" dy="7" stdDeviation="6" flood-color="#0B2436" flood-opacity=".22"/></filter>
</defs>
<g fill="none" stroke-linecap="round" stroke-linejoin="round" filter="url(#pShadow)">
 <path d="M48 146 C58 50 144 24 164 101 S255 200 318 83" stroke="#DDFCF6" stroke-width="24" opacity=".16"/>
 <path d="M49 145 C89 102 88 49 123 53 C165 58 121 156 176 157 C226 157 183 46 238 62 C292 78 236 170 313 85" stroke="url(#pRibbon)" stroke-width="15"/>
 <path d="M44 155 C85 115 78 44 122 48 C169 55 119 154 172 157 C229 166 186 44 239 60 C294 79 242 158 316 89" stroke="#E4FDFC" stroke-width="4" opacity=".65"/>
 <path d="M69 100 C133 199 190 173 248 110" stroke="url(#pGold)" stroke-width="11"/>
 <path d="M93 55 C156 6 273 34 299 133" stroke="#5C9DE4" stroke-width="8" opacity=".75"/>
</g>
<circle cx="88" cy="78" r="9" fill="#C7F6F4"/><circle cx="273" cy="133" r="7" fill="#F8D66E"/>
</svg>'''

INIT_SCORE = '<section class="pending-card">La prédiction apparaîtra ici après l’analyse.</section>'
INIT_IDENT = '<section class="pending-card">L’identification PDB/UniProt apparaîtra ici.</section>'

CSS = r'''
:root { --ui-ink: #12263a; --ui-muted: #334155; --ui-border:#cbd5e1; }
.gradio-container { max-width: 1150px !important; margin: auto !important; color:#172637 !important; }
.gradio-container .prose, .gradio-container .prose p, .gradio-container .prose li,
.gradio-container .prose td, .gradio-container .prose strong { color:#172637 !important; opacity:1 !important; }
.gradio-container .prose h1, .gradio-container .prose h2,
.gradio-container .prose h3 { color:#14384d !important; opacity:1 !important; }
#hero { background:linear-gradient(110deg, #F5FAFF, #E7F7F5 75%, #DFEFF7); border:1px solid #C8DEEA; border-radius:20px; margin:0 0 18px; padding:15px 22px; }
#hero .hero-inner { display:flex; align-items:center; justify-content:space-between; gap:20px; min-height:175px; }
#hero h1 { font-size:clamp(24px, 3vw, 37px); font-weight:800; line-height:1.22; margin:0; max-width:650px; color:#112C41 !important; }
#hero .ribbon { width:min(290px,38%); flex-shrink:0; }
#hero svg { max-width:100%; height:auto; display:block; }
#analyze-btn { margin:12px 0 18px; }
#analyze-btn button { background:#0F617E !important; border-color:#0F617E !important; color:white !important; font-weight:800 !important; font-size:17px !important; }
#analyze-btn button:hover { background:#084862 !important; }
.section-label { font-size:19px; font-weight:800; color:#112C41; padding:6px 0 8px; }
.pending-card { background:#F1F5F9; color:#334155 !important; border:1px solid #CBD5E1; border-radius:14px; padding:17px; font-weight:600; }
.score-card { background:#fff; border:2px solid #b8c9d6; border-radius:16px; padding:23px 26px; box-shadow:0 3px 12px rgba(10,44,63,.08); }
.score-card.positive { border-left:8px solid #127443; }
.score-card.negative { border-left:8px solid #B42318; }
.score-kicker { color:#334155 !important; font-size:14px; font-weight:750; }
.score-status { font-size:24px; font-weight:850; color:#112C41 !important; padding-top:5px; }
.score-card.positive .score-status { color:#126438 !important; }
.score-card.negative .score-status { color:#AB241B !important; }
.score-class { display:inline-block; margin-left:9px; font-size:13px; font-weight:600; color:#243B53; vertical-align:middle; }
.score-percentage { font-size:54px; font-weight:900; line-height:1.12; color:#102B40 !important; padding:14px 0 5px; }
.score-percentage span { font-size:30px; }
.score-explainer { color:#334155 !important; font-size:14px; font-weight:600; }
.score-progress { height:12px; width:100%; border-radius:12px; background:#D9E3EB; overflow:hidden; margin-top:14px; }
.score-progress-fill { height:100%; background:#178E72; border-radius:12px; }
.score-card.negative .score-progress-fill { background:#CE5747; }
.score-meta { display:flex; flex-wrap:wrap; gap:24px; padding-top:15px; font-size:14px; color:#223850 !important; }
.score-meta b { color:#142F41 !important; }
.score-warning { margin-top:10px; color:#8A3F08 !important; font-weight:700; }
.id-card { border:1px solid #A7BEC8; border-radius:15px; padding:20px 23px; background:#F5FBFC; color:#1F3446 !important; }
.id-card.muted { background:#F7F8FA; border-color:#CBD5E1; }
.id-heading { color:#173C51 !important; font-size:17px; font-weight:850; }
.id-protein { color:#12334D !important; font-size:23px; font-weight:800; padding:7px 0 2px; }
.id-organism { color:#334155 !important; font-style:italic; font-weight:650; }
.id-metrics { display:flex; flex-wrap:wrap; gap:12px; padding:12px 0; color:#182E44 !important; }
.id-metrics span { background:#E3F0F3; border:1px solid #B1D4DA; padding:8px 10px; border-radius:10px; }
.id-links a { color:#075E99 !important; text-decoration:underline !important; font-weight:700; }
.id-family { margin-top:10px; color:#126438 !important; }
.id-note { margin:9px 0 0; color:#334155 !important; font-size:13px; line-height:1.5; }
#plan-body { border-radius:13px; border:1px solid #CBD5E1; background:white; padding:12px 24px; }
#plan-body .prose { color:#12263a !important; }
#plan-body .prose li { color:#1E354A !important; line-height:1.6; }
#plan-body .prose strong { color:#163448 !important; }
#plan-body .prose a { color:#075E99 !important; }
.small-note { color:#455769; font-size:13px; line-height:1.5; margin:3px 0 12px; }
@media (max-width:680px) { #hero .ribbon { display:none; } .score-percentage { font-size:43px; } #hero { padding:18px 14px; } }
'''

# Gradio 6 : CSS fourni à launch() ; Gradio 5 : CSS dans Blocks().
GRADIO_MAJOR = int(gr.__version__.split('.')[0])
blocks_opts = {'title': 'Solubilité protéique — prédiction et conception expérimentale'}
if GRADIO_MAJOR < 6:
    blocks_opts['css'] = CSS
with gr.Blocks(**blocks_opts) as demo:
    gr.HTML(f'''<div id="hero"><div class="hero-inner">
      <h1>Solubilité protéique — prédiction et conception expérimentale</h1>
      <div class="ribbon">{PROTEIN_RIBBON}</div>
    </div></div>''')
    if MODEL_INIT_ERROR:
        gr.Markdown('**Chargement du modèle indisponible.** Consulter les Logs et les variables du Space.')

    sequence = gr.Textbox(
        label='Séquence protéique',
        lines=5, max_lines=12,
        placeholder='Colle les acides aminés de la protéine (sans en-tête FASTA).',
        elem_id='protein-sequence'
    )
    gr.HTML('<p class="small-note">Un clic sur « Analyser » lance aussi une recherche de similarité sur les API publiques RCSB PDB et, si pertinent, UniProt. La séquence est transmise à ces services : ne pas utiliser de séquence confidentielle. Une absence de résultat PDB ne signifie pas que la protéine est inconnue.</p>')

    with gr.Accordion('Contexte de production (facultatif)', open=False):
        gr.Markdown('Précise ces contraintes seulement si elles sont connues ; aucune information n’est inventée en leur absence.')
        with gr.Row():
            ptm = gr.Dropdown(['Inconnu','Oui (nécessaires pour activité)','Non','Potentiellement'],
                              value='Inconnu', label='Modifications post-traductionnelles nécessaires ?')
            disulfides = gr.Dropdown(['Inconnu','Oui (requis)','Non'], value='Inconnu',
                                     label='Ponts disulfure indispensables ?')
            membrane = gr.Dropdown(['Inconnu','Oui (confirmée)','Non'], value='Inconnu',
                                    label='Protéine membranaire ?')
        with gr.Row():
            objective = gr.Dropdown(['Solubilité et purification','Activité biologique','Cristallographie / biophysique'],
                                     value='Solubilité et purification', label='Objectif expérimental')
            host = gr.Dropdown(['Non défini','E. coli BL21(DE3)','E. coli SHuffle/oxydante',
                                'Sf9 / baculovirus','HEK293 / cellules mammifères'],
                               value='Non défini', label='Hôte déjà envisagé')

    prediction_state = gr.State(value=None)
    identification_state = gr.State(value=None)
    btn = gr.Button('Analyser la séquence', variant='primary', elem_id='analyze-btn')

    gr.HTML('<div class="section-label">01 — Solubilité prédite</div>')
    score_box = gr.HTML(INIT_SCORE)
    gr.HTML('<div class="section-label">02 — Identification de la protéine</div>')
    ident_box = gr.HTML(INIT_IDENT)
    gr.HTML('<div class="section-label">03 — Plan d’expression et de purification</div>')
    with gr.Row():
        with gr.Column(scale=3):
            plan_box = gr.Markdown('Le plan sera généré automatiquement après l’analyse.', elem_id='plan-body')
        with gr.Column(scale=1):
            pdf = gr.File(label='Fiche expérimentale PDF', interactive=False)
    with gr.Accordion('Détails techniques et indicateurs', open=False):
        details = gr.JSON(label='Sortie du modèle ESM-2')
        indicators = gr.JSON(label='Indicateurs du plan expérimental')

    reset_outputs = [score_box, ident_box, plan_box, pdf, details,
                     indicators, prediction_state, identification_state]
    sequence.input(fn=reset_ui, inputs=[], outputs=reset_outputs,
                   queue=False, show_progress='hidden', api_name=False)
    preparation = btn.click(fn=start_ui, inputs=[], outputs=reset_outputs,
                            queue=False, show_progress='hidden', api_name=False)
    # Même checkpoint et calibrateur ; seule la fenêtre d'inférence change pour 513–1022 AA.
    scoring = preparation.success(
        fn=prediction_solubilite, inputs=[sequence],
        outputs=[score_box, details, prediction_state],
        api_name='predict', concurrency_limit=1,
    )
    # Appels PDB/UniProt et PDF hors ZeroGPU : un seul bouton, un seul flux visible.
    scoring.success(
        fn=identification_et_plan,
        inputs=[sequence, prediction_state, ptm, disulfides, membrane, objective, host],
        outputs=[ident_box, plan_box, pdf, indicators, identification_state],
        api_name=False, concurrency_limit=4,
    )
    # Un changement de contexte actualise uniquement les recommandations, sans nouveau calcul GPU.
    for champ in [ptm, disulfides, membrane, objective, host]:
        champ.change(
            fn=actualiser_plan,
            inputs=[sequence, prediction_state, identification_state,
                    ptm, disulfides, membrane, objective, host],
            outputs=[plan_box, pdf, indicators],
            queue=False, show_progress='hidden', api_name=False,
        )

    gr.Markdown('---\n**Utilisation de recherche uniquement.** Le score de solubilité est expérimental, '
                'sans validation externe indépendante de ce fine-tuning. Les propositions de tags, '
                'hôtes et induction sont des pistes de criblage, pas des prédictions validées. '
                'Pour 513–1 022 AA, le score et la classe sont indicatifs car la calibration 512 AA '
                'n’a pas été validée pour cette longueur. '
                'Coûts hypothétiques : FN 1 200 €, FP 600 €.')

if __name__ == '__main__':
    demo.queue().launch(**({'css': CSS} if GRADIO_MAJOR >= 6 else {}))
