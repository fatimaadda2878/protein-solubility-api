"""Moteur V6 de recommandations biochimiques auditable, SANS nouvelle prédiction ML.

Entrées : séquence, score ESM-2, annotations vérifiées UniProt via PDB et
contraintes déclarées. Les sorties sont des pistes d'ESSAIS, jamais des
conditions optimales calculées par le modèle. Les annotations de séquences
homologues ne sont PAS considérées comme preuve expérimentale sur la cible.
"""
from __future__ import annotations

from dataclasses import dataclass

FAMILIES = {
    'GFP / protéine fluorescente', 'Thioredoxine / système redox',
    'Ubiquitine / ubiquitine-like', 'Chaperonne / protéine de repliement',
    'Liaison ADN / ARN', 'Lysozyme / enzyme muramidase',
    'Enzyme (autre)', 'Anticorps / fragment liant un antigène',
    'Protéine membranaire (fonction connue)',
}


def decide(family: str, features: dict, prob: float | None, threshold: float | None,
           ptm: str, disulfides: str, membrane: str, objective: str,
           host: str, annotation: dict | None) -> dict:
    """Retourne les recommandations en fonction de critères explicites.

    IMPORTANT: le seuil sert uniquement à fixer une classe économique.
    Un score de 0.35 ne mesure ni 35% de rendement ni une solubilité physique.
    """
    fam = family if family in FAMILIES else 'Non précisée'
    anno = annotation if isinstance(annotation, dict) else {}
    # Annotations structurales autorisées UNIQUEMENT si match de séquence solide.
    reliable = bool(anno.get('sequence_verified'))
    flags = set(anno.get('feature_types') or []) if reliable else set()
    position_info = anno.get('feature_details') or [] if reliable else []
    location = str(anno.get('location') or '') if reliable else ''
    origin = str(anno.get('organism_class') or '') if reliable else ''

    declared_mem = membrane.startswith('Oui')
    declared_ss = disulfides.startswith('Oui')
    declared_ptm = ptm.startswith('Oui')
    # Ces signaux sont des hypothèses, on évite de conclure d'une Cys isolée.
    evidence_mem = reliable and ('TRANSMEM' in flags or 'INTRAMEM' in flags)
    evidence_signal = reliable and 'SIGNAL' in flags
    evidence_ss = reliable and 'DISULFID' in flags
    evidence_glyc = reliable and ('CARBOHYD' in flags or 'LIPID' in flags)
    mem = declared_mem or evidence_mem or fam == 'Protéine membranaire (fonction connue)'
    ss = declared_ss or evidence_ss
    ptm_needed = declared_ptm  # présence de PTM annotée != indispensabilité pour activité
    warnings = []
    evidence = []
    if evidence_mem: evidence.append('Segment transmembranaire annoté dans UniProt (correspondance de séquence vérifiée).')
    if evidence_signal: evidence.append('Peptide signal annoté dans UniProt (ne prouve pas les besoins de la construction recombinante).')
    if evidence_ss: evidence.append('Pont(s) disulfure annoté(s) dans UniProt ; vérifier la pertinence pour la cible et le contexte redox.')
    if evidence_glyc:
        evidence.append('Modification(s) annotée(s) : vérifier leur nécessité biologique avant de choisir un hôte.')
    if location:
        evidence.append('Localisation UniProt annotée : ' + location[:170] + '.')
    if reliable and anno.get('function'):
        evidence.append('Fonction annotée UniProt (résumé) : ' + str(anno['function'])[:235])
    if 'ACT_SITE' in flags:
        evidence.append('Site(s) actif(s) annoté(s) : éviter une fusion ou une modification qui perturbe leur accessibilité.')
    if features.get('hydrophobic_windows_19aa') and not mem:
        warnings.append('Région(s) hydrophobe(s) détectée(s) : vérifier avec TMHMM/DeepTMHMM ou SignalP ; ce signal seul ne prouve pas une protéine membranaire.')
    if features.get('n_cys',0) >= 4 and not ss:
        warnings.append('Plusieurs cystéines sans annotation redox confirmée : ne pas imposer SHuffle, périplasme ni HEK293 sur ce seul critère.')
    if features.get('length', 0) > features.get('model_aa_analyzed', 512):
        warnings.append(f'Séquence longue : ESM-2 a analysé les '
                        f"{features.get('model_aa_analyzed', 512)} premiers AA sur {features.get('length')} ; "
                        'les autres domaines n’ont pas été vus directement par le réseau.')
    if p_is_known(prob,threshold):
        margin=float(prob)-float(threshold)
        if abs(margin)<=0.08:
            warnings.append('Score proche du seuil métier (±8 points) : la classification coût-dépendante ne constitue pas une preuve du rendement ou du repliement. Éviter les recommandations agressives sur ce seul score.')
    if not reliable:
        warnings.append('Aucune annotation UniProt exploitable avec un alignement suffisant : choix fondés sur les propriétés de séquence et contraintes déclarées, pas sur une fonction supposée.')
    if reliable and not anno.get('reviewed'):
        warnings.append('La fiche UniProt associée n’est pas annotée manuellement (TrEMBL) : confirmer les fonctions et PTM avant décision expérimentale.')

    tags=[]
    hosts=[]
    quality=[]
    rationale=[]
    def tag(rank,label,why): tags.append({'niveau':rank,'essai':label,'pourquoi':why})
    def hostopt(rank,label,why): hosts.append({'priorite':rank,'systeme':label,'justification':why})

    # Les déclarations expérimentales ont la priorité sur une simple heuristique de nom.
    if mem:
        tag('Priorité 1','Tag court compatible avec topologie (His6/Strep), position à déterminer',
            'Préserver les domaines membranaires, le ciblage et l’activité ; tester l’accessibilité du tag.')
        tag('Comparaison si nécessaire','Tag alternatif / extrémité opposée',
            'Dépend de l’orientation membranaire et de l’assemblage ; aucune fusion solubilisante n’est garantie.')
        hostopt('À sélectionner selon cible','Sf9/HEK293 ou système membranaire bactérien adapté',
                'Vérifier topologie, lipides, cofacteurs, partenaires et compatibilité des systèmes.')
        quality.extend(['Vérifier insertion/localisation et topologie par une méthode adaptée.',
                        'Évaluer la fonction et la stabilité en membrane, détergent ou nanodisques selon la cible.'])
        rationale.append('Architecture membranaire prioritaire sur le score de solubilité appris sur DeepSol.')
    elif ptm_needed or fam == 'Anticorps / fragment liant un antigène':
        tag('Priorité 1','Tag compatible avec maturation et sécrétion (His6/Strep ou format sans tag)',
            'Préserver l’adressage, le repliement et les modifications indispensables ; éviter l’obstruction des sites de liaison.')
        tag('Conditionnel','Tag clivable après purification',
            'Comparer seulement si la maturation ou la fonction sont perturbées.')
        hostopt('Priorité 1','HEK293 ou système de sécrétion eucaryote adapté',
                'Prioriser si une glycosylation de type mammifère est fonctionnellement indispensable ou si le format est un anticorps complexe.')
        hostopt('Alternative contextualisée','Sf9 ; E. coli périplasmique pour certains fragments simples',
                'Comparer selon format, glycosylation et qualité de liaison ; ne pas traiter les anticorps entiers comme des scFv.')
        quality.extend(['Vérifier intégrité, maturation/PTM et fraction monodisperse.',
                        'Pour un anticorps : test de liaison antigène (BLI/SPR/ELISA selon disponibilité) et agrégation.'])
        rationale.append('La nécessité biologique déclarée de maturation ou le format de l’anticorps priment sur le score ESM-2.')
    elif ss or (fam == 'Lysozyme / enzyme muramidase' and not (reliable and anno.get('organism_class') == 'Bactérienne')):
        tag('Priorité 1','Construction courte clivable et environnement de repliement adapté',
            'Préserver le domaine mature ; ne pas supposer que His6 améliore le repliement oxydatif.')
        tag('En secours','Fusion clivable après diagnostic de repliement',
            'MBP/SUMO ne remplacent pas un environnement redox adéquat ; contrôler la cible après clivage.')
        hostopt('Priorité 1','E. coli périplasmique ou SHuffle (si ponts disulfure nécessaires)',
                'Essai pertinent si les ponts disulfure sont réellement nécessaires et la cible compatible.')
        hostopt('Alternative','HEK293 ou Sf9 en fonction de la maturation et de l’activité',
                'À justifier par les besoins de la protéine et la reproductibilité, non uniquement par le nombre de cystéines.')
        quality.append('Contrôler les ponts disulfure et l’activité après purification, pas seulement la fraction soluble.')
        if fam == 'Lysozyme / enzyme muramidase':
            quality.append('Mesurer l’activité muramidase sur un substrat approprié avec témoin lysozyme actif et contrôle négatif.')
        rationale.append('La contrainte redox documentée impose des contrôles de repliement indépendants du score ML.')
    elif fam == 'Thioredoxine / système redox':
        tag('Priorité 1','Thioredoxine native ou His6 N-terminal clivable',
            'Une thioredoxine soluble connue ne nécessite pas MBP/SUMO d’emblée ; vérifier la conservation du site actif CXXC.')
        tag('Seulement si problème observé','Variante de position du tag / fusion alternative',
            'Éviter que le partenaire de fusion masque le motif actif ou modifie le cycle redox.')
        hostopt('Priorité 1','E. coli BL21(DE3) si vecteur compatible',
                'Point de départ cohérent pour TrxA bactérienne cytosolique ; les cystéines catalytiques ne justifient pas une souche SHuffle.')
        quality.extend(['Test de fonction redox approprié (p. ex. réduction de l’insuline avec témoins) et état du motif CXXC.',
                        'Vérifier le repliement, l’état redox et l’activité après clivage éventuel du tag.'])
        rationale.append('Thioredoxine : priorité à la fonction redox et au motif CXXC, pas à une stratégie automatique de solubilisation.')
    elif fam == 'Ubiquitine / ubiquitine-like':
        tag('Priorité 1','Ubiquitine native ou His6 N-terminal clivable si affinité requise',
            'Éviter de bloquer l’extrémité C-terminale si la reconnaissance/conjugaison nécessite le di-Gly terminal.')
        tag('En secours','Construction à tag alternatif clivable',
            'MBP/SUMO ne sont pas nécessaires par défaut pour une petite ubiquitine correctement repliée.')
        hostopt('Priorité 1','E. coli pour la production de domaine isolé sans PTM indispensable',
                'La fonction de la machinerie d’ubiquitination sera testée séparément ; l’expression ne prouve pas l’activité.')
        quality.extend(['Si l’objectif est la conjugaison : contrôler l’intégrité du motif C-terminal di-Gly et réaliser un essai E1/E2/E3 adapté.',
                        'Pour une ubiquitine libre : contrôle de repliement/monodispersité et, si pertinent, reconnaissance par une déubiquitinase.'])
        rationale.append('La conservation de l’extrémité C-terminale est souvent plus décisive que l’ajout d’un tag solubilisant.')
    elif fam == 'GFP / protéine fluorescente':
        tag('Priorité 1','His6 clivable (construction simple de référence)',
            'Première construction simple ; n’ajouter MBP/SUMO que si un problème de solubilité ou maturation est observé.')
        tag('Secours','Fusion MBP/SUMO clivable seulement en cas d’échec',
            'Confirmer fluorescence ET solubilité de la cible après clivage.')
        hostopt('Priorité 1','E. coli BL21(DE3) si vecteur T7 compatible',
                'Protéine fluorescente classique généralement compatible avec une expression bactérienne.')
        quality.extend(['Fluorescence/absorbance : mesurer la maturation du chromophore avec témoin GFP positif.',
                        'Comparer la quantité de GFP soluble et la fluorescence spécifique après purification et clivage.'])
        rationale.append('GFP : privilégier une construction simple et un contrôle fonctionnel direct de fluorescence.')
    elif fam == 'Chaperonne / protéine de repliement':
        tag('Priorité 1','Construction native ou tag court clivable',
            'Ne pas gêner l’oligomérisation, le site ATPase éventuel ni l’interaction avec les protéines clientes.')
        tag('Conditionnel','Comparer position du tag si activité ou complexe altérés',
            'MBP/SUMO seulement en cas de difficulté de production documentée.')
        hostopt('Première intention','E. coli si cible bactérienne compatible, sinon hôte adapté à l’assemblage',
                'Les chaperonnes eucaryotes peuvent nécessiter des partenaires ou régulations supplémentaires.')
        quality.append('Mesurer repliement/agrégation d’une protéine cliente ; ATPase et cycle de co-chaperonnes selon la famille.')
        rationale.append('Pour une chaperonne, valider l’activité d’assistance au repliement, pas uniquement sa propre solubilité.')
    elif fam == 'Liaison ADN / ARN':
        tag('Priorité 1','Tag court clivable placé hors domaine de liaison',
            'Préserver la surface de reconnaissance d’acide nucléique et les éventuels domaines d’oligomérisation.')
        tag('Comparaison si nécessaire','N- versus C-terminal clivable',
            'Mesurer l’affinité/activité après clivage ; aucune orientation optimale déduite de la séquence seule.')
        hostopt('Selon organisme','E. coli pour cible simple ; HEK293/Sf9 si assemblage/modifications requis',
                'Le choix dépend des domaines et de la forme fonctionnelle recherchée.')
        quality.append('Contrôle de liaison ADN/ARN par EMSA, anisotropie de fluorescence ou méthode quantitative appropriée.')
        rationale.append('La localisation du site de liaison et l’assemblage déterminent le plan de construction.')
    elif fam == 'Lysozyme / enzyme muramidase':
        tag('Priorité 1','Domaine mature, éventuellement His6 clivable',
            'Vérifier la stabilité et l’accès au site catalytique, sans imposer MBP.')
        hostopt('Selon cible','E. coli pour lysozyme bactérien adapté ; environnement oxydant si disulfures nécessaires',
                'Les lysozymes diffèrent fortement entre espèces et structures.')
        quality.append('Mesurer activité muramidase sur substrat adapté, avec témoins actif et négatif.')
    elif fam == 'Enzyme (autre)':
        tag('Priorité 1','Tag court clivable à distance des domaines fonctionnels',
            'Les contraintes dépendent de l’activité annotée et des sites actifs ; ne pas proposer automatiquement une fusion MBP.')
        tag('Conditionnel','Fusion de solubilisation uniquement après essai de référence décevant',
            'Comparer l’activité spécifique après clivage, pas seulement l’expression.')
        hostopt('Selon fonction/PTM','E. coli si enzyme autonome compatible ; sinon système d’expression adapté',
                'Examiner les cofactors, modifications et partenaires avant de trancher.')
        quality.append('Dosage de l’activité spécifique sur substrat approprié ; contrôle des cofacteurs lorsque requis.')
        rationale.append('Pour une enzyme, l’activité spécifique peut primer sur la quantité soluble.')
    else:
        tag('Priorité 1','Construction simple avec tag court clivable (His6 ou Strep)',
            'Comparer l’expression et la purification ; l’identité biologique et la disponibilité des extrémités restent inconnues.')
        if p_is_known(prob, threshold) and float(prob)<float(threshold) and \
                float(prob) < float(threshold)-0.08 and not features.get('n_terminal_hydrophobic_window'):
            tag('Priorité 2 — essai comparatif','MBP clivable face à un tag court',
                'Piste pour une cible non annotée sous le seuil coût-métier ; comparer la solubilité de la cible après clivage.')
        else:
            tag('Secours','MBP/SUMO clivable après problème d’expression constaté',
                'Ne pas multiplier les fusions solubilisantes sans observations expérimentales justifiant un essai de secours.')
        if features.get('n_terminal_hydrophobic_window'):
            hostopt('À vérifier','Choisir l’hôte après validation peptide signal / TM',
                    'Descriptif hydrophobe N-terminal non spécifique ; ne pas présumer BL21(DE3).')
        else:
            hostopt('Point de départ sous conditions','E. coli BL21(DE3) si protéine soluble sans PTM indispensable et vecteur T7',
                    'Hypothèse pratique, non personnalisable sans identité et contraintes de production connues.')
        quality.append('Définir le véritable essai fonctionnel après annotation ou caractérisation biochimique de la cible.')
        rationale.append('Fonction non déterminée de manière fiable : aucun protocole spécifique n’est inventé.')

    if reliable and origin=='Eucaryote' and fam=='Liaison ADN / ARN':
        warnings.append('Cible eucaryote : vérifier domaines, état oligomérique, partenaires et modifications avant d’interpréter l’expression bactérienne.')
    if evidence_signal and not mem:
        warnings.append('Peptide signal annoté : examiner la frontière de la chaîne mature et la nécessité de sécrétion avant clonage.')
    if evidence_glyc and not declared_ptm:
        warnings.append('Des PTM sont annotées mais leur caractère indispensable est INCONNU : vérifier avant de privilégier HEK293.')
    if host!='Non défini':
        rationale.append('Hôte déclaré par l’utilisateur : '+host+' ; vérifier qu’il permet les fonctions et PTM requises.')

    return {'tags':tags,'hosts':hosts,'specific_validation':quality,
            'evidence':evidence,'warnings':warnings,'rationale':rationale,
            'annotation_used':reliable,'family_used':fam}


def p_is_known(p,threshold):
    return isinstance(p,(float,int)) and isinstance(threshold,(float,int)) and 0<=p<=1 and 0<threshold<1
