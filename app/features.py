"""
app/features.py — Calcul de features à partir de séquences FASTA
=================================================================
Point 4 du plan mentor : endpoint /predict/sequence

Calcule les 13 features physico-chimiques P7 directement depuis
une séquence d'acides aminés (format FASTA ou brut).

Dépendances :
    pip install biopython

Usage :
    from app.features import compute_features_from_sequence
    features = compute_features_from_sequence(">prot1\\nMKLLLPTINY...")
"""
import math
import re
from typing import Optional

import numpy as np

# ── Constantes physico-chimiques des acides aminés ────────────────────────
# Source : Kyte & Doolittle (1982), Chou-Fasman parameters
AA_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

AA_PKA = {  # pKa des groupes ionisables
    "D": 3.86, "E": 4.07, "H": 6.04, "C": 8.18, "Y": 10.46, "K": 10.54, "R": 12.48,
}
AA_AROMATIC = {"F", "Y", "W"}

# Masse moléculaire résiduelle (Da)
AA_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# Paramètres Chou-Fasman (helix/turn/sheet)
CF_HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
CF_SHEET = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}
CF_TURN = {
    "A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.43, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}

# Instabilité Guruprasad et al. (DIWV — dipeptide instability weights)
# Valeurs simplifiées (matrice complète 20×20 → version condensée)
INSTABILITY_WEIGHTS = {
    ("W", "W"): 1.0, ("C", "K"): 1.0, ("C", "W"): 1.0, ("H", "H"): 1.0,
    ("D", "E"): 7.5, ("D", "S"): 6.8, ("E", "K"): 5.0, ("A", "D"): 7.5,
    ("I", "L"): 0.0, ("F", "S"): 0.0, ("V", "I"): 0.0, ("L", "L"): 0.0,
}
DEFAULT_INSTABILITY = 1.0


def parse_fasta(sequence: str) -> str:
    """Extrait la séquence brute d'un format FASTA ou retourne la séquence telle quelle."""
    lines = sequence.strip().split("\n")
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    clean = "".join(seq_lines).upper()
    # Nettoyer les caractères non-AA standard
    clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", clean)
    return clean


def compute_gravy(seq: str) -> float:
    """GRAVY index (Grand Average of HydroPAtHicity)."""
    if not seq:
        return 0.0
    return sum(AA_HYDROPHOBICITY.get(aa, 0.0) for aa in seq) / len(seq)


def compute_isoelectric_point(seq: str) -> float:
    """Point isoélectrique (pI) par méthode dichotomique."""
    aa_counts = {aa: seq.count(aa) for aa in set(seq)}

    def net_charge(pH: float) -> float:
        charge = 0.0
        # N-terminal (alpha-NH2, pKa ≈ 8.0)
        charge += 1.0 / (1 + 10 ** (pH - 8.0))
        # C-terminal (alpha-COOH, pKa ≈ 3.1)
        charge -= 1.0 / (1 + 10 ** (3.1 - pH))
        # Chaînes latérales
        for aa, pka in AA_PKA.items():
            n = aa_counts.get(aa, 0)
            if n == 0:
                continue
            if aa in ("D", "E", "C", "Y"):  # acides → chargés négatif
                charge -= n / (1 + 10 ** (pka - pH))
            else:  # bases → chargés positif
                charge += n / (1 + 10 ** (pH - pka))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if net_charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_molecular_weight(seq: str) -> float:
    """Masse moléculaire en Da."""
    return sum(AA_MASS.get(aa, 110.0) for aa in seq) - 18.02 * (len(seq) - 1)


def compute_aromaticity(seq: str) -> float:
    """Fraction d'acides aminés aromatiques (F, Y, W)."""
    if not seq:
        return 0.0
    return sum(1 for aa in seq if aa in AA_AROMATIC) / len(seq)


def compute_instability(seq: str) -> float:
    """Indice d'instabilité de Guruprasad."""
    if len(seq) < 2:
        return 0.0
    total = sum(
        INSTABILITY_WEIGHTS.get((seq[i], seq[i + 1]), DEFAULT_INSTABILITY)
        for i in range(len(seq) - 1)
    )
    return 10.0 / len(seq) * total


def compute_secondary_structure(seq: str) -> tuple:
    """
    Fraction helix, turn, sheet (prédiction Chou-Fasman simplifiée).
    Retourne (pct_helix, pct_turn, pct_sheet).
    """
    if not seq:
        return 0.0, 0.0, 0.0
    h = sum(CF_HELIX.get(aa, 1.0) for aa in seq) / len(seq)
    t = sum(CF_TURN.get(aa, 1.0) for aa in seq) / len(seq)
    s = sum(CF_SHEET.get(aa, 1.0) for aa in seq) / len(seq)
    total = h + t + s
    if total == 0:
        return 0.33, 0.33, 0.34
    return h / total, t / total, s / total


def compute_features_from_sequence(sequence: str) -> dict:
    """
    Calcule les 13 features P7 à partir d'une séquence d'acides aminés.

    Args:
        sequence: Séquence brute ou format FASTA (avec ou sans header >)

    Returns:
        dict avec les 13 features P7

    Raises:
        ValueError: Si la séquence est trop courte ou invalide
    """
    seq = parse_fasta(sequence)

    if len(seq) < 10:
        raise ValueError(
            f"Séquence trop courte ({len(seq)} aa). Minimum : 10 acides aminés."
        )

    # Features de base
    pI = compute_isoelectric_point(seq)
    mw = compute_molecular_weight(seq)
    gravy = compute_gravy(seq)
    instability = compute_instability(seq)
    aromaticity = compute_aromaticity(seq)
    pct_helix, pct_turn, pct_sheet = compute_secondary_structure(seq)

    # Transformations log (comme dans le pipeline P7)
    log_mw = math.log1p(max(0, mw))
    log_instability = math.log1p(max(0, instability))

    # Normalisation GRAVY — utiliser les stats DeepSol (mean≈-0.2, std≈0.8)
    gravy_norm = (gravy - (-0.2)) / (0.8 + 1e-9)

    # Features dérivées
    pI_distance = abs(pI - 7.0)
    pct_coil = max(0.0, 1 - pct_helix - pct_turn - pct_sheet)
    helix_x_gravy = pct_helix * gravy_norm
    stability_score = 1.0 / (1 + log_instability)
    helix_sheet_ratio = pct_helix / (pct_sheet + 1e-9)

    return {
        "pI": round(pI, 4),
        "log_mw": round(log_mw, 4),
        "gravy_norm": round(gravy_norm, 4),
        "log_instability": round(log_instability, 4),
        "aromaticity": round(aromaticity, 4),
        "pct_helix": round(pct_helix, 4),
        "pct_turn": round(pct_turn, 4),
        "pct_sheet": round(pct_sheet, 4),
        "pI_distance": round(pI_distance, 4),
        "pct_coil": round(pct_coil, 4),
        "helix_x_gravy": round(helix_x_gravy, 4),
        "stability_score": round(stability_score, 4),
        "helix_sheet_ratio": round(helix_sheet_ratio, 4),
    }


def sequence_to_feature_array(sequence: str) -> np.ndarray:
    """Retourne les features sous forme de numpy array (pour la prédiction directe)."""
    features = compute_features_from_sequence(sequence)
    ordered = [
        "pI", "log_mw", "gravy_norm", "log_instability",
        "aromaticity", "pct_helix", "pct_turn", "pct_sheet",
        "pI_distance", "pct_coil", "helix_x_gravy",
        "stability_score", "helix_sheet_ratio",
    ]
    return np.array([[features[f] for f in ordered]], dtype=np.float32)
