"""Règle d'inférence V7, choisie après étude exploratoire sur DeepSol Tune.

Aucun checkpoint, calibrateur ou seuil n'est réentraîné. La calibration 512 n'est PAS
validée pour 1022. Pour > 1022 AA, le repli 512 est conservateur et explicite.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LengthPolicy:
    mode: str
    residue_limit: int
    experimental: bool
    explanation: str


def policy_for_length(length: int) -> LengthPolicy:
    if not isinstance(length, int) or length < 1:
        raise ValueError('Longueur de séquence invalide')
    if length <= 512:
        return LengthPolicy(
            mode='reference_512', residue_limit=512, experimental=False,
            explanation='Inférence de référence, compatible avec la calibration historique à 512 AA.'
        )
    if length <= 1022:
        return LengthPolicy(
            mode='experimental_1022', residue_limit=1022, experimental=True,
            explanation=('Inférence exploratoire sur la séquence complète, jusqu’à 1 022 AA. '
                         'Le calibrateur et le seuil ont été établis pour 512 AA : '
                         'le score et la classe obtenus ici ne sont pas recalibrés/validés indépendamment.')
        )
    return LengthPolicy(
        mode='fallback_512_over_1022', residue_limit=512, experimental=False,
        explanation=('Séquence de plus de 1 022 AA : retour prudent au modèle 512 AA. '
                     'Le réseau ignore la partie après le résidu 512 ; '
                     'Bio13 utilise toujours la séquence entière. '
                     'La fiabilité de ce repli pour les très longues séquences est limitée.')
    )
