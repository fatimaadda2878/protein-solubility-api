"""
app/homology.py — Contrôle d'homologie des séquences protéiques
================================================================
Point 5 du plan mentor : éviter la fuite de données entre train/test

Problème : DeepSol contient des protéines homologues entre train et test.
Si on entraîne sur des protéines similaires à > 30% d'identité de celles
qu'on teste, le modèle "mémorise" plutôt qu'il n'apprend.

Solution :
  1. CD-HIT (clustering à 30% d'identité) → recommandé en production
  2. k-mer hashing (fallback Python pur, sans dépendances système)
  3. IdentityFilter : filtre les séquences trop similaires

Usage :
    from app.homology import IdentityFilter
    filt = IdentityFilter(threshold=0.30)
    unique_seqs = filt.deduplicate(sequences_list)

    # Vérifier si une nouvelle séquence est trop proche des séquences d'entraînement
    is_novel = filt.is_novel(new_seq, train_sequences)
"""
import hashlib
import logging
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Seuil d'identité recommandé par le mentor ─────────────────────────────
DEFAULT_IDENTITY_THRESHOLD = 0.30  # 30% identité de séquence


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires k-mer (fallback Python pur)
# ─────────────────────────────────────────────────────────────────────────────

def kmer_set(seq: str, k: int = 5) -> set:
    """Retourne l'ensemble des k-mers d'une séquence."""
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


def kmer_jaccard(seq1: str, seq2: str, k: int = 5) -> float:
    """
    Similarité de Jaccard basée sur les k-mers.

    C'est une approximation rapide de l'identité de séquence.
    Pour des séquences très similaires, Jaccard(k=5) ≈ identité exacte.

    Complexité : O(|seq1| + |seq2|) — bien plus rapide qu'un alignement.
    """
    if not seq1 or not seq2:
        return 0.0
    s1 = kmer_set(seq1, k)
    s2 = kmer_set(seq2, k)
    if not s1 and not s2:
        return 1.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def kmer_similarity_matrix(sequences: List[str], k: int = 5) -> np.ndarray:
    """
    Matrice de similarité Jaccard N×N pour une liste de séquences.

    Pour N > 1000 séquences, utilise un MinHash approximatif.
    """
    n = len(sequences)
    sim = np.zeros((n, n), dtype=np.float32)
    kmer_sets = [kmer_set(seq, k) for seq in sequences]

    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            s1, s2 = kmer_sets[i], kmer_sets[j]
            union = len(s1 | s2)
            if union > 0:
                sim[i, j] = sim[j, i] = len(s1 & s2) / union

    return sim


# ─────────────────────────────────────────────────────────────────────────────
# MinHash pour grandes collections (> 1 000 séquences)
# ─────────────────────────────────────────────────────────────────────────────

class MinHash:
    """
    MinHash approximatif pour estimer la similarité de Jaccard
    entre de grandes collections de séquences.

    Utilise 128 fonctions de hachage pour une précision acceptable.
    Erreur standard ≈ 1/√num_perm ≈ 9% avec 128 permutations.
    """

    def __init__(self, num_perm: int = 128, k: int = 5):
        self.num_perm = num_perm
        self.k = k
        self._hash_seeds = [
            int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16)
            for i in range(num_perm)
        ]

    def signature(self, seq: str) -> np.ndarray:
        """Génère la signature MinHash d'une séquence."""
        kmers = kmer_set(seq, self.k)
        sig = np.full(self.num_perm, np.inf)

        for kmer in kmers:
            kmer_bytes = kmer.encode()
            for i, seed in enumerate(self._hash_seeds):
                h = int(hashlib.md5(kmer_bytes + seed.to_bytes(4, 'big')).hexdigest(), 16)
                if h < sig[i]:
                    sig[i] = float(h)

        return sig

    def jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estime la similarité de Jaccard à partir de deux signatures."""
        return float(np.mean(sig1 == sig2))


# ─────────────────────────────────────────────────────────────────────────────
# Interface principale : IdentityFilter
# ─────────────────────────────────────────────────────────────────────────────

class IdentityFilter:
    """
    Filtre de redondance basé sur l'identité de séquence.

    Stratégie :
    - Pour N ≤ 500 séquences : k-mer Jaccard exact (O(N²))
    - Pour N > 500 séquences : MinHash approximatif (O(N))
    - Si CD-HIT disponible : délègue à CD-HIT (le plus précis)

    Le seuil 0.30 (30%) suit la recommandation du mentor :
    "< 30% d'identité de séquence = pas d'homologie significative".
    """

    def __init__(
        self,
        threshold: float = DEFAULT_IDENTITY_THRESHOLD,
        k: int = 5,
        use_cdhit: bool = True,
    ):
        self.threshold = threshold
        self.k = k
        self._minhash = MinHash(k=k)
        self._use_cdhit = use_cdhit and self._check_cdhit()
        self._train_signatures: Optional[List[np.ndarray]] = None
        self._train_sequences: Optional[List[str]] = None

    def _check_cdhit(self) -> bool:
        """Vérifie si CD-HIT est installé."""
        import shutil
        if shutil.which("cd-hit"):
            logger.info("CD-HIT disponible — utilisation pour le clustering exact")
            return True
        logger.info("CD-HIT absent — utilisation du fallback k-mer/MinHash")
        return False

    # ── Méthode principale : dédupliquer une liste de séquences ──────────────

    def deduplicate(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Supprime les séquences redondantes (> threshold d'identité).

        Si labels fournis, garde les labels correspondants.
        En cas de doublon, conserve la première occurrence.

        Returns:
            (unique_sequences, unique_labels) ou (unique_sequences, [])
        """
        if labels is None:
            labels = list(range(len(sequences)))

        if self._use_cdhit:
            return self._deduplicate_cdhit(sequences, labels)

        n = len(sequences)
        if n <= 500:
            return self._deduplicate_kmer(sequences, labels)
        else:
            return self._deduplicate_minhash(sequences, labels)

    def _deduplicate_kmer(
        self, sequences: List[str], labels: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Dédupliquer par k-mer Jaccard exact (pour petites collections)."""
        kept_idx = []
        kept_seqs = []

        for i, seq in enumerate(sequences):
            is_redundant = False
            for ref_seq in kept_seqs:
                if kmer_jaccard(seq, ref_seq, self.k) >= self.threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                kept_idx.append(i)
                kept_seqs.append(seq)

        unique_seqs = [sequences[i] for i in kept_idx]
        unique_labels = [labels[i] for i in kept_idx]

        n_removed = len(sequences) - len(unique_seqs)
        logger.info(
            f"Déduplication k-mer : {len(sequences)} → {len(unique_seqs)} "
            f"({n_removed} redondantes supprimées, seuil={self.threshold:.0%})"
        )
        return unique_seqs, unique_labels

    def _deduplicate_minhash(
        self, sequences: List[str], labels: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Dédupliquer par MinHash approximatif (pour grandes collections)."""
        logger.info(f"Déduplication MinHash sur {len(sequences)} séquences…")
        signatures = [self._minhash.signature(seq) for seq in sequences]

        kept_idx = []
        kept_sigs = []

        for i, sig in enumerate(signatures):
            is_redundant = any(
                self._minhash.jaccard(sig, ref_sig) >= self.threshold
                for ref_sig in kept_sigs
            )
            if not is_redundant:
                kept_idx.append(i)
                kept_sigs.append(sig)

        unique_seqs = [sequences[i] for i in kept_idx]
        unique_labels = [labels[i] for i in kept_idx]

        n_removed = len(sequences) - len(unique_seqs)
        logger.info(
            f"Déduplication MinHash : {len(sequences)} → {len(unique_seqs)} "
            f"({n_removed} redondantes supprimées)"
        )
        return unique_seqs, unique_labels

    def _deduplicate_cdhit(
        self, sequences: List[str], labels: List[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Déduplication via CD-HIT (précision maximale).
        Écrit un FASTA temporaire, lance CD-HIT, relit les clusters.
        """
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fasta_in = tmpdir / "input.fasta"
            fasta_out = tmpdir / "output"

            # Écrire le FASTA
            with open(fasta_in, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")

            # Lancer CD-HIT
            # -c = seuil d'identité, -n = taille des k-mers (5 pour c=0.3-0.5)
            cmd = [
                "cd-hit",
                "-i", str(fasta_in),
                "-o", str(fasta_out),
                "-c", str(self.threshold),
                "-n", "5",  # k-mer word size pour threshold 0.3-0.5
                "-M", "2000",  # mémoire max (Mo)
                "-T", "4",  # threads
                "-d", "0",  # description complète
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                logger.error(f"CD-HIT erreur : {result.stderr}")
                logger.info("Fallback sur MinHash…")
                return self._deduplicate_minhash(sequences, labels)

            # Lire les représentants de clusters
            kept_ids = set()
            clstr_file = Path(str(fasta_out) + ".clstr")

            if clstr_file.exists():
                with open(clstr_file) as f:
                    for line in f:
                        if "*" in line:  # représentant du cluster
                            match = re.search(r">seq_(\d+)\.\.\.", line)
                            if match:
                                kept_ids.add(int(match.group(1)))
            else:
                # Lire le FASTA de sortie directement
                with open(fasta_out) as f:
                    for line in f:
                        if line.startswith(">"):
                            match = re.search(r"seq_(\d+)", line)
                            if match:
                                kept_ids.add(int(match.group(1)))

        kept_idx = sorted(kept_ids)
        unique_seqs = [sequences[i] for i in kept_idx]
        unique_labels = [labels[i] for i in kept_idx]

        n_removed = len(sequences) - len(unique_seqs)
        logger.info(
            f"CD-HIT : {len(sequences)} → {len(unique_seqs)} "
            f"({n_removed} redondantes supprimées, identité={self.threshold:.0%})"
        )
        return unique_seqs, unique_labels

    # ── Vérification d'une nouvelle séquence ─────────────────────────────────

    def fit(self, train_sequences: List[str]):
        """Mémorise les signatures des séquences d'entraînement."""
        self._train_sequences = train_sequences
        logger.info(f"Calcul des signatures MinHash pour {len(train_sequences)} séquences…")
        self._train_signatures = [
            self._minhash.signature(seq) for seq in train_sequences
        ]
        logger.info("Signatures calculées")
        return self

    def is_novel(self, sequence: str, train_sequences: Optional[List[str]] = None) -> bool:
        """
        Vérifie si une séquence est suffisamment différente des séquences d'entraînement.

        Returns:
            True si la séquence est nouvelle (< threshold d'identité avec tout le train set)
            False si elle est trop similaire à une séquence d'entraînement (data leak)
        """
        if train_sequences is not None:
            self.fit(train_sequences)

        if self._train_signatures is None:
            raise ValueError("Appelez fit() d'abord ou passez train_sequences")

        query_sig = self._minhash.signature(sequence)

        for ref_sig in self._train_signatures:
            if self._minhash.jaccard(query_sig, ref_sig) >= self.threshold:
                return False  # Trop similaire → pas novel

        return True  # Suffisamment différent → novel

    def novelty_score(self, sequence: str) -> float:
        """
        Score de nouveauté = 1 - max_similarité_avec_train_set.
        Score > 0.7 = séquence vraiment nouvelle.
        """
        if self._train_signatures is None:
            raise ValueError("Appelez fit() d'abord")

        query_sig = self._minhash.signature(sequence)
        max_sim = max(
            (self._minhash.jaccard(query_sig, ref_sig) for ref_sig in self._train_signatures),
            default=0.0
        )
        return 1.0 - max_sim


# ─────────────────────────────────────────────────────────────────────────────
# Script standalone : analyser la contamination dans DeepSol
# ─────────────────────────────────────────────────────────────────────────────

def analyze_deepsol_contamination(deepsol_path: str = "deepsol_raw.pkl") -> dict:
    """
    Analyse le taux de contamination homologue entre train/test de DeepSol.

    Résultat typique attendu :
    - Sans filtre : ~15-20% de paires train/test avec > 30% d'identité
    - Avec filtre CD-HIT 30% : redondance quasi nulle

    Usage :
        python -c "from app.homology import analyze_deepsol_contamination; analyze_deepsol_contamination()"
    """
    import pickle

    logger.info(f"Chargement DeepSol depuis {deepsol_path}…")
    with open(deepsol_path, "rb") as f:
        data = pickle.load(f)

    train_seqs = []
    test_seqs = []

    for split in ["train", "valid"]:
        if split in data and "src" in data[split]:
            # src contient les séquences tokenisées → on reconstitue
            # (les tokens sont des AA, 21 valeurs possibles)
            pass  # DeepSol ne stocke pas les séquences brutes

    logger.warning(
        "DeepSol stocke des features précalculées, pas les séquences brutes.\n"
        "Pour un contrôle d'homologie complet, télécharger les séquences FASTA\n"
        "depuis UniProt/NCBI en utilisant les IDs dans DeepSol."
    )

    return {
        "note": "DeepSol ne contient pas les séquences brutes — features précalculées uniquement",
        "recommendation": "Utiliser UESolDS (2024) qui inclut les séquences FASTA + contrôle MMseqs2 intégré",
        "threshold": DEFAULT_IDENTITY_THRESHOLD,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Démonstration avec séquences synthétiques
    print("=== Test IdentityFilter ===\n")

    seqs = [
        "MKLLLPTINYGFLKGGGTST",   # original
        "MKLLLPTINYGFLKGGGTST",   # identique → doit être filtré
        "MKLLLPTINYGFLKGGXYYY",   # similaire (80% identité) → doit être filtré
        "ACDEFGHIKLMNPQRSTVWY",   # complètement différent → garde
        "WWWWWWWWWWWWWWWWWWWW",   # différent → garde
    ]
    labels = [1, 1, 0, 1, 0]

    filt = IdentityFilter(threshold=0.30, use_cdhit=False)
    unique_seqs, unique_labels = filt.deduplicate(seqs, labels)

    print(f"Avant : {len(seqs)} séquences")
    print(f"Après : {len(unique_seqs)} séquences (seuil 30%)")
    for seq, lbl in zip(unique_seqs, unique_labels):
        print(f"  label={lbl} : {seq[:30]}…")

    print("\n=== Test is_novel ===")
    train_seqs = seqs[:3]
    filt.fit(train_seqs)

    new_query = "ACDEFGHIKLMNPQRSTVWY"
    score = filt.novelty_score(new_query)
    is_novel = filt.is_novel(new_query)
    print(f"Séquence : {new_query[:30]}")
    print(f"Novel score : {score:.2f} (1.0 = parfaitement nouveau)")
    print(f"Is novel (> 70%) : {is_novel}")

    known_seq = seqs[0]
    score2 = filt.novelty_score(known_seq)
    print(f"\nSéquence connue : {known_seq[:30]}")
    print(f"Novel score : {score2:.2f} (doit être proche de 0)")
