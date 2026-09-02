"""
LSH-based similarity + class Prototypes for noise-robust IoC matching.

Two complementary techniques:

1. CLASS PROTOTYPES - share ONE average vector per class per client
   instead of per-sample IoCs. Averaging reduces DP noise variance by
   sqrt(N), so cosine similarity remains usable even after Gaussian/Laplace
   sanitization.

2. LSH (Locality-Sensitive Hashing) - project vectors onto random
   hyperplanes; store signs as binary hash. Hamming distance between
   hashes approximates cosine similarity. Robust to moderate DP noise
   because majority-sign survives.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


# ============================================================
#  Class Prototypes
# ============================================================
def compute_class_prototypes(X: np.ndarray, y: np.ndarray,
                              num_classes: int = 5,
                              prototypes_per_class: int = 1
                              ) -> Dict[int, np.ndarray]:
    """Return a dictionary {class_id: prototype_vector}.

    prototypes_per_class : if > 1, use k-means within the class.
    """
    protos: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        mask = y == c
        if mask.sum() == 0:
            continue
        X_c = X[mask]
        if prototypes_per_class == 1:
            protos[c] = X_c.mean(axis=0)
        else:
            from sklearn.cluster import KMeans
            k = min(prototypes_per_class, len(X_c))
            km = KMeans(n_clusters=k, random_state=42, n_init=4)
            km.fit(X_c)
            for i, ctr in enumerate(km.cluster_centers_):
                protos[f"{c}_{i}"] = ctr   # synthetic key
    return protos


# ============================================================
#  LSH (random projection)
# ============================================================
@dataclass
class LSHIndex:
    n_bits: int                    # signature length
    planes: np.ndarray             # (n_bits, feature_dim)
    seed: int = 42

    @classmethod
    def build(cls, feature_dim: int, n_bits: int = 64,
              seed: int = 42) -> "LSHIndex":
        rng = np.random.default_rng(seed)
        planes = rng.normal(size=(n_bits, feature_dim)).astype(np.float32)
        return cls(n_bits=n_bits, planes=planes, seed=seed)

    def hash(self, X: np.ndarray) -> np.ndarray:
        """Return binary signatures of shape (N, n_bits) as uint8."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Project then sign
        projections = X @ self.planes.T                 # (N, n_bits)
        return (projections > 0).astype(np.uint8)

    def hamming_similarity(self, sig1: np.ndarray, sig2: np.ndarray
                            ) -> float:
        """1 - hamming_distance / n_bits → [0, 1]."""
        return 1.0 - float(np.abs(sig1 - sig2).sum()) / self.n_bits

    def query(self, X_query: np.ndarray, signatures: np.ndarray,
              threshold: float = 0.70) -> List[List[Tuple[int, float]]]:
        """For each query row, return list of (index, similarity) in
        signatures with sim >= threshold."""
        q_sigs = self.hash(X_query)
        results = []
        for q in q_sigs:
            # Broadcast XOR to compute distances
            distances = np.abs(signatures.astype(np.int8) - q).sum(axis=1)
            sims = 1.0 - distances / self.n_bits
            hits = np.where(sims >= threshold)[0]
            ranked = sorted(hits, key=lambda i: -sims[i])
            results.append([(int(i), float(sims[i])) for i in ranked])
        return results
