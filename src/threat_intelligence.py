"""
HCAPT-Defense - Layer 3: Threat Intelligence Integration

Sits between Layer 2 (Ensemble) and Layer 4 (Zero-Trust). Queries the
Blockchain for known IoCs that match the current sample, and uses the
hits to ENRICH the ensemble's prediction:

  - if ensemble predicted Normal but blockchain has a high-similarity
    IoC labelled 'attack', raise the suspicion
  - if ensemble predicted attack and blockchain confirms, boost
    confidence
  - otherwise pass through unchanged

This is the data-plane counterpart to Zero-Trust (Innovation #3):
Zero-Trust uses a decorrelated classifier, the TI layer uses a
decorrelated knowledge base.

API mirrors ZeroTrustLayer.validate_batch() so they can be chained.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from blockchain import Blockchain, IoC
from config import NUM_CLASSES, LABEL_MAP
from differential_privacy import LaplaceDPSanitizer, DPConfig

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class TIEnrichment:
    """Per-sample result of a threat-intel lookup."""
    matched: bool = False
    top_similarity: float = 0.0
    top_class: int = 0             # 0 if no match
    top_severity: int = 0
    contributing_nodes: List[str] = field(default_factory=list)
    n_matches: int = 0


class ThreatIntelligenceLayer:
    def __init__(self,
                 blockchain: Blockchain,
                 similarity_threshold: float = 0.90,
                 class_vote_required: int = 2,
                 min_confidence_boost: float = 0.05):
        """
        similarity_threshold : float - cosine sim required to call a hit
        class_vote_required  : int   - how many IoCs must agree on class
                                       for a strong override
        min_confidence_boost : float - by how much to increase confidence
                                       when blockchain confirms the class
        """
        self.blockchain = blockchain
        self.similarity_threshold = similarity_threshold
        self.class_vote_required = class_vote_required
        self.min_confidence_boost = min_confidence_boost
        # Stats
        self.n_queries = 0
        self.n_matches = 0
        self.n_overrides = 0
        self.n_boosts = 0

    # ----- Lookups -----
    def lookup(self, X: np.ndarray) -> List[TIEnrichment]:
        """For each sample, query blockchain; return an enrichment record."""
        hits_per_sample = self.blockchain.query_batch(
            X, min_similarity=self.similarity_threshold,
        )
        out = []
        for hits in hits_per_sample:
            self.n_queries += 1
            if not hits:
                out.append(TIEnrichment())
                continue
            self.n_matches += 1
            # Majority vote on class
            votes: Dict[int, int] = {}
            sev_sum, sev_count = 0, 0
            nodes = set()
            for sim, ioc in hits:
                votes[ioc.attack_class] = votes.get(ioc.attack_class, 0) + 1
                sev_sum += ioc.severity; sev_count += 1
                nodes.add(ioc.source_node)
            top_class = max(votes, key=votes.get)
            out.append(TIEnrichment(
                matched=True,
                top_similarity=hits[0][0],
                top_class=top_class,
                top_severity=int(round(sev_sum / max(sev_count, 1))),
                contributing_nodes=sorted(nodes),
                n_matches=len(hits),
            ))
        return out

    # ----- Enrichment of predictions -----
    def enrich(self, X: np.ndarray,
               ens_preds: np.ndarray,
               ens_confs: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                  List[TIEnrichment]]:
        """Use blockchain matches to modify predictions.

        Rules:
          - If blockchain vote says 'attack' with enough agreement and
            ensemble said Normal with low-to-mid confidence, OVERRIDE.
          - If blockchain agrees with ensemble's attack class, BOOST
            confidence by min_confidence_boost.
          - Else leave unchanged.
        """
        enriched_preds = ens_preds.copy()
        enriched_confs = ens_confs.copy()
        enrichments = self.lookup(X)

        for i, enr in enumerate(enrichments):
            if not enr.matched:
                continue
            if enr.n_matches < self.class_vote_required:
                continue

            is_ens_attack = ens_preds[i] != 0
            ti_says_attack = enr.top_class != 0

            # Override: ensemble said Normal but blockchain says attack
            if not is_ens_attack and ti_says_attack and ens_confs[i] < 0.95:
                enriched_preds[i] = enr.top_class
                # Confidence = sim * (sev/5)
                enriched_confs[i] = min(
                    1.0,
                    max(enr.top_similarity * (enr.top_severity / 5.0),
                        ens_confs[i] + self.min_confidence_boost)
                )
                self.n_overrides += 1

            # Boost: both agree on attack class
            elif is_ens_attack and ti_says_attack and \
                    enr.top_class == ens_preds[i]:
                enriched_confs[i] = min(1.0,
                                         ens_confs[i] + self.min_confidence_boost)
                self.n_boosts += 1

        return enriched_preds, enriched_confs, enrichments

    def stats(self) -> Dict:
        match_rate = self.n_matches / max(self.n_queries, 1)
        return {
            "queries": self.n_queries,
            "matches": self.n_matches,
            "match_rate": match_rate,
            "overrides": self.n_overrides,
            "boosts": self.n_boosts,
            "similarity_threshold": self.similarity_threshold,
            "blockchain_size": len(self.blockchain.chain),
            "blockchain_iocs": len(self.blockchain.all_iocs()),
        }


# ============================================================
#  Helper: harvest IoCs from an ensemble's correct attack predictions
# ============================================================
def harvest_iocs_from_node(
    node_id: str,
    X: np.ndarray,
    y_true: np.ndarray,
    ens_preds: np.ndarray,
    ens_confs: np.ndarray,
    min_confidence: float = 0.85,
    max_iocs: int = 500,
    severity_map: Optional[Dict[int, int]] = None,
    dp_config: Optional[DPConfig] = None,
) -> List[IoC]:
    """Produce IoCs from samples where the ensemble correctly detected
    an attack with high confidence.

    severity_map : {attack_class: severity 1-5}. Default: DoS=3, Probe=2,
                   R2L=4, U2R=5.
    dp_config    : if provided, apply Laplace-DP noise to each feature
                   vector before storing it as an IoC.
    """
    severity_map = severity_map or {0: 0, 1: 3, 2: 2, 3: 4, 4: 5}
    mask = (ens_preds == y_true) & (ens_preds != 0) & (ens_confs >= min_confidence)
    idx = np.where(mask)[0]
    # Randomly sub-sample if too many
    rng = np.random.default_rng(42)
    if len(idx) > max_iocs:
        idx = rng.choice(idx, size=max_iocs, replace=False)

    # Apply DP sanitization if requested (privacy-preserving IoCs)
    sanitizer = LaplaceDPSanitizer(dp_config) if dp_config is not None else None

    iocs = []
    for i in idx:
        vec = X[i]
        if sanitizer is not None:
            vec = sanitizer.sanitize(vec)
        iocs.append(IoC(
            feature_vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
            attack_class=int(ens_preds[i]),
            severity=severity_map.get(int(ens_preds[i]), 3),
            confidence=float(ens_confs[i]),
            source_node=node_id,
            timestamp=time.time(),
        ))
    return iocs
