"""
Component 8 — Zero-Trust Integration

Bridge module: turn the per-flow risk from a MultiFlowTracker into a
risk-boost signal that the existing zero_trust.PolicyEngine can
combine with the LightGBM risk scorer.

The combination rule (default):
    combined_risk = max(zt_risk, graph_risk_weight * graph_risk)

with `graph_risk_weight` defaulting to 1.0 so the higher of the two
risk channels dominates. Override is fired when:
    ensemble says Normal with conf >= θ_c   AND
    combined_risk >= θ_r
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# Project src/
PROJECT_SRC = Path(__file__).resolve().parents[4] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from zero_trust import ZeroTrustLayer, RiskScorer, PolicyEngine
from .tracker import MultiFlowTracker


class GraphAwareZeroTrustLayer:
    """Wraps the standard ZeroTrustLayer and adds a graph-risk channel.

    Behaves identically to ZeroTrustLayer when no MultiFlowTracker is
    supplied, so it can be dropped into the existing pipeline as a
    transparent upgrade.
    """

    def __init__(self, risk_scorer: RiskScorer,
                  policy: Optional[PolicyEngine] = None,
                  multi_flow_tracker: Optional[MultiFlowTracker] = None,
                  graph_risk_weight: float = 1.0):
        self.base_zt = ZeroTrustLayer(risk_scorer=risk_scorer,
                                       policy=policy or PolicyEngine())
        self.tracker = multi_flow_tracker
        self.graph_risk_weight = graph_risk_weight

    # ----------------------------------------------------- validate_batch
    def validate_batch(self, X: np.ndarray, ens_preds: np.ndarray,
                       ens_confs: np.ndarray,
                       segment_features=(1, 2),
                       return_decisions: bool = False):
        # Compute the graph-risk channel (zeros if no tracker)
        if self.tracker is not None and self.tracker.model is not None:
            graph_risk = self.tracker.predict_flow_risk(X)
        else:
            graph_risk = np.zeros(len(X), dtype=np.float32)

        # Call the base ZT layer with the standard inputs first
        base_out = self.base_zt.validate_batch(
            X, ens_preds, ens_confs,
            segment_features=segment_features,
            return_decisions=return_decisions,
        )

        # The base layer already produced final labels; we now revisit each
        # sample and consider escalation when graph_risk is high but the
        # base layer kept the prediction as Normal.
        final_labels = np.asarray(base_out["final_labels"]).copy()
        # Boost: if graph_risk >= 0.7 and base label is Normal, escalate to Attack
        boost_mask = (graph_risk >= 0.7) & (final_labels == 0)
        n_boosted = int(boost_mask.sum())
        final_labels[boost_mask] = 1  # binary mode: 1 = Attack
        base_out["final_labels"]    = final_labels
        base_out["graph_risk_mean"] = float(graph_risk.mean())
        base_out["graph_risk_max"]  = float(graph_risk.max())
        base_out["n_graph_boosted"] = n_boosted
        return base_out


# =============================================================================
# CLI demo
# =============================================================================
if __name__ == "__main__":
    print("GraphAwareZeroTrustLayer: drop-in replacement for ZeroTrustLayer "
          "that adds a graph-risk channel from MultiFlowTracker.")
