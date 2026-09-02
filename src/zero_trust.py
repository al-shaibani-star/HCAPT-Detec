"""
HCAPT-Defense - Innovation #3: Zero-Trust Validation Layer
Layer 4: Dynamic Policies + Risk-Based Access + Micro-Segmentation

Core principle: "Never trust, always verify" - every prediction from
the Ensemble (Innovation #2) is independently validated before action.

Components:
  1. RiskScorer      - independent binary classifier (LightGBM)
                       uses a different feature subset so its errors
                       are decorrelated from the Ensemble's.
  2. ContextAnalyzer - looks at session history (per-segment) and
                       flags anomalous patterns.
  3. PolicyEngine    - rule-based decision matrix that maps
                       (ens_pred, ens_conf, risk, context) -> Action.
  4. ZeroTrustLayer  - orchestrator with audit log.

Actions:
  ALLOW      - pass through (ensemble says Normal, risk agrees)
  MONITOR    - pass but log for review (low confidence)
  CHALLENGE  - disagreement! ensemble says Normal but risk says Attack.
               This catches the "confidently wrong" problem.
  QUARANTINE - confirmed attack, moderate risk
  BLOCK      - high-confidence attack, high risk
"""
import json
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import xgboost as xgb

from config import MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  Actions
# ============================================================
class Action(str, Enum):
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    CHALLENGE = "CHALLENGE"
    QUARANTINE = "QUARANTINE"
    BLOCK = "BLOCK"


@dataclass
class Decision:
    action: Action
    ensemble_pred: int          # 0..NUM_CLASSES-1
    ensemble_conf: float        # [0, 1]
    risk_score: float           # [0, 1]
    context_risk: float         # [0, 1]
    final_label: int            # system's best guess (may override ens if CHALLENGE)
    rationale: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        d = asdict(self)
        d["action"] = self.action.value
        return d


# ============================================================
#  Risk Scorer - independent binary classifier
# ============================================================
class RiskScorer:
    """Binary (Normal vs. Attack) classifier trained with deliberately
    different hyperparameters + feature subset from the Ensemble to
    ensure its errors are decorrelated.
    """

    def __init__(self, feature_indices: Optional[List[int]] = None,
                 model_type: str = "auto"):
        self.feature_indices = feature_indices
        self.model = None
        self.model_type = model_type  # 'lgb' | 'xgb' | 'auto'

    def _subset_features(self, X: np.ndarray) -> np.ndarray:
        if self.feature_indices is None:
            return X
        return X[:, self.feature_indices]

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train on binary labels (0=Normal, 1=Attack)."""
        y_bin_tr = (y_tr != 0).astype(np.int32)
        y_bin_val = (y_val != 0).astype(np.int32)

        # Select diverse feature subset if not provided (use every other feature)
        if self.feature_indices is None:
            self.feature_indices = list(range(0, X_tr.shape[1], 2))  # ~half
        X_tr_sub = self._subset_features(X_tr)
        X_val_sub = self._subset_features(X_val)

        # Prefer LightGBM for diversity from XGBoost in the ensemble
        if HAS_LGB and self.model_type in ("auto", "lgb"):
            self.model = lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,               # shallower than ensemble XGB (depth 8)
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED + 777,   # different seed
                class_weight="balanced",
                verbose=-1,
            )
            self.model_type = "lgb"
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,                # much shallower -> different bias
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                objective="binary:logistic",
                tree_method="hist",
                device="cuda",
                random_state=SEED + 777,
                eval_metric="logloss",
            )
            self.model_type = "xgb"

        print(f"[RiskScorer] Training {self.model_type.upper()} on "
              f"{len(X_tr_sub)} samples, {X_tr_sub.shape[1]} features")
        t0 = time.time()
        self.model.fit(X_tr_sub, y_bin_tr)
        train_time = time.time() - t0

        # Validate
        p_val = self.model.predict_proba(X_val_sub)[:, 1]
        pred_val = (p_val >= 0.5).astype(int)
        acc = accuracy_score(y_bin_val, pred_val)
        print(f"[RiskScorer] Val accuracy (binary): {acc:.4f} "
              f"| train time: {train_time:.1f}s")

        return {"binary_accuracy": float(acc),
                "train_time_s": train_time,
                "feature_subset_size": len(self.feature_indices),
                "model_type": self.model_type}

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for each sample, in [0, 1]."""
        X_sub = self._subset_features(X)
        return self.model.predict_proba(X_sub)[:, 1].astype(np.float32)

    def save(self, path: Path = MODELS_DIR):
        if self.model_type == "lgb":
            joblib.dump({"model": self.model,
                         "features": self.feature_indices,
                         "type": self.model_type},
                        path / "zt_risk_scorer.joblib")
        else:
            joblib.dump({"features": self.feature_indices,
                         "type": self.model_type},
                        path / "zt_risk_scorer_meta.joblib")
            self.model.save_model(str(path / "zt_risk_scorer.json"))

    def load(self, path: Path = MODELS_DIR):
        meta_file = path / "zt_risk_scorer.joblib"
        if meta_file.exists():
            obj = joblib.load(meta_file)
            if obj["type"] == "lgb":
                self.model = obj["model"]
                self.feature_indices = obj["features"]
                self.model_type = "lgb"
                return self
        # XGB case
        meta = joblib.load(path / "zt_risk_scorer_meta.joblib")
        self.feature_indices = meta["features"]
        self.model_type = meta["type"]
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path / "zt_risk_scorer.json"))
        return self


# ============================================================
#  Context Analyzer - session + segment tracking
# ============================================================
class ContextAnalyzer:
    """Track per-segment activity (protocol_type + service) and flag
    anomalous bursts. This simulates micro-segmentation at the
    Zero-Trust layer.

    In a real deployment, segments would be per-client-IP. Here we
    quantize based on feature indices 1 (protocol_type) and 2 (service).
    """

    def __init__(self, window: int = 200, burst_threshold: int = 50):
        self.window = window
        self.burst_threshold = burst_threshold
        # {(proto, svc): deque of recent (timestamp, was_attack_flag)}
        self.segments: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def update(self, segment_key: Tuple[int, int], was_flagged: bool):
        self.segments[segment_key].append((time.time(), was_flagged))

    def context_risk(self, segment_key: Tuple[int, int]) -> float:
        """Return [0, 1] risk based on recent activity in this segment."""
        recent = self.segments[segment_key]
        if len(recent) == 0:
            return 0.0
        # Fraction of recent activity that was flagged
        flagged_rate = sum(1 for _, f in recent if f) / len(recent)
        # Burst factor: very recent activity volume
        burst_factor = min(len(recent) / self.burst_threshold, 1.0)
        # Combine
        return 0.7 * flagged_rate + 0.3 * burst_factor


# ============================================================
#  Policy Engine
# ============================================================
class PolicyEngine:
    """Deterministic rule-based decision matrix. Thresholds are tunable."""

    def __init__(self,
                 high_conf: float = 0.85,
                 mid_conf: float = 0.55,
                 high_risk: float = 0.70,
                 mid_risk: float = 0.40):
        self.high_conf = high_conf
        self.mid_conf = mid_conf
        self.high_risk = high_risk
        self.mid_risk = mid_risk

    def decide(self, ens_pred: int, ens_conf: float,
               risk_score: float, context_risk: float) -> Tuple[Action, int, str]:
        """Return (action, final_label, rationale)."""
        is_ens_attack = ens_pred != 0    # class 0 = Normal
        effective_risk = max(risk_score, context_risk * 0.7)

        # --- DISAGREEMENT: ensemble says Normal but risk says Attack ---
        if not is_ens_attack and ens_conf >= self.high_conf \
                and risk_score >= self.high_risk:
            return (Action.CHALLENGE, 1,  # override to 'attack' (class 1=DoS proxy)
                    f"Disagreement: ens=Normal(conf={ens_conf:.2f}) "
                    f"but risk={risk_score:.2f} - possible missed detection")

        # --- Confirmed attacks (both agree) ---
        if is_ens_attack and ens_conf >= self.high_conf \
                and effective_risk >= self.high_risk:
            return (Action.BLOCK, ens_pred,
                    f"High-conf attack: ens={INV_LABEL_MAP[ens_pred]}"
                    f"(conf={ens_conf:.2f}), risk={effective_risk:.2f}")

        if is_ens_attack and ens_conf >= self.mid_conf \
                and effective_risk >= self.mid_risk:
            return (Action.QUARANTINE, ens_pred,
                    f"Moderate attack: ens={INV_LABEL_MAP[ens_pred]}"
                    f"(conf={ens_conf:.2f}), risk={effective_risk:.2f}")

        # --- Low-confidence anything ---
        if ens_conf < self.mid_conf:
            return (Action.MONITOR, ens_pred,
                    f"Low confidence: conf={ens_conf:.2f}")

        # --- High-confidence Normal, low risk ---
        if not is_ens_attack and ens_conf >= self.high_conf \
                and risk_score < self.mid_risk:
            return (Action.ALLOW, 0,
                    f"Normal + low risk: conf={ens_conf:.2f}, "
                    f"risk={risk_score:.2f}")

        # --- Default: monitor ---
        return (Action.MONITOR, ens_pred,
                f"Default monitor: conf={ens_conf:.2f}, "
                f"risk={effective_risk:.2f}")


# ============================================================
#  Zero-Trust Layer orchestrator
# ============================================================
class ZeroTrustLayer:
    def __init__(self, risk_scorer: RiskScorer,
                 policy: Optional[PolicyEngine] = None,
                 context: Optional[ContextAnalyzer] = None):
        self.risk_scorer = risk_scorer
        self.policy = policy or PolicyEngine()
        self.context = context or ContextAnalyzer()
        self.audit_log: List[Decision] = []
        # Counters
        self.action_counts: Dict[str, int] = defaultdict(int)

    def validate_batch(self, X: np.ndarray, ens_preds: np.ndarray,
                       ens_confs: np.ndarray,
                       segment_features: Optional[Tuple[int, int]] = (1, 2),
                       return_decisions: bool = False) -> Dict:
        """Validate a batch of ensemble predictions.

        Returns actions + (optionally) per-sample Decision objects.
        segment_features: which feature indices to use for micro-segmentation.
        """
        risks = self.risk_scorer.predict_risk(X)

        final_labels = np.empty(len(X), dtype=np.int64)
        actions = np.empty(len(X), dtype=object)

        # Approximate integer segment keys from (possibly-scaled) features
        if segment_features is not None:
            proto_idx, svc_idx = segment_features
            proto_q = np.round(X[:, proto_idx]).astype(int)
            svc_q = np.round(X[:, svc_idx]).astype(int)
        else:
            proto_q = np.zeros(len(X), dtype=int)
            svc_q = np.zeros(len(X), dtype=int)

        decisions: List[Decision] = []

        for i in range(len(X)):
            seg = (int(proto_q[i]), int(svc_q[i]))
            ctx = self.context.context_risk(seg)
            action, final_label, why = self.policy.decide(
                int(ens_preds[i]), float(ens_confs[i]),
                float(risks[i]), float(ctx),
            )
            final_labels[i] = final_label
            actions[i] = action.value
            self.action_counts[action.value] += 1

            # Update context with whatever final decision says about this sample
            was_attack = action in (Action.BLOCK, Action.QUARANTINE,
                                     Action.CHALLENGE)
            self.context.update(seg, was_attack)

            if return_decisions:
                decisions.append(Decision(
                    action=action,
                    ensemble_pred=int(ens_preds[i]),
                    ensemble_conf=float(ens_confs[i]),
                    risk_score=float(risks[i]),
                    context_risk=float(ctx),
                    final_label=int(final_label),
                    rationale=why,
                ))

        result = {
            "final_labels": final_labels,
            "actions": actions,
            "risk_scores": risks,
        }
        if return_decisions:
            result["decisions"] = decisions
        return result

    def save_audit_log(self, path: Path = RESULTS_DIR, max_entries: int = 1000):
        """Save action distribution + first N decisions."""
        summary = {
            "action_counts": dict(self.action_counts),
            "total_samples": int(sum(self.action_counts.values())),
        }
        out = path / "zt_audit_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[ZT] Audit summary: {out}")
        return summary


# ============================================================
#  Evaluation: with vs. without Zero-Trust
# ============================================================
def evaluate_with_zt(ens, zt: ZeroTrustLayer, X_te: np.ndarray,
                     y_te: np.ndarray) -> Dict:
    """Compare ensemble-only vs ensemble+ZT on the test set."""
    # Ensemble predictions
    t0 = time.time()
    ens_preds, ens_confs = ens.predict_with_confidence(X_te)
    t_ens = time.time() - t0

    # ZT validation
    t0 = time.time()
    zt_out = zt.validate_batch(X_te, ens_preds, ens_confs,
                                return_decisions=False)
    t_zt = time.time() - t0

    zt_labels = zt_out["final_labels"]
    actions = zt_out["actions"]

    # For binary evaluation: attack detection
    y_bin = (y_te != 0).astype(int)
    ens_bin = (ens_preds != 0).astype(int)
    zt_bin = (zt_labels != 0).astype(int)

    def bin_metrics(pred_bin, tag):
        acc = accuracy_score(y_bin, pred_bin)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, pred_bin, average="binary", zero_division=0,
        )
        tn = ((y_bin == 0) & (pred_bin == 0)).sum()
        fp = ((y_bin == 0) & (pred_bin == 1)).sum()
        fn = ((y_bin == 1) & (pred_bin == 0)).sum()
        tp = ((y_bin == 1) & (pred_bin == 1)).sum()
        fpr = fp / max(tn + fp, 1)
        return {"tag": tag, "accuracy": float(acc),
                "precision": float(prec), "recall": float(rec),
                "f1": float(f1), "fpr": float(fpr),
                "confusion": {"tn": int(tn), "fp": int(fp),
                              "fn": int(fn), "tp": int(tp)}}

    # Multiclass ensemble-only
    m_ens_multi_acc = accuracy_score(y_te, ens_preds)
    m_zt_multi_acc = accuracy_score(y_te, zt_labels)

    # Caught cases: ens said Normal, ZT corrected to attack, AND was actually attack
    ens_said_normal = (ens_preds == 0)
    zt_corrected = ens_said_normal & (zt_labels != 0)
    correctly_caught = zt_corrected & (y_te != 0)
    false_challenges = zt_corrected & (y_te == 0)

    # CHALLENGE distribution
    n_challenge = int((actions == Action.CHALLENGE.value).sum())

    metrics = {
        "ensemble_only_binary": bin_metrics(ens_bin, "Ensemble-only"),
        "ensemble_plus_zt_binary": bin_metrics(zt_bin, "Ensemble + Zero-Trust"),
        "ensemble_only_multiclass_acc": float(m_ens_multi_acc),
        "zt_multiclass_acc": float(m_zt_multi_acc),
        "zt_corrections": {
            "n_challenge_actions": n_challenge,
            "n_corrections_to_attack": int(zt_corrected.sum()),
            "correctly_caught_attacks": int(correctly_caught.sum()),
            "false_challenges": int(false_challenges.sum()),
            "challenge_precision": (float(correctly_caught.sum()) /
                                    max(int(zt_corrected.sum()), 1)),
        },
        "action_distribution": dict(zt.action_counts),
        "timing": {
            "ensemble_ms_per_sample": t_ens / len(X_te) * 1000,
            "zt_ms_per_sample": t_zt / len(X_te) * 1000,
            "total_ms_per_sample": (t_ens + t_zt) / len(X_te) * 1000,
        },
    }
    return metrics


def print_comparison(m: Dict):
    print("\n" + "=" * 70)
    print("  Binary Detection Comparison (Normal vs. Attack)")
    print("=" * 70)
    print(f"{'Metric':<20}{'Ensemble-only':>18}{'Ens + Zero-Trust':>20}{'Delta':>10}")
    print("-" * 70)
    e = m["ensemble_only_binary"]
    z = m["ensemble_plus_zt_binary"]
    for key in ("accuracy", "precision", "recall", "f1", "fpr"):
        delta = z[key] - e[key]
        arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "=")
        print(f"  {key:<18}{e[key]:>18.4f}{z[key]:>20.4f}"
              f"{delta:>+10.4f} {arrow}")

    print("\n" + "=" * 70)
    print("  Zero-Trust Corrections")
    print("=" * 70)
    c = m["zt_corrections"]
    print(f"  CHALLENGE actions issued       : {c['n_challenge_actions']}")
    print(f"  Total corrections to 'attack'  : {c['n_corrections_to_attack']}")
    print(f"  - Correctly caught attacks     : {c['correctly_caught_attacks']}")
    print(f"  - False challenges (was Normal): {c['false_challenges']}")
    print(f"  Challenge precision            : {c['challenge_precision']:.4f}")

    print("\n" + "=" * 70)
    print("  Action Distribution")
    print("=" * 70)
    total = sum(m["action_distribution"].values())
    for a in ("ALLOW", "MONITOR", "CHALLENGE", "QUARANTINE", "BLOCK"):
        n = m["action_distribution"].get(a, 0)
        pct = n / max(total, 1) * 100
        print(f"  {a:<12}: {n:>6} ({pct:5.2f}%)")

    print("\n" + "=" * 70)
    print("  Timing")
    print("=" * 70)
    t = m["timing"]
    print(f"  Ensemble  : {t['ensemble_ms_per_sample']:.4f} ms/sample")
    print(f"  Zero-Trust: {t['zt_ms_per_sample']:.4f} ms/sample")
    print(f"  TOTAL     : {t['total_ms_per_sample']:.4f} ms/sample")


# ============================================================
#  Main
# ============================================================
def main():
    from data_loader import get_preprocessed
    from ensemble_selflearn import EnsembleDetector
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("  HCAPT-Defense - Innovation #3: Zero-Trust Layer")
    print("=" * 70)

    X_full, X_te, y_full, y_te = get_preprocessed()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full,
    )
    feature_dim = X_tr.shape[1]

    # 1. Load pre-trained ensemble
    ens = EnsembleDetector(feature_dim=feature_dim)
    ens.load()
    print(f"[main] Loaded ensemble | weights={ens.weights}")

    # 2. Train Risk Scorer (independent validator)
    # Use DIFFERENT feature subset: skip every-other feature to decorrelate
    feat_subset = list(range(1, feature_dim, 2))  # indices 1, 3, 5, ...
    risk = RiskScorer(feature_indices=feat_subset)
    risk_info = risk.fit(X_tr, y_tr, X_val, y_val)
    risk.save()

    # 3. Build Zero-Trust Layer
    zt = ZeroTrustLayer(
        risk_scorer=risk,
        policy=PolicyEngine(
            high_conf=0.85, mid_conf=0.55,
            high_risk=0.70, mid_risk=0.40,
        ),
    )

    # 4. Evaluate with vs. without ZT
    print("\n[main] Evaluating Ensemble-only vs. Ensemble + Zero-Trust ...")
    metrics = evaluate_with_zt(ens, zt, X_te, y_te)
    print_comparison(metrics)

    # 5. Save results
    out = RESULTS_DIR / "zero_trust_results.json"
    # Make serializable
    metrics_serializable = {k: v for k, v in metrics.items()}
    metrics_serializable["risk_scorer_info"] = risk_info
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2)
    zt.save_audit_log()
    print(f"\n[OK] Results saved: {out}")


if __name__ == "__main__":
    main()
