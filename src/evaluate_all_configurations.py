"""
HCAPT-Defense - Full Ablation Evaluation

Implements the evaluation matrix from UNIFIED_ARCHITECTURE.md §5.4:

  | Configuration              | NN trained | Ensemble | BC enrich | ZT |
  |----------------------------|-----------|----------|-----------|----|
  | Baseline XGB               |    -      |   -      |   -       | -  |
  | Ens (central NN)           | central   |   yes    |   no      | no |
  | Ens + GAN                  | central   |   yes    |   no      | no |
  | Ens + GAN + ZT             | central   |   yes    |   no      | yes|
  | Ens + GAN + FL             | federated |   yes    |   no      | no |
  | Ens + GAN + FL + BC        | federated |   yes    |   yes     | no |
  | Ens + GAN + FL + BC + ZT   | federated |   yes    |   yes     | yes|

Reports: accuracy, precision, recall, F1, FPR, per-class recall,
         inference latency, BC match rate.
"""
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import joblib
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP,
)
from data_loader import get_preprocessed
from ensemble_selflearn import EnsembleDetector, DeepDetector
from blockchain import Blockchain
from threat_intelligence import ThreatIntelligenceLayer
from zero_trust import RiskScorer, PolicyEngine, ZeroTrustLayer
from differential_privacy import DPConfig

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  Metric helpers
# ============================================================
def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_bin = (y_true != 0).astype(int)
    p_bin = (y_pred != 0).astype(int)
    tn = int(((y_bin == 0) & (p_bin == 0)).sum())
    fp = int(((y_bin == 0) & (p_bin == 1)).sum())
    fn = int(((y_bin == 1) & (p_bin == 0)).sum())
    tp = int(((y_bin == 1) & (p_bin == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr  = fp / max(tn + fp, 1)
    return {"accuracy": float((y_bin == p_bin).mean()),
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "fpr": float(fpr),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    p_pc, r_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    per_class = {
        INV_LABEL_MAP[i]: {"precision": float(p_pc[i]),
                            "recall": float(r_pc[i]),
                            "f1": float(f1_pc[i]),
                            "support": int(sup_pc[i])}
        for i in range(NUM_CLASSES)
    }
    return {"accuracy": float(acc),
            "precision_weighted": float(p),
            "recall_weighted": float(r),
            "f1_weighted": float(f1),
            "per_class": per_class}


# ============================================================
#  Load models
# ============================================================
def load_master_ensemble(feature_dim: int) -> EnsembleDetector:
    """Load the ensemble assembled with FL-NN."""
    ens = EnsembleDetector(feature_dim=feature_dim)
    ens.rf = joblib.load(MODELS_DIR / "master_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "master_ens_xgb.json"))
    ens.nn = DeepDetector(feature_dim).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "master_ens_nn.pt",
                                        map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "master_ens_weights.npy")
    return ens


def load_central_ensemble(feature_dim: int) -> EnsembleDetector:
    """Load the original (centrally-trained, GAN-augmented) ensemble."""
    ens = EnsembleDetector(feature_dim=feature_dim)
    ens.rf = joblib.load(MODELS_DIR / "gan_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "gan_ens_xgb.json"))
    ens.nn = DeepDetector(feature_dim).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "gan_ens_nn.pt",
                                        map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "gan_ens_weights.npy")
    return ens


def load_no_gan_ensemble(feature_dim: int) -> EnsembleDetector:
    """Load the non-augmented baseline ensemble."""
    ens = EnsembleDetector(feature_dim=feature_dim)
    ens.rf = joblib.load(MODELS_DIR / "ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "ens_xgb.json"))
    ens.nn = DeepDetector(feature_dim).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "ens_nn.pt",
                                        map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "ens_weights.npy")
    return ens


# ============================================================
#  Evaluate a single configuration
# ============================================================
def evaluate_config(name: str,
                    X_te: np.ndarray, y_te: np.ndarray,
                    ens: EnsembleDetector,
                    blockchain: Blockchain = None,
                    zero_trust: ZeroTrustLayer = None,
                    bc_threshold: float = 0.85) -> Dict:
    """Run a single configuration end-to-end, time it, return metrics."""
    t0 = time.time()
    ens_preds, ens_confs = ens.predict_with_confidence(X_te)
    t_ens = time.time() - t0

    preds, confs = ens_preds.copy(), ens_confs.copy()
    t_bc = 0.0
    bc_stats = None
    if blockchain is not None:
        t0 = time.time()
        ti = ThreatIntelligenceLayer(
            blockchain,
            similarity_threshold=bc_threshold,
            class_vote_required=1,
            min_confidence_boost=0.05,
        )
        preds, confs, _ = ti.enrich(X_te, preds, confs)
        t_bc = time.time() - t0
        bc_stats = ti.stats()

    t_zt = 0.0
    zt_stats = None
    if zero_trust is not None:
        t0 = time.time()
        zt_out = zero_trust.validate_batch(
            X_te, preds, confs, return_decisions=False,
        )
        preds = zt_out["final_labels"]
        t_zt = time.time() - t0
        zt_stats = {"action_counts": dict(zero_trust.action_counts)}

    # Metrics
    bin_m = metrics_binary(y_te, preds)
    mc_m = metrics_multiclass(y_te, preds)

    n = len(X_te)
    return {
        "name": name,
        "binary": bin_m,
        "multiclass": mc_m,
        "timing_ms_per_sample": {
            "ensemble": t_ens / n * 1000,
            "blockchain": t_bc / n * 1000,
            "zero_trust": t_zt / n * 1000,
            "total": (t_ens + t_bc + t_zt) / n * 1000,
        },
        "bc_stats": bc_stats,
        "zt_stats": zt_stats,
    }


# ============================================================
#  Main
# ============================================================
def main():
    print("#" * 76)
    print("#  HCAPT-Defense - ABLATION EVALUATION MATRIX")
    print("#" * 76)

    X_full, X_te, y_full, y_te = get_preprocessed()
    feature_dim = X_te.shape[1]

    # Load blockchain for enrichment configs
    bc_path = RESULTS_DIR / "master_blockchain.json"
    blockchain = None
    if bc_path.exists():
        blockchain = Blockchain.load(bc_path)
        ok, _ = blockchain.verify()
        stats = blockchain.stats()
        print(f"\n[eval] Blockchain loaded: {stats['blocks']} blocks, "
              f"{stats['iocs']} IoCs, consensus={stats['consensus']}, "
              f"verified={ok}")

    # Load Risk Scorer for ZT configs (from master pipeline)
    zt_model_path = MODELS_DIR / "zt_risk_scorer.joblib"
    zero_trust = None
    if zt_model_path.exists():
        risk = RiskScorer()
        risk.load(MODELS_DIR)
        zero_trust = ZeroTrustLayer(
            risk_scorer=risk,
            policy=PolicyEngine(
                high_conf=0.85, mid_conf=0.55,
                high_risk=0.70, mid_risk=0.40,
            ),
        )
        print(f"[eval] Risk scorer loaded ({risk.model_type})")

    results = []

    # --- Configuration 1: No-GAN Ensemble ---
    print("\n" + "-" * 76)
    print("  [1/7] Ens (central NN, no GAN)")
    print("-" * 76)
    try:
        ens = load_no_gan_ensemble(feature_dim)
        results.append(evaluate_config(
            "Ens (no GAN)", X_te, y_te, ens))
    except Exception as e:
        print(f"[eval] skipped: {e}")

    # --- Configuration 2: Central Ens + GAN ---
    print("\n" + "-" * 76)
    print("  [2/7] Ens + GAN (central NN)")
    print("-" * 76)
    try:
        ens = load_central_ensemble(feature_dim)
        results.append(evaluate_config(
            "Ens + GAN (central)", X_te, y_te, ens))
    except Exception as e:
        print(f"[eval] skipped: {e}")

    # --- Configuration 3: Central Ens + GAN + ZT ---
    print("\n" + "-" * 76)
    print("  [3/7] Ens + GAN + ZT (central NN)")
    print("-" * 76)
    try:
        ens = load_central_ensemble(feature_dim)
        if zero_trust:
            zero_trust.audit_log.clear()
            zero_trust.action_counts.clear()
        results.append(evaluate_config(
            "Ens + GAN + ZT (central)", X_te, y_te,
            ens, zero_trust=zero_trust))
    except Exception as e:
        print(f"[eval] skipped: {e}")

    # --- Configuration 4: Master Ens (FL) ---
    print("\n" + "-" * 76)
    print("  [4/7] Ens + GAN + FL (federated NN)")
    print("-" * 76)
    try:
        ens = load_master_ensemble(feature_dim)
        results.append(evaluate_config(
            "Ens + GAN + FL", X_te, y_te, ens))
    except Exception as e:
        print(f"[eval] skipped: {e}")

    # --- Configuration 5: Master Ens + BC ---
    print("\n" + "-" * 76)
    print("  [5/7] Ens + GAN + FL + Blockchain enrichment")
    print("-" * 76)
    try:
        ens = load_master_ensemble(feature_dim)
        results.append(evaluate_config(
            "Ens + GAN + FL + BC", X_te, y_te, ens,
            blockchain=blockchain))
    except Exception as e:
        print(f"[eval] skipped: {e}")

    # --- Configuration 6: Master Ens + BC + ZT (FULL STACK) ---
    print("\n" + "-" * 76)
    print("  [6/7] Ens + GAN + FL + BC + ZT (FULL HCAPT-Defense)")
    print("-" * 76)
    try:
        ens = load_master_ensemble(feature_dim)
        if zero_trust:
            zero_trust.audit_log.clear()
            zero_trust.action_counts.clear()
        results.append(evaluate_config(
            "FULL: Ens + GAN + FL + BC + ZT", X_te, y_te, ens,
            blockchain=blockchain, zero_trust=zero_trust))
    except Exception as e:
        print(f"[eval] skipped: {e}")

    # --- Pretty-print comparison table ---
    print("\n" + "=" * 110)
    print("  CONSOLIDATED ABLATION RESULTS")
    print("=" * 110)
    hdr = f"{'Configuration':<40}{'Acc':>8}{'Prec':>8}{'Rec':>8}" \
          f"{'F1':>8}{'FPR':>8}{'Total-ms':>12}{'BC-match':>10}"
    print(hdr)
    print("-" * 110)
    for r in results:
        b = r["binary"]
        t = r["timing_ms_per_sample"]
        bc_m = "-"
        if r["bc_stats"]:
            bc_m = f"{r['bc_stats']['match_rate']:.1%}"
        print(f"  {r['name']:<38}{b['accuracy']:>8.4f}{b['precision']:>8.4f}"
              f"{b['recall']:>8.4f}{b['f1']:>8.4f}{b['fpr']:>8.4f}"
              f"{t['total']:>12.4f}{bc_m:>10}")

    # --- Per-class recall table ---
    print("\n" + "=" * 110)
    print("  PER-CLASS RECALL BY CONFIGURATION")
    print("=" * 110)
    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    hdr = f"{'Configuration':<40}" + "".join(f"{c:>12}" for c in classes)
    print(hdr)
    print("-" * 110)
    for r in results:
        pc = r["multiclass"]["per_class"]
        row = f"  {r['name']:<38}"
        for c in classes:
            row += f"{pc[c]['recall']:>12.4f}"
        print(row)

    # Save JSON
    out_path = RESULTS_DIR / "ablation_matrix.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved ablation matrix -> {out_path}")


if __name__ == "__main__":
    main()
