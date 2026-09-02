"""
HCAPT-Defense - MASTER PIPELINE V2 (with all improvements)

Fixes the four failure modes from v1:
  1. R2L/U2R recall     -> Focal Loss + SMOTE + Threshold calibration
  2. Mode collapse      -> CTGAN (tabular-specific)
  3. BC match 0%        -> Gaussian DP + Class prototypes + LSH
  4. Distribution shift -> Test-Time Adaptation (BN stats)

All improvements are OPT-IN so ablation vs v1 remains possible.
"""
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import joblib
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP,
)
from data_loader import get_preprocessed
from cache import CACHE_DIR

# Innovation #1 alternatives
from wgan_gp_augment import train_wgan_gp, balance_dataset as wgangp_balance
try:
    from ctgan_augment import train_ctgan, balance_dataset as ctgan_balance
    HAS_CTGAN = True
except Exception:
    HAS_CTGAN = False

# Innovation #2 (ensemble)
from ensemble_selflearn import EnsembleDetector, DeepDetector

# Layer 3a (FL)
from federated_learning import run_federated_training

# Layer 3b (Blockchain) - with Gaussian DP alternative
from blockchain import Blockchain, IoC
from threat_intelligence import ThreatIntelligenceLayer
from differential_privacy import DPConfig  # Laplace (v1)

# Innovation #3 (Zero-Trust)
from zero_trust import RiskScorer, PolicyEngine, ZeroTrustLayer

# ===== NEW IMPROVEMENTS =====
from improvements.smote_balance import smote_minority
from improvements.gaussian_dp import GaussianDPSanitizer, GaussianDPConfig
from improvements.lsh_prototypes import (
    compute_class_prototypes, LSHIndex,
)
from improvements.tta import tta_predict
from improvements.threshold_calibration import (
    find_class_thresholds, apply_thresholds,
)

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  V2 Auditor (shares logic with v1)
# ============================================================
class FLAuditorV2:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain

    def on_round_end(self, round_idx: int, server, client_updates):
        wh = server.weight_hash()
        payload = np.array([int(wh[2*i:2*i+2], 16) / 255.0
                             for i in range(16)], dtype=float)
        feature_vec = np.concatenate(
            [payload, np.zeros(server.feature_dim - len(payload))])
        audit_ioc = IoC(
            feature_vector=feature_vec.tolist(),
            attack_class=0, severity=0, confidence=0.0,
            source_node="FL_AGGREGATOR",
            timestamp=time.time(),
            attack_name=f"v2_round_{round_idx+1}_hash_{wh[:12]}",
        )
        self.blockchain.submit_ioc(audit_ioc)


# ============================================================
#  Harvest class PROTOTYPES (not per-sample IoCs) - DP-friendly
# ============================================================
def harvest_prototype_iocs(clients, bc: Blockchain,
                            dp_config: GaussianDPConfig) -> int:
    """For each client, compute ONE prototype vector per attack class
    it observed. Apply Gaussian DP. Submit to blockchain.

    This reduces DP noise impact by sqrt(n_per_class).
    """
    san = GaussianDPSanitizer(dp_config)
    total = 0
    for client in clients:
        client.model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(client.X).float().to(DEVICE)
            probs = torch.softmax(client.model(Xt), dim=1).cpu().numpy()
        preds = probs.argmax(1); confs = probs.max(1)
        correct = (preds == client.y) & (confs >= 0.85) & (preds != 0)

        for cls in range(1, NUM_CLASSES):    # skip Normal
            mask = correct & (client.y == cls)
            if mask.sum() < 5:
                continue
            proto = client.X[mask].mean(axis=0)
            proto = san.sanitize(proto)       # Gaussian DP
            sev_map = {1: 3, 2: 2, 3: 4, 4: 5}
            bc.submit_ioc(IoC(
                feature_vector=proto.tolist(),
                attack_class=int(cls),
                severity=sev_map[cls],
                confidence=float(confs[mask].mean()),
                source_node=client.client_id,
                timestamp=time.time(),
                attack_name=f"proto_{INV_LABEL_MAP[cls]}",
            ))
            total += 1
    bc.mine_pending(force=True)
    return total


# ============================================================
#  Main v2 pipeline
# ============================================================
def main():
    t_start = time.time()
    print("#" * 76)
    print("#  HCAPT-Defense V2 - IMPROVED MASTER PIPELINE")
    print("#  Fixes: R2L/U2R recall, mode collapse, BC match, drift")
    print("#" * 76)

    # ---- Phase A.1: data ----
    print("\n[A.1] Preprocessing NSL-KDD")
    X_full, X_te, y_full, y_te = get_preprocessed()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full,
    )
    feature_dim = X_tr.shape[1]

    # ---- Phase A.2a: GAN (try CTGAN, fallback to WGAN-GP cache) ----
    print("\n[A.2a] Synthetic data via CTGAN (or cached WGAN-GP)")
    balanced_cache = CACHE_DIR / "v2_balanced_data.npz"
    wgp_cache = CACHE_DIR / "wgangp_balanced_data.npz"
    ctgan_cache = CACHE_DIR / "ctgan_balanced_data.npz"

    if balanced_cache.exists():
        print(f"[A.2a] v2 cache hit")
        d = np.load(balanced_cache)
        X_gan, y_gan = d["X_bal"], d["y_bal"]
    elif ctgan_cache.exists():
        print(f"[A.2a] loading CTGAN output")
        d = np.load(ctgan_cache)
        X_gan, y_gan = d["X_bal"], d["y_bal"]
    elif wgp_cache.exists():
        print(f"[A.2a] fallback to cached WGAN-GP")
        d = np.load(wgp_cache)
        X_gan, y_gan = d["X_bal"], d["y_bal"]
    else:
        print(f"[A.2a] no cache - training CTGAN now")
        if not HAS_CTGAN:
            raise RuntimeError("ctgan unavailable")
        ctgan = train_ctgan(X_tr, y_tr, epochs=20, batch_size=500)
        X_gan, y_gan = ctgan_balance(X_tr, y_tr, ctgan)
        np.savez(ctgan_cache, X_bal=X_gan, y_bal=y_gan)

    # ---- Phase A.2b: SMOTE for extreme minorities ON TOP of GAN ----
    print("\n[A.2b] SMOTE boost for R2L/U2R")
    X_bal, y_bal = smote_minority(
        X_gan, y_gan, targets={3: 70000, 4: 70000}, k_neighbors=5,
    )
    np.savez(balanced_cache, X_bal=X_bal, y_bal=y_bal)
    print(f"[A.2b] final class counts: "
          f"{np.bincount(y_bal, minlength=NUM_CLASSES).tolist()}")

    X_bal_tr, X_bal_val, y_bal_tr, y_bal_val = train_test_split(
        X_bal, y_bal, test_size=0.1, random_state=SEED, stratify=y_bal,
    )

    # ---- Phase A.3: Federated Learning with Focal Loss ----
    print("\n[A.3] Federated Learning with Focal Loss (γ=2.0)")
    bc = Blockchain(consensus="poa", block_size=15)
    auditor = FLAuditorV2(bc)
    server, clients, fl_history = run_federated_training(
        X_bal_tr, y_bal_tr, X_bal_val, y_bal_val,
        n_clients=5, n_rounds=15, local_epochs=3,
        partition="non_iid", concentration=0.5,
        use_focal=True, focal_gamma=2.0,        # ← NEW
        on_round_end=auditor.on_round_end,
    )
    fl_test_info = server.evaluate(X_te, y_te)
    print(f"[A.3] FL global TEST accuracy (raw): "
          f"{fl_test_info['accuracy']:.4f}")
    torch.save(server.global_model.state_dict(),
                MODELS_DIR / "v2_fl_global_nn.pt")

    # ---- Phase A.4: Ensemble assembly (same as v1) ----
    print("\n[A.4] Ensemble assembly with FL-NN")
    ens = EnsembleDetector(feature_dim=feature_dim)
    fl_state = {k: v.clone()
                 for k, v in server.global_model.state_dict().items()}
    ens_info = ens.fit_with_fl_nn(
        X_bal_tr, y_bal_tr, X_bal_val, y_bal_val, fl_state,
    )
    joblib.dump(ens.rf, MODELS_DIR / "v2_ens_rf.joblib")
    ens.xgb.save_model(str(MODELS_DIR / "v2_ens_xgb.json"))
    torch.save(ens.nn.state_dict(), MODELS_DIR / "v2_ens_nn.pt")
    np.save(MODELS_DIR / "v2_ens_weights.npy", ens.weights)

    # ---- Phase A.5: Gaussian DP + Prototypes to Blockchain ----
    print("\n[A.5] Harvesting CLASS PROTOTYPES with Gaussian DP")
    dp = GaussianDPConfig(epsilon=5.0, delta=1e-5, clip_norm=1.0)
    print(f"[A.5] Gaussian σ = {2*dp.clip_norm * np.sqrt(2*np.log(1.25/dp.delta)) / dp.epsilon:.3f}")
    n_protos = harvest_prototype_iocs(clients, bc, dp)
    ok, reason = bc.verify()
    print(f"[A.5] submitted {n_protos} prototype IoCs; "
          f"chain verified: {ok} ({reason or 'ok'})")
    bc.save(RESULTS_DIR / "v2_blockchain.json")

    # Build LSH index from prototypes for fast similarity
    all_iocs = bc.all_iocs()
    attack_iocs = [i for i in all_iocs if i.attack_class != 0]
    if attack_iocs:
        V = np.array([i.feature_vector for i in attack_iocs],
                      dtype=np.float32)
        lsh = LSHIndex.build(feature_dim=feature_dim, n_bits=64)
        sigs = lsh.hash(V)
        np.savez(CACHE_DIR / "v2_lsh.npz",
                 planes=lsh.planes, sigs=sigs)
        print(f"[A.5] built LSH index on {len(attack_iocs)} prototypes")

    # ---- Phase A.6: Zero-Trust Risk Scorer ----
    print("\n[A.6] Zero-Trust Risk Scorer")
    feat_subset = list(range(1, feature_dim, 2))
    risk = RiskScorer(feature_indices=feat_subset)
    risk_info = risk.fit(X_bal_tr, y_bal_tr, X_bal_val, y_bal_val)
    risk.save(MODELS_DIR)   # zt_risk_scorer.*

    # ---- Phase A.7: Threshold calibration (on validation) ----
    print("\n[A.7] Threshold calibration on validation set")
    val_probs = ens.predict_proba(X_bal_val)
    thresholds = find_class_thresholds(
        y_bal_val, val_probs, target_metric="f1",
    )
    print(f"[A.7] optimised thresholds: {thresholds}")
    with open(MODELS_DIR / "v2_thresholds.json", "w") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)

    # ---- Phase B preview: TTA + calibrated thresholds on test ----
    print("\n[B-preview] TTA + calibrated thresholds on KDDTest+")
    from sklearn.metrics import accuracy_score, f1_score
    # Raw ensemble prediction (no TTA, argmax)
    raw_preds = ens.predict(X_te)
    raw_acc = accuracy_score(y_te, raw_preds)

    # With TTA: adapt NN BN stats on test, then re-run ensemble
    print("  Applying TTA to Deep NN ...")
    tta_probs_nn = tta_predict(ens.nn, X_te, batch_size=512)
    # Combine NN TTA output with RF/XGB via learned weights
    p_rf = ens.rf.predict_proba(X_te)
    p_xgb = ens.xgb.predict_proba(X_te)
    tta_probs = (ens.weights[0] * p_rf
                  + ens.weights[1] * p_xgb
                  + ens.weights[2] * tta_probs_nn)
    tta_preds = tta_probs.argmax(1)
    tta_acc = accuracy_score(y_te, tta_preds)

    # With thresholds on top of TTA
    cal_preds = apply_thresholds(tta_probs, thresholds)
    cal_acc = accuracy_score(y_te, cal_preds)

    print(f"  raw ensemble:      acc={raw_acc:.4f}")
    print(f"  + TTA:             acc={tta_acc:.4f}")
    print(f"  + TTA + thresholds:acc={cal_acc:.4f}")

    # Per-class recall for the best
    from sklearn.metrics import recall_score
    rec_raw = recall_score(y_te, raw_preds, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)
    rec_cal = recall_score(y_te, cal_preds, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)
    print(f"\n  Per-class recall:")
    for c in range(NUM_CLASSES):
        print(f"    {INV_LABEL_MAP[c]:<7}: raw={rec_raw[c]:.4f}  "
              f"cal={rec_cal[c]:.4f}")

    elapsed = time.time() - t_start
    summary = {
        "total_time_s": elapsed,
        "phase_fl_test_acc": fl_test_info["accuracy"],
        "ensemble_train_info": ens_info,
        "risk_scorer": risk_info,
        "blockchain": {
            "blocks": len(bc.chain),
            "iocs": n_protos,
            "verified": ok,
            "consensus": bc.consensus,
        },
        "dp": {"mechanism": "Gaussian", "epsilon": dp.epsilon,
                "delta": dp.delta, "sigma": (
                    dp.clip_norm * np.sqrt(2*np.log(1.25/dp.delta))/dp.epsilon
                )},
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "test_accuracy": {
            "raw":   float(raw_acc),
            "tta":   float(tta_acc),
            "final": float(cal_acc),
        },
        "per_class_recall_final": {
            INV_LABEL_MAP[c]: float(rec_cal[c])
            for c in range(NUM_CLASSES)
        },
    }
    with open(RESULTS_DIR / "v2_pipeline_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "#" * 76)
    print(f"#  V2 pipeline complete in {elapsed/60:.1f} min")
    print(f"#  RAW accuracy:  {raw_acc:.4f}")
    print(f"#  FINAL accuracy:{cal_acc:.4f}")
    print("#" * 76)


if __name__ == "__main__":
    main()
