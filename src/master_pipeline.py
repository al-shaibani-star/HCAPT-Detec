"""
HCAPT-Defense - Master Pipeline (end-to-end Phase A)

Executes the full offline setup described in
`docs/UNIFIED_ARCHITECTURE.md §3`:

    Phase A.1  Preprocess + cache (Layer 1)
    Phase A.2  WGAN-GP training (Innovation #1)
    Phase A.3  Federated Learning (Layer 3a) on balanced data
    Phase A.4  Assemble ensemble with FL-trained NN (Innovation #2)
    Phase A.5  Harvest DP-noised IoCs from clients -> Blockchain
    Phase A.6  Train Zero-Trust Risk Scorer (Innovation #3)

Idempotent via cache: re-running picks up where it left off.
"""
import json
import time
from pathlib import Path
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP,
)
from data_loader import get_preprocessed
from cache import mark_phase_done, is_phase_done, CACHE_DIR

# Innovation #1 - WGAN-GP
from wgan_gp_augment import train_wgan_gp, balance_dataset

# Layer 3a - Federated Learning
from federated_learning import run_federated_training

# Innovation #2 - Ensemble
from ensemble_selflearn import EnsembleDetector

# Layer 3b - Blockchain + Threat Intelligence
from blockchain import Blockchain, IoC
from threat_intelligence import (
    ThreatIntelligenceLayer, harvest_iocs_from_node,
)
from differential_privacy import DPConfig

# Innovation #3 - Zero-Trust
from zero_trust import RiskScorer, PolicyEngine, ZeroTrustLayer


INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  Audit callback for FL rounds
# ============================================================
class FLAuditor:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.rounds_logged = 0

    def on_round_end(self, round_idx: int, server, client_updates):
        """Write an audit IoC summarising this FL round's hash."""
        wh = server.weight_hash()
        payload = np.array([int(wh[2*i:2*i+2], 16) / 255.0
                             for i in range(16)], dtype=float)
        feature_vec = np.concatenate([payload,
                                        np.zeros(server.feature_dim - len(payload))])
        audit_ioc = IoC(
            feature_vector=feature_vec.tolist(),
            attack_class=0, severity=0, confidence=0.0,
            source_node="FL_AGGREGATOR",
            timestamp=time.time(),
            attack_name=f"fl_round_{round_idx+1}_hash_{wh[:12]}",
        )
        self.blockchain.submit_ioc(audit_ioc)
        self.rounds_logged += 1


# ============================================================
#  Main pipeline
# ============================================================
def main():
    t_start = time.time()
    print("#" * 76)
    print("#  HCAPT-Defense - MASTER PIPELINE (Phase A)")
    print("#  Innovations 1-3 integrated with FL + Blockchain")
    print("#" * 76)

    # ========== Phase A.1: Preprocess ==========
    print("\n" + "=" * 76)
    print("  Phase A.1 - Preprocess NSL-KDD")
    print("=" * 76)
    X_full, X_te, y_full, y_te = get_preprocessed()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full,
    )
    feature_dim = X_tr.shape[1]
    mark_phase_done("A1_preprocess",
                     {"n_train": len(y_tr), "n_val": len(y_val),
                      "n_test": len(y_te), "features": feature_dim})

    # ========== Phase A.2: WGAN-GP ==========
    print("\n" + "=" * 76)
    print("  Phase A.2 - WGAN-GP training (Innovation #1)")
    print("=" * 76)
    wgan_gen_path = MODELS_DIR / "wgan_gp_generator.pt"
    balanced_cache = CACHE_DIR / "wgangp_balanced_data.npz"
    if balanced_cache.exists() and wgan_gen_path.exists():
        print(f"[A.2] cache hit - loading balanced data")
        d = np.load(balanced_cache)
        X_bal, y_bal = d["X_bal"], d["y_bal"]
    else:
        print(f"[A.2] training WGAN-GP from scratch (~35 min on GPU)")
        G, C, hist = train_wgan_gp(X_tr, y_tr, epochs=100)
        X_bal, y_bal = balance_dataset(X_tr, y_tr, G)
        np.savez(balanced_cache, X_bal=X_bal, y_bal=y_bal)
    counts = np.bincount(y_bal, minlength=NUM_CLASSES).tolist()
    print(f"[A.2] balanced class counts: {counts}")
    mark_phase_done("A2_wgan_gp", {"balanced_total": int(len(y_bal))})

    # Split balanced data for FL clients and ensemble training
    X_bal_tr, X_bal_val, y_bal_tr, y_bal_val = train_test_split(
        X_bal, y_bal, test_size=0.1, random_state=SEED, stratify=y_bal,
    )

    # ========== Phase A.3: Federated Learning ==========
    print("\n" + "=" * 76)
    print("  Phase A.3 - Federated Learning (Layer 3a)")
    print("=" * 76)

    # Initialise blockchain with PoA consensus
    bc = Blockchain(consensus="poa", block_size=15)
    auditor = FLAuditor(bc)

    server, clients, fl_history = run_federated_training(
        X_bal_tr, y_bal_tr, X_bal_val, y_bal_val,
        n_clients=5, n_rounds=20, local_epochs=3,
        partition="non_iid", concentration=0.5,
        on_round_end=auditor.on_round_end,
    )
    fl_test_info = server.evaluate(X_te, y_te)
    print(f"\n[A.3] FL global model TEST accuracy: "
          f"{fl_test_info['accuracy']:.4f}")
    torch.save(server.global_model.state_dict(),
                MODELS_DIR / "master_fl_global_nn.pt")
    mark_phase_done("A3_federated_learning",
                     {"final_val_acc": fl_history["round_acc"][-1],
                      "test_acc": fl_test_info["accuracy"]})

    # ========== Phase A.4: Assemble Ensemble with FL NN ==========
    print("\n" + "=" * 76)
    print("  Phase A.4 - Ensemble assembly (Innovation #2 + FL NN)")
    print("=" * 76)
    ens = EnsembleDetector(feature_dim=feature_dim)
    fl_state = {k: v.clone() for k, v in server.global_model.state_dict().items()}
    ens_info = ens.fit_with_fl_nn(
        X_bal_tr, y_bal_tr, X_bal_val, y_bal_val, fl_state,
    )
    # Save
    joblib.dump(ens.rf, MODELS_DIR / "master_ens_rf.joblib")
    ens.xgb.save_model(str(MODELS_DIR / "master_ens_xgb.json"))
    torch.save(ens.nn.state_dict(), MODELS_DIR / "master_ens_nn.pt")
    np.save(MODELS_DIR / "master_ens_weights.npy", ens.weights)
    mark_phase_done("A4_ensemble_assembly", ens_info)

    # ========== Phase A.5: IoC harvest with DP ==========
    print("\n" + "=" * 76)
    print("  Phase A.5 - IoC harvest with Differential Privacy")
    print("=" * 76)
    dp_cfg = DPConfig(epsilon=5.0, clip_norm=1.0, seed=SEED)
    print(f"[A.5] DP config: epsilon={dp_cfg.epsilon}, "
          f"clip_norm={dp_cfg.clip_norm}, "
          f"noise_scale={2*dp_cfg.clip_norm/dp_cfg.epsilon:.3f}")

    total_iocs = 0
    for client in clients:
        # Client runs FINAL global model on LOCAL data
        client.model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(client.X).float().to(DEVICE)
            probs = torch.softmax(client.model(Xt), dim=1).cpu().numpy()
        preds = probs.argmax(1)
        confs = probs.max(1)
        iocs = harvest_iocs_from_node(
            node_id=client.client_id,
            X=client.X, y_true=client.y,
            ens_preds=preds, ens_confs=confs,
            min_confidence=0.85, max_iocs=80,
            dp_config=dp_cfg,   # ← DP applied
        )
        for ioc in iocs:
            bc.submit_ioc(ioc)
        total_iocs += len(iocs)
        print(f"[A.5] {client.client_id}: +{len(iocs)} IoCs")
    bc.mine_pending(force=True)
    ok, reason = bc.verify()
    print(f"\n[A.5] Blockchain verified: {ok} ({reason or 'ok'})")
    print(f"[A.5] Total IoCs on chain: {total_iocs}")
    bc.save(RESULTS_DIR / "master_blockchain.json")
    mark_phase_done("A5_ioc_harvest",
                     {"total_iocs": total_iocs, "verified": ok,
                      "consensus": bc.consensus,
                      "dp_epsilon": dp_cfg.epsilon})

    # ========== Phase A.6: Zero-Trust Risk Scorer ==========
    print("\n" + "=" * 76)
    print("  Phase A.6 - Zero-Trust Risk Scorer (Innovation #3)")
    print("=" * 76)
    feat_subset = list(range(1, feature_dim, 2))
    risk = RiskScorer(feature_indices=feat_subset)
    risk_info = risk.fit(X_bal_tr, y_bal_tr, X_bal_val, y_bal_val)
    risk.save(MODELS_DIR)   # saves as zt_risk_scorer.*
    mark_phase_done("A6_zero_trust", risk_info)

    # ========== Summary ==========
    elapsed = time.time() - t_start
    summary = {
        "total_time_s": elapsed,
        "phase_A3_fl": {
            "final_val_acc": fl_history["round_acc"][-1],
            "test_acc": fl_test_info["accuracy"],
        },
        "phase_A4_ensemble": ens_info,
        "phase_A5_blockchain": {
            "blocks": len(bc.chain), "iocs": total_iocs,
            "verified": ok, "consensus": bc.consensus,
        },
        "phase_A6_zt": risk_info,
    }
    with open(RESULTS_DIR / "master_pipeline_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "#" * 76)
    print(f"#  Phase A COMPLETE in {elapsed/60:.1f} min")
    print(f"#  Artifacts saved to: {MODELS_DIR}")
    print(f"#  Next: run evaluate_all_configurations.py")
    print("#" * 76)


if __name__ == "__main__":
    main()
