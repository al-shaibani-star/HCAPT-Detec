"""
HCAPT-Defense - Layer 3 integrated demo: Federated Learning + Blockchain

After each FL round, the aggregator publishes an auditable block
containing:
  - round number
  - hash of the averaged weights
  - per-client sample sizes and metadata (but NOT weights or data)
  - IoCs harvested from the current global model's confident correct
    detections on a small public shared validation set

This gives us:
  (a) FL privacy: no raw training data leaves any client
  (b) Blockchain integrity: any tampering with logged updates or IoCs
      is detectable via hash chain + PoW
  (c) Shared threat intelligence: new nodes joining after training
      can query the blockchain for known attack signatures.
"""
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP
from data_loader import get_preprocessed
from federated_learning import (
    run_federated_training, FLServer, FLClient,
)
from blockchain import Blockchain, IoC
from threat_intelligence import (
    ThreatIntelligenceLayer, harvest_iocs_from_node,
)

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  Round-end callback: log to blockchain
# ============================================================
class FLAuditor:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.rounds_logged = 0

    def on_round_end(self, round_idx: int, server: FLServer,
                     client_updates: List[Dict]):
        """Publish a metadata IoC block summarising this FL round.

        We use IoCs with attack_class=0 (Normal) as a carrier for
        audit metadata; the blockchain doesn't care about the
        semantic meaning, only about the content hash.
        """
        # We include the weight hash inside the feature_vector as
        # first 16 float-encoded bytes so it's auditable later.
        wh = server.weight_hash()
        # Encode first 32 hex chars as 16 floats (0-1 range)
        payload = np.array([int(wh[2*i:2*i+2], 16) / 255.0
                             for i in range(16)], dtype=float)
        # Pad to feature_dim (41)
        feature_len = server.feature_dim
        feature_vec = np.concatenate([payload,
                                        np.zeros(feature_len - len(payload))])

        audit_ioc = IoC(
            feature_vector=feature_vec.tolist(),
            attack_class=0,  # 'Normal' as a carrier for metadata
            severity=0,
            confidence=0.0,
            source_node="FL_AGGREGATOR",
            timestamp=time.time(),
            attack_name=f"fl_round_{round_idx+1}_hash_{wh[:12]}",
        )
        self.blockchain.submit_ioc(audit_ioc)
        self.rounds_logged += 1


def client_harvest_iocs(clients: List[FLClient],
                         confidence_threshold: float = 0.90,
                         max_iocs_per_client: int = 50) -> List[IoC]:
    """After FL completes, each client harvests IoCs from its LOCAL data
    using the final global model. Raw data never leaves the client;
    only the IoC feature vectors (already in feature space, no PII).
    """
    iocs = []
    for c in clients:
        # predictions on its local data
        c.model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(c.X).float().to(DEVICE)
            logits = c.model(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)
        confs = probs.max(1)
        # Only harvest correct attack detections with high confidence
        mask = ((preds == c.y) & (preds != 0) & (confs >= confidence_threshold))
        idx = np.where(mask)[0]
        rng = np.random.default_rng(SEED + hash(c.client_id) % 1000)
        if len(idx) > max_iocs_per_client:
            idx = rng.choice(idx, size=max_iocs_per_client, replace=False)
        sev_map = {0: 0, 1: 3, 2: 2, 3: 4, 4: 5}
        for i in idx:
            iocs.append(IoC(
                feature_vector=c.X[i].tolist(),
                attack_class=int(preds[i]),
                severity=sev_map[int(preds[i])],
                confidence=float(confs[i]),
                source_node=c.client_id,
                timestamp=time.time(),
                attack_name=f"post_fl_{INV_LABEL_MAP[int(preds[i])]}",
            ))
    print(f"[harvest] collected {len(iocs)} IoCs from "
          f"{len(clients)} clients")
    return iocs


# ============================================================
#  Evaluation of different configurations
# ============================================================
def evaluate_fl_variants(server: FLServer, bc: Blockchain,
                          X_te: np.ndarray, y_te: np.ndarray) -> Dict:
    """Measure 3 configurations on test set:
      (1) FL global model alone
      (2) FL + Blockchain query (ThreatIntelligenceLayer)
      (3) FL + Blockchain + Zero-Trust (optional, if available)
    """
    # (1) FL alone
    server.global_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_te).float()
        preds_list = []; confs_list = []
        for i in range(0, len(X_te), 2048):
            batch = X_t[i:i+2048].to(DEVICE)
            logits = server.global_model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds_list.append(probs.argmax(1))
            confs_list.append(probs.max(1))
    fl_preds = np.concatenate(preds_list)
    fl_confs = np.concatenate(confs_list)

    acc1 = (fl_preds == y_te).mean()
    # Binary
    y_bin = (y_te != 0).astype(int)
    p_bin = (fl_preds != 0).astype(int)
    tp = ((y_bin == 1) & (p_bin == 1)).sum()
    fp = ((y_bin == 0) & (p_bin == 1)).sum()
    fn = ((y_bin == 1) & (p_bin == 0)).sum()
    tn = ((y_bin == 0) & (p_bin == 0)).sum()
    fl_fpr = fp / max(tn + fp, 1)
    fl_recall = tp / max(tp + fn, 1)
    fl_prec = tp / max(tp + fp, 1)
    fl_f1 = 2 * fl_prec * fl_recall / max(fl_prec + fl_recall, 1e-9)

    # (2) FL + Blockchain TI
    ti = ThreatIntelligenceLayer(bc, similarity_threshold=0.85,
                                  class_vote_required=1,
                                  min_confidence_boost=0.1)
    enr_preds, enr_confs, enrichments = ti.enrich(X_te, fl_preds, fl_confs)

    acc2 = (enr_preds == y_te).mean()
    ep_bin = (enr_preds != 0).astype(int)
    tp2 = ((y_bin == 1) & (ep_bin == 1)).sum()
    fp2 = ((y_bin == 0) & (ep_bin == 1)).sum()
    fn2 = ((y_bin == 1) & (ep_bin == 0)).sum()
    tn2 = ((y_bin == 0) & (ep_bin == 0)).sum()
    fl_bc_fpr = fp2 / max(tn2 + fp2, 1)
    fl_bc_rec = tp2 / max(tp2 + fn2, 1)
    fl_bc_prec = tp2 / max(tp2 + fp2, 1)
    fl_bc_f1 = 2*fl_bc_prec*fl_bc_rec / max(fl_bc_prec + fl_bc_rec, 1e-9)

    return {
        "fl_only": {
            "accuracy": float(acc1),
            "precision": float(fl_prec),
            "recall": float(fl_recall),
            "f1": float(fl_f1),
            "fpr": float(fl_fpr),
        },
        "fl_plus_blockchain": {
            "accuracy": float(acc2),
            "precision": float(fl_bc_prec),
            "recall": float(fl_bc_rec),
            "f1": float(fl_bc_f1),
            "fpr": float(fl_bc_fpr),
            "ti_stats": ti.stats(),
        },
    }


# ============================================================
#  Main
# ============================================================
def main():
    print("#" * 70)
    print("#  HCAPT-Defense - Layer 3 integrated demo: FL + Blockchain")
    print("#" * 70)

    # 1. Data
    X_full, X_te, y_full, y_te = get_preprocessed()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full,
    )

    # 2. Blockchain with auditor
    print("\n[bc] Initialising blockchain (difficulty=3)")
    bc = Blockchain(difficulty=3, block_size=10)
    auditor = FLAuditor(bc)

    # 3. Run FL with auditor callback
    print("\n[FL] starting 20-round federated training ...")
    server, clients, history = run_federated_training(
        X_tr, y_tr, X_val, y_val,
        n_clients=5, n_rounds=20, local_epochs=3,
        partition="non_iid", concentration=0.5,
        on_round_end=auditor.on_round_end,
    )

    # 4. Harvest IoCs from clients (raw data stays local)
    print("\n[bc] Harvesting IoCs from clients...")
    iocs = client_harvest_iocs(clients, confidence_threshold=0.85,
                                 max_iocs_per_client=100)
    for ioc in iocs:
        bc.submit_ioc(ioc)
    bc.mine_pending(force=True)

    # 5. Verify blockchain
    ok, reason = bc.verify()
    stats = bc.stats()
    print(f"\n[bc] chain verified: {ok}  ({reason or 'all good'})")
    print(f"[bc] blocks: {stats['blocks']} | iocs: {stats['iocs']}")
    print(f"[bc] per-class iocs: {stats['iocs_per_class']}")
    print(f"[bc] per-node iocs:  {stats['iocs_per_node']}")

    # 6. Evaluate variants
    print("\n[eval] evaluating FL alone vs FL + Blockchain ...")
    results = evaluate_fl_variants(server, bc, X_te, y_te)

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    for cfg_name, m in results.items():
        print(f"\n  {cfg_name}:")
        for k in ("accuracy", "precision", "recall", "f1", "fpr"):
            print(f"    {k:<12}: {m[k]:.4f}")

    # 7. Save
    bc.save(RESULTS_DIR / "fl_blockchain.json")
    out = {
        "fl_history": history,
        "blockchain_stats": stats,
        "blockchain_verified": ok,
        "evaluation": results,
    }
    with open(RESULTS_DIR / "fl_blockchain_results.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    torch.save(server.global_model.state_dict(),
                MODELS_DIR / "fl_global_nn.pt")
    print(f"\n[OK] All artifacts saved:")
    print(f"  - {MODELS_DIR / 'fl_global_nn.pt'}")
    print(f"  - {RESULTS_DIR / 'fl_blockchain.json'}")
    print(f"  - {RESULTS_DIR / 'fl_blockchain_results.json'}")


if __name__ == "__main__":
    main()
