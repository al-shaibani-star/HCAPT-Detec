"""
HCAPT-Defense - Layer 3a: Federated Learning (FedAvg)

Implements a minimal but faithful Federated Learning simulation for
the Deep NN component of the ensemble. Raw data never leaves the
clients; only model weight tensors are shared with the aggregator.

Architecture:
  FLClient   - local model + local dataset + local_train()
  FLServer   - aggregates client weights via FedAvg
  run_federated_training()  - orchestrates N rounds of R round-trips

This pairs with blockchain.py: after each round the aggregator
can publish a hash of the global weights + IoCs to the blockchain
for a tamper-evident audit trail (see fl_blockchain_demo.py).

Reference:
  McMahan et al. 2017 "Communication-Efficient Learning of Deep
  Networks from Decentralized Data" (FedAvg)
"""
import time
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, BATCH_SIZE,
    LABEL_MAP,
)
from ensemble_selflearn import DeepDetector

# Optional Focal Loss support (Innovation v2)
try:
    from improvements.focal_loss import FocalLoss
    HAS_FOCAL = True
except ImportError:
    HAS_FOCAL = False

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
#  Data partitioning (non-IID support)
# ============================================================
def partition_iid(X: np.ndarray, y: np.ndarray, n_clients: int,
                  seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simple IID split: shuffle and slice."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, n_clients)
    return [(X[s], y[s]) for s in splits]


def partition_non_iid(X: np.ndarray, y: np.ndarray, n_clients: int,
                       concentration: float = 0.5,
                       seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Dirichlet-based non-IID partition: each client gets a skewed
    class mix. concentration < 1.0 = more skewed."""
    rng = np.random.default_rng(seed)
    n_classes = NUM_CLASSES
    client_idx: List[List[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        # Sample proportions from Dirichlet
        props = rng.dirichlet(alpha=np.full(n_clients, concentration))
        # Cumulative boundaries
        bounds = (np.cumsum(props) * len(cls_idx)).astype(int)
        start = 0
        for i, end in enumerate(bounds):
            client_idx[i].extend(cls_idx[start:end].tolist())
            start = end

    return [(X[idx], y[idx]) for idx in client_idx]


def partition_stats(partitions: List[Tuple[np.ndarray, np.ndarray]]
                    ) -> List[Dict]:
    out = []
    for i, (Xi, yi) in enumerate(partitions):
        counts = np.bincount(yi, minlength=NUM_CLASSES).tolist()
        out.append({
            "client": i, "n_samples": int(len(yi)),
            "class_counts": counts,
        })
    return out


# ============================================================
#  FLClient
# ============================================================
class FLClient:
    def __init__(self, client_id: str, X: np.ndarray, y: np.ndarray,
                 feature_dim: int, lr: float = 1e-3, batch_size: int = 128):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.feature_dim = feature_dim
        self.lr = lr
        self.batch_size = batch_size
        self.model = DeepDetector(feature_dim).to(DEVICE)

    def n_samples(self) -> int:
        return len(self.y)

    def set_global_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load the latest global model weights."""
        # Clone tensors to avoid shared-memory issues across rounds
        self.model.load_state_dict(
            {k: v.clone().to(DEVICE) for k, v in state_dict.items()}
        )

    def local_train(self, local_epochs: int = 3,
                    class_weights: Optional[np.ndarray] = None,
                    use_focal: bool = False,
                    focal_gamma: float = 2.0) -> Dict:
        """One round of local training, returns metrics + final weights."""
        self.model.train()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(self.X).float(),
                          torch.from_numpy(self.y).long()),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
        )
        weight_tensor = (torch.from_numpy(class_weights).float().to(DEVICE)
                          if class_weights is not None else None)
        if use_focal and HAS_FOCAL:
            criterion = FocalLoss(gamma=focal_gamma, alpha=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                 weight_decay=1e-4)

        running_loss, correct, total = 0.0, 0, 0
        for epoch in range(local_epochs):
            for xb, yb in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                running_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)

        return {
            "client_id": self.client_id,
            "n_samples": total // max(local_epochs, 1),
            "loss": running_loss / max(total, 1),
            "acc": correct / max(total, 1),
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in self.model.state_dict().items()},
        }


# ============================================================
#  FLServer - FedAvg
# ============================================================
class FLServer:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.global_model = DeepDetector(feature_dim).to(DEVICE)
        self.round = 0

    def global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone()
                for k, v in self.global_model.state_dict().items()}

    def fedavg(self, client_updates: List[Dict]) -> None:
        """Weighted average of client weights (weights ∝ n_samples).

        Long-type tensors (e.g. BatchNorm.num_batches_tracked) are
        not averaged - we take the maximum value across clients
        (so the counter only advances, matching torch's semantics).
        """
        total = sum(u["n_samples"] for u in client_updates)
        if total == 0:
            return
        first_state = client_updates[0]["state_dict"]

        avg_state = {}
        for k, v in first_state.items():
            if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                # Non-averagable - take max across clients
                avg_state[k] = torch.stack(
                    [u["state_dict"][k] for u in client_updates]
                ).max(dim=0).values
            else:
                acc = torch.zeros_like(v, dtype=torch.float32)
                for u in client_updates:
                    w = u["n_samples"] / total
                    acc += w * u["state_dict"][k].float()
                avg_state[k] = acc.to(v.dtype)

        self.global_model.load_state_dict(
            {k: v.to(DEVICE) for k, v in avg_state.items()}
        )
        self.round += 1

    def weight_hash(self) -> str:
        """Deterministic hash of current global weights (for audit)."""
        h = hashlib.sha256()
        for k in sorted(self.global_model.state_dict().keys()):
            v = self.global_model.state_dict()[k]
            h.update(k.encode())
            # quantize to 6 decimals for cross-platform determinism
            h.update(np.round(v.detach().cpu().numpy(), 6).tobytes())
        return h.hexdigest()

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        self.global_model.eval()
        Xt = torch.from_numpy(X).float()
        ys = []
        for i in range(0, len(X), 2048):
            batch = Xt[i:i+2048].to(DEVICE)
            logits = self.global_model(batch)
            ys.append(logits.argmax(1).cpu().numpy())
        preds = np.concatenate(ys)
        return {
            "accuracy": float((preds == y).mean()),
            "n_samples": int(len(y)),
        }


# ============================================================
#  Orchestrator
# ============================================================
def run_federated_training(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_clients: int = 5,
    n_rounds: int = 20,
    local_epochs: int = 3,
    partition: str = "non_iid",
    concentration: float = 0.5,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    on_round_end: Optional[Callable[[int, "FLServer", List[Dict]], None]] = None,
) -> Tuple["FLServer", List[FLClient], Dict]:
    """
    on_round_end(round_idx, server, client_updates) - called after each
    round; used by the Blockchain demo to log a ModelUpdate block.
    """
    feature_dim = X_train.shape[1]

    # 1. Partition
    if partition == "iid":
        parts = partition_iid(X_train, y_train, n_clients)
    else:
        parts = partition_non_iid(X_train, y_train, n_clients, concentration)
    stats = partition_stats(parts)
    print(f"[FL] partition = {partition} | concentration = {concentration}")
    for s in stats:
        print(f"  client {s['client']}: n={s['n_samples']}  "
              f"classes={s['class_counts']}")

    # 2. Instantiate
    clients = [FLClient(f"client_{i}", Xi, yi, feature_dim)
               for i, (Xi, yi) in enumerate(parts)]
    server = FLServer(feature_dim)

    # class weights based on ALL training data (public statistic)
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = (counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
                      ).astype(np.float32)

    # 3. Run rounds
    history = {"round_acc": [], "round_loss": [],
               "per_client_acc": [], "weight_hashes": [],
               "partition_stats": stats}
    for r in range(n_rounds):
        t0 = time.time()
        global_state = server.global_state()
        client_updates = []
        for c in clients:
            c.set_global_weights(global_state)
            info = c.local_train(local_epochs=local_epochs,
                                 class_weights=class_weights,
                                 use_focal=use_focal,
                                 focal_gamma=focal_gamma)
            client_updates.append(info)

        server.fedavg(client_updates)

        # Evaluate
        val_info = server.evaluate(X_val, y_val)
        wh = server.weight_hash()
        round_loss = float(np.mean([u["loss"] for u in client_updates]))
        history["round_acc"].append(val_info["accuracy"])
        history["round_loss"].append(round_loss)
        history["per_client_acc"].append(
            [u["acc"] for u in client_updates]
        )
        history["weight_hashes"].append(wh)

        elapsed = time.time() - t0
        print(f"  Round {r+1:2d}/{n_rounds} | "
              f"global val acc = {val_info['accuracy']:.4f} | "
              f"loss = {round_loss:.4f} | "
              f"{elapsed:.1f}s | "
              f"hash={wh[:8]}")

        # Callback - lets blockchain demo write audit blocks
        if on_round_end is not None:
            on_round_end(r, server, client_updates)

    return server, clients, history


# ============================================================
#  Main (stand-alone demo)
# ============================================================
def main():
    from data_loader import get_preprocessed
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("  HCAPT-Defense - Layer 3a: Federated Learning")
    print("=" * 70)

    X_full, X_te, y_full, y_te = get_preprocessed()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full,
    )

    server, clients, history = run_federated_training(
        X_tr, y_tr, X_val, y_val,
        n_clients=5, n_rounds=20, local_epochs=3,
        partition="non_iid", concentration=0.5,
    )

    # Final test evaluation
    test_info = server.evaluate(X_te, y_te)
    print(f"\n[FL] Final global model TEST accuracy: "
          f"{test_info['accuracy']:.4f}")

    # Save
    torch.save(server.global_model.state_dict(),
                MODELS_DIR / "fl_global_nn.pt")

    out = {
        "test_accuracy": test_info["accuracy"],
        "final_round_val_acc": history["round_acc"][-1],
        "history": history,
    }
    with open(RESULTS_DIR / "fl_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[FL] Saved -> {RESULTS_DIR / 'fl_results.json'}")


if __name__ == "__main__":
    main()
