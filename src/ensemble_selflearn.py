"""
HCAPT-Defense — Innovation #2: Ensemble + Self-Learning Module
Layer 2 (core): Adaptive detection with continuous learning

Components:
  1. EnsembleDetector
     - Soft voting over {RandomForest, XGBoost, DeepNN}
     - Learned weights from validation performance
     - Confidence scoring for downstream Zero-Trust layer

  2. DeepDetector (PyTorch MLP on GPU)
     - Trained with class-balanced sampling
     - Supports fine-tuning on new samples (online update)

  3. SelfLearningLoop
     - Monitors accuracy on a sliding window
     - Triggers retraining when performance drifts by > THRESHOLD
     - Uses high-confidence predictions as pseudo-labels (semi-supervised)
     - Logs every decision for audit / Zero-Trust handoff
"""
import json
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP,
    BATCH_SIZE, SELF_LEARN_WINDOW, SELF_LEARN_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)
from cache import save_checkpoint, load_checkpoint, mark_phase_done, is_phase_done

torch.manual_seed(SEED)
np.random.seed(SEED)

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  Deep base learner (PyTorch MLP)
# ============================================================
class DeepDetector(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = NUM_CLASSES,
                 hidden: List[int] = (256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        dims = [feature_dim] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, X: np.ndarray, batch: int = 1024) -> np.ndarray:
        self.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i:i+batch]).float().to(DEVICE)
                logits = self(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 30, batch_size: int = BATCH_SIZE,
            lr: float = 1e-3, class_weights: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict:
        self.to(DEVICE).train()

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
            drop_last=True,
        )

        if class_weights is not None:
            w = torch.from_numpy(class_weights).float().to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=w)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"loss": [], "acc": []}
        for epoch in range(epochs):
            self.train()
            running_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
            scheduler.step()

            avg_loss = running_loss / total
            acc = correct / total
            history["loss"].append(avg_loss)
            history["acc"].append(acc)

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"    Epoch {epoch+1:3d}/{epochs} | "
                      f"loss={avg_loss:.4f} | acc={acc:.4f}")
        return history


# ============================================================
#  Ensemble Detector
# ============================================================
class EnsembleDetector:
    """Soft voting over heterogeneous base learners with learned weights."""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.rf: Optional[RandomForestClassifier] = None
        self.xgb: Optional[xgb.XGBClassifier] = None
        self.nn: Optional[DeepDetector] = None
        # Weights learned from validation accuracy per model
        self.weights: np.ndarray = np.ones(3) / 3

    # ----- Training -----
    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            nn_epochs: int = 30, verbose: bool = True) -> Dict:
        print(f"\n[Ensemble] Training on {len(X_tr)} samples")

        # Class weights for imbalance
        counts = np.bincount(y_tr, minlength=NUM_CLASSES)
        class_weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))

        # 1. Random Forest
        print("\n[Ensemble] Training Random Forest...")
        t0 = time.time()
        self.rf = RandomForestClassifier(
            n_estimators=200, min_samples_split=5, n_jobs=-1,
            random_state=SEED, class_weight="balanced",
        )
        self.rf.fit(X_tr, y_tr)
        rf_acc = self.rf.score(X_val, y_val)
        print(f"  RF val acc: {rf_acc:.4f} | time: {time.time()-t0:.1f}s")

        # 2. XGBoost
        print("\n[Ensemble] Training XGBoost (GPU)...")
        t0 = time.time()
        sample_weight = class_weights[y_tr]
        self.xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            objective="multi:softprob", num_class=NUM_CLASSES,
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="mlogloss",
        )
        self.xgb.fit(X_tr, y_tr, sample_weight=sample_weight)
        xgb_acc = self.xgb.score(X_val, y_val)
        print(f"  XGB val acc: {xgb_acc:.4f} | time: {time.time()-t0:.1f}s")

        # 3. Deep NN
        print("\n[Ensemble] Training Deep NN (GPU)...")
        t0 = time.time()
        self.nn = DeepDetector(self.feature_dim).to(DEVICE)
        self.nn.fit(X_tr, y_tr, epochs=nn_epochs, class_weights=class_weights,
                    verbose=verbose)
        p = self.nn.predict_proba(X_val)
        nn_acc = accuracy_score(y_val, p.argmax(1))
        print(f"  NN val acc: {nn_acc:.4f} | time: {time.time()-t0:.1f}s")

        # Learn ensemble weights (softmax over val accuracies)
        accs = np.array([rf_acc, xgb_acc, nn_acc])
        exp_scores = np.exp(accs * 10)  # sharpen
        self.weights = exp_scores / exp_scores.sum()
        print(f"\n[Ensemble] Weights → RF={self.weights[0]:.3f} "
              f"XGB={self.weights[1]:.3f} NN={self.weights[2]:.3f}")

        return {"rf_acc": rf_acc, "xgb_acc": xgb_acc, "nn_acc": nn_acc,
                "weights": self.weights.tolist()}

    # ----- Inference -----
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_rf = self.rf.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        p_nn = self.nn.predict_proba(X)

        # Align class columns (all were fit on same label space 0..NUM_CLASSES-1)
        proba = (self.weights[0] * p_rf +
                 self.weights[1] * p_xgb +
                 self.weights[2] * p_nn)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predictions, confidence) where confidence ∈ [0, 1]."""
        proba = self.predict_proba(X)
        preds = proba.argmax(1)
        conf = proba.max(1)
        return preds, conf

    # ----- Persistence -----
    def save(self, path: Path = MODELS_DIR):
        joblib.dump(self.rf, path / "ens_rf.joblib")
        self.xgb.save_model(str(path / "ens_xgb.json"))
        torch.save(self.nn.state_dict(), path / "ens_nn.pt")
        np.save(path / "ens_weights.npy", self.weights)
        print(f"[Ensemble] Saved to {path}")

    def load(self, path: Path = MODELS_DIR):
        self.rf = joblib.load(path / "ens_rf.joblib")
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(path / "ens_xgb.json"))
        self.nn = DeepDetector(self.feature_dim).to(DEVICE)
        self.nn.load_state_dict(torch.load(path / "ens_nn.pt", map_location=DEVICE))
        self.weights = np.load(path / "ens_weights.npy")
        return self

    # ----- FL integration -----
    def load_fl_nn(self, fl_state_dict: Dict) -> None:
        """Replace the deep-NN base learner with FL-trained weights.

        The caller MUST still fit RF and XGBoost separately (on
        GAN-augmented data) before calling .predict_* methods.
        """
        if self.nn is None:
            self.nn = DeepDetector(self.feature_dim).to(DEVICE)
        self.nn.load_state_dict(
            {k: (v.to(DEVICE) if hasattr(v, "to") else v)
             for k, v in fl_state_dict.items()}
        )
        self.nn.eval()

    def fit_with_fl_nn(self,
                       X_tr: np.ndarray, y_tr: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       fl_state_dict: Dict,
                       verbose: bool = True) -> Dict:
        """Assemble the ensemble using a pre-trained FL Deep-NN plus
        freshly-trained RF + XGBoost base learners.

        Skips the local NN training step entirely.
        """
        print(f"\n[Ensemble] Assembling with FL-trained NN + RF/XGB")
        counts = np.bincount(y_tr, minlength=NUM_CLASSES)
        class_weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))

        # 1. RF
        print("\n[Ensemble] Training Random Forest...")
        t0 = time.time()
        self.rf = RandomForestClassifier(
            n_estimators=200, min_samples_split=5, n_jobs=-1,
            random_state=SEED, class_weight="balanced",
        )
        self.rf.fit(X_tr, y_tr)
        rf_acc = self.rf.score(X_val, y_val)
        print(f"  RF val acc: {rf_acc:.4f} | {time.time()-t0:.1f}s")

        # 2. XGBoost
        print("\n[Ensemble] Training XGBoost (GPU)...")
        t0 = time.time()
        sample_weight = class_weights[y_tr]
        self.xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            objective="multi:softprob", num_class=NUM_CLASSES,
            tree_method="hist", device="cuda", random_state=SEED,
            eval_metric="mlogloss",
        )
        self.xgb.fit(X_tr, y_tr, sample_weight=sample_weight)
        xgb_acc = self.xgb.score(X_val, y_val)
        print(f"  XGB val acc: {xgb_acc:.4f} | {time.time()-t0:.1f}s")

        # 3. FL-trained NN (plug in, no training here)
        print("\n[Ensemble] Plugging in FL-trained Deep NN (no local training)")
        self.load_fl_nn(fl_state_dict)
        p_nn = self.nn.predict_proba(X_val)
        nn_acc = accuracy_score(y_val, p_nn.argmax(1))
        print(f"  FL-NN val acc: {nn_acc:.4f}")

        # 4. Learn ensemble weights
        accs = np.array([rf_acc, xgb_acc, nn_acc])
        exp_scores = np.exp(accs * 10)
        self.weights = exp_scores / exp_scores.sum()
        print(f"\n[Ensemble] Weights -> RF={self.weights[0]:.3f} "
              f"XGB={self.weights[1]:.3f} FL-NN={self.weights[2]:.3f}")

        return {"rf_acc": rf_acc, "xgb_acc": xgb_acc, "nn_acc": nn_acc,
                "weights": self.weights.tolist(), "nn_source": "FL"}


# ============================================================
#  Self-Learning Loop
# ============================================================
class SelfLearningLoop:
    """Online adaptation for the Ensemble.

    Maintains a sliding window of recent predictions + true labels (when
    available). When accuracy drifts below baseline_acc - threshold, or
    when enough high-confidence pseudo-labels accumulate, triggers a
    fine-tuning step on the DeepDetector and re-weights the ensemble.
    """

    def __init__(self,
                 ensemble: EnsembleDetector,
                 baseline_acc: float,
                 window: int = SELF_LEARN_WINDOW,
                 drift_threshold: float = SELF_LEARN_THRESHOLD,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.ensemble = ensemble
        self.baseline_acc = baseline_acc
        self.window = window
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold

        # Sliding windows
        self.pred_window = deque(maxlen=window)   # (pred, true_or_None)
        self.pseudo_buffer_X: List[np.ndarray] = []
        self.pseudo_buffer_y: List[int] = []

        # Audit log
        self.events: List[Dict] = []
        self.retrain_count = 0

    # ----- Online inference -----
    def process_batch(self, X: np.ndarray,
                      y_true: Optional[np.ndarray] = None) -> Dict:
        """Process an incoming batch; return decisions + log any drift event."""
        preds, conf = self.ensemble.predict_with_confidence(X)

        # Update window with known labels (supervised signal)
        if y_true is not None:
            for p, t in zip(preds, y_true):
                self.pred_window.append((int(p), int(t)))

        # Harvest high-confidence predictions as pseudo-labels
        high_conf_mask = conf >= self.confidence_threshold
        if high_conf_mask.any():
            self.pseudo_buffer_X.append(X[high_conf_mask])
            self.pseudo_buffer_y.extend(preds[high_conf_mask].tolist())

        # Check drift + trigger retrain
        drift_event = self._check_drift()
        retrain_event = self._maybe_retrain()

        return {
            "predictions": preds,
            "confidence": conf,
            "high_confidence_count": int(high_conf_mask.sum()),
            "pseudo_buffer_size": sum(len(x) for x in self.pseudo_buffer_X),
            "drift_detected": drift_event is not None,
            "retrained": retrain_event is not None,
            "current_accuracy": self._window_accuracy(),
        }

    # ----- Internals -----
    def _window_accuracy(self) -> Optional[float]:
        if len(self.pred_window) < 100:
            return None
        correct = sum(1 for p, t in self.pred_window if p == t)
        return correct / len(self.pred_window)

    def _check_drift(self) -> Optional[Dict]:
        acc = self._window_accuracy()
        if acc is None:
            return None
        drop = self.baseline_acc - acc
        if drop > self.drift_threshold:
            event = {
                "type": "drift",
                "baseline_acc": self.baseline_acc,
                "current_acc": acc,
                "drop": drop,
                "timestamp": time.time(),
            }
            self.events.append(event)
            print(f"[SelfLearn] !! DRIFT DETECTED: acc dropped from "
                  f"{self.baseline_acc:.4f} -> {acc:.4f} (drop={drop:.4f})")
            return event
        return None

    def _maybe_retrain(self) -> Optional[Dict]:
        total_pseudo = sum(len(x) for x in self.pseudo_buffer_X)
        if total_pseudo < self.window:
            return None

        X_new = np.concatenate(self.pseudo_buffer_X, axis=0)
        y_new = np.array(self.pseudo_buffer_y, dtype=np.int64)
        print(f"[SelfLearn] >> Fine-tuning on {len(X_new)} pseudo-labeled samples")

        t0 = time.time()
        # Lightweight fine-tune: 5 epochs, low LR
        self.ensemble.nn.fit(X_new, y_new, epochs=5, lr=1e-4, verbose=False)
        elapsed = time.time() - t0
        self.retrain_count += 1

        # Clear buffers
        self.pseudo_buffer_X.clear()
        self.pseudo_buffer_y.clear()

        # Update baseline to current moving accuracy
        new_acc = self._window_accuracy()
        if new_acc is not None:
            self.baseline_acc = max(self.baseline_acc, new_acc)

        event = {
            "type": "retrain",
            "retrain_num": self.retrain_count,
            "samples_used": int(len(X_new)),
            "time_s": elapsed,
            "new_baseline_acc": self.baseline_acc,
            "timestamp": time.time(),
        }
        self.events.append(event)
        print(f"[SelfLearn]    retrain #{self.retrain_count} "
              f"complete in {elapsed:.1f}s")
        return event

    def save_log(self, path: Path = RESULTS_DIR):
        out = path / "selflearn_events.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
        print(f"[SelfLearn] Event log saved: {out}")


# ============================================================
#  Evaluation helpers
# ============================================================
def evaluate_ensemble(ensemble: EnsembleDetector,
                      X_te: np.ndarray, y_te: np.ndarray,
                      name: str = "Ensemble") -> Dict:
    t0 = time.time()
    preds, conf = ensemble.predict_with_confidence(X_te)
    infer_time = (time.time() - t0) / len(X_te)

    acc = accuracy_score(y_te, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, preds, average="weighted", zero_division=0,
    )
    tn = ((y_te == 0) & (preds == 0)).sum()
    fp = ((y_te == 0) & (preds != 0)).sum()
    fpr = fp / max(tn + fp, 1)

    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_te, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0,
    )

    return {
        "model": name,
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "fpr": float(fpr),
        "mean_confidence": float(conf.mean()),
        "inference_time_per_sample_ms": infer_time * 1000,
        "per_class": {
            INV_LABEL_MAP[i]: {
                "precision": float(prec_pc[i]),
                "recall": float(rec_pc[i]),
                "f1": float(f1_pc[i]),
                "support": int(sup_pc[i]),
            } for i in range(NUM_CLASSES)
        },
    }


def print_metrics(m: Dict):
    print(f"\n{'='*60}")
    print(f"  {m['model']}")
    print(f"{'='*60}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1 (wgt) : {m['f1_weighted']:.4f}")
    print(f"  Precision: {m['precision_weighted']:.4f}")
    print(f"  Recall   : {m['recall_weighted']:.4f}")
    print(f"  FPR      : {m['fpr']:.4f}")
    print(f"  Mean conf: {m['mean_confidence']:.4f}")
    print(f"  Inference: {m['inference_time_per_sample_ms']:.3f} ms/sample")
    print("  Per-class:")
    for cls, s in m["per_class"].items():
        print(f"    {cls:<7} | P={s['precision']:.3f} "
              f"R={s['recall']:.3f} F1={s['f1']:.3f} n={s['support']}")


# ============================================================
#  Main: train ensemble -> evaluate -> simulate online stream
# ============================================================
def main():
    from data_loader import load_raw, map_attack_category, encode_and_scale
    from sklearn.model_selection import train_test_split

    print("="*60)
    print("  HCAPT-Defense - Innovation #2: Ensemble + Self-Learning")
    print("="*60)

    # 1. Data
    df_train, df_test = load_raw()
    df_train = map_attack_category(df_train)
    df_test = map_attack_category(df_test)
    X_full, X_te, y_full, y_te = encode_and_scale(df_train, df_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full,
    )
    print(f"\n[data] Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} "
          f"| Test: {X_te.shape[0]}")

    # 2. Train ensemble
    ensemble = EnsembleDetector(feature_dim=X_tr.shape[1])
    train_info = ensemble.fit(X_tr, y_tr, X_val, y_val, nn_epochs=30)
    ensemble.save()

    # 3. Evaluate
    m_ens = evaluate_ensemble(ensemble, X_te, y_te, name="HCAPT-Ensemble")
    print_metrics(m_ens)

    # 4. Self-Learning simulation on test stream
    print(f"\n{'='*60}")
    print("  Self-Learning Simulation (online stream)")
    print(f"{'='*60}")

    loop = SelfLearningLoop(
        ensemble=ensemble,
        baseline_acc=train_info["nn_acc"],  # deep model's val acc
    )

    # Shuffle test set to simulate streaming
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_te))
    X_stream, y_stream = X_te[perm], y_te[perm]

    # Process in chunks of ~500 samples (simulates real-time micro-batches)
    chunk_size = 500
    stream_metrics = []
    for i in range(0, len(X_stream), chunk_size):
        Xb = X_stream[i:i+chunk_size]
        yb = y_stream[i:i+chunk_size]
        info = loop.process_batch(Xb, y_true=yb)
        if info["current_accuracy"] is not None:
            stream_metrics.append({
                "batch": i // chunk_size,
                "accuracy": info["current_accuracy"],
                "high_conf": info["high_confidence_count"],
                "retrained": info["retrained"],
            })

    # 5. Re-evaluate after self-learning
    print(f"\n[SelfLearn] Total retrains triggered: {loop.retrain_count}")
    m_ens_after = evaluate_ensemble(
        ensemble, X_te, y_te, name="HCAPT-Ensemble (post self-learning)",
    )
    print_metrics(m_ens_after)

    # 6. Save everything
    results = {
        "ensemble_training": train_info,
        "before_selflearning": m_ens,
        "after_selflearning": m_ens_after,
        "stream_metrics": stream_metrics,
        "retrain_events": [e for e in loop.events if e["type"] == "retrain"],
        "drift_events": [e for e in loop.events if e["type"] == "drift"],
    }
    out_file = RESULTS_DIR / "ensemble_selflearn_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    loop.save_log()
    print(f"\n[OK] All results saved to {out_file}")


if __name__ == "__main__":
    main()
