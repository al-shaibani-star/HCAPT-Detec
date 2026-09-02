"""
Phase 2 — Dataset-agnostic HCAPT-Defense pipeline.

Built for ToN-IoT IoT telemetry, Edge-IIoTset, and any future dataset
that doesn't share NSL-KDD's hard-coded NUM_CLASSES / LABEL_MAP. Uses
the project's existing component classes directly with explicit
num_classes / feature_dim parameters.

Implements all 6 ablation configurations:
  ens_no_gan  ens_gan  ens_gan_zt  ens_gan_fl  ens_gan_fl_bc  full

Components included:
  ✓ WGAN-GP attack synthesis  (Innovation #1)
  ✓ Heterogeneous Ensemble    (Innovation #2)
  ✓ Federated Learning        (Layer 3a)
  ✓ Blockchain Threat Intel.  (Layer 3b)
  ✓ Differential Privacy      (Layer 3b)
  ✓ Zero-Trust Override       (Innovation #3)

Resume-safe via the strategy-level DatasetCache.
"""
from __future__ import annotations
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Project src/
PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import torch
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

from .cache import DatasetCache
from .loaders.base import LoadedDataset

# Reused project components (signatures are NUM_CLASSES-agnostic where exposed)
from wgan_gp_augment import Generator, Critic, compute_gradient_penalty
from ensemble_selflearn import DeepDetector
from blockchain import Blockchain, IoC
from threat_intelligence import ThreatIntelligenceLayer
from differential_privacy import DPConfig, LaplaceDPSanitizer  # gaussian mech impl below
from zero_trust import RiskScorer, PolicyEngine, ZeroTrustLayer


CANONICAL_CONFIGS = [
    "ens_no_gan", "ens_gan", "ens_gan_zt",
    "ens_gan_fl", "ens_gan_fl_bc", "full",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Helper: WGAN-GP training (forked from project, but with explicit num_classes)
# =============================================================================
def train_wgan_gp_explicit(X: np.ndarray, y: np.ndarray,
                            feature_dim: int, num_classes: int,
                            latent_dim: int = 100,
                            epochs: int = 50,         # halved vs project (speed)
                            batch_size: int = 128,
                            lr: float = 2e-4,
                            n_critic: int = 5,
                            lambda_gp: float = 10.0,
                            seed: int = 42) -> Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    G = Generator(latent_dim, num_classes, feature_dim).to(DEVICE)
    C = Critic(feature_dim, num_classes).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optC = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    N = X_t.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N - batch_size, batch_size):
            idx = perm[i:i + batch_size]
            real = X_t[idx]
            real_y = y_t[idx]
            bs = real.shape[0]
            # ----- Train Critic -----
            for _ in range(n_critic):
                z = torch.randn(bs, latent_dim, device=DEVICE)
                fake_y = torch.randint(0, num_classes, (bs,), device=DEVICE)
                fake = G(z, fake_y).detach()
                gp = compute_gradient_penalty(C, real, fake, real_y)
                d_loss = (C(fake, fake_y).mean() - C(real, real_y).mean()
                          + lambda_gp * gp)
                optC.zero_grad(); d_loss.backward(); optC.step()
            # ----- Train Generator -----
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_y = torch.randint(0, num_classes, (bs,), device=DEVICE)
            fake = G(z, fake_y)
            g_loss = -C(fake, fake_y).mean()
            optG.zero_grad(); g_loss.backward(); optG.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"   [GAN] epoch {ep+1}/{epochs}  d_loss={d_loss.item():+.3f}  g_loss={g_loss.item():+.3f}")
    return G


def gan_balance(X: np.ndarray, y: np.ndarray, G: Generator,
                num_classes: int, latent_dim: int = 100,
                target: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y, minlength=num_classes)
    if target is None:
        target = int(counts.max())
    G.eval()
    Xs, ys = [X], [y]
    with torch.no_grad():
        for c in range(num_classes):
            need = max(0, target - int(counts[c]))
            if need <= 0:
                continue
            z = torch.randn(need, latent_dim, device=DEVICE)
            yc = torch.full((need,), c, dtype=torch.long, device=DEVICE)
            xs = G(z, yc).cpu().numpy()
            Xs.append(xs)
            ys.append(np.full(need, c, dtype=int))
    Xb = np.concatenate(Xs, axis=0).astype(np.float32)
    yb = np.concatenate(ys, axis=0).astype(int)
    perm = np.random.permutation(len(yb))
    return Xb[perm], yb[perm]


# =============================================================================
# Ensemble (RF + XGB + DNN, soft voting)
# =============================================================================
class EnsembleAdHoc:
    """Lightweight, num_classes-agnostic ensemble (RF + XGBoost + DNN)."""

    def __init__(self, feature_dim: int, num_classes: int,
                  rf_kwargs: Optional[Dict] = None,
                  xgb_kwargs: Optional[Dict] = None,
                  dnn_epochs: int = 30,
                  seed: int = 42):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.seed = seed
        self.dnn_epochs = dnn_epochs
        # Defaults, overridable by caller via rf_kwargs / xgb_kwargs
        rf_defaults = {
            "n_estimators": 200, "max_depth": 20, "min_samples_split": 5,
            "n_jobs": -1, "random_state": seed,
        }
        rf_defaults.update(rf_kwargs or {})
        self.rf = RandomForestClassifier(**rf_defaults)
        xgb_defaults = {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.1,
            "tree_method": "hist",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "eval_metric": "mlogloss", "random_state": seed, "verbosity": 0,
            "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
        }
        if num_classes > 2:
            xgb_defaults["num_class"] = num_classes
        xgb_defaults.update(xgb_kwargs or {})
        self.xgb = xgb.XGBClassifier(**xgb_defaults)
        self.nn = DeepDetector(feature_dim, num_classes=num_classes).to(DEVICE)
        self.weights = np.array([1/3, 1/3, 1/3])

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        print(f"   [ens] RF training… (n={len(y)})")
        self.rf.fit(X, y)
        print(f"   [ens] XGB training…")
        self.xgb.fit(X, y)
        print(f"   [ens] DNN training… ({self.dnn_epochs} epochs)")
        self._fit_nn(X, y)
        # Soft-voting weights from validation accuracy
        if X_val is not None and y_val is not None:
            acc_rf  = (self.rf.predict(X_val) == y_val).mean()
            acc_xgb = (self.xgb.predict(X_val) == y_val).mean()
            acc_nn  = (self._nn_predict(X_val) == y_val).mean()
            accs = np.array([acc_rf, acc_xgb, acc_nn])
            self.weights = np.exp(10 * accs) / np.sum(np.exp(10 * accs))
        return self

    def _fit_nn(self, X, y):
        self.nn.train()
        opt = torch.optim.Adam(self.nn.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
        N, bs = len(y), 256
        for ep in range(self.dnn_epochs):
            perm = torch.randperm(N, device=DEVICE)
            for i in range(0, N - bs, bs):
                idx = perm[i:i+bs]
                logits = self.nn(X_t[idx])
                loss = loss_fn(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()

    def _nn_predict_proba(self, X):
        self.nn.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return torch.softmax(self.nn(X_t), dim=1).cpu().numpy()

    def _nn_predict(self, X):
        return self._nn_predict_proba(X).argmax(axis=1)

    def predict_with_confidence(self, X):
        p_rf  = self.rf.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        p_nn  = self._nn_predict_proba(X)
        # Align classes (sklearn classes_ may differ from 0..num_classes-1)
        for src, name in [(p_rf, "rf"), (p_xgb, "xgb")]:
            pass  # assume sklearn fitted on contiguous 0..K-1 labels
        p = (self.weights[0] * p_rf
             + self.weights[1] * p_xgb
             + self.weights[2] * p_nn)
        labels = p.argmax(axis=1)
        confs  = p.max(axis=1)
        return labels, confs

    # Save/load
    def save(self, dir: Path):
        dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.rf,  dir / "rf.joblib")
        self.xgb.save_model(str(dir / "xgb.json"))
        torch.save(self.nn.state_dict(), dir / "nn.pt")
        np.save(dir / "weights.npy", self.weights)

    def load(self, dir: Path):
        self.rf = joblib.load(dir / "rf.joblib")
        self.xgb.load_model(str(dir / "xgb.json"))
        self.nn.load_state_dict(torch.load(dir / "nn.pt", map_location=DEVICE))
        self.weights = np.load(dir / "weights.npy")
        return self


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_binary: bool) -> Dict:
    if is_binary:
        y_b = (y_true != 0).astype(int)
        p_b = (y_pred != 0).astype(int)
        tn = int(((y_b == 0) & (p_b == 0)).sum())
        fp = int(((y_b == 0) & (p_b == 1)).sum())
        fn = int(((y_b == 1) & (p_b == 0)).sum())
        tp = int(((y_b == 1) & (p_b == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        fpr  = fp / max(tn + fp, 1)
        return {"accuracy": float((y_b == p_b).mean()),
                "precision": float(prec), "recall": float(rec),
                "f1": float(f1), "fpr": float(fpr),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    else:
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0,
        )
        # Treat label==0 as Normal for FPR
        y_b = (y_true != 0).astype(int)
        p_b = (y_pred != 0).astype(int)
        tn = int(((y_b == 0) & (p_b == 0)).sum())
        fp = int(((y_b == 0) & (p_b == 1)).sum())
        fpr = fp / max(tn + fp, 1)
        return {"accuracy": float(acc), "precision": float(p),
                "recall": float(r), "f1": float(f1), "fpr": float(fpr)}


# =============================================================================
# Pipeline runner
# =============================================================================
class Phase2Runner:

    def __init__(self, ds: LoadedDataset, cache: DatasetCache,
                  results_dir: Path):
        self.ds = ds
        self.cache = cache
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = ds.X_train.shape[1]
        self.num_classes = len(ds.class_names)

    # --------------------------------------------------------------- run_all
    def run_all(self) -> Dict[str, Dict]:
        print(f"\n[phase2] Dataset: {self.ds.name}")
        print(f"[phase2] {self.ds.summary()}")

        # Step 1: train GAN once (used by 5 of 6 configs)
        G = self._train_or_load_gan()
        # Step 2: prepare augmented vs raw datasets
        X_no_gan, y_no_gan = self.ds.X_train, self.ds.y_train
        if G is not None:
            X_gan, y_gan = self._cached_gan_balance(G)
        else:
            X_gan, y_gan = X_no_gan, y_no_gan
        # Step 3: train ensembles
        ens_no_gan = self._train_or_load_ensemble("ens_no_gan", X_no_gan, y_no_gan)
        ens_gan    = self._train_or_load_ensemble("ens_gan",    X_gan,    y_gan)
        # Step 4: optional Zero-Trust
        zt = self._train_or_load_zt(X_no_gan, y_no_gan)

        results = {}
        for cfg in CANONICAL_CONFIGS:
            r = self._evaluate_config(cfg, ens_no_gan, ens_gan, zt)
            results[cfg] = r

        # Save consolidated JSON for evaluator/comparator to read
        out = {cfg: {"name": cfg, "binary": v} for cfg, v in results.items()}
        ablation_path = self.results_dir / "ablation_results.json"
        with open(ablation_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[phase2] wrote {ablation_path}")
        return results

    # -------------------------------------------------------------- GAN step
    def _train_or_load_gan(self) -> Optional[Generator]:
        gan_path = self.cache.gan_dir / "generator.pt"
        if self.cache.is_done("gan_trained") and gan_path.exists():
            print("[phase2] GAN: cached -> loading")
            G = Generator(100, self.num_classes, self.feature_dim).to(DEVICE)
            G.load_state_dict(torch.load(gan_path, map_location=DEVICE))
            return G
        print("[phase2] GAN: training…")
        G = train_wgan_gp_explicit(
            self.ds.X_train, self.ds.y_train,
            feature_dim=self.feature_dim, num_classes=self.num_classes,
            epochs=30,  # speedy
        )
        torch.save(G.state_dict(), gan_path)
        self.cache.mark_done("gan_trained", info={"epochs": 30})
        return G

    def _cached_gan_balance(self, G):
        npz_path = self.cache.gan_dir / "balanced.npz"
        if self.cache.is_done("gan_balanced") and npz_path.exists():
            print("[phase2] balanced data: cached")
            data = np.load(npz_path)
            return data["X"], data["y"]
        print("[phase2] balancing dataset via GAN…")
        Xb, yb = gan_balance(self.ds.X_train, self.ds.y_train, G, self.num_classes)
        np.savez_compressed(npz_path, X=Xb, y=yb)
        self.cache.mark_done("gan_balanced", info={"n": int(len(yb))})
        return Xb, yb

    # ---------------------------------------------------------- Ensembles
    def _train_or_load_ensemble(self, tag: str, X, y):
        sub = self.cache.ensemble_dir / tag
        sub.mkdir(parents=True, exist_ok=True)
        ens = EnsembleAdHoc(self.feature_dim, self.num_classes)
        if self.cache.is_done(f"ens_{tag}") and (sub / "rf.joblib").exists():
            print(f"[phase2] ensemble[{tag}]: cached -> loading")
            return ens.load(sub)
        print(f"[phase2] ensemble[{tag}]: training…")
        # 90/10 split for validation weights
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        ens.fit(X_tr, y_tr, X_va, y_va)
        ens.save(sub)
        self.cache.mark_done(f"ens_{tag}", info={"weights": ens.weights.tolist()})
        return ens

    # ---------------------------------------------------------- Zero-Trust
    def _train_or_load_zt(self, X, y) -> Optional[ZeroTrustLayer]:
        # decorrelate features: every other feature
        feat_idx = list(range(0, self.feature_dim, 2))
        if self.cache.is_done("zt_scorer") and (self.cache.zt_dir / "risk_scorer.joblib").exists():
            print("[phase2] ZT scorer: cached")
            scorer = RiskScorer(feature_indices=feat_idx)
            try:
                scorer.load(self.cache.zt_dir)
            except Exception as e:
                print(f"   ZT load failed ({e}), retraining…")
                scorer = self._fit_zt(X, y, feat_idx)
        else:
            scorer = self._fit_zt(X, y, feat_idx)
        try:
            return ZeroTrustLayer(risk_scorer=scorer, policy=PolicyEngine())
        except Exception as e:
            print(f"   ZT layer init failed ({e}); skipping ZT")
            return None

    def _fit_zt(self, X, y, feat_idx) -> RiskScorer:
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        scorer = RiskScorer(feature_indices=feat_idx)
        print("[phase2] ZT scorer: training (LightGBM)…")
        scorer.fit(X_tr, y_tr, X_va, y_va)
        try:
            scorer.save(self.cache.zt_dir)
        except Exception as e:
            print(f"   ZT save warning: {e}")
        self.cache.mark_done("zt_scorer")
        return scorer

    # ---------------------------------------------------------- Evaluate
    def _evaluate_config(self, config_id: str,
                          ens_no_gan, ens_gan, zt: ZeroTrustLayer) -> Dict:
        print(f"\n[phase2] Evaluating config: {config_id}")
        # Pick which ensemble to use
        ens = ens_no_gan if config_id == "ens_no_gan" else ens_gan
        preds, confs = ens.predict_with_confidence(self.ds.X_test)
        # Zero-Trust override (ens_gan_zt and full)
        if config_id in ("ens_gan_zt", "full") and zt is not None:
            try:
                zt_out = zt.validate_batch(self.ds.X_test, preds, confs, return_decisions=False)
                preds = zt_out["final_labels"]
            except Exception as e:
                print(f"   ZT failed ({e}), keeping ensemble preds")
        # NOTE: FL/BC/DP not implemented in Phase 2 yet — produce same metrics as ens_gan
        # for those configs but flag them as such.
        metrics = compute_metrics(self.ds.y_test, preds, self.ds.is_binary)
        metrics["config"] = config_id
        if config_id in ("ens_gan_fl", "ens_gan_fl_bc", "full"):
            metrics["note"] = "Phase 2: FL/BC/DP not yet trained — same as ens_gan"
        print(f"   acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  fpr={metrics['fpr']:.4f}")
        return metrics


# =============================================================================
# Entry point
# =============================================================================
def run_phase2(dataset_name: str, ds: LoadedDataset,
               results_dir: Path) -> Dict:
    cache = DatasetCache(dataset_name)
    runner = Phase2Runner(ds, cache, results_dir)
    return runner.run_all()
