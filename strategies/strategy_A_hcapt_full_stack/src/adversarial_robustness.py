"""
Adversarial Robustness Module — FGSM, PGD, CW-L2 attacks on HCAPT-Defense.

Pure PyTorch implementations (no torchattacks dependency for portability).
Tests the DNN component of the ensemble against:
  * FGSM (Fast Gradient Sign Method) — Goodfellow et al. 2015
  * PGD  (Projected Gradient Descent) — Madry et al. 2018
  * CW-L2 (Carlini-Wagner) — Carlini & Wagner 2017

Since RF/XGBoost are non-differentiable, attacks target the DNN only,
then the full ensemble is evaluated on the adversarial examples. This
measures whether HCAPT's tree learners provide *robust voting* even
when one ensemble member is fooled.

Writes:
  results/_adversarial/<dataset>_adversarial.json
  results/_adversarial/<dataset>_adversarial.csv
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .cache import DatasetCache
from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import EnsembleAdHoc, compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_adversarial"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


# =============================================================================
# Attack implementations
# =============================================================================
def fgsm_attack(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                epsilon: float) -> torch.Tensor:
    """Fast Gradient Sign Method (single-step)."""
    X_adv = X.clone().detach().requires_grad_(True)
    logits = model(X_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, X_adv)[0]
    X_adv = X_adv + epsilon * grad.sign()
    return X_adv.detach()


def pgd_attack(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
               epsilon: float, alpha: float, n_steps: int) -> torch.Tensor:
    """Projected Gradient Descent (iterative)."""
    X_orig = X.clone().detach()
    X_adv = X.clone().detach()
    # random init within epsilon ball
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    for _ in range(n_steps):
        X_adv = X_adv.detach().requires_grad_(True)
        logits = model(X_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv + alpha * grad.sign()
        # project back into epsilon-ball around original
        X_adv = torch.max(torch.min(X_adv, X_orig + epsilon), X_orig - epsilon)
    return X_adv.detach()


def cw_l2_attack(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                 c: float = 1.0, n_steps: int = 100, lr: float = 0.01) -> torch.Tensor:
    """Carlini-Wagner L2 attack (simplified, untargeted)."""
    X_orig = X.clone().detach()
    # parameterise via tanh to keep values bounded
    w = (X_orig.clone() / (1 + 1e-6)).atanh().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    best_adv = X_orig.clone()
    for _ in range(n_steps):
        X_adv = w.tanh()
        logits = model(X_adv)
        true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        other_max = logits.scatter(1, y.unsqueeze(1), float("-inf")).max(dim=1)[0]
        adv_loss = torch.clamp(true_logit - other_max, min=0).sum()
        l2 = ((X_adv - X_orig) ** 2).sum()
        loss = l2 + c * adv_loss
        opt.zero_grad(); loss.backward(); opt.step()
    return w.tanh().detach()


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_one(ens: EnsembleAdHoc, ds_name: str,
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate clean + 3 adversarial conditions on a trained ensemble.

    Adversarial examples are generated targeting the DNN, then the
    full ensemble (RF + XGB + DNN) predicts on them.
    """
    X_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    print(f"\n[{ds_name}] Generating adversarial examples…")
    results = {}

    # Clean baseline
    clean_preds, _ = ens.predict_with_confidence(X_test)
    results["clean"] = compute_metrics(y_test, clean_preds, is_binary=True)
    print(f"   clean:   acc={results['clean']['accuracy']:.4f}  f1={results['clean']['f1']:.4f}")

    # Note: NN must be in eval mode but we need gradients for the attack.
    ens.nn.eval()
    for p in ens.nn.parameters():
        p.requires_grad = False

    # FGSM at multiple epsilons
    for eps in [0.05, 0.1, 0.3]:
        print(f"   FGSM eps={eps}…", end=" ")
        t0 = time.time()
        X_adv_t = fgsm_attack(ens.nn, X_t, y_t, epsilon=eps)
        X_adv = X_adv_t.cpu().numpy()
        preds, _ = ens.predict_with_confidence(X_adv)
        m = compute_metrics(y_test, preds, is_binary=True)
        m["attack"] = "fgsm"; m["epsilon"] = eps
        m["time_s"] = round(time.time() - t0, 1)
        results[f"fgsm_eps{eps}"] = m
        print(f"acc={m['accuracy']:.4f} ({m['time_s']:.1f}s)")

    # PGD eps=0.1, 10 steps
    print(f"   PGD eps=0.1 (10 steps)…", end=" ")
    t0 = time.time()
    X_adv_t = pgd_attack(ens.nn, X_t, y_t,
                          epsilon=0.1, alpha=0.01, n_steps=10)
    X_adv = X_adv_t.cpu().numpy()
    preds, _ = ens.predict_with_confidence(X_adv)
    m = compute_metrics(y_test, preds, is_binary=True)
    m["attack"] = "pgd"; m["epsilon"] = 0.1; m["n_steps"] = 10
    m["time_s"] = round(time.time() - t0, 1)
    results["pgd_eps0.1_10steps"] = m
    print(f"acc={m['accuracy']:.4f} ({m['time_s']:.1f}s)")

    # CW-L2 on FULL test set, batched (memory-bounded; iterative attack)
    n_full = len(X_t)
    print(f"   CW-L2 (FULL {n_full} samples, batched)…", flush=True)
    t0 = time.time()
    batch_size = 2000
    X_adv_full = torch.empty_like(X_t)
    for start in range(0, n_full, batch_size):
        end = min(start + batch_size, n_full)
        Xb = X_t[start:end]
        yb = y_t[start:end]
        X_adv_b = cw_l2_attack(ens.nn, Xb, yb, c=1.0, n_steps=50)
        X_adv_full[start:end] = X_adv_b
        if (start // batch_size) % 5 == 0:
            print(f"     CW batch {start//batch_size+1}/{(n_full+batch_size-1)//batch_size}"
                   f" ({(end/n_full)*100:.0f}%) elapsed={time.time()-t0:.0f}s", flush=True)
    X_adv = X_adv_full.cpu().numpy()
    preds, _ = ens.predict_with_confidence(X_adv)
    m = compute_metrics(y_test, preds, is_binary=True)
    m["attack"] = "cw_l2"; m["n_samples"] = n_full
    m["time_s"] = round(time.time() - t0, 1)
    results["cw_l2"] = m
    print(f"   CW-L2 final: acc={m['accuracy']:.4f} ({m['time_s']:.1f}s)")

    return results


# =============================================================================
# Per-dataset runner
# =============================================================================
def run_dataset(loader_fn, ds_name: str, dnn_epochs: int = 10):
    print(f"\n{'='*72}\n  {ds_name}\n{'='*72}")
    ds = loader_fn()
    num_classes = len(ds.class_names)
    # Train ensemble (small DNN for speed since we only need it for attacks)
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, random_state=42, stratify=ds.y_train,
    )
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=dnn_epochs)
    ens.fit(X_t, y_t, X_v, y_v)
    # FULL test set (no subsampling)
    X_test = ds.X_test; y_test = ds.y_test
    n = len(X_test)
    print(f"  using FULL test set: n={n}", flush=True)
    results = evaluate_one(ens, ds_name, X_test, y_test)
    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{ds_name}_adversarial.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": ds_name, "n_test_samples": int(n),
                    "subsampling": "NONE (full test set)",
                    "results": results}, f, indent=2)
    print(f"\n[done] wrote {out_path.name}")
    return results


def main():
    import os
    only = os.environ.get("DATASET", "").strip()
    datasets = [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
        ("dapt-2020",         lambda: Dapt2020Loader(DATA_ROOT).load("binary")),
    ]
    if only:
        datasets = [(n, l) for n, l in datasets if n == only]
    for name, loader in datasets:
        run_dataset(loader, name, dnn_epochs=10)


if __name__ == "__main__":
    main()
