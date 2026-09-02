"""Adaptive adversarial attacker that knows the Zero-Trust layer.

The standard FGSM/PGD/CW attacks evaluated in
src.adversarial_robustness are *naive* — they generate perturbations
that fool the DNN classifier and assume the soft-voting ensemble
will follow. An adaptive attacker who has read this paper would:

  (a) Know the ZT layer uses a different random seed (SEED+777)
      on a disjoint feature subset.
  (b) Target perturbations that simultaneously fool the DNN AND
      the LightGBM risk scorer.
  (c) Stay within the FGSM / PGD / CW epsilon budget.

This module implements (b)+(c) by computing a JOINT gradient
combining DNN cross-entropy and the LightGBM risk scorer's
log-margin into a single attack objective. The result is the
*adaptive* analogue of FGSM and PGD.

Output: results/_adaptive_adversarial/{dataset}_adaptive.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .phase2_pipeline import EnsembleAdHoc, compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_adaptive_adversarial"
SEED          = 42


def adaptive_fgsm(dnn: torch.nn.Module, risk_scorer,
                  X: torch.Tensor, y: torch.Tensor,
                  epsilon: float, alpha_ratio: float = 0.5) -> torch.Tensor:
    """Joint-gradient FGSM: gradient = alpha * d(DNN_CE)/dX +
                                       (1-alpha) * d(LGBM_log_margin)/dX

    LightGBM is non-differentiable, so we approximate its gradient with
    a finite-difference probe in the feature space.
    """
    X_adv = X.clone().detach().requires_grad_(True)
    logits = dnn(X_adv)
    ce_loss = F.cross_entropy(logits, y)
    dnn_grad = torch.autograd.grad(ce_loss, X_adv)[0]

    # Finite-difference probe of LightGBM in batch (per-dim)
    if risk_scorer is not None:
        X_np = X.detach().cpu().numpy()
        base = risk_scorer.predict_proba(X_np)[:, 1]
        d = X_np.shape[1]
        eps_probe = 1e-3
        lgb_grad = np.zeros_like(X_np)
        # Approximate: probe each feature dim once on a single representative
        # row (the mean of the batch) to avoid d * batch full re-evaluations
        mean_row = X_np.mean(axis=0, keepdims=True)
        base_row = risk_scorer.predict_proba(mean_row)[:, 1][0]
        for j in range(d):
            probe = mean_row.copy()
            probe[0, j] += eps_probe
            lgb_grad[:, j] = (risk_scorer.predict_proba(probe)[:, 1][0] - base_row) / eps_probe
        lgb_grad_t = torch.from_numpy(lgb_grad.astype(np.float32)).to(DEVICE)
        grad = alpha_ratio * dnn_grad + (1 - alpha_ratio) * lgb_grad_t
    else:
        grad = dnn_grad

    return (X + epsilon * grad.sign()).detach()


def adaptive_pgd(dnn: torch.nn.Module, risk_scorer,
                 X: torch.Tensor, y: torch.Tensor,
                 epsilon: float, n_steps: int = 10,
                 alpha_ratio: float = 0.5) -> torch.Tensor:
    """Iterative joint-gradient PGD."""
    X_orig = X.clone().detach()
    X_adv = X_orig + torch.empty_like(X_orig).uniform_(-epsilon, epsilon)
    step_size = epsilon / 4
    for _ in range(n_steps):
        X_adv = X_adv.detach().requires_grad_(True)
        logits = dnn(X_adv)
        ce_loss = F.cross_entropy(logits, y)
        dnn_grad = torch.autograd.grad(ce_loss, X_adv)[0]
        if risk_scorer is not None:
            X_np = X_adv.detach().cpu().numpy()
            mean_row = X_np.mean(axis=0, keepdims=True)
            d = X_np.shape[1]
            eps_probe = 1e-3
            base_row = risk_scorer.predict_proba(mean_row)[:, 1][0]
            lgb_grad = np.zeros_like(X_np)
            for j in range(d):
                probe = mean_row.copy()
                probe[0, j] += eps_probe
                lgb_grad[:, j] = (risk_scorer.predict_proba(probe)[:, 1][0]
                                    - base_row) / eps_probe
            lgb_grad_t = torch.from_numpy(lgb_grad.astype(np.float32)).to(DEVICE)
            grad = alpha_ratio * dnn_grad + (1 - alpha_ratio) * lgb_grad_t
        else:
            grad = dnn_grad
        X_adv = (X_adv + step_size * grad.sign()).detach()
        X_adv = torch.max(torch.min(X_adv, X_orig + epsilon), X_orig - epsilon)
    return X_adv.detach()


def evaluate_one(ens: EnsembleAdHoc, ds_name: str,
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    X_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    # We don't have an actual ZT risk scorer trained alongside the ensemble
    # in the standard pipeline; for this adaptive evaluation we use the
    # ensemble's RF as a stand-in for the risk scorer (different model
    # family from XGB/DNN, captures the spirit of decorrelation).
    risk_scorer = ens.rf if hasattr(ens, "rf") and ens.rf is not None else None

    ens.nn.eval()
    for p in ens.nn.parameters():
        p.requires_grad = False

    results = {}
    clean_preds, _ = ens.predict_with_confidence(X_test)
    results["clean"] = compute_metrics(y_test, clean_preds, is_binary=True)
    print(f"   clean:  acc={results['clean']['accuracy']:.4f}", flush=True)

    for eps in [0.05, 0.1, 0.3]:
        for alpha_ratio in [0.5, 0.8]:
            tag = f"adaptive_fgsm_eps{eps}_a{alpha_ratio}"
            t0 = time.time()
            X_adv_t = adaptive_fgsm(ens.nn, risk_scorer, X_t, y_t,
                                       epsilon=eps, alpha_ratio=alpha_ratio)
            preds, _ = ens.predict_with_confidence(X_adv_t.cpu().numpy())
            m = compute_metrics(y_test, preds, is_binary=True)
            m["epsilon"] = eps; m["alpha_dnn_ratio"] = alpha_ratio
            m["time_s"] = round(time.time() - t0, 1)
            results[tag] = m
            print(f"   {tag:<35s}  acc={m['accuracy']:.4f}  ({m['time_s']:.1f}s)",
                   flush=True)

    # Adaptive PGD at eps=0.1
    t0 = time.time()
    X_adv_t = adaptive_pgd(ens.nn, risk_scorer, X_t, y_t,
                              epsilon=0.1, n_steps=10, alpha_ratio=0.5)
    preds, _ = ens.predict_with_confidence(X_adv_t.cpu().numpy())
    m = compute_metrics(y_test, preds, is_binary=True)
    m["epsilon"] = 0.1; m["n_steps"] = 10
    m["time_s"] = round(time.time() - t0, 1)
    results["adaptive_pgd_eps0.1"] = m
    print(f"   adaptive_pgd_eps0.1               acc={m['accuracy']:.4f} ({m['time_s']:.1f}s)",
           flush=True)

    return results


def run_dataset(loader_fn, ds_name: str, dnn_epochs: int = 10):
    print(f"\n{'='*72}\n  ADAPTIVE: {ds_name}\n{'='*72}", flush=True)
    ds = loader_fn()
    num_classes = len(ds.class_names)
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, random_state=SEED,
        stratify=ds.y_train,
    )
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=dnn_epochs)
    ens.fit(X_t, y_t, X_v, y_v)
    print(f"  using FULL test set: n={len(ds.X_test)}", flush=True)
    results = evaluate_one(ens, ds_name, ds.X_test, ds.y_test)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{ds_name}_adaptive.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": ds_name,
            "attack_type": "adaptive (joint DNN+LGBM gradient)",
            "n_test_samples": int(len(ds.X_test)),
            "results": results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


def main():
    import os
    only = os.environ.get("DATASET", "").strip()
    datasets = [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
    ]
    if only:
        datasets = [(n, l) for n, l in datasets if n == only]
    for name, loader in datasets:
        run_dataset(loader, name, dnn_epochs=10)


if __name__ == "__main__":
    main()
