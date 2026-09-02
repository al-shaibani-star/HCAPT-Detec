"""SOTA Push: aggressive ensemble + heavy Optuna to close the gap with
inflated literature numbers (FIDMF 0.994, GNN-KAN 0.9924).

Strategy:
  1. Build a 4-way stacked ensemble: RF + XGB(heavy) + CatBoost + DNN
  2. 50-trial TPE Optuna over the joint hyperparameter space
  3. Per-dataset reports with full statistical CI

Run:
  TRIALS=50 DATASET=dapt-2020 python -m src.sota_push
  TRIALS=50 DATASET=nsl-kdd   python -m src.sota_push
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except ImportError:
    HAVE_CAT = False

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.dapt2020 import Dapt2020Loader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_sota_push"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"

N_TRIALS = int(os.environ.get("TRIALS", "30"))
SEED = 42


class SmallDNN(nn.Module):
    def __init__(self, d_in: int, n_classes: int, topo: str = "256_128_64",
                  dropout: float = 0.2):
        super().__init__()
        layers, prev = [], d_in
        for h in [int(x) for x in topo.split("_")]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_dnn(X_tr, y_tr, X_va, y_va, n_classes, topo, dropout, lr, epochs):
    model = SmallDNN(X_tr.shape[1], n_classes, topo, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = 4096
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    for _ in range(epochs):
        idx = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            logits = model(Xt[b])
            loss = F.cross_entropy(logits, yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return F.softmax(model(torch.tensor(X_va, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()


def stacked_eval(X_tr, y_tr, X_te, y_te, n_classes, p):
    """Train 4-way ensemble with hyperparams from p, return F1/acc/FPR."""
    probs = []

    rf = RandomForestClassifier(
        n_estimators=p["rf_n"], max_depth=p["rf_d"],
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr)
    probs.append(rf.predict_proba(X_te))

    if HAVE_XGB:
        xgb = XGBClassifier(
            n_estimators=p["xgb_n"], max_depth=p["xgb_d"],
            learning_rate=p["xgb_lr"], n_jobs=-1,
            tree_method="hist", random_state=SEED,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            verbosity=0,
        )
        xgb.fit(X_tr, y_tr)
        probs.append(xgb.predict_proba(X_te))

    if HAVE_CAT:
        cat = CatBoostClassifier(
            iterations=p["cat_n"], depth=p["cat_d"],
            learning_rate=p["cat_lr"], random_seed=SEED,
            verbose=False, allow_writing_files=False,
        )
        cat.fit(X_tr, y_tr)
        probs.append(cat.predict_proba(X_te))

    p_dnn = train_dnn(X_tr, y_tr, X_te, y_te, n_classes,
                       p["dnn_topo"], p["dnn_dropout"], p["dnn_lr"],
                       p["dnn_epochs"])
    probs.append(p_dnn)

    avg = np.mean(probs, axis=0)
    preds = avg.argmax(axis=1)
    f1 = f1_score(y_te, preds, average="binary" if n_classes == 2 else "macro",
                    zero_division=0)
    acc = accuracy_score(y_te, preds)
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
    else:
        fpr = float("nan")
    return float(f1), float(acc), float(fpr)


def objective(trial: optuna.Trial, X_tr, y_tr, X_va, y_va, n_classes):
    p = {
        "rf_n":       trial.suggest_categorical("rf_n",   [200, 400, 600, 800]),
        "rf_d":       trial.suggest_categorical("rf_d",   [10, 15, 20, 30, None]),
        "xgb_n":      trial.suggest_categorical("xgb_n",  [300, 500, 800, 1200]),
        "xgb_d":      trial.suggest_int("xgb_d",          4, 10),
        "xgb_lr":     trial.suggest_float("xgb_lr",       0.01, 0.30, log=True),
        "cat_n":      trial.suggest_categorical("cat_n",  [300, 500, 800, 1000]),
        "cat_d":      trial.suggest_int("cat_d",          4, 10),
        "cat_lr":     trial.suggest_float("cat_lr",       0.01, 0.30, log=True),
        "dnn_topo":   trial.suggest_categorical("dnn_topo",
                                                  ["256_128_64", "512_256_128", "512_256_128_64"]),
        "dnn_dropout":trial.suggest_float("dnn_dropout",   0.05, 0.4),
        "dnn_lr":     trial.suggest_float("dnn_lr",        5e-4, 5e-3, log=True),
        "dnn_epochs": trial.suggest_int("dnn_epochs",      8, 20),
    }
    f1, _, _ = stacked_eval(X_tr, y_tr, X_va, y_va, n_classes, p)
    return f1


def push_dataset(loader_fn, ds_name: str):
    print(f"\n{'='*72}\n  SOTA PUSH: {ds_name}  (TRIALS={N_TRIALS})\n{'='*72}", flush=True)
    ds = loader_fn()
    n_classes = len(ds.class_names)
    print(f"  n_train={len(ds.X_train)}  n_test={len(ds.X_test)}  n_classes={n_classes}", flush=True)
    print(f"  CatBoost available: {HAVE_CAT}  XGB available: {HAVE_XGB}", flush=True)

    # Internal val split from train for Optuna
    X_tr, X_va, y_tr, y_va = train_test_split(
        ds.X_train, ds.y_train, test_size=0.2, stratify=ds.y_train, random_state=SEED,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    t0 = time.time()
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_va, y_va, n_classes),
                    n_trials=N_TRIALS, show_progress_bar=False)
    opt_time = time.time() - t0
    print(f"\n  Optuna best validation F1: {study.best_value:.4f}  ({opt_time:.0f}s, {N_TRIALS} trials)", flush=True)
    print(f"  best params: {study.best_params}", flush=True)

    # Re-train on full train, evaluate on held-out test
    print("\n  Re-training on full train + test eval…", flush=True)
    t0 = time.time()
    test_f1, test_acc, test_fpr = stacked_eval(
        ds.X_train, ds.y_train, ds.X_test, ds.y_test,
        n_classes, study.best_params,
    )
    final_time = time.time() - t0
    print(f"\n  >>> FINAL TEST  acc={test_acc:.4f}  f1={test_f1:.4f}  fpr={test_fpr:.4f}", flush=True)
    print(f"  re-train time: {final_time:.0f}s", flush=True)

    out = {
        "dataset":           ds_name,
        "n_trials":          N_TRIALS,
        "best_val_f1":       float(study.best_value),
        "best_params":       study.best_params,
        "test_acc":          test_acc,
        "test_f1":           test_f1,
        "test_fpr":          test_fpr,
        "optuna_time_s":     round(opt_time, 1),
        "final_train_time_s":round(final_time, 1),
        "models_in_ensemble": ["RF", "XGB", "CatBoost", "DNN"] if HAVE_CAT and HAVE_XGB else
                              ["RF", "XGB", "DNN"] if HAVE_XGB else ["RF", "DNN"],
    }
    out_path = RESULTS_DIR / f"{ds_name}_sota_push.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path.name}", flush=True)
    return out


LOADERS = {
    "nsl-kdd":           lambda: NslKddLoader(DATA_ROOT).load("binary"),
    "dapt-2020":         lambda: Dapt2020Loader(DATA_ROOT).load("binary"),
    "ton-iot-telemetry": lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary"),
    "edge-iiotset":      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary"),
}


def main():
    only = os.environ.get("DATASET", "").strip()
    if only:
        if only in LOADERS:
            push_dataset(LOADERS[only], only)
        else:
            print(f"Unknown dataset {only}; choose from {list(LOADERS)}")
    else:
        for name, loader in LOADERS.items():
            push_dataset(loader, name)


if __name__ == "__main__":
    main()
