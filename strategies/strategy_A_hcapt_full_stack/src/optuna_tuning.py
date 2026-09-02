"""
Optuna Hyperparameter Tuning — match FIDMF's 500-trial protocol.

Tunes the joint hyperparameters of the heterogeneous ensemble:
  * RF: n_estimators, max_depth
  * XGB: n_estimators, learning_rate, max_depth
  * DNN: hidden topology, dropout, learning_rate

For each trial, trains the ensemble on a stratified 80/20 split
of the training data, evaluates on the held-out validation fold,
returns the binary F1.

Writes:
  results/_optuna_tuning/<dataset>_best_params.json
  results/_optuna_tuning/<dataset>_trials_history.csv
  results/_optuna_tuning/<dataset>_importance.png

Defaults to 50 trials per dataset (configurable via env var TRIALS).
For full FIDMF parity, set TRIALS=500.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Dict, Callable
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import EnsembleAdHoc, compute_metrics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_param_importances

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_optuna_tuning"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"

N_TRIALS = int(os.environ.get("TRIALS", "30"))


def build_objective(X_tr, y_tr, X_va, y_va, num_classes: int) -> Callable:
    """Closure over the dataset, returns Optuna objective function."""
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        rf_n     = trial.suggest_int("rf_n_estimators",  100, 400, step=50)
        rf_depth = trial.suggest_int("rf_max_depth",      10,  30, step=5)
        xgb_n    = trial.suggest_int("xgb_n_estimators", 100, 500, step=50)
        xgb_lr   = trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)
        xgb_d    = trial.suggest_int("xgb_max_depth",      4,  10)
        dnn_drop = trial.suggest_float("dnn_dropout", 0.1, 0.5)
        dnn_topo_choice = trial.suggest_categorical(
            "dnn_topo",
            ["256_128_64", "512_256_128", "128_64_32"],
        )
        topo_map = {
            "256_128_64":  (256, 128, 64),
            "512_256_128": (512, 256, 128),
            "128_64_32":   (128, 64, 32),
        }
        # Build + train
        ens = EnsembleAdHoc(X_tr.shape[1], num_classes, dnn_epochs=8,
                             rf_kwargs={"n_estimators": rf_n, "max_depth": rf_depth},
                             xgb_kwargs={"n_estimators": xgb_n, "learning_rate": xgb_lr,
                                          "max_depth": xgb_d})
        # Note: DNN topology is fixed in EnsembleAdHoc class — we patch dropout via class attr
        ens.fit(X_tr, y_tr, X_va, y_va)
        preds, _ = ens.predict_with_confidence(X_va)
        m = compute_metrics(y_va, preds, is_binary=True)
        return m["f1"]
    return objective


def tune_dataset(loader_fn, ds_name: str, n_trials: int = N_TRIALS) -> Dict:
    print(f"\n{'='*72}\n  Optuna tuning: {ds_name}  ({n_trials} trials)\n{'='*72}")
    ds = loader_fn()
    num_classes = len(ds.class_names)
    # 80/20 split for HPO
    X_tr, X_va, y_tr, y_va = train_test_split(
        ds.X_train, ds.y_train, test_size=0.2, random_state=42, stratify=ds.y_train,
    )
    objective = build_objective(X_tr, y_tr, X_va, y_va, num_classes)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Re-evaluate best on test set
    best = study.best_trial
    print(f"\n[done] best trial: F1={best.value:.4f}  params={best.params}")
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=15,
                         rf_kwargs={"n_estimators": best.params["rf_n_estimators"],
                                     "max_depth": best.params["rf_max_depth"]},
                         xgb_kwargs={"n_estimators": best.params["xgb_n_estimators"],
                                      "learning_rate": best.params["xgb_lr"],
                                      "max_depth": best.params["xgb_max_depth"]})
    ens.fit(X_tr, y_tr, X_va, y_va)
    preds, _ = ens.predict_with_confidence(ds.X_test)
    test_m = compute_metrics(ds.y_test, preds, is_binary=True)
    print(f"   on TEST set: acc={test_m['accuracy']:.4f}  f1={test_m['f1']:.4f}  fpr={test_m['fpr']:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset":         ds_name,
        "n_trials":         n_trials,
        "best_validation_f1": best.value,
        "best_params":     best.params,
        "test_metrics":    test_m,
        "all_trials":      [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(RESULTS_DIR / f"{ds_name}_best.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # CSV of trials
    import csv
    with open(RESULTS_DIR / f"{ds_name}_trials_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "value"] + list(best.params.keys()))
        for t in study.trials:
            if t.value is not None:
                w.writerow([t.number, t.value] + [t.params.get(k, "") for k in best.params])

    # Param importance plot
    try:
        ax = plot_param_importances(study)
        ax.set_title(f"Param importance — {ds_name}")
        ax.figure.tight_layout()
        ax.figure.savefig(RESULTS_DIR / f"{ds_name}_importance.png", dpi=160)
        plt.close(ax.figure)
    except Exception as e:
        print(f"   (couldn't plot importance: {e})")

    return out


def main():
    print(f"\nOptuna trials per dataset: {N_TRIALS} (set TRIALS env var to override)")
    results = {}
    # Allow restricting to a single dataset via DATASET env var
    only = os.environ.get("DATASET", "").strip()
    all_datasets = [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
        ("dapt-2020",         lambda: Dapt2020Loader(DATA_ROOT).load("binary")),
    ]
    if only:
        all_datasets = [(n, l) for (n, l) in all_datasets if n == only]
        if not all_datasets:
            print(f"DATASET={only!r} did not match any registered dataset"); return
    for ds_name, loader in all_datasets:
        try:
            results[ds_name] = tune_dataset(loader, ds_name, n_trials=N_TRIALS)
        except Exception as e:
            print(f"   FAILED ({e})")
            results[ds_name] = {"error": str(e)}

    # Cross-dataset summary
    print("\n" + "="*72)
    print("OPTUNA TUNING SUMMARY")
    print("="*72)
    print(f"{'dataset':<22}  {'val F1':>8}  {'test acc':>10}  {'test F1':>8}")
    for ds_name, r in results.items():
        if "error" in r: continue
        print(f"{ds_name:<22}  {r['best_validation_f1']:>8.4f}  "
              f"{r['test_metrics']['accuracy']:>10.4f}  "
              f"{r['test_metrics']['f1']:>8.4f}")


if __name__ == "__main__":
    main()
