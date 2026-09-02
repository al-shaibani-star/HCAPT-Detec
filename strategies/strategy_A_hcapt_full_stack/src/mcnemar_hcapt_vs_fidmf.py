"""McNemar test comparing HCAPT vs FIDMF re-implementation per-sample
correctness on the held-out test set of all four datasets.

Re-trains both models (since we don't ship saved predictions per row)
and runs scipy's McNemar exact test on the 2x2 contingency table.

Output: results/_mcnemar_hcapt_fidmf.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy import stats

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import EnsembleAdHoc
from .fidmf_reimpl import train_attlstm, predict_attlstm

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"
RESULTS_DIR   = STRATEGY_ROOT / "results"
SEED = 42


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar exact test (binomial) for small discordants."""
    n = b + c
    if n == 0:
        return 1.0
    # two-sided p-value
    k = min(b, c)
    return 2.0 * stats.binom.cdf(k, n, 0.5)


def run_one(ds_name, loader, dnn_epochs=8):
    print(f"\n{'='*72}\n  {ds_name}  --  HCAPT vs FIDMF\n{'='*72}", flush=True)
    ds = loader()
    num_classes = len(ds.class_names)

    # HCAPT ensemble (RF+XGB+DNN)
    print("  training HCAPT ensemble...", flush=True)
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=dnn_epochs)
    ens.fit(ds.X_train, ds.y_train, None, None)
    hcapt_pred, _ = ens.predict_with_confidence(ds.X_test)

    # FIDMF re-impl (Attention-LSTM only — strongest of the 3 FIDMF variants
    # on most datasets per our prior re-impl results)
    print("  training FIDMF Att-LSTM...", flush=True)
    X_tr, X_va, y_tr, y_va = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, stratify=ds.y_train, random_state=SEED,
    )
    model = train_attlstm(X_tr.astype(np.float32), y_tr.astype(int),
                            X_va.astype(np.float32), y_va.astype(int),
                            num_classes, epochs=12)
    fidmf_pred = predict_attlstm(model, ds.X_test)

    # Contingency: rows=HCAPT, cols=FIDMF (correctness)
    h_correct = (hcapt_pred == ds.y_test).astype(int)
    f_correct = (fidmf_pred == ds.y_test).astype(int)
    a = int(((h_correct == 1) & (f_correct == 1)).sum())
    b = int(((h_correct == 1) & (f_correct == 0)).sum())  # HCAPT right, FIDMF wrong
    c = int(((h_correct == 0) & (f_correct == 1)).sum())  # HCAPT wrong, FIDMF right
    d = int(((h_correct == 0) & (f_correct == 0)).sum())

    # McNemar exact + asymptotic
    p_exact = mcnemar_exact(b, c)
    # asymptotic chi-square with continuity correction
    if b + c == 0:
        chi2 = 0.0
        p_asym = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
        p_asym = float(1 - stats.chi2.cdf(chi2, df=1))

    h_acc = h_correct.mean()
    f_acc = f_correct.mean()

    out = {
        "dataset": ds_name,
        "n_test": int(len(ds.y_test)),
        "hcapt_accuracy": float(h_acc),
        "fidmf_accuracy": float(f_acc),
        "contingency": {"both_right": a, "hcapt_only": b, "fidmf_only": c, "both_wrong": d},
        "mcnemar_exact_p": float(p_exact),
        "mcnemar_asymp_chi2": float(chi2),
        "mcnemar_asymp_p":    float(p_asym),
        "winner": "HCAPT" if h_acc > f_acc else ("FIDMF" if f_acc > h_acc else "tie"),
        "delta_acc_pp": float((h_acc - f_acc) * 100),
    }
    print(f"  HCAPT acc={h_acc:.4f} | FIDMF acc={f_acc:.4f} | delta={out['delta_acc_pp']:+.2f}pp",
           flush=True)
    print(f"  contingency: both_right={a} HCAPT_only={b} FIDMF_only={c} both_wrong={d}",
           flush=True)
    print(f"  McNemar exact p = {p_exact:.4e}   asymp chi2 = {chi2:.2f}  p = {p_asym:.4e}",
           flush=True)
    return out


def main():
    results = []
    datasets = [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
        ("dapt-2020",         lambda: Dapt2020Loader(DATA_ROOT).load("binary")),
    ]
    for name, loader in datasets:
        try:
            results.append(run_one(name, loader))
        except Exception as e:
            print(f"  !! {name} failed: {e}", flush=True)
            results.append({"dataset": name, "error": str(e)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "_mcnemar_hcapt_fidmf.json"
    with open(out_path, "w") as f:
        json.dump({
            "test": "McNemar's test, HCAPT-ensemble vs FIDMF-Att-LSTM",
            "alpha_bonferroni": 0.05 / 4,
            "results": results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
