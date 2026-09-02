"""DAPT 2020 multi-class (5-stage) evaluation.

Stages: Benign, Reconnaissance, Establish Foothold, Lateral
Movement, Data Exfiltration.

This closes the Paper-1 gap "DAPT 2020 binary only — no per-stage
evaluation" by running the HCAPT ensemble in 5-way classification
mode and reporting per-stage precision/recall/F1.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import EnsembleAdHoc

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"
RESULTS_DIR   = STRATEGY_ROOT / "results" / "dapt-2020"
SEED = 42


def main():
    print("\n" + "="*72)
    print("  DAPT 2020 -- 5-stage multi-class evaluation")
    print("="*72, flush=True)

    ds = Dapt2020Loader(DATA_ROOT).load("multiclass")
    num_classes = len(ds.class_names)
    print(f"  classes ({num_classes}): {ds.class_names}", flush=True)
    print(f"  train n={len(ds.X_train)}  test n={len(ds.X_test)}", flush=True)

    # Print per-class distribution
    import collections
    tr_dist = collections.Counter(ds.y_train.tolist())
    te_dist = collections.Counter(ds.y_test.tolist())
    print(f"  per-class TRAIN counts: " +
           ", ".join(f"{ds.class_names[i]}={tr_dist.get(i,0)}" for i in range(num_classes)),
           flush=True)
    print(f"  per-class TEST  counts: " +
           ", ".join(f"{ds.class_names[i]}={te_dist.get(i,0)}" for i in range(num_classes)),
           flush=True)

    # Train ensemble in multi-class mode
    t0 = time.time()
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=12)
    ens.fit(ds.X_train, ds.y_train, None, None)
    train_time = time.time() - t0
    print(f"  trained ensemble in {train_time:.1f}s", flush=True)

    # Evaluate
    preds, conf = ens.predict_with_confidence(ds.X_test)

    # Per-class metrics
    report = classification_report(
        ds.y_test, preds, target_names=ds.class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(ds.y_test, preds, labels=list(range(num_classes)))

    print("\n--- Per-class report ---", flush=True)
    for name in ds.class_names:
        m = report[name]
        print(f"  {name:<22s}  prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
               f"f1={m['f1-score']:.4f}  support={int(m['support'])}",
               flush=True)
    print(f"\n  macro-F1   = {report['macro avg']['f1-score']:.4f}", flush=True)
    print(f"  weighted-F1 = {report['weighted avg']['f1-score']:.4f}", flush=True)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset":            "dapt-2020",
        "task":               "multiclass-5way",
        "class_names":        list(ds.class_names),
        "n_train":            int(len(ds.X_train)),
        "n_test":             int(len(ds.X_test)),
        "train_time_s":       round(train_time, 1),
        "per_class_report":   report,
        "confusion_matrix":   cm.tolist(),
        "macro_f1":           float(report["macro avg"]["f1-score"]),
        "weighted_f1":        float(report["weighted avg"]["f1-score"]),
    }
    out_path = RESULTS_DIR / "multiclass_5stage_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
