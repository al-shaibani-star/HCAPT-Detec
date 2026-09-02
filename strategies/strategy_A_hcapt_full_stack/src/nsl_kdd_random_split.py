"""
NSL-KDD with FIDMF-style random 90:10 split — apples-to-apples baseline.

This script reproduces the protocol used by FIDMF [N] and several
other 2024-2026 papers: combine KDDTrain+ and KDDTest+ into one
158,517-row pool, random 90:10 split, train an ensemble, report
binary metrics. This is the *only* fair head-to-head with their
99.40% accuracy claim, and our number from this protocol becomes
Appendix A of the paper.

Writes results/_random_split_baseline/nsl_kdd_random.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Project src/ for component reuse
PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import (
    NSL_KDD_COLUMNS, CATEGORICAL_COLS, ATTACK_CATEGORIES,
)
from .phase2_pipeline import EnsembleAdHoc, compute_metrics

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_random_split_baseline"


def load_combined_pool():
    train_p = DATA_ROOT / "nsl-kdd" / "train.txt"
    test_p  = DATA_ROOT / "nsl-kdd" / "test.txt"
    df_tr = pd.read_csv(train_p, header=None, names=NSL_KDD_COLUMNS).drop(columns=["difficulty"])
    df_te = pd.read_csv(test_p,  header=None, names=NSL_KDD_COLUMNS).drop(columns=["difficulty"])
    combined = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
    combined["category"] = combined["label"].map(lambda s: ATTACK_CATEGORIES.get(s, "Unknown"))
    return combined


def main():
    print("\n" + "#"*72)
    print("#  NSL-KDD random 90:10 baseline  (FIDMF-comparable protocol)")
    print("#"*72)

    df = load_combined_pool()
    print(f"[random-split] combined pool: {df.shape}")

    # Binary labels (Normal=0, Attack=1)
    y = (df["category"] != "Normal").astype(int).values

    # Drop label cols, encode categoricals on the COMBINED pool (FIDMF practice)
    X_df = df.drop(columns=["label", "category"])
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        encoders[col] = list(le.classes_)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    # Scale on the combined pool BEFORE splitting (FIDMF practice, leaks distribution)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Random 90:10 split (the FIDMF protocol per their Table 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y,
    )
    print(f"[random-split] train: {X_train.shape}, test: {X_test.shape}")
    print(f"  attack ratio train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # Train ensemble (RF + XGBoost + DNN) — matches our Phase 2 stack
    feature_dim = X_train.shape[1]
    num_classes = 2
    t0 = time.time()
    ens = EnsembleAdHoc(feature_dim, num_classes, dnn_epochs=20)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train,
    )
    ens.fit(X_tr, y_tr, X_va, y_va)
    train_time = time.time() - t0
    print(f"[random-split] train time: {train_time:.1f}s")

    preds, _ = ens.predict_with_confidence(X_test)
    m = compute_metrics(y_test, preds, is_binary=True)
    m["protocol"]         = "FIDMF-style random 90:10 of combined Train+/Test+"
    m["n_train"]          = int(len(y_train))
    m["n_test"]           = int(len(y_test))
    m["ensemble_weights"] = ens.weights.tolist()
    m["train_time_s"]     = round(train_time, 1)
    m["note"]             = ("Reported as Appendix A baseline. Headline number "
                              "uses official KDDTrain+/KDDTest+ split, see "
                              "Section 6.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "nsl_kdd_random.json"
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\n[random-split] wrote {out_path}")
    print("\n=== Random-split baseline ===")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  FPR      : {m['fpr']:.4f}")
    print(f"\nFIDMF reports 99.40% on the same protocol. HCAPT achieves "
          f"{m['accuracy']:.4f} → {('parity' if abs(m['accuracy']-0.994)<0.01 else 'gap')}.")
    return m


if __name__ == "__main__":
    main()
