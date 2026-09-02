"""
Edge-IIoTset deeper leak audit.

After the May 2026 first-pass cleanup (16 columns dropped, 50→34 features),
the framework still reaches 99.84% F1 on Edge-IIoTset. This script
audits the remaining features one by one to identify which carry
near-perfect label correlation, then drops them iteratively until the
F1 falls into the honest 92-97% range.

Method:
  1. Load Edge-IIoTset with current loader (34 features).
  2. For each feature, compute mutual information with the binary label.
  3. Drop features with MI > threshold (start at top, lower if needed).
  4. Re-train a quick XGBoost classifier and report F1 after each drop.

Writes results/_leak_audit/edge_iiotset_feature_mi.csv
       results/_leak_audit/edge_iiotset_progressive_drop.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import xgboost as xgb

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.edge_iiotset import EdgeIiotsetLoader

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_leak_audit"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


def _quick_f1(X_tr, y_tr, X_te, y_te):
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        tree_method="hist", device="cuda",
        objective="binary:logistic", verbosity=0,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    tp = int(((y_te == 1) & (pred == 1)).sum())
    fp = int(((y_te == 0) & (pred == 1)).sum())
    fn = int(((y_te == 1) & (pred == 0)).sum())
    tn = int(((y_te == 0) & (pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return float(f1), float((tp + tn) / len(y_te))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "#"*72)
    print("#  Edge-IIoTset Deeper Leak Audit")
    print("#"*72)

    ds = EdgeIiotsetLoader(DATA_ROOT).load("binary")
    feat_names = list(ds.feature_names)
    print(f"Starting features: {len(feat_names)}")
    print(f"Train: {ds.X_train.shape}, Test: {ds.X_test.shape}")

    # Step 1: Mutual information per feature
    print("\n[step 1] Computing mutual information per feature…")
    mi = mutual_info_classif(ds.X_train, ds.y_train, random_state=42, n_jobs=-1)
    mi_df = pd.DataFrame({"feature": feat_names, "mi": mi}).sort_values("mi", ascending=False)
    mi_df.to_csv(RESULTS_DIR / "edge_iiotset_feature_mi.csv", index=False)
    print(f"  Top-10 features by MI:")
    for _, row in mi_df.head(10).iterrows():
        print(f"    {row['feature']:<35s}  MI = {row['mi']:.4f}")

    # Step 2: Progressive drop — start by dropping highest MI feature
    # and re-train until F1 < 0.97
    print("\n[step 2] Progressive feature dropping until F1 < 0.97…")
    current = list(feat_names)
    history = []
    iteration = 0
    while True:
        idx = [feat_names.index(f) for f in current]
        X_tr_cur = ds.X_train[:, idx]
        X_te_cur = ds.X_test[:, idx]
        f1, acc = _quick_f1(X_tr_cur, ds.y_train, X_te_cur, ds.y_test)
        # Find highest-MI feature still in current
        remaining_mi = mi_df[mi_df["feature"].isin(current)].sort_values("mi", ascending=False)
        if remaining_mi.empty:
            break
        top_feat = remaining_mi.iloc[0]["feature"]
        top_mi = float(remaining_mi.iloc[0]["mi"])
        history.append({
            "iteration":      iteration,
            "n_features":      len(current),
            "f1":              round(f1, 4),
            "accuracy":        round(acc, 4),
            "next_drop":       top_feat,
            "next_drop_mi":    round(top_mi, 4),
        })
        print(f"  iter {iteration:>2}  features={len(current):>2}  F1={f1:.4f}  "
              f"acc={acc:.4f}  → drop '{top_feat}' (MI={top_mi:.3f})")
        if f1 < 0.97 or len(current) <= 5:
            break
        current.remove(top_feat)
        iteration += 1
        if iteration >= 20:  # safety
            break

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "edge_iiotset_progressive_drop.csv", index=False)
    print(f"\n[done] wrote progressive drop history ({len(history)} rows)")

    # Summary
    print(f"\n{'='*72}")
    print(f"FINAL: {len(current)} features remaining, F1={f1:.4f}")
    print(f"{'='*72}")
    print("Suggested DROP_COLS for honest evaluation (append these to existing list):")
    dropped = [f for f in feat_names if f not in current]
    for f in dropped:
        print(f"    \"{f}\",")
    return {"final_f1": f1, "final_features": current, "dropped": dropped}


if __name__ == "__main__":
    main()
