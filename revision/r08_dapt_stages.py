"""
Step 6 of the IJIES revision (Reviewer 2, comments 4 and 5) - DAPT 2020.

Facts that shape the design (see the per-file table this script prints):
  every APT stage occurs on exactly one capture day
  (Tue = Reconnaissance, Wed = Establish Foothold, Thu = Lateral Movement,
   Fri = Data Exfiltration; Monday is benign only), so stage and day are
  confounded and a temporal split cannot train a per-stage classifier.

Protocols
  P1 random     stratified 5-fold CV over flows (the original, optimistic protocol)
  P2 grouped    stratified group 5-fold CV, group = Flow ID (5-tuple), so flows of
                the same connection never sit on both sides of a split
  P3 day-out    binary detection of an UNSEEN stage: train on every other day,
                test on one attack day (its benign + its attack flows)
P1/P2 are 5-class (stage) tasks with out-of-fold predictions pooled over folds, so
every flow - including all 137 Lateral and 15 Exfiltration flows - is scored once.
Per-stage recall is reported with Wilson 95 % intervals.

Models (fixed hyper-parameters, no tuning, preprocessing fitted per training fold):
  lr   logistic regression, class-weighted
  xgb  XGBoost, sample-weighted
  ens  RF + XGBoost + DNN, class-weighted, equal-weight soft vote
Usage (PowerShell, from 05-Implementation):
  python -u revision\\r08_dapt_stages.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import rev_common as rc
from rev_common import OUT_DIR
import ensemble_selflearn as es

CSV_DIR = rc.SRC_DIR.parent / "data" / "dapt-2020" / "csv"
STAGES = ["Normal", "Reconnaissance", "Establish Foothold", "Lateral Movement", "Data Exfiltration"]
STAGE_TO_INT = {s.lower(): i for i, s in enumerate(STAGES)} | {"benign": 0}
DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Activity", "Stage"]
DAY_OF = {"monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu", "friday": "Fri"}
SEED = 0


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def load():
    frames = []
    for p in sorted(CSV_DIR.glob("*.pcap_Flow.csv")):
        df = pd.read_csv(p, low_memory=False)
        if "Stage" not in df.columns:
            print(f"[dapt] {p.name}: no Stage column, skipped"); continue
        df["_day"] = next(v for k, v in DAY_OF.items() if k in p.name)
        df["_file"] = p.name
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["_y"] = big["Stage"].astype(str).str.strip().str.lower().map(STAGE_TO_INT)
    big = big.dropna(subset=["_y"]).reset_index(drop=True)
    big["_y"] = big["_y"].astype(int)
    feats = big.drop(columns=[c for c in DROP + ["_day", "_file", "_y"] if c in big.columns])
    feats = feats.apply(pd.to_numeric, errors="coerce")
    X = np.nan_to_num(feats.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e12, 1e12).astype(np.float32)
    return big, X, list(feats.columns)


def release_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_predict(model, Xtr, ytr, Xte, k, seed, rf_kw=None):
    """rf_kw: optional RandomForest overrides (r14 caps bootstrap size and jobs for memory)."""
    keep = Xtr.std(0) > 1e-7
    sc = StandardScaler().fit(Xtr[:, keep])
    A, B = sc.transform(Xtr[:, keep]).astype(np.float32), sc.transform(Xte[:, keep]).astype(np.float32)
    A, B = np.clip(A, -50, 50), np.clip(B, -50, 50)
    counts = np.bincount(ytr, minlength=k); cw = counts.sum() / (k * np.maximum(counts, 1))

    def full(p, classes):   # learners may not see every class in a fold
        out = np.zeros((len(p), k), np.float32); out[:, classes] = p; return out
    if model == "lr":
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(A, ytr)
        return full(m.predict_proba(B), m.classes_)
    xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                           device="cuda", random_state=seed)
    present = np.unique(ytr); remap = {c: i for i, c in enumerate(present)}
    xg.fit(A, np.vectorize(remap.get)(ytr), sample_weight=cw[ytr])
    xg.set_params(device="cpu")   # predict on CPU: avoids a second GPU copy of the data
    p_xgb = full(xg.predict_proba(B), present)
    del xg; release_gpu()   # GPU memory otherwise accumulates across folds (CUDA OOM)
    if model == "xgb":
        return p_xgb
    kw = {**dict(n_estimators=200, min_samples_split=5, n_jobs=-1,
                 class_weight="balanced", random_state=seed), **(rf_kw or {})}
    if isinstance(kw.get("max_samples"), int) and kw["max_samples"] >= len(A):
        kw["max_samples"] = None          # cap only applies to larger training sets
    rf = RandomForestClassifier(**kw).fit(A, ytr)
    p_rf = full(rf.predict_proba(B), rf.classes_)
    torch.manual_seed(seed)
    nn = es.DeepDetector(A.shape[1], num_classes=k).to(es.DEVICE)
    nn.fit(A, ytr, epochs=20, class_weights=cw, verbose=False)
    p_nn = nn.predict_proba(B)
    del nn, rf; release_gpu()
    return (p_rf + p_xgb + p_nn) / 3


def main():
    big, X, names = load()
    y = big["_y"].values; groups = big["Flow ID"].astype(str).values
    tab = pd.crosstab([big["_day"], big["_file"]], big["_y"].map(dict(enumerate(STAGES))))
    print(tab.to_string())
    dup = pd.DataFrame(X).duplicated().mean()
    print(f"\n[dapt] {len(y)} flows, {X.shape[1]} features, exact-duplicate feature rows: {dup:.1%}, "
          f"distinct Flow IDs: {len(set(groups))}")

    rows, stage_rows = [], []
    for proto in ("P1 random", "P2 grouped by Flow ID"):
        if proto.startswith("P1"):
            splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)
        else:
            splits = StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups)
        splits = list(splits)
        for model in ("lr", "xgb", "ens"):
            t0 = time.time(); oof = np.zeros((len(y), 5), np.float32)
            for f, (tr, te) in enumerate(splits):
                oof[te] = fit_predict(model, X[tr], y[tr], X[te], 5, SEED + f)
            pred = oof.argmax(1)
            b = rc.binary_metrics(y, pred)
            rows.append({"protocol": proto, "model": model, "binary_f1": b["f1"], "binary_fpr": b["fpr"],
                         "binary_recall": b["recall"], "stage_accuracy": float((pred == y).mean()),
                         "macro_f1_stage": float(pd.Series(
                             [2 * ((pred == c) & (y == c)).sum() / max((pred == c).sum() + (y == c).sum(), 1)
                              for c in range(5)]).mean()),
                         "seconds": round(time.time() - t0, 1)})
            for c in range(1, 5):
                m = y == c
                stage_ok = wilson(int((pred[m] == c).sum()), int(m.sum()))   # correct stage
                any_alert = wilson(int((pred[m] != 0).sum()), int(m.sum()))   # flagged as any attack
                stage_rows.append({"protocol": proto, "model": model, "stage": STAGES[c], "n": int(m.sum()),
                                   "stage_recall": stage_ok[0], "stage_ci95": f"[{stage_ok[1]:.3f}, {stage_ok[2]:.3f}]",
                                   "detected_as_attack": any_alert[0],
                                   "detected_ci95": f"[{any_alert[1]:.3f}, {any_alert[2]:.3f}]"})
            print(f"[dapt] {proto} | {model} done ({time.time() - t0:.0f}s)")

    # P3: leave one attack day out, binary
    day_rows = []
    yb = (y != 0).astype(int); days = big["_day"].values
    for day, stage in (("Tue", 1), ("Wed", 2), ("Thu", 3), ("Fri", 4)):
        tr, te = days != day, days == day
        for model in ("lr", "xgb", "ens"):
            p = fit_predict(model, X[tr], yb[tr], X[te], 2, SEED)
            pr = p.argmax(1); yt = y[te]
            att = yt != 0
            rec = wilson(int((pr[att] == 1).sum()), int(att.sum()))
            fpr = wilson(int((pr[~att] == 1).sum()), int((~att).sum()))
            day_rows.append({"held_out_day": day, "unseen_stages": ", ".join(
                                 sorted({STAGES[c] for c in np.unique(yt[att])})),
                             "model": model, "attack_flows": int(att.sum()),
                             "recall": rec[0], "recall_ci95": f"[{rec[1]:.3f}, {rec[2]:.3f}]",
                             "fpr": fpr[0], "fpr_ci95": f"[{fpr[1]:.4f}, {fpr[2]:.4f}]"})
        print(f"[dapt] P3 day-out {day} done")

    tdir = OUT_DIR / "tables"; tdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(tdir / "dapt_overall.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(tdir / "dapt_per_stage.csv", index=False)
    pd.DataFrame(day_rows).to_csv(tdir / "dapt_day_out.csv", index=False)
    tab.to_csv(tdir / "dapt_stage_by_file.csv")
    pd.set_option("display.width", 250)
    print("\nOverall (out-of-fold)\n", pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nPer stage\n", pd.DataFrame(stage_rows).round(3).to_string(index=False))
    print("\nUnseen-stage detection (leave one day out)\n", pd.DataFrame(day_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
