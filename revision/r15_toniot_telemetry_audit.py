"""
ToN-IoT IoT-telemetry audit (Reviewer 2, comment 4: device/time boundaries).

The submitted paper pooled the seven ToN-IoT telemetry files and reported F1 0.847
under a random split.  This script documents why that pool cannot support a
deployment-oriented evaluation:
  1. time: per device, the capture days of normal and attack rows
  2. identifiability: share of rows whose exact feature vector occurs with both labels
     (such rows bound the achievable accuracy) and number of distinct vectors
  3. encoding artefacts: per device, accuracy of a depth-3 decision tree that sees ONLY
     the string format of each value (e.g. "0"/"1" vs "false"/"true"), not its meaning
  4. duplicate rate
Output: revision/out/tables/toniot_telemetry_audit.csv
Usage:  python revision\\r15_toniot_telemetry_audit.py
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import rev_common as rc
from rev_common import OUT_DIR

DIR = rc.SRC_DIR.parent / "data" / "ton-iot" / "iot-telemetry"


def fmt(v):
    """String *format* of a value, with its content removed."""
    s = str(v).strip()
    if re.fullmatch(r"-?\d+", s): return "int"
    if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?", s): return "float"
    if s.lower() in ("true", "false"): return "bool-word"
    return "word" if s.isalpha() else "other"


def main():
    rows = []
    for p in sorted(DIR.glob("*.csv")):
        df = pd.read_csv(p); df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        feats = [c for c in df.columns if c not in ("ts", "date", "time", "label", "type")]
        for c in feats:
            df[c] = df[c].astype(str).str.strip()
        t = pd.to_datetime(df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
                           format="%d-%b-%y %H:%M:%S", errors="coerce")
        y = df["label"].astype(int).values
        g = df.groupby(feats)["label"].nunique().rename("nl").reset_index()
        conflict = df.merge(g, on=feats)["nl"].gt(1).mean()
        F = pd.get_dummies(df[feats].apply(lambda col: col.map(fmt))).values
        fmt_acc = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), F, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        rows.append({
            "device": p.stem, "rows": len(df), "features": ", ".join(feats),
            "normal_days": f"{t[y == 0].min():%d %b}–{t[y == 0].max():%d %b}",
            "attack_days": f"{t[y == 1].min():%d %b}–{t[y == 1].max():%d %b}",
            "time_overlap": bool(t[y == 0].max() >= t[y == 1].min()),
            "distinct_vectors": int(len(g)), "label_conflict_rows": float(conflict),
            "duplicate_rows": float(df[feats + ["label"]].duplicated().mean()),
            "format_only_accuracy": float(fmt_acc), "majority_rate": float(max(y.mean(), 1 - y.mean()))})
    out = pd.DataFrame(rows)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "tables" / "toniot_telemetry_audit.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
