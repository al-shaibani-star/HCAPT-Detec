"""
ToN-IoT timestamp counterfactual — task 3 of FINAL_PLAN.md.

The ToN-IoT telemetry loader discards two columns, `date` and `time`, before
building the feature matrix.  Every published false-positive floor for this
benchmark is computed on what is left.  This script asks what those two
columns were holding.

Four column sets, one pipeline, one split:

    base           the 18 features the detector actually used
    base + date
    base + time
    base + date + time

For each: cell structure, the majority-rule floor, the split-level frontier at
the detector's own recall, the train/test twin rate, and a per-device
attribution of the forced false positives.

Everything runs through the loader's own pipeline — same construction, same
seed-42 stratified split, `StandardScaler` fitted on train, cast to float32 —
so the base row reproduces the authoritative number in
`results/_attainability/ton-iot-telemetry.json` exactly, and the other three
differ from it only by the added columns.

BOTH HORNS MUST BE REPORTED.  If the floor collapses when the timestamps are
restored, that does not mean the timestamps should be restored: a per-record
timestamp is an identifier, and a detector that reads one is exploiting
collection order rather than behaviour.  What it means is that the floor is a
property of a preprocessing recipe, not of the phenomenon — which is the
paper's thesis, stated against the paper's own instrument.

    python -m src.attainability_ton_iot_timestamps
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

STRATEGY_ROOT = Path(__file__).resolve().parents[1]
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

from src.attainability import (                                   # noqa: E402
    DATA_DIR, OUT_DIR, SPEC_VERSION, _sha256, _at_recall, _frontier)
from src.loaders.ton_iot_telemetry import (                       # noqa: E402
    TonIotTelemetryLoader, DEVICE_FILES, FEATURE_COLS_BY_DEVICE, CAT_COLS_BY_DEVICE)

# The detector's own operating point, from results/ton-iot-telemetry/ablation_table.csv
OBS_CONFIG, OBS_FPR, OBS_RECALL = "full", 0.5290, 0.9667

HERE = Path(__file__).resolve()


def build_frame():
    """The loader's construction, plus `date` and `time` carried alongside."""
    ld = TonIotTelemetryLoader(DATA_DIR)
    frames, devices = [], []
    for fname, dname in DEVICE_FILES.items():
        p = ld.dir / fname
        if not p.exists():
            print(f"  missing device file: {fname} (skip)")
            continue
        frames.append(ld._load_one_device(p, dname))
        devices.append(dname)

    num = sorted({c for v in FEATURE_COLS_BY_DEVICE.values() for c in v})
    cat = sorted({c for v in CAT_COLS_BY_DEVICE.values() for c in v})

    rows = []
    for df in frames:
        for c in num:
            if c not in df.columns:
                df[c] = 0.0
        for c in cat:
            if c not in df.columns:
                df[c] = "missing"
        for c in ("date", "time"):
            if c not in df.columns:
                df[c] = "missing"
        rows.append(df[["__device", "label", "date", "time"] + num + cat])

    big = pd.concat(rows, ignore_index=True)
    big["label"] = pd.to_numeric(big["label"], errors="coerce").fillna(0).astype(int)
    device_raw = big["__device"].copy()

    # Encoding order matches the loader: categoricals and device first, then
    # numerics with the string fallback. `date`/`time` are encoded the same way
    # any other categorical would be.
    for col in list(cat) + ["__device", "date", "time"]:
        big[col] = LabelEncoder().fit_transform(big[col].astype(str))
    for col in num:
        co = pd.to_numeric(big[col], errors="coerce")
        big[col] = LabelEncoder().fit_transform(big[col].astype(str)) if co.isna().any() else co

    base_feats = list(num) + list(cat) + ["__device"]
    return big, base_feats, device_raw


def pipeline(big: pd.DataFrame, feats: list, y: np.ndarray):
    """The loader's pipeline: float32 -> seed-42 stratified split -> scale on train."""
    X = big[feats].astype(np.float32).values
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[i_tr]).astype(np.float32)
    Xte = sc.transform(X[i_te]).astype(np.float32)
    return np.vstack([Xtr, Xte]), np.concatenate([y[i_tr], y[i_te]]), i_tr, i_te


def cell_report(X: np.ndarray, y: np.ndarray) -> dict:
    _, inv = np.unique(X, axis=0, return_inverse=True)
    m = int(inv.max()) + 1
    na = np.bincount(inv, weights=y.astype(float), minlength=m)
    nt = np.bincount(inv, minlength=m).astype(float)
    nb = nt - na
    P, N = float(y.sum()), float(len(y) - y.sum())
    pred = na >= nb
    rec, fpr = _frontier(na, nb, P, N)
    return {"cells": m,
            "duplicate_rate": float((nt[inv] > 1).mean()),
            "conflict_rate": float((((na > 0) & (nb > 0))[inv]).mean()),
            "forced_fp": float(nb[pred].sum()),
            "floor_fpr": float(nb[pred].sum() / N),
            "floor_recall": float(na[pred].sum() / P),
            "_inv": inv, "_na": na, "_nb": nb, "_rec": rec, "_fpr": fpr}


def per_device(inv: np.ndarray, na: np.ndarray, nb: np.ndarray,
               device_raw: pd.Series, y: np.ndarray) -> list:
    """Which devices the forced false positives come from.

    A forced false positive is a benign record sitting in a cell whose majority
    is attack: no rule on this representation can spare it.
    """
    pred_attack = (na >= nb)[inv]
    benign = y == 0
    forced = benign & pred_attack
    total = int(forced.sum())
    out = []
    for dev in sorted(device_raw.unique()):
        m = (device_raw.values == dev)
        n_benign = int((benign & m).sum())
        n_forced = int((forced & m).sum())
        out.append({
            "device": dev,
            "rows": int(m.sum()),
            "benign_rows": n_benign,
            "forced_false_positives": n_forced,
            "device_floor_fpr": n_forced / n_benign if n_benign else None,
            "share_of_total_floor": n_forced / total if total else 0.0,
        })
    return sorted(out, key=lambda d: -d["forced_false_positives"])


def label_oracle_report() -> dict:
    """Why the floor collapses: is a timestamp column a label oracle?

    Read the raw `date` strings back (the feature build label-encodes them) and
    tabulate the label against each value.  If every value carries one label,
    the column determines the label outright and no amount of modelling is
    involved in the collapse.
    """
    ld = TonIotTelemetryLoader(DATA_DIR)
    frames = []
    for fname, dname in DEVICE_FILES.items():
        p = ld.dir / fname
        if not p.exists():
            continue
        df = ld._load_one_device(p, dname)
        if "date" in df.columns:
            frames.append(df[["__device", "date", "time", "label"]])
    big = pd.concat(frames, ignore_index=True)
    big["label"] = pd.to_numeric(big["label"], errors="coerce").fillna(0).astype(int)

    by_date = big.groupby("date")["label"].agg(["size", "sum", "mean", ])
    table = [{"date": d,
              "rows": int(r["size"]),
              "attack": int(r["sum"]),
              "benign": int(r["size"] - r["sum"]),
              "attack_share": float(r["mean"]),
              "devices": int(big.loc[big.date == d, "__device"].nunique())}
             for d, r in by_date.iterrows()]
    pure = [t for t in table if t["attack_share"] in (0.0, 1.0)]

    def conflict_of(cols):
        X = big[cols].astype(str).agg("|".join, axis=1)
        g = big.assign(_k=X).groupby("_k")["label"].mean()
        k = big.assign(_k=X)["_k"]
        mixed = set(g[(g > 0) & (g < 1)].index)
        return {"distinct_values": int(len(g)),
                "label_conflicting_rows": float(k.isin(mixed).mean())}

    return {
        "question": "does a timestamp column determine the label on its own?",
        "date": conflict_of(["date"]),
        "time": conflict_of(["time"]),
        "date_time": conflict_of(["date", "time"]),
        "device_date": conflict_of(["__device", "date"]),
        "date_table": table,
        "date_values_that_are_single_label": len(pure),
        "rows_covered_by_single_label_dates": int(sum(t["rows"] for t in pure)),
        "total_rows": int(len(big)),
        "verdict": None,        # filled by main()
    }


def main() -> int:
    print("ToN-IoT timestamp counterfactual\n")
    big, base_feats, device_raw = build_frame()
    y = big["label"].astype(int).values
    print(f"  {len(y)} rows, {int(y.sum())} attack, {int(len(y)-y.sum())} benign, "
          f"{len(base_feats)} base features\n")

    variants = {
        "base": base_feats,
        "base+date": base_feats + ["date"],
        "base+time": base_feats + ["time"],
        "base+date+time": base_feats + ["date", "time"],
    }

    results, summary = {}, []
    for name, feats in variants.items():
        Xp, yp, i_tr, i_te = pipeline(big, feats, y)
        pooled = cell_report(Xp, yp)

        n_tr = len(i_tr)
        split = cell_report(Xp[n_tr:], yp[n_tr:])
        at = _at_recall(split["_rec"], split["_fpr"], OBS_RECALL)
        frontier = at["frontier_fpr"] if at else None
        slack = (OBS_FPR - frontier) if frontier is not None else None

        inv_tr, inv_te = pooled["_inv"][:n_tr], pooled["_inv"][n_tr:]
        twin = float(np.isin(inv_te, np.unique(inv_tr)).mean())

        dev = per_device(pooled["_inv"], pooled["_na"], pooled["_nb"],
                         device_raw.iloc[np.concatenate([i_tr, i_te])].reset_index(drop=True),
                         yp)

        row = {
            "variant": name,
            "n_features": len(feats),
            "added_columns": [c for c in feats if c not in base_feats],
            "pooled": {k: v for k, v in pooled.items() if not k.startswith("_")},
            "split": {k: v for k, v in split.items() if not k.startswith("_")},
            "split_frontier_at_detector_recall": frontier,
            "detector": {"config": OBS_CONFIG, "fpr": OBS_FPR, "recall": OBS_RECALL},
            "slack_pp": 100 * slack if slack is not None else None,
            "slack_share": slack / OBS_FPR if slack is not None else None,
            "twin_rate": twin,
            "per_device": dev,
        }
        results[name] = row
        summary.append(row)

        print(f"  {name:<15} cells {pooled['cells']:>7}  dup {100*pooled['duplicate_rate']:5.2f}%  "
              f"conflict {100*pooled['conflict_rate']:5.2f}%  floor {pooled['floor_fpr']:.6f}")
        print(f"  {'':<15} split floor {split['floor_fpr']:.6f}  "
              f"frontier@{OBS_RECALL:.4f} {frontier:.6f}  "
              f"slack {100*slack:+.2f} pp ({100*slack/OBS_FPR:.1f}%)  twin {100*twin:.2f}%")

    base, full = results["base"], results["base+date+time"]
    print(f"\n  conflict rate  {100*base['pooled']['conflict_rate']:.2f}%  ->  "
          f"{100*full['pooled']['conflict_rate']:.2f}%")
    print(f"  pooled floor   {base['pooled']['floor_fpr']:.6f}  ->  "
          f"{full['pooled']['floor_fpr']:.6f}")
    print(f"  twin rate      {100*base['twin_rate']:.2f}%  ->  {100*full['twin_rate']:.2f}%")

    print(f"\n  per-device forced false positives ({'base':>14}):")
    for d in base["per_device"][:4]:
        print(f"    {d['device']:<14} {d['forced_false_positives']:>7}  "
              f"({100*d['share_of_total_floor']:5.1f}% of the floor, "
              f"device FPR {d['device_floor_fpr']:.4f})")

    oracle = label_oracle_report()
    n_pure = oracle["date_values_that_are_single_label"]
    n_vals = oracle["date"]["distinct_values"]
    oracle["verdict"] = (
        f"`date` takes {n_vals} distinct values and {n_pure} of them carry a single "
        f"label, covering {oracle['rows_covered_by_single_label_dates']} of "
        f"{oracle['total_rows']} rows. The column is a label oracle: benign telemetry "
        f"was collected on one set of days and attack telemetry on another, with no "
        f"overlap. Dropping it is therefore CORRECT, and the 29.5% floor is the true "
        f"cost of a sound protocol on this corpus - not an artefact to be undone."
        if n_pure == n_vals else
        f"`date` takes {n_vals} values, {n_pure} single-label. Not a complete oracle.")

    print(f"\n  why the floor collapses:")
    print(f"    date: {n_vals} distinct values, "
          f"{100*oracle['date']['label_conflicting_rows']:.2f}% label-conflicting rows")
    print(f"    time: {oracle['time']['distinct_values']} distinct values, "
          f"{100*oracle['time']['label_conflicting_rows']:.2f}% label-conflicting rows")
    for t in oracle["date_table"]:
        print(f"      {t['date']:<11} {t['rows']:>7} rows  "
              f"{'ATTACK' if t['attack_share'] == 1.0 else 'benign' if t['attack_share'] == 0.0 else 'mixed':<6}"
              f"  {t['devices']} devices")
    print(f"    -> {oracle['verdict']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "ton_iot_timestamp_counterfactual.json"
    out.write_text(json.dumps({
        "spec_version": SPEC_VERSION,
        "analysis": "ton-iot timestamp counterfactual",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": str(HERE), "sha256": _sha256(HERE)},
        "pipeline": ("loader construction; float32; seed-42 stratified 80/20; "
                     "StandardScaler fitted on train; identical across variants"),
        "both_horns": (
            "A collapse of the floor when `date`/`time` are restored is NOT an argument "
            "for restoring them. A per-record timestamp is an identifier: a detector "
            "reading one exploits collection order, not behaviour. The finding is that "
            "the floor is a property of the preprocessing recipe rather than of the "
            "phenomenon, which is the claim this paper makes about benchmarks generally, "
            "here turned on its own instrument."),
        "variants": summary,
        "label_oracle": oracle,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
