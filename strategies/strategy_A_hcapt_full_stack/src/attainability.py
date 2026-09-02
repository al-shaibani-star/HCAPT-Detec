"""
Attainability frontier — the frozen reference implementation.

This is the single source of truth for every floor / frontier number in the
manuscript.  Nothing else may compute one.  If a number appears in the paper,
it came out of `results/_attainability/*.json`, which came out of this file.

--------------------------------------------------------------------------
WHAT IT COMPUTES
--------------------------------------------------------------------------
Given a labelled corpus in a fixed feature representation, group records into
*cells* of identical feature vectors.  Within a cell no function of the
representation can separate the records, so the label mix inside each cell
bounds what any detector reading that representation can attain.

  * majority rule   — assign each cell its majority label.  Yields one
                      (FPR, recall) point: the error-minimising rule.
  * frontier        — order cells by attack density (Neyman-Pearson) and
                      sweep.  Yields the whole curve: for every recall, the
                      lowest false-positive rate attainable at that recall.

The frontier, not the scalar floor, is the object the paper needs: a detector
operating at recall 0.80 must be compared against the frontier at recall 0.80,
never against the floor at recall 1.00.

--------------------------------------------------------------------------
PINNED CHOICES  (changing any of these invalidates every published number;
                 bump SPEC_VERSION and re-run all four benchmarks)
--------------------------------------------------------------------------
P1 REPRESENTATION  Cells are computed on exactly the array the detector
                   consumed: the loader's own `X_train` / `X_test` output.
                   For loaders that scale, that means POST-scaling.  Column-
                   wise affine scaling is injective, so it cannot merge two
                   distinct records — except through float32 rounding, which
                   is why P2 is pinned too.  Computing cells on the raw frame
                   instead would bound a representation no detector ever saw.
P2 DTYPE           float32, matching `LoadedDataset`.  Reported alongside a
                   float64 recount so any rounding-induced cell merge is
                   visible rather than silent.
P3 NAN POLICY      The loader's: `np.nan_to_num(..., nan=0.0, posinf=0.0,
                   neginf=0.0)`.  Re-applied defensively here.
P4 TIE RULE        A cell with equal attack and benign counts is assigned to
                   ATTACK.  Both tie rules are emitted; the paper quotes
                   ties-to-attack, which is the conservative choice for a
                   floor on FPR (it can only raise it).
P5 SPLIT           Whatever the loader defines.  NSL-KDD = the official
                   KDDTest+.  ToN-IoT / Edge-IIoTset / DAPT 2020 = the
                   loader's own stratified split, seed 42, test_size 0.2.
P6 SCOPE           Every quantity is emitted twice: over the pooled corpus
                   (train + test) and over the evaluation split alone.  The
                   detector's FPR is measured on the split, so the split-level
                   frontier is the one that may be differenced against it.

--------------------------------------------------------------------------
CAVEAT THAT MUST TRAVEL WITH EVERY NUMBER
--------------------------------------------------------------------------
The bound is computed in-sample: it is a certificate about *this labelled
corpus in this representation*, not an estimate of a population Bayes error.
It is informative only where records collide, so the duplicate rate is
emitted beside every floor and a corpus with no duplicates yields a vacuous
(zero) floor.  `train_fitted_lookup` is the honest counterweight: the same
rule fitted on train and applied to test is an *achievable* classifier, and
the gap between it and the in-sample floor is the memorisation component.

Usage
-----
    python -m src.attainability                     # all four benchmarks
    python -m src.attainability --benchmarks nsl-kdd edge-iiotset
    python -m src.attainability --ton-iot-variant rebuild_raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SPEC_VERSION = "attainability-1.0.0"

HERE = Path(__file__).resolve()
STRATEGY_ROOT = HERE.parents[1]
DATA_DIR = STRATEGY_ROOT.parents[1] / "data"
RESULTS_DIR = STRATEGY_ROOT / "results"
OUT_DIR = RESULTS_DIR / "_attainability"

if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

BENCHMARKS = ["nsl-kdd", "ton-iot-telemetry", "edge-iiotset", "dapt-2020"]

# Curve stored in JSON is downsampled to this many points; the exact curve
# goes to the companion .npz, and exact values at every queried recall are
# always stored at full precision.
CURVE_POINTS = 1000


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def _cells(X: np.ndarray) -> np.ndarray:
    """Map each row to a cell id.  Identical feature vectors share an id."""
    _, inv = np.unique(X, axis=0, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def _counts(inv: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = int(inv.max()) + 1 if len(inv) else 0
    n_attack = np.bincount(inv, weights=y, minlength=m)
    n_total = np.bincount(inv, minlength=m).astype(float)
    return n_attack, n_total - n_attack


def _majority(n_attack: np.ndarray, n_benign: np.ndarray, P: float, N: float,
              ties_to_attack: bool) -> Dict:
    pred = n_attack >= n_benign if ties_to_attack else n_attack > n_benign
    tp = float(n_attack[pred].sum())
    fp = float(n_benign[pred].sum())
    fn = float(n_attack[~pred].sum())
    tn = float(n_benign[~pred].sum())
    return {
        "tie_rule": "attack" if ties_to_attack else "benign",
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": fp / N if N else None,
        "recall": tp / P if P else None,
        "accuracy": (tp + tn) / (P + N) if (P + N) else None,
    }


def _frontier(n_attack: np.ndarray, n_benign: np.ndarray,
              P: float, N: float) -> Tuple[np.ndarray, np.ndarray]:
    """Neyman-Pearson sweep: cells ordered by attack density, cumulated.

    Returns (recall, fpr), both monotone non-decreasing.  Point i is the best
    attainable false-positive rate once the i-th cell has been admitted.
    """
    density = n_attack / (n_benign + 1e-12)
    order = np.argsort(-density, kind="stable")
    recall = np.cumsum(n_attack[order]) / P
    fpr = np.cumsum(n_benign[order]) / N
    return recall, fpr


def _at_recall(recall: np.ndarray, fpr: np.ndarray, target: float) -> Optional[Dict]:
    """Lowest attainable FPR at recall >= target."""
    i = int(np.searchsorted(recall, target, side="left"))
    if i >= len(recall):
        return None
    return {"requested_recall": float(target),
            "attained_recall": float(recall[i]),
            "frontier_fpr": float(fpr[i])}


def _downsample(recall: np.ndarray, fpr: np.ndarray, k: int) -> Dict[str, List[float]]:
    if len(recall) <= k:
        idx = np.arange(len(recall))
    else:
        idx = np.unique(np.linspace(0, len(recall) - 1, k).astype(int))
    return {"recall": [float(v) for v in recall[idx]],
            "fpr": [float(v) for v in fpr[idx]]}


def analyse(X: np.ndarray, y: np.ndarray, query_recalls: List[float]) -> Tuple[Dict, np.ndarray, Dict]:
    """Full attainability report for one (X, y).  Returns (report, cell_ids, curve)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)   # P3
    y = np.asarray(y).astype(np.int64)

    n = len(y)
    P = float(y.sum())
    N = float(n - P)

    inv = _cells(X)
    n_attack, n_benign = _counts(inv, y.astype(float))
    n_total = n_attack + n_benign

    duplicate_rate = float((n_total[inv] > 1).mean())
    conflict_rate = float((((n_attack > 0) & (n_benign > 0))[inv]).mean())

    recall, fpr = _frontier(n_attack, n_benign, P, N)

    # P2: does float32 rounding merge cells that float64 keeps apart?
    cells_f64 = len(np.unique(np.ascontiguousarray(X, dtype=np.float64), axis=0))

    report = {
        "n_rows": int(n),
        "n_attack": int(P),
        "n_benign": int(N),
        "n_cells": int(len(n_total)),
        "n_cells_float64": int(cells_f64),
        "float32_merged_cells": int(cells_f64 - len(n_total)),
        "duplicate_rate": duplicate_rate,
        "conflict_rate": conflict_rate,
        "rows_in_conflicting_cells": int(round(conflict_rate * n)),
        "majority_ties_to_attack": _majority(n_attack, n_benign, P, N, True),
        "majority_ties_to_benign": _majority(n_attack, n_benign, P, N, False),
        "frontier_at": [r for r in (_at_recall(recall, fpr, q) for q in query_recalls) if r],
        "frontier_curve_downsampled": _downsample(recall, fpr, CURVE_POINTS),
    }
    return report, inv, {"recall": recall, "fpr": fpr}


def train_fitted_lookup(inv_train: np.ndarray, y_train: np.ndarray,
                        inv_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """The floor's achievable counterpart: majority rule FITTED ON TRAIN ONLY,
    applied to test.  Unseen cells fall back to the training prior.  The gap
    between this and the in-sample floor is the memorisation component."""
    u, inv2 = np.unique(inv_train, return_inverse=True)
    a = np.bincount(inv2, weights=y_train.astype(float), minlength=len(u))
    t = np.bincount(inv2, minlength=len(u)).astype(float)
    rule = dict(zip(u.tolist(), (a >= (t - a)).tolist()))
    default = bool(y_train.mean() >= 0.5)
    pred = np.fromiter((rule.get(int(k), default) for k in inv_test),
                       dtype=bool, count=len(inv_test))
    yt = y_test.astype(bool)
    tp = int((pred & yt).sum()); fp = int((pred & ~yt).sum())
    fn = int((~pred & yt).sum()); tn = int((~pred & ~yt).sum())
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "accuracy": (tp + tn) / len(yt),
        "unseen_test_cells": int(sum(1 for k in np.unique(inv_test) if int(k) not in rule)),
    }


def twin_rate(inv_train: np.ndarray, inv_test: np.ndarray) -> Dict:
    """Share of test rows whose exact feature vector also occurs in train."""
    seen = np.isin(inv_test, np.unique(inv_train))
    return {"test_rows": int(len(inv_test)),
            "test_rows_with_training_twin": int(seen.sum()),
            "twin_rate": float(seen.mean())}


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def load_via_loader(benchmark: str):
    """Load through the project's own loader — the representation the detector saw."""
    if benchmark == "nsl-kdd":
        from src.loaders.nsl_kdd import NslKddLoader as L
    elif benchmark == "ton-iot-telemetry":
        from src.loaders.ton_iot_telemetry import TonIotTelemetryLoader as L
    elif benchmark == "edge-iiotset":
        from src.loaders.edge_iiotset import EdgeIiotsetLoader as L
    elif benchmark == "dapt-2020":
        from src.loaders.dapt2020 import Dapt2020Loader as L
    else:
        raise ValueError(benchmark)
    loader = L(DATA_DIR)
    ds = loader.load("binary")
    return ds, loader


def load_ton_iot_rebuild(with_timestamps: bool = False):
    """ToN-IoT rebuilt from raw CSVs, unscaled — the variant used by the
    gatekeeper audit.  Kept so the two constructions can be adjudicated
    (see task 2 of FINAL_PLAN.md); NOT the representation the detector saw."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from src.loaders.ton_iot_telemetry import (
        DEVICE_FILES, FEATURE_COLS_BY_DEVICE, CAT_COLS_BY_DEVICE)

    d = DATA_DIR / "ton-iot" / "iot-telemetry"
    frames = []
    for fname, device in DEVICE_FILES.items():
        df = pd.read_csv(d / fname)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
        df["__device"] = device
        frames.append(df)

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
                df[c] = "?"
        rows.append(df[["__device", "label", "date", "time"] + num + cat])

    big = pd.concat(rows, ignore_index=True)
    big["label"] = pd.to_numeric(big["label"], errors="coerce").fillna(0).astype(int)
    for c in list(cat) + ["__device", "date", "time"]:
        big[c] = LabelEncoder().fit_transform(big[c].astype(str))
    for c in num:
        co = pd.to_numeric(big[c], errors="coerce")
        big[c] = LabelEncoder().fit_transform(big[c].astype(str)) if co.isna().any() else co

    feats = list(num) + list(cat) + ["__device"]
    if with_timestamps:
        feats += ["date", "time"]

    y = big["label"].astype(int).values
    X = big[feats].astype(np.float32).values
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    return X[i_tr], X[i_te], y[i_tr], y[i_te], feats


def observed_operating_points(benchmark: str) -> List[Dict]:
    """The detector's own (recall, FPR) points, read from the stored ablation
    table.  These are what the frontier must be evaluated AT."""
    path = RESULTS_DIR / benchmark / "ablation_table.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [{"config": str(r.config),
             "observed_recall": float(r.recall),
             "observed_fpr": float(r.fpr),
             "observed_f1": float(r.f1)} for r in df.itertuples()]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _attach_slack(scope: Dict, points: List[Dict], curve: Dict) -> None:
    """For every observed operating point, the frontier at ITS recall and the
    slack above it.  Quoting an FPR without its recall is what this prevents."""
    out = []
    for p in points:
        f = _at_recall(curve["recall"], curve["fpr"], p["observed_recall"])
        row = dict(p)
        if f is None:
            row.update({"frontier_fpr_at_observed_recall": None,
                        "slack_pp": None, "slack_share": None})
        else:
            slack = p["observed_fpr"] - f["frontier_fpr"]
            row.update({
                "frontier_fpr_at_observed_recall": f["frontier_fpr"],
                "frontier_attained_recall": f["attained_recall"],
                "slack_pp": 100.0 * slack,
                "slack_share": slack / p["observed_fpr"] if p["observed_fpr"] else None,
            })
        out.append(row)
    scope["observed_vs_frontier"] = out


def run_benchmark(benchmark: str, ton_iot_variant: str = "loader") -> Dict:
    points = observed_operating_points(benchmark)
    query = sorted({round(p["observed_recall"], 6) for p in points} | {0.90, 0.99, 1.0})

    rebuild = benchmark == "ton-iot-telemetry" and ton_iot_variant != "loader"
    if rebuild:
        Xtr, Xte, ytr, yte, feats = load_ton_iot_rebuild(
            with_timestamps=(ton_iot_variant == "rebuild_raw_timestamps"))
        source = {"variant": ton_iot_variant, "scaled": False,
                  "n_features": len(feats), "feature_names": feats}
    else:
        ds, loader = load_via_loader(benchmark)
        Xtr, Xte, ytr, yte = ds.X_train, ds.X_test, ds.y_train, ds.y_test
        source = {"variant": "loader", "loader": type(loader).__name__,
                  "scaled": bool(ds.metadata.get("scaler")),
                  "n_features": int(ds.X_train.shape[1]),
                  "source_paths": [str(p) for p in loader.source_paths()]}

    X_pool = np.vstack([Xtr, Xte])
    y_pool = np.concatenate([ytr, yte])

    pooled, inv_pool, curve_pool = analyse(X_pool, y_pool, query)
    split, inv_split, curve_split = analyse(Xte, yte, query)
    _attach_slack(split, points, curve_split)          # P6: differencing happens here
    _attach_slack(pooled, points, curve_pool)          # reference only

    inv_tr = inv_pool[:len(ytr)]
    inv_te = inv_pool[len(ytr):]
    split["leakage"] = twin_rate(inv_tr, inv_te)
    split["train_fitted_lookup"] = train_fitted_lookup(inv_tr, ytr, inv_te, yte)

    report = {
        "spec_version": SPEC_VERSION,
        "benchmark": benchmark,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": str(HERE), "sha256": _sha256(HERE)},
        "pinned": {
            "P1_representation": "loader output X (post-scaling where the loader scales)",
            "P2_dtype": "float32",
            "P3_nan_policy": "nan->0.0, +-inf->0.0",
            "P4_tie_rule": "ties to attack (both emitted)",
            "P5_split": "loader-defined; NSL-KDD official KDDTest+, others seed 42 / 0.2",
            "P6_scope": "pooled corpus and evaluation split, both emitted",
        },
        "caveat": ("In-sample certificate about this labelled corpus in this "
                   "representation, not a population Bayes error. Informative "
                   "only where records collide; read the duplicate rate beside "
                   "every floor, and train_fitted_lookup as the achievable "
                   "counterpart."),
        "source": source,
        "pooled": pooled,
        "split": split,
    }
    return report, curve_pool, curve_split


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Attainability frontier (frozen).")
    ap.add_argument("--benchmarks", nargs="*", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--ton-iot-variant", default="loader",
                    choices=["loader", "rebuild_raw", "rebuild_raw_timestamps"])
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = []
    for b in args.benchmarks:
        print(f"[{b}] computing ...", flush=True)
        try:
            report, cp, cs = run_benchmark(b, args.ton_iot_variant)
        except Exception as exc:                      # a missing corpus must not
            print(f"[{b}] FAILED: {exc!r}")           # kill the other three
            index.append({"benchmark": b, "status": "failed", "error": repr(exc)})
            continue

        tag = b if not (b == "ton-iot-telemetry" and args.ton_iot_variant != "loader") \
            else f"{b}__{args.ton_iot_variant}"
        (out / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(out / f"{tag}_curve.npz",
                            pooled_recall=cp["recall"], pooled_fpr=cp["fpr"],
                            split_recall=cs["recall"], split_fpr=cs["fpr"])

        p, s = report["pooled"], report["split"]
        print(f"  pooled: {p['n_rows']:>7} rows  {p['n_cells']:>7} cells  "
              f"dup {100*p['duplicate_rate']:5.2f}%  conflict {100*p['conflict_rate']:5.2f}%  "
              f"floor {p['majority_ties_to_attack']['fpr']:.6f} "
              f"@rec {p['majority_ties_to_attack']['recall']:.6f}")
        print(f"  split : {s['n_rows']:>7} rows  {s['n_cells']:>7} cells  "
              f"dup {100*s['duplicate_rate']:5.2f}%  conflict {100*s['conflict_rate']:5.2f}%  "
              f"floor {s['majority_ties_to_attack']['fpr']:.6f} "
              f"@rec {s['majority_ties_to_attack']['recall']:.6f}")
        print(f"  twin rate {100*s['leakage']['twin_rate']:.2f}%   "
              f"train-fitted lookup FPR {s['train_fitted_lookup']['fpr']:.6f} "
              f"@rec {s['train_fitted_lookup']['recall']:.6f}")
        for r in s["observed_vs_frontier"]:
            if r.get("frontier_fpr_at_observed_recall") is None:
                print(f"    {r['config']:<14} FPR {r['observed_fpr']:.4f} "
                      f"@rec {r['observed_recall']:.4f}  frontier: recall unreachable")
            else:
                print(f"    {r['config']:<14} FPR {r['observed_fpr']:.4f} "
                      f"@rec {r['observed_recall']:.4f}  frontier "
                      f"{r['frontier_fpr_at_observed_recall']:.4f}  "
                      f"slack {r['slack_pp']:+.2f} pp ({100*r['slack_share']:.1f}%)")
        index.append({"benchmark": b, "status": "ok", "file": f"{tag}.json",
                      "spec_version": SPEC_VERSION})

    # Merge, so running in batches does not erase earlier entries. Keyed on the
    # output file, which already encodes the ToN-IoT variant.
    index_path = out / "index.json"
    merged: Dict[str, Dict] = {}
    if index_path.exists():
        try:
            prev = json.loads(index_path.read_text(encoding="utf-8"))
            for e in prev.get("benchmarks", []):
                merged[e.get("file") or e["benchmark"]] = e
        except (json.JSONDecodeError, KeyError):
            pass                                       # a corrupt index is not fatal
    for e in index:
        merged[e.get("file") or e["benchmark"]] = e

    index_path.write_text(json.dumps(
        {"spec_version": SPEC_VERSION,
         "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "script_sha256": _sha256(HERE),
         "last_run_ton_iot_variant": args.ton_iot_variant,
         "benchmarks": sorted(merged.values(), key=lambda e: e["benchmark"])},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
