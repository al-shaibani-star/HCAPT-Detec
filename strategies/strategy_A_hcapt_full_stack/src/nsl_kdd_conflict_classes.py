"""Which attack families the NSL-KDD forced false positives collide with.

Section 5.7 reported this breakdown from no stored artefact, and its figures did
not reconcile: it stated 88 forced false positives, enumerated 92 colliding attack
records, and quoted a share that required 88 as the denominator. This regenerates
the whole thing from the prescribed test partition and writes it to
results/_attainability/nsl_kdd_conflict_by_class.json, so the manuscript quotes a
file rather than a recollection.

A *forced false positive* is a benign record in a cell the attainability rule
admits -- attack count at least benign count, the attack-tie convention pinned as
P4 in attainability.py. Cells are formed on the loader's output array, after
scaling and the cast to single precision, matching P1 and P2.

The taxonomy is the loader's own ATTACK_CATEGORIES, not a hand-typed list.

    python -m src.nsl_kdd_conflict_classes
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STRATEGY_ROOT = HERE.parent
DATA_DIR = STRATEGY_ROOT.parents[1] / "data"
OUT = STRATEGY_ROOT / "results" / "_attainability" / "nsl_kdd_conflict_by_class.json"


def main() -> int:
    from src.loaders.nsl_kdd import (ATTACK_CATEGORIES, NSL_KDD_COLUMNS,
                                     NslKddLoader)

    loader = NslKddLoader(DATA_DIR)
    ds = loader.load("binary")            # binary drops no rows, so order is the CSV's
    X = np.asarray(ds.X_test, dtype=np.float32)
    y = np.asarray(ds.y_test).astype(int).ravel()

    raw = pd.read_csv(loader.test_path, header=None, names=NSL_KDD_COLUMNS)
    family = raw["label"].map(lambda s: ATTACK_CATEGORIES.get(s, "Unknown")).to_numpy()
    if len(family) != len(y):
        print(f"row mismatch: {len(family)} labels vs {len(y)} rows")
        return 1

    _, inverse = np.unique(X, axis=0, return_inverse=True)
    attack_n, benign_n = Counter(), Counter()
    for cell, label in zip(inverse.tolist(), y.tolist()):
        (attack_n if label == 1 else benign_n)[cell] += 1

    forced = 0
    colliding = defaultdict(int)
    conflicting_cells = 0
    for cell in range(inverse.max() + 1):
        a, b = attack_n[cell], benign_n[cell]
        if not (a and b):
            continue
        conflicting_cells += 1
        if a >= b:                                   # admitted under the tie rule
            forced += b
            for idx in np.flatnonzero(inverse == cell):
                if y[idx] == 1:
                    colliding[str(family[idx])] += 1

    total = sum(colliding.values())
    rows_in_conflicting = sum(attack_n[c] + benign_n[c] for c in range(inverse.max() + 1)
                              if attack_n[c] and benign_n[c])
    report = {
        "note": "regenerated for section 5.7; supersedes the unsourced 88/63/27 figures",
        "partition": "prescribed KDDTest+",
        "tie_rule": "attack (P4)",
        "n_test_rows": int(len(y)),
        "n_cells": int(inverse.max() + 1),
        "n_conflicting_cells": conflicting_cells,
        "rows_in_conflicting_cells": int(rows_in_conflicting),
        "conflict_rate": round(rows_in_conflicting / len(y), 6),
        "forced_false_positives": int(forced),
        "colliding_attack_records": int(total),
        "by_family": dict(sorted(colliding.items(), key=lambda kv: -kv[1])),
        "share_of_colliding_attack_records": (
            {k: round(v / total, 4) for k, v in
             sorted(colliding.items(), key=lambda kv: -kv[1])} if total else {}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
