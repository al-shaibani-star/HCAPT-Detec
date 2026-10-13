"""
Compare regenerated result tables with the shipped snapshot (used by reproduce_from_predictions.ps1).
Numbers are compared with a relative tolerance of 1e-9; text fields must match exactly.
Usage: python revision\\verify_tables.py <shipped_dir> <regenerated_dir>
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

FILES = ["nslkdd_summary.json", "nslkdd_per_seed.json", "learning_curve_summary.csv", "learning_curve_changeover.json",
         "calibration_summary.csv", "calibration_per_seed.json", "validator_independence.csv",
         "mcnemar_proposed_vs_alternatives.csv", "paper_numbers.json"]


def same(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [m for k in a for m in same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in same(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
            return []
        return [] if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def load(p):
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(pd.read_csv(p).to_json(orient="records"))


def main():
    shipped, regen = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    for f in FILES:
        if not (shipped / f).exists():
            print(f"  skip  {f} (not shipped)"); continue
        diffs = same(load(shipped / f), load(regen / f))
        print(f"  {'OK  ' if not diffs else 'DIFF'}  {f}" + ("" if not diffs else f"  ({len(diffs)} values, e.g. {diffs[0]})"))
        bad += bool(diffs)
    print("ALL TABLES REPRODUCED" if not bad else f"{bad} file(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
