"""Read the support sweep and test the two competing explanations.

The dissertation's explanation is SUPPORT: augmentation stops helping a class
once that class has too few real samples to train the generator on.

The five-class snapshot suggests a rival: HEADROOM. The one class that gained
(DoS) is also the abundant class the ensemble was worst at -- control recall
0.7753 against Normal's 0.9720. A class the model already recalls well has
nothing for augmentation to add, whatever its support.

The sweep separates them, because along the ladder the subject class's support
changes while its identity does not. This script reports the gain against both
axes and says which one it tracks.

    python -m src.support_sweep_report --class DoS
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent / "results" / "_support_sweep"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls_name", default="DoS")
    args = ap.parse_args()

    p = IN_DIR / f"{args.cls_name}.json"
    if not p.exists():
        print(f"no sweep yet at {p}")
        return 1
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["rows"]

    by_n: dict[int, list] = {}
    for r in rows:
        by_n.setdefault(r["support"], []).append(r)

    print(f"subject class {d['subject_class']}   full support {d['full_support']:,}")
    print(f"test partition: {d['test_partition']}\n")
    print(f"{'support':>9}{'runs':>6}{'ctrl recall':>13}{'aug recall':>12}"
          f"{'delta pp':>11}{'spread':>9}")
    ladder = []
    for n in sorted(by_n):
        rs = by_n[n]
        c = st.fmean(r["recall_control"] for r in rs)
        a = st.fmean(r["recall_augmented"] for r in rs)
        ds = [r["delta_pp"] for r in rs]
        spread = (max(ds) - min(ds)) if len(ds) > 1 else float("nan")
        ladder.append((n, c, a, st.fmean(ds), spread))
        print(f"{n:>9,}{len(rs):>6}{c:>13.4f}{a:>12.4f}{st.fmean(ds):>+11.2f}"
              f"{spread:>9.2f}")

    if len(ladder) < 2:
        print("\nonly one level so far; the ladder cannot be read yet")
        return 0

    # Does the gain track support, or does it track how much room the control left?
    def corr(xs, ys):
        mx, my = st.fmean(xs), st.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return num / den if den else float("nan")

    import math
    ns = [math.log10(n) for n, *_ in ladder]
    deltas = [dl for *_, dl, _ in ladder]
    headroom = [1.0 - c for _, c, *_ in ladder]

    r_support = corr(ns, deltas)
    r_headroom = corr(headroom, deltas)
    print(f"\n  gain vs log10(support) : r = {r_support:+.3f}")
    print(f"  gain vs control headroom: r = {r_headroom:+.3f}")
    print("\n  " + ("support explains it better" if abs(r_support) > abs(r_headroom)
                    else "headroom explains it better")
          + " on the levels measured so far.")

    signs = {n: (dl > 0) for n, *_, dl, _ in ladder}
    pos = [n for n, ok in signs.items() if ok]
    neg = [n for n, ok in signs.items() if not ok]
    if pos and neg and max(neg) < min(pos):
        print(f"  sign change bracketed: gain turns positive between "
              f"{max(neg):,} and {min(pos):,} samples")
    elif pos and not neg:
        print("  gain is positive at every level measured: no threshold in this range")
    elif neg and not pos:
        print("  gain is negative at every level measured")
    else:
        print("  the sign does not change monotonically along the ladder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
