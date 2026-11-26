"""
Editor's conditional-acceptance items (IJIES 20267615, 2026-11-26), numerical parts.

  M  exact McNemar p-values computed in log space (no underflow), with Holm adjustment in log space:
     NSL-KDD Table 8 (from the per-seed discordant counts behind it) and UNSW-NB15 hypothesis H2 (Table 23).
     The exact two-sided p is 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2), capped at 1; it is strictly positive
     for every finite count, so it is reported as a number (mantissa and exponent) taken from log10 p.
  L  streaming simulation of Table D2 re-run with the warm single-flow median of Table D1 (500 calls) as the
     service time of a one-flow call; the other batch sizes keep the median of 30 calls each, log-linearly
     interpolated in the batch size exactly as in r21.

Usage
  python r22_editor_round3.py --impl <05-Implementation> --out <folder>
Inputs
  <impl>/revision/out/tables/mcnemar_proposed_vs_alternatives.csv   (written by r11 from the frozen predictions)
  <impl>/revision/out/tables/unsw_mcnemar.csv                        (written by r17)
  <out>/L_latency.json                                               (written by r21, part L)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom, binomtest

LN10 = math.log(10)


def log_p_exact(b, c):
    """Natural log of the exact two-sided McNemar p for discordant counts b, c."""
    n, k = int(b + c), int(min(b, c))
    if n == 0:
        return 0.0
    lp = math.log(2) + float(logsumexp(binom.logpmf(np.arange(k + 1), n, 0.5)))
    return min(0.0, lp)


def log_holm(logp):
    """Holm step-down adjustment of natural-log p-values (same order as the input)."""
    logp = np.asarray(logp, float); m = len(logp); order = np.argsort(logp)
    adj = np.empty(m); run = -np.inf
    for rank, i in enumerate(order):
        run = max(run, math.log(m - rank) + logp[i]); adj[i] = min(0.0, run)
    return adj


def fmt(lp, digits=1):
    """Format exp(lp) as 'm.me-XX' without leaving log space."""
    if lp >= math.log(1e-3):
        return f"{math.exp(lp):.{digits}e}"
    l10 = lp / LN10; e = math.floor(l10); m = 10 ** (l10 - e)
    if round(m, digits) >= 10:
        m /= 10; e += 1
    return f"{m:.{digits}f}e{e:+03d}".replace("e+", "e+").replace("e-0", "e-")


def part_m(impl, out):
    t = impl / "revision/out/tables"
    mc = pd.read_csv(t / "mcnemar_proposed_vs_alternatives.csv")
    rows = []
    for s, g in mc.groupby("seed"):
        lp = [log_p_exact(b, c) for b, c in zip(g.a_right_b_wrong, g.a_wrong_b_right)]
        la = log_holm(lp)
        for (_, r), x, y in zip(g.iterrows(), lp, la):
            # cross-check against scipy where the double-precision value does not underflow
            ref = binomtest(int(min(r.a_right_b_wrong, r.a_wrong_b_right)), int(r.a_right_b_wrong + r.a_wrong_b_right), 0.5).pvalue
            rows.append({"seed": s, "comparator": r.comparator, "only_proposed_correct": r.a_right_b_wrong,
                         "only_comparator_correct": r.a_wrong_b_right, "log10_p_exact": x / LN10,
                         "log10_p_holm": y / LN10, "p_exact": fmt(x), "p_holm": fmt(y),
                         "scipy_p_exact": ref, "matches_scipy": bool(ref == 0 or abs(math.log(ref) - x) < 1e-6)})
    nsl = pd.DataFrame(rows)
    nsl.to_csv(out / "M_mcnemar_nslkdd_logspace.csv", index=False)
    summ = []
    for comp, g in nsl.groupby("comparator", sort=False):
        lo, hi = g.log10_p_holm.min(), g.log10_p_holm.max()
        summ.append({"comparator": comp, "mean_only_proposed": g.only_proposed_correct.mean(),
                     "mean_only_comparator": g.only_comparator_correct.mean(),
                     "p_holm_min": fmt(lo * LN10), "p_holm_max": fmt(hi * LN10),
                     "scipy_underflow_in_old_table": bool((g.scipy_p_exact == 0).any())})
    summ = pd.DataFrame(summ); summ.to_csv(out / "M_mcnemar_nslkdd_table8.csv", index=False)
    print(summ.to_string(index=False)); print("all finite p match scipy:", bool(nsl.matches_scipy.all()))

    un = pd.read_csv(t / "unsw_mcnemar.csv")
    rows = []
    for s, g in un.groupby("seed"):
        lp = [log_p_exact(b, c) for b, c in zip(g.proposed_right, g.comparator_right)]
        la = log_holm(lp)
        for (_, r), x, y in zip(g.iterrows(), lp, la):
            rows.append({"seed": s, "comparator": r.comparator, "only_proposed_correct": r.proposed_right,
                         "only_comparator_correct": r.comparator_right, "log10_p_exact": x / LN10,
                         "log10_p_holm": y / LN10, "p_holm": fmt(y), "old_p_holm": r.p_holm})
    u = pd.DataFrame(rows); u.to_csv(out / "M_mcnemar_unsw_logspace.csv", index=False)
    print(u[u.comparator == "ens_smote"].to_string(index=False))


def part_l(out):
    res = json.loads((out / "L_latency.json").read_text())
    sizes = [1, 4, 16, 64, 256, 1024]
    sim = []
    for where in ("gpu", "cpu"):
        P = res["placements"][where]
        st_ = np.array([P["batch_seconds"][str(b)] for b in sizes])
        st_[0] = P["warm_single_median_ms"] / 1e3          # one-flow call: warm single-flow median (500 calls)
        bs = np.array(sizes, float)

        def service(n):
            return float(np.interp(np.log(n), np.log(bs), st_))
        for lam in (100, 1000, 5000, 20000):
            for policy, bmax in (("one flow per call", 1), ("micro-batching", 1024)):
                cap = bmax / service(bmax)
                arr = np.cumsum(np.random.default_rng(lam).exponential(1 / lam, 100000))
                t, i, lat = 0.0, 0, []
                while i < len(arr):
                    t = max(t, arr[i]); j = i
                    while j < len(arr) and arr[j] <= t and j - i < bmax:
                        j += 1
                    t += service(j - i); lat.extend(t - arr[i:j]); i = j
                lat = np.array(lat) * 1e3
                sim.append({"placement": where, "arrival_rate_per_s": lam, "serving": policy,
                            "service_time_one_flow_ms": service(1) * 1e3,
                            "capacity_flows_per_s": cap, "stable": bool(lam < cap),
                            "p50_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                            "p99_ms": float(np.percentile(lat, 99))})
    df = pd.DataFrame(sim); df.to_csv(out / "L2_streaming_simulation.csv", index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--impl", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); impl, out = Path(a.impl), Path(a.out)
    part_m(impl, out); part_l(out)
