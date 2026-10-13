"""
Step 3 (b): analyse revision/out/learning_curve.csv.

For each target class and support level n: mean and 95 % t-interval over seeds of
VAL and TEST recall, C2ST AUC and coverage per method.

Changeover estimate (Reviewer 2: "estimate the uncertainty of the changeover
rather than promoting one observed count into a decision rule"):
  delta(n) = recall_GAN(n) - recall_SMOTE(n), paired by seed.
  n* = first support level, by log-linear interpolation, at which the mean
  delta crosses from < 0 to >= 0 and stays >= 0 for all larger n.
  The 95 % interval of n* comes from 2,000 bootstrap resamples of seeds.
  "none" is reported when no crossover happens inside the tested range.

Outputs: revision/out/tables/learning_curve_summary.csv,
         revision/out/tables/learning_curve_changeover.json,
         revision/out/figures/fig_learning_curve.png
Usage:   python revision\\r05_learning_curve_analysis.py [--smoke]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from rev_common import OUT_DIR

METHOD_STYLE = {"none": ("No augmentation", "#7f7f7f", ":"), "smote": ("SMOTE", "#1f77b4", "-"),
                "cgan": ("CGAN", "#d62728", "--"), "wgan": ("WGAN-GP", "#2ca02c", "-.")}


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def crossover(levels, delta):
    """First interpolated n where delta goes from <0 to >=0 and stays >=0."""
    levels = np.asarray(levels, float); delta = np.asarray(delta, float)
    for i in range(len(levels)):
        if np.all(delta[i:] >= 0):
            if i == 0:
                return float(levels[0])          # GAN never worse in the tested range
            x0, x1 = np.log(levels[i - 1]), np.log(levels[i])
            d0, d1 = delta[i - 1], delta[i]
            return float(np.exp(x0 + (0 - d0) * (x1 - x0) / (d1 - d0)))
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    src = OUT_DIR / ("learning_curve_smoke.csv" if a.smoke else "learning_curve.csv")
    tdir = OUT_DIR / "tables"; fdir = OUT_DIR / "figures"
    tdir.mkdir(exist_ok=True); fdir.mkdir(exist_ok=True)
    df = pd.read_csv(src)
    for col in ("c2st_auc", "coverage"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []
    for (cls, n, m), g in df.groupby(["class_name", "n_real", "method"]):
        r = {"class": cls, "n_real": n, "method": m, "seeds": len(g)}
        for k in ("recall_val", "recall_test", "macro_f1_test", "c2st_auc", "coverage"):
            r[f"{k}_mean"], r[f"{k}_ci95"] = ci95(g[k])
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values(["class", "method", "n_real"])
    summ.to_csv(tdir / "learning_curve_summary.csv", index=False)

    change = {}
    rng = np.random.default_rng(0)
    for cls, g in df.groupby("class_name"):
        change[cls] = {}
        for gan in ("cgan", "wgan"):
            for split in ("recall_val", "recall_test"):
                piv = g[g.method.isin([gan, "smote"])].pivot_table(
                    index=["seed", "n_real"], columns="method", values=split).dropna()
                if piv.empty:
                    continue
                d = (piv[gan] - piv["smote"]).unstack("n_real")      # seeds x levels
                d = d.dropna(axis=1)
                levels = list(d.columns)
                point = crossover(levels, d.mean(0).values)
                boots = []
                for _ in range(2000):
                    s = d.iloc[rng.integers(0, len(d), len(d))]
                    boots.append(crossover(levels, s.mean(0).values))
                found = [b for b in boots if b is not None]
                change[cls][f"{gan}_vs_smote_{split}"] = {
                    "levels": levels, "mean_delta": d.mean(0).round(4).tolist(),
                    "n_star": point,
                    "n_star_ci95": ([float(np.percentile(found, 2.5)), float(np.percentile(found, 97.5))]
                                    if len(found) > 20 else None),
                    "share_of_bootstraps_with_crossover": len(found) / len(boots)}
    (tdir / "learning_curve_changeover.json").write_text(json.dumps(change, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = sorted(df.class_name.unique())
    fig, axes = plt.subplots(2, len(classes), figsize=(4.2 * len(classes), 6.4), squeeze=False)
    for j, cls in enumerate(classes):
        for i, split in enumerate(("recall_val", "recall_test")):
            ax = axes[i, j]
            for m, (lab, col, ls) in METHOD_STYLE.items():
                s = summ[(summ["class"] == cls) & (summ.method == m)]
                if s.empty:
                    continue
                ax.errorbar(s.n_real, s[f"{split}_mean"], yerr=s[f"{split}_ci95"], label=lab,
                            color=col, ls=ls, marker="o", ms=3, capsize=2, lw=1.2)
            ax.axvline(1000, color="k", lw=0.6, alpha=0.4)
            ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{cls} - {'VAL' if i == 0 else 'KDDTest+'}", fontsize=9)
            ax.set_xlabel("real training samples of the class (n)", fontsize=8)
            if j == 0:
                ax.set_ylabel("recall of the class", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fdir / "fig_learning_curve.png", dpi=300)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(change, indent=1))


if __name__ == "__main__":
    main()
