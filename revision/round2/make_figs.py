"""
Redraws every manuscript figure at its exact print width with >= 10-pt lettering (IJIES format check 6).
The data and statistics are those of the authors' scripts (r05, r06, r10); only size and typography change,
except Fig. 2, which is regenerated with the leakage-safe pipeline (PCA fitted on FIT).

Usage: python make_figs.py --impl <05-Implementation> --out <folder>
Print widths (inches) must match those used in build_revised.py (WIDTH dict below).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.metrics import roc_curve

FS = 10
plt.rcParams.update({"font.family": "Times New Roman", "font.size": FS, "axes.titlesize": FS,
                     "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.linewidth": 0.6, "savefig.dpi": 300})
WIDTH = {"fig1": 6.7, "fig2": 6.7, "fig3": 6.7, "fig4": 6.7, "fig5": 6.7, "fig6": 4.0, "fig7": 6.7,
         "fig8": 3.2, "fig9": 5.6, "fig10": 6.2}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")          # no bbox='tight': the printed width stays exact
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1
def box(ax, x, y, w, h, text, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
                                fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS,
            fontweight="bold" if bold else "normal", linespacing=1.1)


def arrow(ax, a, b, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.9, ls=ls, color="#333"))


def fig1(impl, out):
    W, H = WIDTH["fig1"], 4.0
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    box(ax, 0.05, 2.85, 1.25, 0.6, "KDDTrain+\n125,973 flows", "#e8e8e8", bold=True)
    box(ax, 1.55, 3.15, 1.15, 0.55, "FIT (80%)\n100,778", "#e3eefc")
    box(ax, 1.55, 2.45, 1.15, 0.55, "VAL (20%)\n25,195", "#e6f4e6")
    arrow(ax, (1.30, 3.25), (1.55, 3.42)); arrow(ax, (1.30, 3.05), (1.55, 2.72))
    ax.text(3.9, 3.8, "fitted on", ha="center", fontsize=FS, style="italic")
    box(ax, 2.95, 3.05, 1.9, 0.68, "encoders, scaler,\ngenerators, learners,\nvalidator", "#e3eefc")
    box(ax, 2.95, 2.3, 1.9, 0.6, "ensemble weights,\nthresholds, calibrators", "#e6f4e6")
    arrow(ax, (2.70, 3.42), (2.95, 3.39)); arrow(ax, (2.70, 2.72), (2.95, 2.6))
    box(ax, 5.1, 2.45, 1.55, 1.05, "frozen\nper-sample\npredictions\n(5 seeds)", "#fff3d6", bold=True)
    arrow(ax, (4.85, 3.39), (5.1, 3.2)); arrow(ax, (4.85, 2.6), (5.1, 2.75))
    y = 1.35
    box(ax, 1.55, y - 0.3, 1.0, 0.6, "x → encode\n+ scale", "#f2f2f2")
    box(ax, 2.8, y - 0.3, 1.45, 0.6, "RF + XGB + DNN\nsoft vote (ŷ, c)", "#d9f2f2")
    box(ax, 4.5, y - 0.3, 0.95, 0.6, "validator\nrisk r", "#fde9d2")
    box(ax, 5.65, y - 0.3, 1.0, 0.6, "policy\n(5 actions)", "#f2f2f2")
    box(ax, 0.05, 0.1, 1.25, 0.75, "KDDTest+\n22,544 flows\n(scored once)", "#fde2e2", bold=True)
    arrow(ax, (1.30, 0.6), (1.75, y - 0.3)); arrow(ax, (2.55, y), (2.8, y))
    arrow(ax, (4.25, y), (4.5, y)); arrow(ax, (5.45, y), (5.65, y))
    arrow(ax, (5.875, 2.45), (5.875, y + 0.3), ls="--")
    box(ax, 4.5, 0.12, 2.15, 0.42, "metrics, CIs, McNemar", "#fde2e2")
    arrow(ax, (6.15, y - 0.3), (6.15, 0.54))
    ax.text(1.5, 0.3, "test labels touch no fitted object", fontsize=FS, style="italic", color="#a00")
    save(fig, out, "fig1_protocol")


# ------------------------------------------------------------------ Fig. 2
def fig2(impl, out):
    sys.path.insert(0, str(impl / "revision"))
    import rev_common as rc
    from sklearn.decomposition import PCA
    D = rc.load_nslkdd()
    pca = PCA(2, random_state=0).fit(D["X_fit"])
    rng = np.random.default_rng(0)
    cols = ["#1f77b4", "#d95f02", "#1b9e77", "#e7298a", "#e6ab02"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH["fig2"], 2.9), sharex=True, sharey=True)
    for ax, (X, y, name) in zip(axes, ((D["X_fit"], D["y_fit"], "KDDTrain+ (FIT)"),
                                       (D["X_test"], D["y_test"], "KDDTest+"))):
        i = rng.choice(len(X), 3000, replace=False); Z = pca.transform(X[i]); yy = y[i]
        for c in range(5):
            m = yy == c
            ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.55, color=cols[c], label=CLASSES[c], lw=0)
        ax.set_title(f"{name}, 3,000 flows"); ax.set_xlabel("PC 1"); ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("PC 2")
    h = [Line2D([], [], marker="o", ls="", color=cols[c], ms=5, label=CLASSES[c]) for c in range(5)]
    axes[1].legend(handles=h, frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, out, "fig2_pca")
    return float(pca.explained_variance_ratio_.sum())


# ------------------------------------------------------------------ Fig. 3
def fig3(impl, out):
    s = json.loads((impl / "revision/out/tables/nslkdd_summary.json").read_text())["ablation"]
    names = list(s); acc = np.array([s[n]["accuracy"]["mean"] for n in names])
    sd = np.array([s[n]["accuracy"]["std"] for n in names]); f1 = np.array([s[n]["f1"]["mean"] for n in names])
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(WIDTH["fig3"], 4.9))
    for i, k in enumerate(order):
        c = "#c0392b" if "proposed" in names[k] else ("#1f4e79" if names[k].startswith("Ensemble +") else "#555")
        ax.errorbar(acc[k], i, xerr=sd[k], fmt="o", color=c, ms=4, capsize=2, lw=0.8)
        ax.plot(f1[k], i, marker="|", color=c, ms=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order])
    ax.set_xlabel("KDDTest+ binary accuracy (dot, mean ± s.d., 5 seeds) and F1 (bar)")
    ax.grid(axis="x", lw=0.3, alpha=0.6)
    fig.tight_layout(); save(fig, out, "fig3_comparison")


# ------------------------------------------------------------------ Fig. 4
def fig4(impl, out):
    df = pd.read_csv(impl / "revision/out/learning_curve.csv")
    sty = {"none": ("No augmentation", "#7f7f7f", ":"), "smote": ("SMOTE", "#1f77b4", "-"),
           "cgan": ("CGAN", "#d62728", "--"), "wgan": ("WGAN-GP", "#2ca02c", "-.")}

    def ci(x):
        x = np.asarray(x, float); return x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    fig, ax = plt.subplots(2, 3, figsize=(WIDTH["fig4"], 4.6), sharey=True)
    for j, c in enumerate(["DoS", "Probe", "R2L"]):
        for i, sp in enumerate(("recall_val", "recall_test")):
            a = ax[i, j]
            for m, (lab, col, ls) in sty.items():
                g = df[(df.class_name == c) & (df.method == m)].groupby("n_real")[sp].apply(list)
                a.errorbar(g.index, [ci(v)[0] for v in g], yerr=[ci(v)[1] for v in g], label=lab, color=col,
                           ls=ls, marker="o", ms=2.5, capsize=2, lw=1)
            a.axvline(1000, color="k", lw=0.6, alpha=0.4); a.set_xscale("log"); a.set_ylim(-0.02, 1.02)
            a.set_title(f"{c}, {'VAL' if i == 0 else 'KDDTest+'}")
            if i == 1: a.set_xlabel("real samples of the class (n)")
            if j == 0: a.set_ylabel("recall of the class")
    ax[1, 2].legend(frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, out, "fig4_learning_curve")


# ------------------------------------------------------------------ Fig. 5
def fig5(impl, out):
    s = pd.read_csv(impl / "revision/out/tables/learning_curve_summary.csv")
    sty = {"smote": ("SMOTE", "#1f77b4", "-"), "cgan": ("CGAN", "#d62728", "--"), "wgan": ("WGAN-GP", "#2ca02c", "-.")}
    mk = {"DoS": "o", "Probe": "s", "R2L": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH["fig5"], 3.3))
    for ax, met, lab in zip(axes, ("c2st_auc", "coverage"),
                            ("C2ST AUC (0.5 = indistinguishable)", "coverage (1 = full support)")):
        for m, (ml, col, ls) in sty.items():
            for cls in ("DoS", "Probe", "R2L"):
                d = s[(s.method == m) & (s["class"] == cls)]
                ax.plot(d.n_real, d[f"{met}_mean"], color=col, ls=ls, marker=mk[cls], ms=3.5, lw=0.9)
        ax.set_xscale("log"); ax.set_xlabel("real samples of the class (n)"); ax.set_ylabel(lab)
        ax.grid(lw=0.3, alpha=0.5)
    h = [Line2D([], [], color=c, ls=l, label=n) for n, c, l in sty.values()] + \
        [Line2D([], [], color="#444", marker=m, ls="", label=k) for k, m in mk.items()]
    fig.legend(handles=h, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, out, "fig5_fidelity")


# ------------------------------------------------------------------ Fig. 6
def fig6(impl, out):
    d = np.load(impl / "revision/out/nslkdd_data.npz"); y = d["y_test"] != 0
    P = np.load(impl / "revision/out/preds/nslkdd_s0_hybrid.npz"); R = np.load(impl / "revision/out/preds/nslkdd_s0_risk.npz")
    s = 1 - P["p_test"][:, 0]; r = R["r_test"]
    fig, ax = plt.subplots(figsize=(WIDTH["fig6"], 3.3))
    for sc, lab, col in ((s, "ensemble score 1 − p(Normal)", "#1f4e79"), (r, "validator risk alone", "#e67e22")):
        f, t, _ = roc_curve(y, sc); ax.plot(f, t, color=col, lw=1.1, label=lab)
    pred = P["p_test"].argmax(1); conf = P["p_test"].max(1)
    pts = {"argmax": pred != 0, "Zero-Trust policy": (pred != 0) | ((pred == 0) & (conf >= 0.85) & (r >= 0.70))}
    for (lab, fl), mkr in zip(pts.items(), ("o", "*")):
        ax.plot((fl & ~y).sum() / (~y).sum(), (fl & y).sum() / y.sum(), mkr, color="#c0392b", ms=7, label=lab)
    ax.set_xlim(0, 0.15); ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("false-positive rate (KDDTest+)"); ax.set_ylabel("attack recall")
    ax.legend(frameon=False, loc="lower right"); ax.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(); save(fig, out, "fig6_roc")


# ------------------------------------------------------------------ Fig. 7
def fig7(impl, out):
    sys.path.insert(0, str(impl / "revision"))
    import r06_calibration_nslkdd as r6
    d = np.load(impl / "revision/out/nslkdd_data.npz"); yv, yt = d["y_val"], d["y_test"]
    pv, pt = r6.load(0, "hybrid"); T = r6.fit_temperature(pv, yv)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH["fig7"], 3.2))
    for ax, (name, p, y) in zip(axes, [("VAL (in distribution)", pv, yv), ("KDDTest+ (shifted)", pt, yt)]):
        for lab, q, col in [("uncalibrated", p, "#d62728"), (f"T = {T:.2f}", r6.apply_temperature(p, T), "#1f77b4")]:
            conf = q.max(1); ok = q.argmax(1) == y
            edges = np.linspace(0, 1, 11); xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mm = (conf > lo) & (conf <= hi)
                if mm.sum() > 20:
                    xs.append(conf[mm].mean()); ys.append(ok[mm].mean())
            ax.plot(xs, ys, marker="o", ms=3, color=col, label=f"{lab} (ECE {r6.ece(q, y):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_title(name); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, out, "fig7_reliability")


# ------------------------------------------------------------------ Fig. 8
def fig8(impl, out):
    ps = json.loads((impl / "revision/out/tables/nslkdd_per_seed.json").read_text())
    cm = np.array(ps["0"]["confusion_zt_tuned"]); norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(WIDTH["fig8"], 3.35))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2f})", ha="center", va="center", fontsize=FS,
                    linespacing=1.0, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("decision (CHALLENGE = attack)"); ax.set_ylabel("true class")
    fig.subplots_adjust(left=0.2, right=0.99, top=0.99, bottom=0.27); save(fig, out, "fig8_confusion")


# ------------------------------------------------------------------ Fig. 9
def fig9(impl, out):
    d = pd.read_csv(impl / "revision/out/tables/dapt_day_out.csv")
    fig, ax = plt.subplots(figsize=(WIDTH["fig9"], 2.9))
    days = list(dict.fromkeys(d.held_out_day))
    lab = {"lr": "Logistic regression", "xgb": "XGBoost", "ens": "Ensemble"}
    col = {"lr": "#999", "xgb": "#5b8fc9", "ens": "#1f4e79"}
    for k, m in enumerate(("lr", "xgb", "ens")):
        sub = d[d.model == m].set_index("held_out_day").loc[days]
        lo = sub.recall_ci95.str.strip("[]").str.split(",").map(lambda v: float(v[0]))
        hi = sub.recall_ci95.str.strip("[]").str.split(",").map(lambda v: float(v[1]))
        x = np.arange(len(days)) + (k - 1) * 0.26
        ax.bar(x, sub.recall, 0.25, color=col[m], label=lab[m], yerr=[sub.recall - lo, hi - sub.recall],
               capsize=2, error_kw={"lw": 0.7})
    tick = {"Tue": "Tue: Recon.", "Wed": "Wed: Foothold, Recon.", "Thu": "Thu: Lateral", "Fri": "Fri: Exfil., Foothold"}
    ax.set_xticks(range(len(days))); ax.set_xticklabels([tick[x] for x in days])
    ax.set_ylabel("recall on held-out day"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout(); save(fig, out, "fig9_dapt")


# ------------------------------------------------------------------ Fig. 10
def fig10(impl, out):
    d = pd.read_csv(impl / "revision/out/tables/adversarial_nslkdd.csv")
    clean = d[d.attack == "clean"].iloc[0]
    cols = {"DNN (attacked)": "#c0392b", "RF (transfer)": "#7f7f7f", "XGBoost (transfer)": "#5b8fc9",
            "Ensemble (transfer)": "#1f4e79"}
    fig, ax = plt.subplots(figsize=(WIDTH["fig10"], 3.1))
    for c, colr in cols.items():
        for att, ls in (("FGSM", "--"), ("PGD-20", "-")):
            sub = d[d.attack == att]
            ax.plot([0] + sub.eps.tolist(), [clean[c]] + sub[c].tolist(), ls=ls, marker="o", ms=3, color=colr, lw=1)
    ax.set_xlabel("L∞ budget ε (standardized units)"); ax.set_ylabel("binary accuracy (KDDTest+)")
    ax.grid(lw=0.3, alpha=0.5)
    h = [Line2D([], [], color=c, lw=2, label=k) for k, c in cols.items()] + \
        [Line2D([], [], color="#333", ls="--", label="FGSM"), Line2D([], [], color="#333", ls="-", label="PGD-20")]
    ax.legend(handles=h, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); save(fig, out, "fig10_adversarial")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--impl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    info = {}
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
        r = f(a.impl, a.out)
        if r is not None:
            info[f.__name__] = r
        print("ok", f.__name__, flush=True)
    (a.out / "figure_info.json").write_text(json.dumps({"widths_in": WIDTH, **info}, indent=1))
    print(info)
