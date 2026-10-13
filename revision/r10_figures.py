"""
Revision figures (IEEE style, 300 dpi) built only from revision/out/ result files.
Usage: python revision\\r10_figures.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

from rev_common import OUT_DIR, PRED_DIR, CLASS_NAMES

T, FD = OUT_DIR / "tables", OUT_DIR / "figures"
FD.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})


def box(ax, x, y, w, h, text, fc, ec="#333", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, lw=0.8, ls=ls, color="#333"))


def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 3.5, 2.2, 0.8, "KDDTrain+\n125,973 flows", "#e8e8e8", bold=True)
    box(ax, 0.1, 0.3, 2.2, 0.8, "KDDTest+\n22,544 flows\n(scored once)", "#fde2e2", bold=True)
    box(ax, 2.9, 3.9, 2.0, 0.6, "FIT (80 %)\n100,778", "#e3eefc")
    box(ax, 2.9, 3.0, 2.0, 0.6, "VAL (20 %)\n25,195", "#e6f4e6")
    arrow(ax, (2.3, 3.95), (2.9, 4.2)); arrow(ax, (2.3, 3.85), (2.9, 3.3))
    box(ax, 5.4, 3.75, 2.1, 0.75, "Encoders, scaler,\nSMOTE / GAN generators,\nRF, XGBoost, DNN, validator", "#e3eefc", fs=6.8)
    box(ax, 5.4, 2.85, 2.1, 0.75, "Ensemble weights,\nZero-Trust thresholds,\ncalibrators, alert budgets", "#e6f4e6", fs=6.8)
    arrow(ax, (4.9, 4.2), (5.4, 4.12)); arrow(ax, (4.9, 3.3), (5.4, 3.22))
    ax.text(6.45, 4.62, "fitted on", ha="center", fontsize=7, style="italic")
    box(ax, 8.0, 3.2, 1.9, 1.0, "Frozen per-sample\npredictions\n(5 seeds)", "#fff3d6", bold=True, fs=7)
    arrow(ax, (7.5, 4.1), (8.0, 3.9)); arrow(ax, (7.5, 3.22), (8.0, 3.5))
    # inference path
    y = 1.7
    box(ax, 2.9, y - 0.35, 1.5, 0.7, "x → encode\n+ scale", "#f2f2f2")
    box(ax, 4.75, y - 0.35, 1.7, 0.7, "RF + XGB + DNN\nsoft vote (ŷ, c)", "#d9f2f2")
    box(ax, 6.8, y - 0.35, 1.5, 0.7, "Validator\nrisk r (LightGBM)", "#fde9d2")
    box(ax, 8.55, y - 0.35, 1.35, 0.7, "Policy\n(5 actions)", "#f2f2f2")
    arrow(ax, (2.3, 0.7), (3.65, y - 0.35)); arrow(ax, (4.4, y), (4.75, y))
    arrow(ax, (6.45, y), (6.8, y)); arrow(ax, (8.3, y), (8.55, y))
    arrow(ax, (9.2, y - 0.35), (9.2, 0.75)); box(ax, 7.9, 0.3, 2.0, 0.45, "metrics, CIs, McNemar", "#fde2e2", fs=6.8)
    arrow(ax, (8.95, 3.2), (8.95, y + 0.35), ls="--")
    ax.text(5.6, 0.95, "test labels touch no fitted object", fontsize=7, style="italic", color="#a00")
    fig.savefig(FD / "fig_protocol.png"); plt.close(fig)


def fig_main_comparison():
    s = json.loads((T / "nslkdd_summary.json").read_text())
    ab = s["ablation"]
    names = list(ab.keys())
    acc = np.array([ab[n]["accuracy"]["mean"] for n in names]); sd = np.array([ab[n]["accuracy"]["std"] for n in names])
    f1 = np.array([ab[n]["f1"]["mean"] for n in names])
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for i, k in enumerate(order):
        c = "#c0392b" if "proposed" in names[k] else ("#1f4e79" if names[k].startswith("Ensemble") else "#555")
        ax.errorbar(acc[k], i, xerr=sd[k], fmt="o", color=c, ms=4, capsize=2, lw=0.8)
        ax.plot(f1[k], i, marker="|", color=c, ms=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order], fontsize=7)
    ax.set_xlabel("KDDTest+ binary accuracy (dot, mean ± s.d., 5 seeds) and F1 (bar)")
    ax.grid(axis="x", lw=0.3, alpha=0.6)
    fig.savefig(FD / "fig_main_comparison.png"); plt.close(fig)


def fig_fidelity():
    s = pd.read_csv(T / "learning_curve_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    style = {"smote": ("SMOTE", "#1f77b4", "-"), "cgan": ("CGAN", "#d62728", "--"), "wgan": ("WGAN-GP", "#2ca02c", "-.")}
    mk = {"DoS": "o", "Probe": "s", "R2L": "^"}
    for ax, met, lab in zip(axes, ("c2st_auc", "coverage"), ("C2ST AUC (0.5 = indistinguishable)", "coverage (1 = full support)")):
        for m, (ml, col, ls) in style.items():
            for cls in ("DoS", "Probe", "R2L"):
                d = s[(s.method == m) & (s["class"] == cls)]
                ax.plot(d.n_real, d[f"{met}_mean"], color=col, ls=ls, marker=mk[cls], ms=3, lw=0.9,
                        label=f"{ml} / {cls}")
        ax.set_xscale("log"); ax.set_xlabel("real samples of the class (n)"); ax.set_ylabel(lab)
        ax.grid(lw=0.3, alpha=0.5)
    axes[1].legend(fontsize=5.5, ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.45))
    fig.tight_layout(); fig.savefig(FD / "fig_fidelity.png"); plt.close(fig)


def fig_roc_operating_points():
    d = np.load(OUT_DIR / "nslkdd_data.npz"); y = d["y_test"] != 0
    P = np.load(PRED_DIR / "nslkdd_s0_hybrid.npz"); R = np.load(PRED_DIR / "nslkdd_s0_risk.npz")
    s = 1 - P["p_test"][:, 0]; r = R["r_test"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for sc, lab, col in ((s, "ensemble score 1−p(Normal)", "#1f4e79"), (r, "validator risk alone", "#e67e22")):
        f, t, _ = roc_curve(y, sc); ax.plot(f, t, color=col, lw=1, label=lab)
    pred = P["p_test"].argmax(1); conf = P["p_test"].max(1)
    pts = {"argmax": pred != 0,
           "Zero-Trust policy": (pred != 0) | ((pred == 0) & (conf >= 0.85) & (r >= 0.70))}
    for (lab, fl), mk in zip(pts.items(), ("o", "*")):
        ax.plot((fl & ~y).sum() / (~y).sum(), (fl & y).sum() / y.sum(), mk, color="#c0392b", ms=6, label=lab)
    ax.set_xlim(0, 0.15); ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("false-positive rate (KDDTest+)"); ax.set_ylabel("attack recall")
    ax.legend(fontsize=6, frameon=False, loc="lower right"); ax.grid(lw=0.3, alpha=0.5)
    fig.savefig(FD / "fig_roc_points.png"); plt.close(fig)


def fig_confusion():
    ps = json.loads((T / "nslkdd_per_seed.json").read_text())
    cm = np.array(ps["0"]["confusion_zt_tuned"])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2f})", ha="center", va="center", fontsize=6,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, fontsize=7); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("decision (CHALLENGE counted as attack)"); ax.set_ylabel("true class")
    fig.savefig(FD / "fig_confusion_zt.png"); plt.close(fig)


def fig_dapt():
    d = pd.read_csv(T / "dapt_day_out.csv")
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    days = list(dict.fromkeys(d.held_out_day)); models = ["lr", "xgb", "ens"]
    lab = {"lr": "Logistic regression", "xgb": "XGBoost", "ens": "Ensemble"}
    col = {"lr": "#999", "xgb": "#5b8fc9", "ens": "#1f4e79"}
    for k, m in enumerate(models):
        sub = d[d.model == m].set_index("held_out_day").loc[days]
        lo = sub.recall_ci95.str.strip("[]").str.split(",").map(lambda v: float(v[0]))
        hi = sub.recall_ci95.str.strip("[]").str.split(",").map(lambda v: float(v[1]))
        x = np.arange(len(days)) + (k - 1) * 0.26
        ax.bar(x, sub.recall, 0.25, color=col[m], label=lab[m],
               yerr=[sub.recall - lo, hi - sub.recall], capsize=2, error_kw={"lw": 0.6})
    stage = {"Tue": "Tue\nReconnaissance", "Wed": "Wed\nFoothold", "Thu": "Thu\nLateral mov.", "Fri": "Fri\nExfiltration"}
    ax.set_xticks(range(len(days))); ax.set_xticklabels([stage[x] for x in days], fontsize=7)
    ax.set_ylabel("recall on the unseen day"); ax.set_ylim(0, 1); ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.savefig(FD / "fig_dapt_dayout.png"); plt.close(fig)


def fig_adversarial():
    d = pd.read_csv(T / "adversarial_nslkdd.csv")
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    clean = d[d.attack == "clean"].iloc[0]
    cols = {"DNN (attacked)": "#c0392b", "RF (transfer)": "#7f7f7f", "XGBoost (transfer)": "#5b8fc9",
            "Ensemble (transfer)": "#1f4e79"}
    for c, colr in cols.items():
        for att, ls in (("FGSM", "--"), ("PGD-20", "-")):
            sub = d[d.attack == att]
            ax.plot([0] + sub.eps.tolist(), [clean[c]] + sub[c].tolist(), ls=ls, marker="o", ms=2.5,
                    color=colr, lw=0.9, label=f"{c}, {att}")
    ax.set_xlabel("L∞ budget ε (standardized units)"); ax.set_ylabel("binary accuracy (KDDTest+)")
    ax.legend(fontsize=5, frameon=False, ncol=1, loc="lower left"); ax.grid(lw=0.3, alpha=0.5)
    fig.savefig(FD / "fig_adversarial.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_protocol, fig_main_comparison, fig_fidelity, fig_roc_operating_points,
              fig_confusion, fig_dapt, fig_adversarial):
        f(); print("ok", f.__name__)
