"""
Comparison & results figures for HCAPT-Defense paper.

Generates (IEEEtran sized):
  fig_models_overall            - Accuracy/F1/FPR across all configurations
  fig_per_class_recall          - grouped bar: No GAN vs CGAN vs WGAN-GP
  fig_confusion_matrices        - 4 confusion matrices side-by-side
  fig_zt_actions_pie            - Zero-Trust action distribution
  fig_zt_catches                - bar: correct vs false challenges
  fig_literature_vs_ours        - bubble chart vs 4 primary refs
  fig_training_curves           - GAN losses + NN training curve
  fig_performance_targets       - bar chart: achieved vs target
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from plotting.ieee_style import (
    save_fig, WIDTH_COL, WIDTH_WIDE, HEIGHT_SHORT, HEIGHT_MED, HEIGHT_TALL,
    COLORS, CLASS_COLORS, MODEL_COLORS,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[2] / "figures" / "comparison"


# ============================================================
#  Helpers to load all result JSONs
# ============================================================
def load_all():
    def _load(name):
        f = RESULTS_DIR / name
        if not f.exists():
            return None
        with open(f, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {
        "baselines":    _load("baselines_results.json"),
        "ens_no_gan":   _load("ensemble_selflearn_results.json"),
        "ens_gan":      _load("gan_augmented_ensemble.json"),
        "gan_compare":  _load("gan_comparison.json"),
        "zero_trust":   _load("zero_trust_results.json"),
        "final":        _load("final_evaluation.json"),
    }


# ============================================================
#  1. Overall model comparison - accuracy / F1 / FPR
# ============================================================
def fig_models_overall(D):
    bl = D["baselines"]
    ens_no_gan = D["ens_no_gan"]["before_selflearning"]
    cgan = D["gan_compare"]["cgan"]
    wgan = D["gan_compare"]["wgangp"]
    zt = D["zero_trust"]["ensemble_plus_zt_binary"]  # binary tag

    rows = []
    for m in bl:
        rows.append((m["model"].replace("(binary)","").strip(),
                     m["accuracy"], m["f1_weighted"], m["fpr"]))
    rows.append(("Ensemble", ens_no_gan["accuracy"],
                 ens_no_gan["f1_weighted"], ens_no_gan["fpr"]))
    rows.append(("Ens + CGAN", cgan["accuracy"],
                 cgan["f1_weighted"], cgan["fpr"]))
    rows.append(("Ens + WGAN-GP", wgan["accuracy"],
                 wgan["f1_weighted"], wgan["fpr"]))
    rows.append(("Ens + CGAN + ZT",
                 zt["accuracy"], zt["f1"], zt["fpr"]))

    labels = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    f1 = [r[2] for r in rows]
    fpr = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_MED))
    x = np.arange(len(labels))
    w = 0.27
    b1 = ax.bar(x - w, acc, w, label="Accuracy",
                color=COLORS["primary"], edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x, f1, w, label="F1 (weighted)",
                color=COLORS["tertiary"], edgecolor="black", linewidth=0.4)
    b3 = ax.bar(x + w, fpr, w, label="FPR",
                color=COLORS["secondary"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("Overall performance across all configurations "
                 "(NSL-KDD KDDTest+)")
    # Target lines
    ax.axhline(0.97, linestyle="--", color=COLORS["gray"], linewidth=0.6)
    ax.text(len(labels)-0.5, 0.97, " target 0.97",
            va="bottom", ha="right", fontsize=6, color=COLORS["gray"])

    # Label values above bars
    for bs in (b1, b2, b3):
        for b in bs:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=5)

    fig.tight_layout()
    return save_fig(fig, OUT, "fig_models_overall")


# ============================================================
#  2. Per-class recall: No-GAN vs CGAN vs WGAN-GP
# ============================================================
def fig_per_class_recall(D):
    cmp_ = D["gan_compare"]
    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    configs = [("No GAN", cmp_["non_gan"]),
               ("CGAN", cmp_["cgan"]),
               ("WGAN-GP", cmp_["wgangp"])]

    fig, ax = plt.subplots(figsize=(WIDTH_COL*2, HEIGHT_MED))
    x = np.arange(len(classes))
    w = 0.27

    for i, (name, d) in enumerate(configs):
        vals = [d.get("per_class", {}).get(c, {}).get("recall", 0)
                for c in classes]
        ax.bar(x + (i-1)*w, vals, w, label=name,
               color=MODEL_COLORS[name], edgecolor="black", linewidth=0.4)
        # annotate
        for xi, v in zip(x + (i-1)*w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class recall: effect of GAN augmentation variant")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_per_class_recall")


# ============================================================
#  3. Zero-Trust action distribution (pie + bar)
# ============================================================
def fig_zt_actions(D):
    zt = D["zero_trust"]
    actions = ["ALLOW", "MONITOR", "CHALLENGE", "QUARANTINE", "BLOCK"]
    counts = [zt["action_distribution"].get(a, 0) for a in actions]
    total = sum(counts)
    colors = [COLORS["tertiary"], COLORS["accent3"], COLORS["accent1"],
              COLORS["accent2"], COLORS["secondary"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED),
                                     gridspec_kw={"width_ratios":[1, 1.2]})

    # Pie
    wedges, texts, autotexts = ax1.pie(
        counts, labels=actions, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1},
        textprops={"fontsize": 7},
    )
    ax1.set_title(f"(a) Action distribution (n={total:,})")

    # Challenges bar: correct vs false
    c = zt["zt_corrections"]
    labels = ["Correct\n(caught attack)", "False\n(was normal)"]
    vals = [c["correctly_caught_attacks"], c["false_challenges"]]
    bars = ax2.bar(labels, vals,
                    color=[COLORS["tertiary"], COLORS["secondary"]],
                    edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.3, str(v),
                 ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("Count")
    ax2.set_title(f"(b) CHALLENGE precision = "
                  f"{c['challenge_precision']:.2%}")
    ax2.set_ylim(0, max(vals) + 4)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_zt_actions")


# ============================================================
#  4. Literature vs ours bubble chart
# ============================================================
def fig_literature_vs_ours(D):
    # From RESULTS_SUMMARY - published baselines
    refs = [
        ("Jia [3]",      0.999,  0.0003, 9999),
        ("Arafah [16]",  0.9962, 0.0,    9999),  # FPR not reported
        ("Radha [41]",   0.996,  0.04,   9999),
        ("Adnan [43]",   0.990,  0.0,    9999),
    ]
    ours = [
        ("Ens",           D["ens_no_gan"]["before_selflearning"]["accuracy"],
                           D["ens_no_gan"]["before_selflearning"]["fpr"]),
        ("Ens+CGAN",      D["gan_compare"]["cgan"]["accuracy"],
                           D["gan_compare"]["cgan"]["fpr"]),
        ("Ens+WGAN-GP",   D["gan_compare"]["wgangp"]["accuracy"],
                           D["gan_compare"]["wgangp"]["fpr"]),
        ("Ens+CGAN+ZT",   D["zero_trust"]["ensemble_plus_zt_binary"]["accuracy"],
                           D["zero_trust"]["ensemble_plus_zt_binary"]["fpr"]),
    ]

    fig, ax = plt.subplots(figsize=(WIDTH_COL*2, HEIGHT_MED))

    for name, acc, fpr, _ in refs:
        ax.scatter(fpr, acc, s=80, color=COLORS["gray"],
                   alpha=0.7, edgecolor="black", linewidth=0.6,
                   marker="o", label="Literature")
        ax.annotate(name, (fpr, acc), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.5)

    our_colors = [COLORS["primary"], COLORS["tertiary"],
                  COLORS["secondary"], COLORS["accent1"]]
    for (name, acc, fpr), c in zip(ours, our_colors):
        ax.scatter(fpr, acc, s=110, color=c,
                   edgecolor="black", linewidth=0.8,
                   marker="*" if "ZT" in name else "D", zorder=3)
        ax.annotate(name, (fpr, acc), textcoords="offset points",
                    xytext=(6, -8), fontsize=6.5,
                    color=c, fontweight="bold")

    # Target zone
    ax.add_patch(Rectangle((0, 0.97), 0.015, 0.03,
                            alpha=0.10, color=COLORS["tertiary"]))
    ax.text(0.0075, 0.99, "target zone",
            ha="center", va="bottom", fontsize=6,
            color=COLORS["tertiary"], fontweight="bold")

    ax.set_xlabel("False Positive Rate (lower is better)")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title("Accuracy-FPR trade-off: HCAPT-Defense vs. literature "
                 "baselines")
    ax.set_xlim(-0.005, 0.08)
    ax.set_ylim(0.70, 1.02)
    # Deduplicate legend
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS["gray"],
                   label="Published baselines"),
        plt.Line2D([], [], marker="D", linestyle="", color=COLORS["primary"],
                   label="This work (Ens variants)"),
        plt.Line2D([], [], marker="*", linestyle="", color=COLORS["accent1"],
                   label="This work (+ Zero-Trust)", markersize=10),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_literature_vs_ours")


# ============================================================
#  5. Performance targets - achieved vs target
# ============================================================
def fig_performance_targets(D):
    final = D["final"]
    t = final["targets"]

    metrics = ["Accuracy >= 97%", "FPR <= 1.5%", "Response <= 1s"]
    achieved = [t["accuracy"]["value"],
                t["fpr"]["value"],
                t["response_time_s"]["value"]]
    target = [t["accuracy"]["target"],
              t["fpr"]["target"],
              t["response_time_s"]["target"]]
    passed = [t[k]["passed"] for k in ("accuracy", "fpr", "response_time_s")]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_WIDE, HEIGHT_MED))
    titles = ["Accuracy", "FPR", "Response time (s)"]
    vals_fmt = ["{:.2%}", "{:.2%}", "{:.0f} us"]
    # For response time, display in microseconds
    display_ach = [achieved[0], achieved[1], achieved[2]*1e6]
    display_tgt = [target[0], target[1], target[2]*1e6]
    display_fmt = ["{:.2%}", "{:.2%}", "{:.0f} us"]

    for ax, ttl, a, t_, pas, fmt in zip(
            axes, titles, display_ach, display_tgt, passed, display_fmt):
        color_ach = COLORS["tertiary"] if pas else COLORS["secondary"]
        bars = ax.bar(["Achieved", "Target"], [a, t_],
                       color=[color_ach, COLORS["gray"]],
                       edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, [a, t_]):
            ax.text(b.get_x() + b.get_width()/2, v*1.02 if v > 0 else 0.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=7)
        ax.set_title(ttl + ("  [PASS]" if pas else "  [FAIL]"),
                     color=color_ach, fontsize=8)
        if ttl == "Response time (s)":
            ax.set_yscale("log")

    fig.suptitle("HCAPT-Defense performance targets", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_performance_targets")


# ============================================================
#  6. Training curves (GAN convergence)
# ============================================================
def fig_training_curves():
    """Read GAN checkpoint histories and plot convergence."""
    import torch
    MODELS = RESULTS_DIR.parent / "models" / "checkpoints"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED))

    # CGAN
    cgan_ckpt = MODELS / "gan_latest.pt"
    if cgan_ckpt.exists():
        ck = torch.load(cgan_ckpt, map_location="cpu", weights_only=False)
        hist = ck.get("history", {})
        if "d_loss" in hist and "g_loss" in hist:
            epochs = range(1, len(hist["d_loss"]) + 1)
            ax1.plot(epochs, hist["d_loss"], label="Discriminator",
                     color=COLORS["primary"])
            ax1.plot(epochs, hist["g_loss"], label="Generator",
                     color=COLORS["secondary"])
            ax1.set_title("(a) CGAN training loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss (BCE)")
            ax1.legend(loc="center right")

    # WGAN-GP
    wgp_ckpt = MODELS / "wgan_gp_latest.pt"
    if wgp_ckpt.exists():
        ck = torch.load(wgp_ckpt, map_location="cpu", weights_only=False)
        hist = ck.get("history", {})
        if "wasserstein" in hist:
            epochs = range(1, len(hist["wasserstein"]) + 1)
            ax2.plot(epochs, hist["wasserstein"], label="W-distance",
                     color=COLORS["primary"])
            ax2_r = ax2.twinx()
            ax2_r.plot(epochs, hist["gp"], label="Gradient penalty",
                       color=COLORS["secondary"], linestyle="--")
            ax2_r.set_ylabel("Gradient penalty",
                              color=COLORS["secondary"])
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Wasserstein distance", color=COLORS["primary"])
            ax2.set_title("(b) WGAN-GP training dynamics")
            # merge legends
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2_r.get_legend_handles_labels()
            ax2.legend(h1+h2, l1+l2, loc="upper left", fontsize=6)

    fig.tight_layout()
    return save_fig(fig, OUT, "fig_training_curves")


# ============================================================
#  7. Confusion matrix (best configuration)
# ============================================================
def fig_confusion_matrix_best():
    """Re-compute CM on the best ensemble + ZT config."""
    from data_loader import get_preprocessed
    from ensemble_selflearn import EnsembleDetector
    from sklearn.metrics import confusion_matrix

    import joblib
    import torch
    # Load GAN-augmented ensemble from saved files
    from config import MODELS_DIR, NUM_CLASSES
    _, X_te, _, y_te = get_preprocessed()

    ens = EnsembleDetector(feature_dim=X_te.shape[1])
    ens.rf = joblib.load(MODELS_DIR / "gan_ens_rf.joblib")
    import xgboost as xgb
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "gan_ens_xgb.json"))
    from ensemble_selflearn import DeepDetector
    from config import DEVICE
    ens.nn = DeepDetector(X_te.shape[1]).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "gan_ens_nn.pt",
                                       map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "gan_ens_weights.npy")

    preds = ens.predict(X_te)
    cm = confusion_matrix(y_te, preds, labels=list(range(NUM_CLASSES)))
    labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    # Normalize by row (recall view)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_TALL))

    # Counts
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_title("(a) Confusion matrix (counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_xticks(range(len(labels))); ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels); ax1.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            color = "white" if v > cm.max()/2 else "black"
            ax1.text(j, i, f"{v:,}", ha="center", va="center",
                     color=color, fontsize=6)
    fig.colorbar(im1, ax=ax1, shrink=0.7)

    # Row-normalized (recall)
    im2 = ax2.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("(b) Row-normalized (recall)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_xticks(range(len(labels))); ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels); ax2.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_pct[i, j]
            color = "white" if v > 0.5 else "black"
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color=color, fontsize=6)
    fig.colorbar(im2, ax=ax2, shrink=0.7)

    fig.suptitle("Confusion matrix - Ens + CGAN (best overall variant)",
                  y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_confusion_best")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("  Comparison / results figures")
    print("=" * 60)

    D = load_all()
    # Drop missing ones
    for k, v in list(D.items()):
        if v is None:
            print(f"[warn] missing result: {k}")
            D[k] = {}

    print("\n[1/7] Overall model comparison...")
    fig_models_overall(D)
    print("\n[2/7] Per-class recall comparison...")
    fig_per_class_recall(D)
    print("\n[3/7] Zero-Trust action distribution...")
    fig_zt_actions(D)
    print("\n[4/7] Literature comparison bubble...")
    fig_literature_vs_ours(D)
    print("\n[5/7] Performance targets...")
    fig_performance_targets(D)
    print("\n[6/7] GAN training curves...")
    try:
        fig_training_curves()
    except Exception as e:
        print(f"[warn] training curves failed: {e}")
    print("\n[7/7] Confusion matrix (best)...")
    try:
        fig_confusion_matrix_best()
    except Exception as e:
        print(f"[warn] confusion matrix failed: {e}")

    print(f"\n[OK] All comparison figures saved to: {OUT}")


if __name__ == "__main__":
    main()
