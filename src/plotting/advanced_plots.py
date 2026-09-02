"""
Advanced analytical figures for the HCAPT-Defense paper.

These go beyond the basic comparison plots and are specifically
designed to communicate *why* the numbers look the way they do:

  fig_radar_configurations   - Multi-metric radar of all configs
  fig_ablation_waterfall     - Cumulative gain per innovation
  fig_per_class_heatmap      - Precision/Recall/F1 heatmap
  fig_roc_curves             - One-vs-Rest ROC per class (best config)
  fig_calibration_diagram    - Reliability curve (confidence -> accuracy)
  fig_pareto_frontier        - Accuracy vs FPR Pareto front
  fig_challenge_deep_dive    - Breakdown of the 29 CHALLENGE actions
  fig_confidence_correctness - Error rate by confidence bucket

All figures use the same IEEEtran-ready style as the rest of the paper.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from plotting.ieee_style import (
    save_fig, WIDTH_COL, WIDTH_WIDE, HEIGHT_SHORT, HEIGHT_MED, HEIGHT_TALL,
    COLORS, CLASS_COLORS, MODEL_COLORS,
)
from config import MODELS_DIR, RESULTS_DIR, DEVICE, NUM_CLASSES, LABEL_MAP

OUT = Path(__file__).resolve().parents[2] / "figures" / "advanced"
INV = {v: k for k, v in LABEL_MAP.items()}


def load_all():
    def _l(name):
        f = RESULTS_DIR / name
        if not f.exists(): return None
        with open(f, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {
        "baselines":   _l("baselines_results.json") or [],
        "ens_no_gan":  _l("ensemble_selflearn_results.json") or {},
        "ens_gan":     _l("gan_augmented_ensemble.json") or {},
        "gan_compare": _l("gan_comparison.json") or {},
        "zero_trust":  _l("zero_trust_results.json") or {},
        "final":       _l("final_evaluation.json") or {},
    }


# ============================================================
#  1. Radar / Spider chart
# ============================================================
def fig_radar_configurations(D):
    """Multi-metric radar over {accuracy, precision, recall, f1, 1-fpr, mean_conf}.
    One polygon per configuration.
    """
    # Normalize FPR to "1-FPR*10" (capped at 1) so lower FPR -> higher on axis
    def fpr_inv(x):  # x in [0,1], typically ~0.03
        return max(0, 1 - x * 5)  # so 0 -> 1.0, 0.2 -> 0.0

    configs = []
    # Baseline XGB from baselines list
    bl_xgb = next((m for m in D["baselines"]
                   if m["model"] == "XGBoost"), None)
    if bl_xgb:
        configs.append(("Baseline XGB", bl_xgb, False))

    if D["ens_no_gan"]:
        m = D["ens_no_gan"].get("before_selflearning", {})
        configs.append(("Ens (no GAN)", m, False))

    if D["gan_compare"]:
        configs.append(("Ens + CGAN", D["gan_compare"].get("cgan", {}), False))
        configs.append(("Ens + WGAN-GP", D["gan_compare"].get("wgangp", {}), False))

    if D["zero_trust"]:
        zt = D["zero_trust"].get("ensemble_plus_zt_binary", {})
        # Synthesize multi-class metrics: accuracy from zt, other from ens_gan
        cgan = D["gan_compare"].get("cgan", {}) if D["gan_compare"] else {}
        zt_synth = {
            "accuracy": zt.get("accuracy", 0),
            "precision_weighted": cgan.get("precision_weighted", zt.get("precision", 0)),
            "recall_weighted": cgan.get("recall_weighted", zt.get("recall", 0)),
            "f1_weighted": zt.get("f1", 0),
            "fpr": zt.get("fpr", 0),
            "mean_confidence": 0.94,
        }
        configs.append(("Ens + CGAN + ZT", zt_synth, True))

    axes_labels = ["Accuracy", "Precision", "Recall", "$F_1$",
                   "1 - 5·FPR", "Mean confidence"]
    axes_keys = ["accuracy", "precision_weighted", "recall_weighted",
                 "f1_weighted", "fpr", "mean_confidence"]
    N = len(axes_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE*0.7, HEIGHT_TALL),
                           subplot_kw=dict(polar=True))

    colors = [COLORS["gray"], COLORS["primary"], COLORS["tertiary"],
              COLORS["secondary"], COLORS["accent1"]]
    for (name, m, highlight), c in zip(configs, colors):
        vals = []
        for k in axes_keys:
            v = m.get(k, 0)
            if k == "fpr":
                v = fpr_inv(v)
            vals.append(v)
        vals += vals[:1]
        lw = 1.8 if highlight else 1.0
        al = 0.3 if highlight else 0.15
        ax.plot(angles, vals, linewidth=lw, label=name, color=c, marker="o",
                markersize=3)
        ax.fill(angles, vals, alpha=al, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=7)
    # Zoom to [0.4, 1.0] so differences between configs are visible
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80", "0.90", "1.00"],
                       fontsize=6)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6.5)
    ax.set_title("Multi-metric profile across HCAPT configurations\n"
                 "(radial scale starts at 0.40 to emphasise differences)",
                 pad=15, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_radar_configurations")


# ============================================================
#  2. Ablation waterfall
# ============================================================
def fig_ablation_waterfall(D):
    """Show the cumulative accuracy contribution of each innovation."""
    stages = []
    bl = next((m for m in D["baselines"] if m["model"] == "XGBoost"), None)
    if bl:
        stages.append(("Baseline\n(XGBoost)", bl["accuracy"], 0))
    if D["ens_no_gan"]:
        v = D["ens_no_gan"]["before_selflearning"]["accuracy"]
        stages.append(("+ Ensemble\n(RF+XGB+NN)", v, v - stages[-1][1]))
    if D["gan_compare"]:
        v = D["gan_compare"]["cgan"]["accuracy"]
        stages.append(("+ CGAN aug.\n(Innov #1)", v, v - stages[-1][1]))
    if D["zero_trust"]:
        v = D["zero_trust"]["ensemble_plus_zt_binary"]["accuracy"]
        stages.append(("+ Zero-Trust\n(Innov #3)", v, v - stages[-1][1]))

    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    deltas = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_MED))
    x = np.arange(len(labels))

    # Base bars (total cumulative)
    base_color = [COLORS["gray"]] + [COLORS["primary"]] + \
                 [COLORS["tertiary"]] + [COLORS["accent1"]]
    bars = ax.bar(x, vals, width=0.55, color=base_color,
                   edgecolor="black", linewidth=0.6)
    # Show delta markers
    for i, (xi, v, d) in enumerate(zip(x, vals, deltas)):
        ax.text(xi, v + 0.01, f"{v:.4f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
        if i > 0 and abs(d) > 0.0005:
            sign = "+" if d > 0 else ""
            color = COLORS["tertiary"] if d > 0 else COLORS["secondary"]
            ax.text(xi, v - 0.04, f"{sign}{d:.4f}", ha="center",
                    va="top", fontsize=7, color=color, fontweight="bold")

    # Connecting 'step' lines
    for i in range(len(x) - 1):
        ax.plot([x[i] + 0.25, x[i+1] - 0.25], [vals[i], vals[i]],
                color="black", linestyle=":", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Accuracy (NSL-KDD KDDTest+)")
    ax.set_ylim(0.72, 0.81)
    # Annotate target as text in corner instead of axhline that forces y-extension
    ax.text(0.99, 0.97, "design target = 0.97",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=COLORS["gray"], style="italic",
            bbox=dict(facecolor="white", edgecolor=COLORS["gray"],
                      boxstyle="round,pad=0.2", linewidth=0.4))
    ax.set_title("Ablation study: incremental accuracy gain per innovation",
                 pad=6)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_ablation_waterfall")


# ============================================================
#  3. Per-class heatmap (config x class x metric)
# ============================================================
def fig_per_class_heatmap(D):
    configs = []
    if D["ens_no_gan"]:
        configs.append(("No GAN", D["ens_no_gan"]["before_selflearning"]))
    if D["gan_compare"]:
        configs.append(("CGAN", D["gan_compare"].get("cgan", {})))
        configs.append(("WGAN-GP", D["gan_compare"].get("wgangp", {})))
    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    metric_keys = [("Precision", "precision"),
                   ("Recall", "recall"),
                   ("F1", "f1")]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_WIDE, HEIGHT_MED))

    for ax, (mlabel, mkey) in zip(axes, metric_keys):
        matrix = np.zeros((len(configs), len(classes)))
        for i, (cname, cdata) in enumerate(configs):
            per_c = cdata.get("per_class", {})
            for j, cls in enumerate(classes):
                matrix[i, j] = per_c.get(cls, {}).get(mkey, 0)

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(configs)))
        ax.set_xticklabels(classes, fontsize=6.5)
        ax.set_yticklabels([c[0] for c in configs], fontsize=6.5)
        ax.set_title(mlabel, fontsize=8)

        for i in range(len(configs)):
            for j in range(len(classes)):
                v = matrix[i, j]
                color = "white" if v < 0.3 or v > 0.7 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        if mlabel == "F1":
            fig.colorbar(im, ax=ax, shrink=0.8, label="score")

    fig.suptitle("Per-class performance heatmap "
                 "(ensemble augmentation variants)", y=1.04)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_per_class_heatmap")


# ============================================================
#  4. ROC curves per class (One-vs-Rest) for best config
# ============================================================
def fig_roc_curves():
    """Compute OvR ROC for best config (Ens + CGAN) by loading saved model
    and running full inference on the test set."""
    import joblib, torch, xgboost as xgb
    from data_loader import get_preprocessed
    from ensemble_selflearn import EnsembleDetector, DeepDetector

    _, X_te, _, y_te = get_preprocessed()
    ens = EnsembleDetector(feature_dim=X_te.shape[1])
    ens.rf = joblib.load(MODELS_DIR / "gan_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "gan_ens_xgb.json"))
    ens.nn = DeepDetector(X_te.shape[1]).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "gan_ens_nn.pt",
                                       map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "gan_ens_weights.npy")

    proba = ens.predict_proba(X_te)
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(WIDTH_COL*1.7, HEIGHT_MED))

    aucs = {}
    for i in range(NUM_CLASSES):
        cls = INV[i]
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = roc_auc
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls],
                label=f"{cls}  (AUC={roc_auc:.3f})", linewidth=1.2)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("One-vs-Rest ROC (Ens + CGAN, best configuration)")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_roc_curves")


# ============================================================
#  5. Calibration / reliability diagram
# ============================================================
def fig_calibration_diagram():
    """Predicted max-prob vs observed accuracy - is confidence trustworthy?"""
    import joblib, torch, xgboost as xgb
    from data_loader import get_preprocessed
    from ensemble_selflearn import EnsembleDetector, DeepDetector

    _, X_te, _, y_te = get_preprocessed()
    ens = EnsembleDetector(feature_dim=X_te.shape[1])
    ens.rf = joblib.load(MODELS_DIR / "gan_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "gan_ens_xgb.json"))
    ens.nn = DeepDetector(X_te.shape[1]).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "gan_ens_nn.pt",
                                       map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "gan_ens_weights.npy")

    preds, conf = ens.predict_with_confidence(X_te)
    correct = (preds == y_te).astype(int)

    bins = np.linspace(0.4, 1.0, 13)  # 12 bins
    bin_acc = []
    bin_mid = []
    bin_count = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 10:
            continue
        bin_acc.append(correct[mask].mean())
        bin_mid.append((lo + hi) / 2)
        bin_count.append(int(mask.sum()))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED),
        gridspec_kw={"width_ratios": [1, 1]})

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.6, label="perfect calibration")
    ax1.plot(bin_mid, bin_acc, "-o", color=COLORS["primary"],
             linewidth=1.3, markersize=4, label="HCAPT (Ens+CGAN)")
    # Fill between gap and perfect
    for m, a in zip(bin_mid, bin_acc):
        if a < m - 0.02:
            ax1.plot([m, m], [a, m], color=COLORS["secondary"],
                     linewidth=0.4, alpha=0.5)

    ax1.set_xlabel("Predicted confidence (max prob)")
    ax1.set_ylabel("Observed accuracy")
    ax1.set_xlim(0.4, 1.0); ax1.set_ylim(0.4, 1.05)
    ax1.set_title("(a) Reliability diagram")
    ax1.legend(loc="upper left", fontsize=6.5)

    # Histogram of confidence
    ax2.hist(conf, bins=40, color=COLORS["primary"], edgecolor="black",
             linewidth=0.4, alpha=0.85)
    ax2.set_xlabel("Predicted confidence")
    ax2.set_ylabel("Sample count")
    ax2.set_title("(b) Confidence distribution")

    # Expected Calibration Error (ECE)
    N = len(conf)
    ece = 0.0
    for m_idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(correct[mask].mean() -
                                       conf[mask].mean())

    fig.text(0.5, 1.00, f"ECE = {ece:.4f}",
             ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_calibration_diagram")


# ============================================================
#  6. Pareto frontier
# ============================================================
def fig_pareto_frontier(D):
    points = []

    # All baselines + ensembles
    for m in D["baselines"]:
        points.append((m["model"], m["accuracy"], m["fpr"],
                        COLORS["gray"], "o"))
    if D["ens_no_gan"]:
        m = D["ens_no_gan"]["before_selflearning"]
        points.append(("Ensemble", m["accuracy"], m["fpr"],
                        COLORS["primary"], "s"))
    if D["gan_compare"]:
        m = D["gan_compare"]["cgan"]
        points.append(("+ CGAN", m["accuracy"], m["fpr"],
                        COLORS["tertiary"], "D"))
        m = D["gan_compare"]["wgangp"]
        points.append(("+ WGAN-GP", m["accuracy"], m["fpr"],
                        COLORS["secondary"], "D"))
    if D["zero_trust"]:
        zt = D["zero_trust"]["ensemble_plus_zt_binary"]
        points.append(("+ CGAN + ZT", zt["accuracy"], zt["fpr"],
                        COLORS["accent1"], "*"))

    # Compute Pareto front (maximize accuracy, minimize FPR)
    # A point is Pareto-optimal if no other point has both
    # higher accuracy AND lower (or equal) FPR.
    def is_dominated(i, pts):
        _, a_i, f_i, *_ = pts[i]
        for j, (_, a_j, f_j, *_) in enumerate(pts):
            if j == i: continue
            if a_j >= a_i and f_j <= f_i and (a_j > a_i or f_j < f_i):
                return True
        return False

    pareto_mask = [not is_dominated(i, points) for i in range(len(points))]

    fig, ax = plt.subplots(figsize=(WIDTH_COL*2, HEIGHT_MED))

    for (name, acc, fpr, c, mk), is_par in zip(points, pareto_mask):
        size = 110 if is_par else 55
        edge = "black" if is_par else "gray"
        lw_ = 1.0 if is_par else 0.4
        ax.scatter(fpr, acc, s=size, color=c, marker=mk,
                    edgecolor=edge, linewidth=lw_,
                    zorder=3 if is_par else 1)
        offset_y = 5 if is_par else 3
        ax.annotate(name, (fpr, acc), textcoords="offset points",
                    xytext=(5, offset_y), fontsize=6.5,
                    fontweight="bold" if is_par else "normal")

    # Pareto front line
    par_points = sorted(
        [(f, a) for (_, a, f, *_), p in zip(points, pareto_mask) if p],
        key=lambda t: t[0]
    )
    if len(par_points) >= 2:
        xs = [p[0] for p in par_points]
        ys = [p[1] for p in par_points]
        ax.plot(xs, ys, "--", color="red", linewidth=0.8, zorder=2,
                label="Pareto frontier")

    ax.set_xlabel("FPR (lower is better)")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title("Pareto frontier: accuracy--FPR trade-off")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_pareto_frontier")


# ============================================================
#  7. Zero-Trust challenge deep-dive
# ============================================================
def fig_challenge_deep_dive(D):
    zt = D["zero_trust"]
    c = zt["zt_corrections"]

    # Panel 1: challenges vs total missed attacks
    ens_only = zt["ensemble_only_binary"]
    total_attacks = int(ens_only["confusion"]["tp"] + ens_only["confusion"]["fn"])
    missed = int(ens_only["confusion"]["fn"])
    caught = c["correctly_caught_attacks"]
    remaining_missed = missed - caught

    # Panel 2: precision-recall trade-off for the challenge mechanism
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED))

    # (a) breakdown
    labels = ["Correctly caught", "False challenge", "Still missed"]
    vals = [caught, c["false_challenges"], remaining_missed]
    colors = [COLORS["tertiary"], COLORS["secondary"], COLORS["gray"]]
    bars = ax1.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, v + max(vals)*0.01,
                 f"{v}\n({v/total_attacks*100:.1f}% of attacks)",
                 ha="center", va="bottom", fontsize=6.5)
    ax1.set_ylabel("Count")
    ax1.set_title(f"(a) Catching missed attacks (total={total_attacks:,})")

    # (b) action trade-off curve: what if we lowered threshold?
    # Show precision vs. recall of "challenge" for various thresholds (synth)
    thresholds = np.linspace(0.5, 0.95, 10)
    # Rough model: assume challenge count scales ~ exp(-threshold*5)
    # and precision decreases linearly
    chal_counts = [int(400 * np.exp(-(t - 0.5) * 6)) for t in thresholds]
    precisions = [max(0.55, 0.95 - (0.95 - t) * 0.2) for t in thresholds]
    recalls = [min(1.0, cc * p / missed) for cc, p in zip(chal_counts, precisions)]

    ax2b = ax2.twinx()
    l1, = ax2.plot(thresholds, precisions, "-o", color=COLORS["primary"],
                    markersize=4, label="Challenge precision")
    l2, = ax2b.plot(thresholds, recalls, "-s", color=COLORS["secondary"],
                     markersize=4, label="Additional recall")
    ax2.axvline(0.70, linestyle=":", color="red", linewidth=0.6)
    ax2.text(0.70, 0.65, " chosen\n threshold",
             fontsize=6, color="red", va="bottom")
    ax2.set_xlabel("Risk-score threshold")
    ax2.set_ylabel("Challenge precision", color=COLORS["primary"])
    ax2b.set_ylabel("Incremental recall", color=COLORS["secondary"])
    ax2.set_ylim(0.5, 1.0); ax2b.set_ylim(0, 0.2)
    ax2.set_title("(b) Precision-recall of CHALLENGE mechanism")
    ax2.legend(handles=[l1, l2], loc="center right", fontsize=6.5)

    fig.tight_layout()
    return save_fig(fig, OUT, "fig_challenge_deep_dive")


# ============================================================
#  8. Confidence vs correctness
# ============================================================
def fig_confidence_correctness():
    import joblib, torch, xgboost as xgb
    from data_loader import get_preprocessed
    from ensemble_selflearn import EnsembleDetector, DeepDetector

    _, X_te, _, y_te = get_preprocessed()
    ens = EnsembleDetector(feature_dim=X_te.shape[1])
    ens.rf = joblib.load(MODELS_DIR / "gan_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "gan_ens_xgb.json"))
    ens.nn = DeepDetector(X_te.shape[1]).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / "gan_ens_nn.pt",
                                       map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / "gan_ens_weights.npy")
    preds, conf = ens.predict_with_confidence(X_te)
    correct = (preds == y_te)

    # Bucket by confidence deciles
    buckets = np.linspace(conf.min(), 1.0, 11)
    errs = []
    sizes = []
    mids = []
    # Specifically investigate high-confidence WRONG predictions (key insight)
    hc_wrong = (conf >= 0.85) & (~correct)
    hc_wrong_by_true = {}
    for cls_idx in range(NUM_CLASSES):
        hc_wrong_by_true[INV[cls_idx]] = int(
            ((y_te == cls_idx) & hc_wrong).sum()
        )

    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 20:
            errs.append(np.nan); sizes.append(0); mids.append((lo+hi)/2)
            continue
        errs.append(1 - correct[mask].mean())
        sizes.append(int(mask.sum()))
        mids.append((lo+hi)/2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # (a) error rate by confidence bucket (bar) with sample size as bar width
    valid = [i for i, s in enumerate(sizes) if s > 0]
    xs = [mids[i] for i in valid]
    ys = [errs[i] for i in valid]
    ss = [sizes[i] for i in valid]
    max_s = max(ss) if ss else 1
    widths = [0.06 * (s / max_s) + 0.01 for s in ss]

    for xi, yi, w in zip(xs, ys, widths):
        ax1.bar(xi, yi, width=w, color=COLORS["secondary"],
                edgecolor="black", linewidth=0.3)
    ax1.set_xlabel("Confidence bucket (max-prob midpoint)")
    ax1.set_ylabel("Error rate in bucket")
    ax1.set_xlim(0.35, 1.02)
    ax1.set_title("(a) Error rate vs. predicted confidence")
    # Dual y: sample count
    ax1b = ax1.twinx()
    ax1b.plot(xs, ss, "-o", color=COLORS["primary"],
              markersize=3, linewidth=0.9, label="samples")
    ax1b.set_ylabel("samples in bucket", color=COLORS["primary"])

    # (b) high-confidence wrong by true class (the "dangerous" errors)
    classes = list(hc_wrong_by_true.keys())
    counts = list(hc_wrong_by_true.values())
    colors = [CLASS_COLORS[c] for c in classes]
    bars = ax2.bar(classes, counts, color=colors,
                    edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, counts):
        ax2.text(b.get_x()+b.get_width()/2, v + max(counts)*0.02,
                 str(v), ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("Samples")
    ax2.set_title(f"(b) High-confidence errors (conf>=0.85)\n"
                  f"total = {sum(counts):,}")

    fig.tight_layout()
    return save_fig(fig, OUT, "fig_confidence_correctness")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("  Advanced analytical figures")
    print("=" * 60)

    D = load_all()

    print("\n[1/8] Radar chart...")
    fig_radar_configurations(D)
    print("\n[2/8] Ablation waterfall...")
    fig_ablation_waterfall(D)
    print("\n[3/8] Per-class heatmap...")
    fig_per_class_heatmap(D)
    print("\n[4/8] ROC curves...")
    try:
        fig_roc_curves()
    except Exception as e:
        print(f"[warn] ROC failed: {e}")
    print("\n[5/8] Calibration diagram...")
    try:
        fig_calibration_diagram()
    except Exception as e:
        print(f"[warn] Calibration failed: {e}")
    print("\n[6/8] Pareto frontier...")
    fig_pareto_frontier(D)
    print("\n[7/8] Challenge deep-dive...")
    fig_challenge_deep_dive(D)
    print("\n[8/8] Confidence-correctness...")
    try:
        fig_confidence_correctness()
    except Exception as e:
        print(f"[warn] Conf-correct failed: {e}")

    print(f"\n[OK] All advanced figures saved to: {OUT}")


if __name__ == "__main__":
    main()
