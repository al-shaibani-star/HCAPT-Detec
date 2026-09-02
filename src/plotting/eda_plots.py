"""
Exploratory Data Analysis figures for HCAPT-Defense paper.

Generates:
  fig_class_distribution        - train/test bar chart + imbalance ratio
  fig_feature_stats             - key numerical features (top-variance)
  fig_categorical_dist          - protocol_type / service / flag
  fig_correlation_heatmap       - Pearson correlation matrix
  fig_train_test_shift          - t-SNE projection showing distribution shift
  fig_class_support_table       - readable data-table image
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

# Allow running from src/
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from plotting.ieee_style import (
    save_fig, WIDTH_COL, WIDTH_WIDE, HEIGHT_SHORT, HEIGHT_MED, HEIGHT_TALL,
    COLORS, CLASS_COLORS,
)
from data_loader import load_raw, map_attack_category
from config import ATTACK_CATEGORIES, LABEL_MAP, NSL_KDD_COLUMNS, CATEGORICAL_COLS

OUT = Path(__file__).resolve().parents[2] / "figures" / "eda"


# ============================================================
#  1. Class distribution (train vs test)
# ============================================================
def fig_class_distribution(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    tr_counts = df_tr["attack_category"].value_counts().reindex(order).fillna(0)
    te_counts = df_te["attack_category"].value_counts().reindex(order).fillna(0)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    # --- Left: grouped bar chart ---
    x = np.arange(len(order))
    w = 0.38
    b1 = ax1.bar(x - w/2, tr_counts.values, w,
                 label="KDDTrain+", color=COLORS["primary"],
                 edgecolor="black", linewidth=0.4)
    b2 = ax1.bar(x + w/2, te_counts.values, w,
                 label="KDDTest+", color=COLORS["secondary"],
                 edgecolor="black", linewidth=0.4)

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    ax1.set_ylabel("Number of samples (log scale)")
    ax1.set_xlabel("Attack category")
    ax1.set_title("(a) Class distribution (log scale)")
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", which="both", alpha=0.3)

    # annotate counts above bars
    for b, v in zip(b1, tr_counts.values):
        ax1.text(b.get_x() + b.get_width()/2, v*1.1, f"{int(v):,}",
                 ha="center", va="bottom", fontsize=6)
    for b, v in zip(b2, te_counts.values):
        ax1.text(b.get_x() + b.get_width()/2, v*1.1, f"{int(v):,}",
                 ha="center", va="bottom", fontsize=6)

    # --- Right: imbalance ratio ---
    max_tr = tr_counts.max()
    ratios = max_tr / tr_counts.replace(0, np.nan)
    colors = [CLASS_COLORS[c] for c in order]
    ax2.barh(order[::-1], ratios.values[::-1], color=colors[::-1],
             edgecolor="black", linewidth=0.4)
    ax2.set_xscale("log")
    ax2.set_xlabel("Imbalance ratio (train)")
    ax2.set_title("(b) Imbalance vs. Normal")
    for i, r in enumerate(ratios.values[::-1]):
        if not np.isnan(r):
            ax2.text(r*1.2, i, f"{r:.0f}:1", va="center", fontsize=6)
    ax2.grid(True, axis="x", which="both", alpha=0.3)

    fig.tight_layout()
    return save_fig(fig, OUT, "fig_class_distribution")


# ============================================================
#  2. Top-variance numerical features distribution
# ============================================================
def fig_feature_distributions(df_tr: pd.DataFrame):
    num_cols = [c for c in df_tr.columns
                if c not in CATEGORICAL_COLS + ["label", "attack_category",
                                                "y", "difficulty"]]
    # Top 6 by variance
    variances = df_tr[num_cols].var().sort_values(ascending=False)
    top = variances.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_WIDE, HEIGHT_TALL))
    axes = axes.flatten()

    for ax, col in zip(axes, top):
        for cls in ("Normal", "DoS", "Probe", "R2L", "U2R"):
            mask = df_tr["attack_category"] == cls
            vals = df_tr.loc[mask, col]
            if len(vals) < 10:
                continue
            # Clip extreme outliers for visibility
            q99 = np.percentile(vals, 99)
            vals = vals[vals <= q99]
            if len(vals) < 10:
                continue
            ax.hist(vals, bins=40, alpha=0.4, label=cls,
                    color=CLASS_COLORS[cls], edgecolor="none", density=True)
        ax.set_title(col, fontsize=8)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=6)

    # Single legend
    handles = [Patch(color=CLASS_COLORS[c], label=c)
               for c in ["Normal", "DoS", "Probe", "R2L", "U2R"]]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Top-variance feature distributions by class "
                 "(density, log Y)", y=1.06)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_feature_distributions")


# ============================================================
#  3. Categorical feature distribution
# ============================================================
def fig_categorical_distribution(df_tr: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_WIDE, HEIGHT_MED))

    for ax, col in zip(axes, CATEGORICAL_COLS):
        top_n = 10
        counts = df_tr[col].value_counts().head(top_n)
        # Overall bar for context
        ax.barh(counts.index[::-1], counts.values[::-1],
                color=COLORS["primary"], edgecolor="black", linewidth=0.4)
        ax.set_title(f"{col}\n(top {top_n} of {df_tr[col].nunique()})")
        ax.set_xscale("log")
        ax.grid(True, axis="x", which="both", alpha=0.3)
        ax.tick_params(axis="y", labelsize=6)

    fig.suptitle("Categorical feature distribution (train)", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_categorical_distribution")


# ============================================================
#  4. Correlation heatmap (numerical features only)
# ============================================================
def fig_correlation_heatmap(df_tr: pd.DataFrame):
    num_cols = [c for c in df_tr.columns
                if c not in CATEGORICAL_COLS + ["label", "attack_category",
                                                "y", "difficulty"]]
    # Restrict to top 20 for readability
    variances = df_tr[num_cols].var().sort_values(ascending=False)
    top = variances.head(20).index.tolist()
    corr = df_tr[top].corr().values

    fig, ax = plt.subplots(figsize=(WIDTH_COL, HEIGHT_TALL))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_xticklabels(top, rotation=90, fontsize=5)
    ax.set_yticklabels(top, fontsize=5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Pearson r")
    ax.set_title("Correlation matrix (top-20 variance features)")
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_correlation_heatmap")


# ============================================================
#  5. Train vs Test distribution shift (PCA projection)
# ============================================================
def fig_train_test_shift(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                          subsample: int = 3000):
    num_cols = [c for c in df_tr.columns
                if c not in CATEGORICAL_COLS + ["label", "attack_category",
                                                "y", "difficulty"]]
    rng = np.random.default_rng(42)
    idx_tr = rng.choice(len(df_tr), size=min(subsample, len(df_tr)),
                         replace=False)
    idx_te = rng.choice(len(df_te), size=min(subsample, len(df_te)),
                         replace=False)

    X_tr = df_tr.iloc[idx_tr][num_cols].values
    X_te = df_te.iloc[idx_te][num_cols].values
    # Normalize
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    pca = PCA(n_components=2, random_state=42)
    all_X = np.concatenate([X_tr_n, X_te_n], axis=0)
    pca.fit(X_tr_n)  # fit on train only
    Z_all = pca.transform(all_X)
    Z_tr, Z_te = Z_all[:len(X_tr_n)], Z_all[len(X_tr_n):]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED),
                                     sharex=True, sharey=True)

    # Color by class
    y_tr = df_tr.iloc[idx_tr]["attack_category"].values
    y_te = df_te.iloc[idx_te]["attack_category"].values
    for cls in ["Normal", "DoS", "Probe", "R2L", "U2R"]:
        m = y_tr == cls
        ax1.scatter(Z_tr[m, 0], Z_tr[m, 1], s=2, alpha=0.45,
                    color=CLASS_COLORS[cls], label=cls, edgecolors="none")
        m = y_te == cls
        ax2.scatter(Z_te[m, 0], Z_te[m, 1], s=2, alpha=0.45,
                    color=CLASS_COLORS[cls], label=cls, edgecolors="none")

    ax1.set_title(f"(a) KDDTrain+ projection (n={len(Z_tr)})")
    ax2.set_title(f"(b) KDDTest+ projection (n={len(Z_te)})")
    for ax in (ax1, ax2):
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(markerscale=3, loc="best", frameon=True, fontsize=6)

    var = pca.explained_variance_ratio_
    fig.suptitle(f"Distribution shift: KDDTrain+ vs KDDTest+ "
                 f"(PCA fit on train, var={var[0]+var[1]:.1%})",
                 y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_train_test_shift")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("  EDA figure generation")
    print("=" * 60)
    df_tr, df_te = load_raw()
    df_tr = map_attack_category(df_tr)
    df_te = map_attack_category(df_te)

    print("\n[1/5] Class distribution...")
    fig_class_distribution(df_tr, df_te)

    print("\n[2/5] Feature distributions...")
    fig_feature_distributions(df_tr)

    print("\n[3/5] Categorical distribution...")
    fig_categorical_distribution(df_tr)

    print("\n[4/5] Correlation heatmap...")
    fig_correlation_heatmap(df_tr)

    print("\n[5/5] Train/Test distribution shift (PCA)...")
    fig_train_test_shift(df_tr, df_te)

    print(f"\n[OK] All EDA figures saved to: {OUT}")


if __name__ == "__main__":
    main()
