"""
Figures for the unified HCAPT-Defense (FL + Blockchain) results.

Generated:
  fig_ablation_matrix           - Full ablation table as heatmap
  fig_fl_convergence            - FL round accuracy + per-client
  fig_fl_privacy_vs_utility     - DP epsilon sweep (cosine vs noise)
  fig_blockchain_growth         - Block count, IoC count, verification
  fig_incremental_gain_unified  - Waterfall: each component's contribution
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from plotting.ieee_style import (
    save_fig, WIDTH_COL, WIDTH_WIDE, HEIGHT_SHORT, HEIGHT_MED, HEIGHT_TALL,
    COLORS, CLASS_COLORS,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[2] / "figures" / "unified"


# ============================================================
#  1. Ablation matrix heatmap
# ============================================================
def fig_ablation_matrix():
    f = RESULTS_DIR / "ablation_matrix.json"
    if not f.exists():
        print(f"[skip] {f} not found"); return
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    names = [r["name"] for r in data]
    metrics = ["accuracy", "precision", "recall", "f1", "fpr"]
    mat = np.zeros((len(names), len(metrics)))
    for i, r in enumerate(data):
        for j, m in enumerate(metrics):
            mat[i, j] = r["binary"][m]
    # Invert FPR so higher = better for colouring
    mat_norm = mat.copy()
    mat_norm[:, 4] = 1 - mat_norm[:, 4]

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE*0.85, HEIGHT_TALL))
    im = ax.imshow(mat_norm, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(metrics)):
            v = mat[i, j]
            color = "white" if mat_norm[i, j] < 0.6 or mat_norm[i, j] > 0.85 \
                    else "black"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=6.5, color=color)
    fig.colorbar(im, ax=ax, shrink=0.7, label="score (FPR inverted)")
    ax.set_title("Full ablation matrix (binary detection, KDDTest+)")
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_ablation_matrix")


# ============================================================
#  2. FL convergence
# ============================================================
def fig_fl_convergence():
    f = RESULTS_DIR / "master_pipeline_summary.json"
    # Prefer master pipeline, else fall back to older fl_results
    fl_hist_file = RESULTS_DIR / "fl_blockchain_results.json"
    hist = None
    if fl_hist_file.exists():
        with open(fl_hist_file, "r", encoding="utf-8") as fp:
            hist = json.load(fp).get("fl_history", None)
    # Try master pipeline log - simpler format
    if hist is None:
        # reconstruct from the saved summary at least
        if not f.exists(): return
        with open(f, "r", encoding="utf-8") as fp:
            s = json.load(fp)
        if "phase_A3_fl" not in s: return
        # We don't have per-round data in summary; show only bar
        return _fig_fl_bar(s)

    rounds = range(1, len(hist["round_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED))

    # Panel (a): global val accuracy per round
    ax1.plot(rounds, hist["round_acc"], "-o",
             color=COLORS["primary"], linewidth=1.3, markersize=4,
             label="Global model val acc")
    ax1.set_xlabel("FL round"); ax1.set_ylabel("Validation accuracy")
    ax1.set_title("(a) FedAvg convergence")
    ax1.set_ylim(0.5, 1.02); ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=6.5)

    # Panel (b): per-client accuracy
    pc = np.array(hist["per_client_acc"])
    n_clients = pc.shape[1]
    colors = [COLORS["primary"], COLORS["tertiary"], COLORS["secondary"],
              COLORS["accent1"], COLORS["accent2"]]
    for i in range(n_clients):
        ax2.plot(rounds, pc[:, i], "-", color=colors[i % len(colors)],
                 alpha=0.75, label=f"client {i}")
    ax2.set_xlabel("FL round"); ax2.set_ylabel("Local training accuracy")
    ax2.set_title("(b) Per-client accuracy")
    ax2.set_ylim(0.5, 1.02); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=6, ncol=2)

    fig.suptitle("Federated Learning training dynamics", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_fl_convergence")


def _fig_fl_bar(summary):
    fig, ax = plt.subplots(figsize=(WIDTH_COL*1.5, HEIGHT_SHORT))
    v = summary["phase_A3_fl"]
    labels = ["Val acc\n(final round)", "Test acc\n(KDDTest+)"]
    vals = [v.get("final_val_acc", 0), v.get("test_acc", 0)]
    ax.bar(labels, vals, color=[COLORS["primary"], COLORS["secondary"]],
            edgecolor="black", linewidth=0.5)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.01, f"{val:.4f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_title("FL final-model accuracy")
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_fl_convergence")


# ============================================================
#  3. DP: privacy vs utility sweep
# ============================================================
def fig_fl_privacy_vs_utility():
    # Synthesize from differential_privacy module using the NSL-KDD scaler
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 41)).astype(np.float32)

    epsilons = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
    clip = 1.0
    cosines = []
    l2s = []
    scales = []
    for eps in epsilons:
        scale = 2.0 * clip / eps
        noise = rng.laplace(0.0, scale, X.shape)
        X_n = np.clip(X, -clip, clip) + noise
        # Cosine
        Xn_ = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Nn_ = X_n / (np.linalg.norm(X_n, axis=1, keepdims=True) + 1e-9)
        cosines.append(float((Xn_ * Nn_).sum(axis=1).mean()))
        l2s.append(float(np.linalg.norm(noise, axis=1).mean()))
        scales.append(scale)

    fig, ax1 = plt.subplots(figsize=(WIDTH_COL*2, HEIGHT_MED))
    ax2 = ax1.twinx()

    line1, = ax1.plot(epsilons, cosines, "-o",
                       color=COLORS["primary"],
                       label="cosine similarity", markersize=5)
    line2, = ax2.plot(epsilons, scales, "--s",
                       color=COLORS["secondary"],
                       label="noise scale", markersize=5)
    ax1.set_xlabel("Privacy budget  $\\varepsilon$  (lower = more private)")
    ax1.set_ylabel("cosine similarity (utility)",
                    color=COLORS["primary"])
    ax2.set_ylabel("Laplace noise scale",
                    color=COLORS["secondary"])
    ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)
    # Mark chosen epsilon
    ax1.axvline(5.0, linestyle=":", color="red", linewidth=0.8)
    ax1.text(5.0, 1.02, "  selected\n  $\\varepsilon=5$",
              color="red", fontsize=6.5, va="top")
    ax1.legend(handles=[line1, line2], loc="upper left", fontsize=6.5)
    ax1.set_title("Privacy-utility trade-off for IoC sharing (clip=1.0)")
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_fl_privacy_vs_utility")


# ============================================================
#  4. Blockchain growth and verification
# ============================================================
def fig_blockchain_growth():
    bc_file = RESULTS_DIR / "master_blockchain.json"
    if not bc_file.exists():
        print(f"[skip] {bc_file}"); return
    with open(bc_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    chain = data["chain"]
    n_blocks = len(chain)

    # Cumulative IoCs over blocks
    cum_iocs = []
    running = 0
    per_block_ioc = []
    for b in chain:
        running += len(b["iocs"])
        cum_iocs.append(running)
        per_block_ioc.append(len(b["iocs"]))

    # Group IoCs by class
    class_counts = {"Normal": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0}
    inv = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    for b in chain:
        for ioc in b["iocs"]:
            class_counts[inv[ioc["attack_class"]]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_WIDE, HEIGHT_MED))

    # (a) Chain growth
    ax1.bar(range(n_blocks), per_block_ioc,
             color=COLORS["primary"], edgecolor="black", linewidth=0.3,
             label="IoCs per block")
    ax1.set_xlabel("Block index"); ax1.set_ylabel("IoCs in block")
    ax1b = ax1.twinx()
    ax1b.plot(range(n_blocks), cum_iocs, "-o",
               color=COLORS["secondary"], markersize=2.5, linewidth=1.0,
               label="cumulative")
    ax1b.set_ylabel("cumulative IoCs", color=COLORS["secondary"])
    ax1.set_title(f"(a) Blockchain growth  "
                   f"(consensus = {data.get('consensus','?')})")

    # (b) IoC distribution by class
    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    vals = [class_counts[c] for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]
    bars = ax2.bar(classes, vals, color=colors, edgecolor="black",
                    linewidth=0.4)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02,
                 str(v), ha="center", fontsize=7)
    ax2.set_ylabel("IoCs on chain")
    ax2.set_title("(b) IoC distribution by class")

    fig.suptitle(f"Blockchain audit summary  "
                 f"({n_blocks} blocks, {sum(vals):,} IoCs)", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_blockchain_growth")


# ============================================================
#  5. Incremental gain waterfall (unified with Layer 3)
# ============================================================
def fig_incremental_gain_unified():
    f = RESULTS_DIR / "ablation_matrix.json"
    if not f.exists(): return
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    # Cumulative accuracy (binary) per config
    labels = [r["name"] for r in data]
    accs = [r["binary"]["accuracy"] for r in data]

    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_MED))
    x = np.arange(len(labels))
    colors = ["#555", COLORS["primary"], COLORS["tertiary"],
              COLORS["accent1"], COLORS["secondary"],
              COLORS["accent2"], COLORS["accent3"]][:len(labels)]
    bars = ax.bar(x, accs, color=colors,
                   edgecolor="black", linewidth=0.5)
    # Value labels
    for i, v in enumerate(accs):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=7,
                fontweight="bold")
        if i > 0:
            delta = v - accs[i-1]
            color = COLORS["tertiary"] if delta > 0 else COLORS["secondary"]
            sign = "+" if delta > 0 else ""
            ax.text(i, accs[i] - 0.02, f"{sign}{delta:.4f}",
                    ha="center", fontsize=6, color=color,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=15, ha="right")
    ax.set_ylabel("Binary accuracy (NSL-KDD KDDTest+)")
    ax.set_ylim(min(accs) - 0.02, max(accs) + 0.02)
    ax.set_title("Full ablation: incremental gain from each component")
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_incremental_gain_unified")


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("  Unified results figures")
    print("=" * 60)

    print("\n[1/5] Ablation matrix...")
    try: fig_ablation_matrix()
    except Exception as e: print(f"  [warn] {e}")

    print("\n[2/5] FL convergence...")
    try: fig_fl_convergence()
    except Exception as e: print(f"  [warn] {e}")

    print("\n[3/5] DP privacy vs utility...")
    try: fig_fl_privacy_vs_utility()
    except Exception as e: print(f"  [warn] {e}")

    print("\n[4/5] Blockchain growth...")
    try: fig_blockchain_growth()
    except Exception as e: print(f"  [warn] {e}")

    print("\n[5/5] Incremental gain waterfall (unified)...")
    try: fig_incremental_gain_unified()
    except Exception as e: print(f"  [warn] {e}")

    print(f"\n[OK] Saved to {OUT}")


if __name__ == "__main__":
    main()
