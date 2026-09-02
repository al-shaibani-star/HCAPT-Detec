"""
Unified HCAPT-Defense architecture diagram (IEEEtran-ready).
Shows the full integration of the 3 innovations with FL + Blockchain.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from plotting.ieee_style import (
    save_fig, WIDTH_WIDE, HEIGHT_TALL, COLORS,
)

OUT = Path(__file__).resolve().parents[2] / "figures" / "architecture"


def _box(ax, x, y, w, h, label, color="#FFFFFF", edge="black",
         fontsize=7, sub=None, subfontsize=6, lw=0.9):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=lw, edgecolor=edge, facecolor=color)
    ax.add_patch(patch)
    if sub:
        ax.text(x + w/2, y + h*0.66, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=subfontsize, style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")


def _arrow(ax, xy1, xy2, color="black", lw=0.9, style="-|>", mut=10,
           label=None, label_offset=(0, 0), label_color=None):
    arr = FancyArrowPatch(xy1, xy2, arrowstyle=style,
                          mutation_scale=mut, linewidth=lw, color=color)
    ax.add_patch(arr)
    if label:
        mx = (xy1[0] + xy2[0]) / 2 + label_offset[0]
        my = (xy1[1] + xy2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=5.5,
                color=label_color or color, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.0))


def fig_unified_architecture():
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_TALL*1.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 16)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # ----- Layer 1 -----
    _box(ax, 4.5, 14.5, 5, 0.9, "Layer 1: Data Preprocessing",
         color="#DDDDDD", sub="NSL-KDD -> scaled features (41-dim)",
         fontsize=8, subfontsize=6.5)

    # ----- Innovation #1 -----
    _box(ax, 4.5, 12.8, 5, 1.2,
         "Innovation #1: GAN / WGAN-GP",
         color=COLORS["primary"] + "30", edge=COLORS["primary"],
         sub="conditional generator + gradient penalty\n"
             "balanced dataset (336K samples)",
         fontsize=8, subfontsize=6)
    _arrow(ax, (7, 14.5), (7, 14.0), label="X_tr, y_tr",
           label_offset=(0.6, 0))

    # ----- Branching to Layer 3a and Innovation #2 (RF/XGB path) -----
    # Layer 3a (FL) - LEFT
    _box(ax, 0.3, 10.5, 4, 1.5,
         "Layer 3a: Federated Learning",
         color=COLORS["tertiary"] + "30", edge=COLORS["tertiary"],
         sub="5 clients, 20 rounds (FedAvg)\n"
             "each client: local train on balanced partition\n"
             "output: global NN weights",
         fontsize=8, subfontsize=6)
    _arrow(ax, (5.5, 12.8), (3, 12.0),
           label="X_bal, y_bal", label_offset=(0, 0.3))

    # Innovation #2 (RF+XGB sub-path) - RIGHT
    _box(ax, 9.5, 11, 4.2, 1.0,
         "I2 baselearners (central)",
         color=COLORS["secondary"] + "20", edge=COLORS["secondary"],
         sub="RF + XGBoost on X_bal",
         fontsize=7, subfontsize=6)
    _arrow(ax, (8.5, 12.8), (11.5, 12.0),
           label="X_bal", label_offset=(0.5, 0.0))

    # ----- Layer 3b (Blockchain) - FAR LEFT ----
    _box(ax, 0.3, 7.8, 4, 1.5,
         "Layer 3b: Blockchain Threat Intel",
         color=COLORS["accent1"] + "30", edge=COLORS["accent1"],
         sub="ModelUpdate blocks (audit trail)\n"
             "IoCs from clients (feature vectors)\n"
             "PoA/PoW consensus, SHA-256",
         fontsize=8, subfontsize=6)
    _arrow(ax, (2.3, 10.5), (2.3, 9.3),
           color=COLORS["accent1"], label="round hashes",
           label_offset=(0, 0.1), label_color=COLORS["accent1"])

    # ----- Innovation #2 (Assembled ensemble) -----
    _box(ax, 5, 8, 4, 2.3,
         "Innovation #2: Ensemble + Self-Learning",
         color=COLORS["secondary"] + "30", edge=COLORS["secondary"],
         sub="  RF  +  XGBoost  +  FL global NN\n"
             "(soft-voting, learned weights)\n\n"
             "Self-Learning loop (drift, pseudo-labels)",
         fontsize=8, subfontsize=6)
    _arrow(ax, (4.3, 11.0), (5.5, 10.3),
           color=COLORS["tertiary"],
           label="FL global NN weights",
           label_offset=(0.5, 0.2), label_color=COLORS["tertiary"])
    _arrow(ax, (11.5, 11.0), (8.5, 10.3),
           color=COLORS["secondary"],
           label="RF + XGB",
           label_offset=(0, 0.2), label_color=COLORS["secondary"])

    # ----- IoC harvest arrow (I2 -> L3b) -----
    _arrow(ax, (5, 8.4), (4.3, 8.5),
           color=COLORS["accent1"],
           label="IoCs from clients",
           label_offset=(-0.3, 0.15), label_color=COLORS["accent1"])

    # ----- Layer 3b enrichment back to I2 -----
    _arrow(ax, (4.3, 8.9), (5, 8.9),
           color=COLORS["accent1"], style="-|>",
           label="enrich", label_offset=(0, 0.13),
           label_color=COLORS["accent1"])

    # ----- Innovation #3 (Zero-Trust) -----
    _box(ax, 5, 4.5, 4, 2.0,
         "Innovation #3: Zero-Trust",
         color=COLORS["accent2"] + "30", edge=COLORS["accent2"],
         sub="Risk Scorer (LightGBM, decorrelated)\n"
             "Policy Engine (5 actions)\n"
             "Micro-segmentation by protocol/service",
         fontsize=8, subfontsize=6)
    _arrow(ax, (7, 8), (7, 6.5),
           label="(enriched_pred, enriched_conf)",
           label_offset=(1.2, 0))

    # ----- Decision output -----
    _box(ax, 2, 2, 10, 1.3, "Final Decision",
         color="#D5F0D5", edge="black",
         sub="ALLOW | MONITOR | CHALLENGE | QUARANTINE | BLOCK",
         fontsize=8, subfontsize=7)
    _arrow(ax, (7, 4.5), (7, 3.3),
           label="action", label_offset=(0.4, 0))

    # ----- Title -----
    fig.suptitle("HCAPT-Defense Unified Architecture\n"
                 "Innovations 1-3 integrated with Federated Learning + Blockchain",
                 fontsize=10, y=0.995, fontweight="bold")

    # Legend below
    ax.text(7, 0.5,
            "Innovations coloured: #1 blue  |  #2 orange  |  #3 yellow  |"
            "  Layer 3 green/purple",
            ha="center", fontsize=6.5, style="italic", color="dimgray")

    fig.tight_layout()
    return save_fig(fig, OUT, "fig_unified_architecture")


def fig_data_flow_timeline():
    """Temporal phases diagram - complementary to the architecture."""
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_TALL*0.9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Horizontal arrow for time
    ax.annotate("", xy=(15.8, 7.2), xytext=(0.2, 7.2),
                arrowprops=dict(arrowstyle="->", linewidth=1.0,
                                color="black"))
    ax.text(14, 7.4, "TIME", fontsize=8, fontweight="bold")

    phases = [
        (0.5, 6.0, 2.0, 0.9, "Phase A.1\nPreprocess", "#DDDDDD"),
        (3.0, 6.0, 2.3, 0.9, "Phase A.2\nGAN training", COLORS["primary"]+"30"),
        (5.8, 6.0, 2.5, 0.9, "Phase A.3\nFL training (20 rounds)",
         COLORS["tertiary"]+"30"),
        (8.8, 6.0, 2.2, 0.9, "Phase A.4\nI2 assemble", COLORS["secondary"]+"30"),
        (11.5, 6.0, 2.0, 0.9, "Phase A.5\nIoC harvest", COLORS["accent1"]+"30"),
        (14.0, 6.0, 1.8, 0.9, "Phase A.6\nZT train", COLORS["accent2"]+"30"),
    ]
    for x, y, w, h, label, c in phases:
        _box(ax, x, y, w, h, label, color=c, fontsize=7, lw=0.8)

    # Phase B
    ax.text(8, 4.8, "--- Offline phase complete ---",
            ha="center", fontsize=6.5, style="italic", color="gray")

    _box(ax, 2, 3.0, 12, 1.3, "Phase B: Online Inference",
         color="#F0F0F0", edge="black",
         sub="x -> I2.predict -> L3b.enrich -> I3.validate -> action",
         fontsize=9, subfontsize=7.5)

    _box(ax, 2, 0.6, 5.5, 1.6,
         "Latency budget (design): 1 s/sample",
         color="#FFFFFF", fontsize=7,
         sub="Achieved: 0.024 ms/sample\n(41,000x headroom)",
         subfontsize=6)
    _box(ax, 8.5, 0.6, 5.5, 1.6,
         "Privacy guarantees",
         color="#FFFFFF", fontsize=7,
         sub="- No raw data leaves clients\n- Tamper-evident audit trail",
         subfontsize=6)

    fig.suptitle("HCAPT-Defense Data Flow Timeline",
                 fontsize=10, fontweight="bold", y=0.99)
    fig.tight_layout()
    return save_fig(fig, OUT, "fig_data_flow_timeline")


def main():
    print("=" * 60)
    print("  Unified Architecture Diagrams")
    print("=" * 60)

    print("\n[1/2] Unified architecture ...")
    fig_unified_architecture()

    print("\n[2/2] Data flow timeline ...")
    fig_data_flow_timeline()

    print(f"\n[OK] Saved to: {OUT}")


if __name__ == "__main__":
    main()
