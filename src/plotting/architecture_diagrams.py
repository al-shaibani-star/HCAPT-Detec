"""
Q1-quality architecture and methodology diagrams for the
HCAPT-Defense paper.

All figures use matplotlib only (no TikZ). The IEEE-Q1 style
config is enforced through plotting/ieee_style.py:
  - 600 DPI output
  - Times New Roman 10pt body, 11pt bold title, 10pt bold labels
  - STIX math fonts
  - 7.16 in default width (IEEE double-column)
  - inward ticks, framed legends, partial spines

Generated:
  fig_arch_7layers          - 7-layer stack with innovation markers
  fig_methodology_pipeline  - end-to-end data flow
  fig_ensemble_selflearn    - Innovation #2 detail
  fig_zero_trust_flow       - Innovation #3 decision tree
  fig_gan_ensemble_combined - GAN + ensemble integration
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from plotting.ieee_style import (
    save_fig, clean_axes,
    WIDTH_COL, WIDTH_WIDE, HEIGHT_SHORT, HEIGHT_MED, HEIGHT_TALL,
    HEIGHT_FULL, COLORS,
)

OUT = Path(__file__).resolve().parents[2] / "figures" / "architecture"


# =============================================================
#  Q1 helpers (cleaner typography + stronger visual hierarchy)
# =============================================================
def box(ax, x, y, w, h, label, *,
        color="#FFFFFF",
        edge="#222222",
        title_size=10,           # Q1: 10pt bold for primary labels
        title_weight="bold",
        sub=None,
        sub_size=8.5,            # Q1: 8.5pt italic for sub-labels
        sub_style="italic",
        sub_color="#444444",
        lw=0.9,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        title_y_frac=0.62, sub_y_frac=0.28,
        title_color="#111111"):
    """Create a styled rounded-corner box with title + optional sub."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                           linewidth=lw, edgecolor=edge,
                           facecolor=color)
    ax.add_patch(patch)
    if sub is None:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=title_size, fontweight=title_weight,
                color=title_color)
    else:
        ax.text(x + w/2, y + h*title_y_frac, label,
                ha="center", va="center",
                fontsize=title_size, fontweight=title_weight,
                color=title_color)
        ax.text(x + w/2, y + h*sub_y_frac, sub,
                ha="center", va="center",
                fontsize=sub_size, style=sub_style,
                color=sub_color)
    return patch


def arrow(ax, xy1, xy2, *, color="#222222", lw=0.9,
          style="-|>", mutation=10, alpha=1.0,
          linestyle="-"):
    """Q1-style arrow: thin, dark, consistent mutation scale."""
    arr = FancyArrowPatch(xy1, xy2, arrowstyle=style,
                          mutation_scale=mutation, linewidth=lw,
                          color=color, alpha=alpha,
                          linestyle=linestyle,
                          shrinkA=2, shrinkB=2)
    ax.add_patch(arr)
    return arr


def annotation_marker(ax, x, y, text, *, color="#C00000",
                       offset=(0.15, 0)):
    """Small numbered/asterisked annotation in red for innovation flags."""
    ax.text(x, y, "*", fontsize=14, color=color, fontweight="bold",
            ha="left", va="center")
    ax.text(x + offset[0], y + offset[1], text, fontsize=8,
            color=color, ha="left", va="center", fontweight="bold")


# =============================================================
#  1. 7-Layer architecture (Q1-grade)
# =============================================================
def fig_arch_7layers():
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_FULL*1.5))
    clean_axes(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.5)

    # Tuned IEEE-grade pastel palette (light enough for B/W print)
    layers = [
        (0, "Layer 0: Hybrid-Cloud Infrastructure",
         "AWS  |  Azure  |  GCP  |  On-Prem  |  Edge",
         "#EFEFEF", "#777777"),
        (1, "Layer 1: Data Collection & Preprocessing",
         "Log capture, normalisation, 41-dim feature engineering",
         "#D5E4FA", COLORS["primary"]),
        (2, "Layer 2: AI-Powered Detection Engine",
         "Innovation #1 (WGAN-GP)  +  Innovation #2 (Ensemble + Self-Learning)",
         "#FFE6B5", COLORS["accent2"]),
        (3, "Layer 3: Threat Intelligence (Federated Learning + Blockchain)",
         "FedAvg over 5 clients  +  PoA chain  +  $(\\varepsilon,\\delta)$-DP IoCs",
         "#E0E0E0", COLORS["gray"]),
        (4, "Layer 4: Zero-Trust Security Enforcement",
         "Innovation #3: independent risk scorer  +  policy engine",
         "#F5D5E4", COLORS["accent1"]),
        (5, "Layer 5: Automated Response & Mitigation",
         "ALLOW  |  MONITOR  |  CHALLENGE  |  QUARANTINE  |  BLOCK",
         "#D5F0D5", COLORS["tertiary"]),
        (6, "Layer 6: Presentation & Reporting",
         "Dashboards, audit logs, SOC alerting",
         "#F0E4D5", "#996633"),
    ]

    box_h = 1.85
    for i, (idx, title, sub, fill, edge) in enumerate(layers):
        y = 15.3 - idx*2.05
        box(ax, 0.5, y, 9.0, box_h, title, color=fill, edge=edge,
            sub=sub, title_size=10.5, sub_size=8.5, lw=1.1,
            title_color=edge if edge != "#777777" else "#222")

    # Vertical flow arrows between layers
    for i in range(len(layers) - 1):
        y_to   = 15.3 - i*2.05
        y_from = 15.3 - (i+1)*2.05 + box_h
        arrow(ax, (5.0, y_from), (5.0, y_to), lw=1.4, mutation=12,
              color="#444")

    # Innovation markers on the right
    annotation_marker(ax,  9.7, 15.3 - 2*2.05 + box_h/2,
                       "Innovations\n#1, #2",
                       color="#C00000", offset=(0.18, 0))
    annotation_marker(ax,  9.7, 15.3 - 4*2.05 + box_h/2,
                       "Innovation\n#3",
                       color="#C00000", offset=(0.18, 0))

    ax.set_title("HCAPT-Defense seven-layer architecture",
                 pad=10, fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return save_fig(fig, OUT, "fig_arch_7layers")


# =============================================================
#  2. Methodology pipeline (Q1-grade)
# =============================================================
def fig_methodology_pipeline():
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_MED*1.3))
    clean_axes(ax)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.2)

    stages = [
        (0.20, "NSL-KDD\n125,973 train\n22,544 test",
         "#EFEFEF",  "#777"),
        (2.30, "Preprocess\nencode, scale,\ncache",
         "#D5E4FA",  COLORS["primary"]),
        (4.50, "WGAN-GP\nbalance\n5 classes",
         "#FFE6B5",  COLORS["accent2"]),
        (6.80, "Federated\nLearning\n5 clients",
         "#E0E0E0",  COLORS["gray"]),
        (9.10, "Ensemble\nRF + XGB +\nFL-NN",
         "#FFE6B5",  COLORS["accent2"]),
        (11.40, "Zero-Trust\nrisk scorer +\npolicy",
         "#F5D5E4",  COLORS["accent1"]),
        (13.65, "Decision\n5 actions",
         "#D5F0D5",  COLORS["tertiary"]),
    ]

    y = 3.2
    w = 1.95
    h = 1.65
    centers_x = []
    for x, label, fill, edge in stages:
        box(ax, x, y, w, h, label, color=fill, edge=edge,
            title_size=8.5, lw=1.0)
        centers_x.append(x + w/2)

    # Connecting arrows
    for i in range(len(stages) - 1):
        x_end   = stages[i][0]   + w
        x_start = stages[i+1][0]
        arrow(ax, (x_end + 0.04, y + h/2),
                  (x_start - 0.04, y + h/2),
              lw=1.1, mutation=11, color="#333")

    # Innovation labels below
    inno_y = 1.1
    inno_positions = [
        (centers_x[2], "Innovation #1"),
        (centers_x[4], "Innovation #2"),
        (centers_x[5], "Innovation #3"),
    ]
    for cx, txt in inno_positions:
        ax.annotate(txt, xy=(cx, y), xytext=(cx, inno_y),
                    fontsize=8.5, color="#C00000", ha="center",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>",
                                     color="#C00000",
                                     lw=0.7,
                                     mutation_scale=8))

    # Layer-3 brace below the FL+Blockchain region
    ax.annotate("Layer 3 (privacy)",
                xy=((centers_x[3]+centers_x[3])/2, y),
                xytext=(centers_x[3], inno_y - 0.6),
                fontsize=8.0, color=COLORS["gray"], ha="center",
                style="italic",
                arrowprops=dict(arrowstyle="-",
                                 color=COLORS["gray"], lw=0.5))

    ax.set_title("HCAPT-Defense end-to-end methodology pipeline",
                 pad=8, fontsize=11.5, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return save_fig(fig, OUT, "fig_methodology_pipeline")


# =============================================================
#  3. Ensemble + Self-Learning detail (Q1)
# =============================================================
def fig_ensemble_selflearn():
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_TALL*1.05))
    clean_axes(ax)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 9.5)

    # Top: input
    box(ax, 5.7, 8.3, 3.1, 0.9, "Features (41-dim)",
        color="#EFEFEF", edge="#666",
        title_size=9.5, lw=0.9)

    # Three base learners
    learners = [
        (1.0, "Random Forest", "#D5F0D5", COLORS["tertiary"],
         "200 trees  |  CPU"),
        (5.7, "XGBoost (GPU)", "#D5E4FA", COLORS["primary"],
         "300 trees  |  depth 8"),
        (10.4, "Deep NN (FL-trained)", "#FFD5D5", COLORS["secondary"],
         "MLP 41-256-128-64-5"),
    ]
    for x, name, fill, edge, hw in learners:
        box(ax, x, 6.2, 3.1, 1.4, name, color=fill, edge=edge,
            sub=hw, title_size=10, sub_size=8.0, lw=1.0,
            title_color=edge)
        # connect from input
        arrow(ax, (7.25, 8.3), (x + 1.55, 7.6),
              lw=1.0, mutation=10, color="#444")

    # Soft-voting fusion
    box(ax, 5.0, 4.5, 4.5, 1.05,
        "Weighted soft voting",
        color="#FFE6B5", edge=COLORS["accent2"],
        sub=("$P = w_{\\mathrm{RF}}\\,P_{\\mathrm{RF}} + "
              "w_{\\mathrm{XGB}}\\,P_{\\mathrm{XGB}} + "
              "w_{\\mathrm{NN}}\\,P_{\\mathrm{NN}}$"),
        title_size=10, sub_size=8.5, lw=1.0,
        title_color=COLORS["accent2"])

    # Arrows from each learner to fusion
    for x in (2.55, 7.25, 11.95):
        arrow(ax, (x, 6.2), (7.25, 5.55), lw=0.9, mutation=10,
              color="#555")

    # Outputs: prediction and confidence
    box(ax, 2.7, 2.85, 3.4, 0.95, "Prediction\n$\\hat{y} = \\arg\\max P$",
        color="#D5E4FA", edge=COLORS["primary"],
        title_size=9.5, lw=0.9, title_color=COLORS["primary"])
    box(ax,  8.4, 2.85, 3.4, 0.95, "Confidence\n$c = \\max P$",
        color="#FFD5C0", edge=COLORS["accent2"],
        title_size=9.5, lw=0.9, title_color=COLORS["accent2"])
    arrow(ax, (7.25, 4.5), (4.4, 3.8), lw=1.0, mutation=10)
    arrow(ax, (7.25, 4.5), (10.1, 3.8), lw=1.0, mutation=10)

    # Self-learning module
    box(ax, 2.5, 0.7, 9.5, 1.7,
        "SELF-LEARNING LOOP",
        color="#F5D5E4", edge=COLORS["accent1"],
        sub=("1\\,000-sample sliding window  |  drift trigger ($\\geq$ 5\\%)  "
              "|  pseudo-labels at $c \\geq 0.75$  |  fine-tune NN (5 epochs, "
              "lr $10^{-4}$)"),
        title_size=10, sub_size=8.0, lw=1.1,
        title_color=COLORS["accent1"])

    # Down arrows from outputs to loop
    arrow(ax, (4.4, 2.85), (4.4, 2.4), lw=1.0, mutation=10)
    arrow(ax, (10.1, 2.85), (10.1, 2.4), lw=1.0, mutation=10)

    # Feedback to NN (curved label only)
    arrow(ax, (12.0, 1.55), (13.5, 1.55), lw=1.0,
          color=COLORS["accent1"], mutation=10)
    arrow(ax, (13.65, 1.55), (13.65, 6.85),
          lw=1.0, color=COLORS["accent1"], style="-",
          mutation=0)
    arrow(ax, (13.65, 6.85), (13.5, 6.85),
          lw=1.0, color=COLORS["accent1"], mutation=10)
    ax.text(13.95, 4.0, "fine-tune\nfeedback",
            fontsize=8, color=COLORS["accent1"],
            rotation=-90, va="center", style="italic")

    ax.set_title("Innovation #2: Ensemble with Self-Learning Loop",
                 pad=9, fontsize=11.5, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return save_fig(fig, OUT, "fig_ensemble_selflearn")


# =============================================================
#  4. Zero-Trust validation flow (Q1)
# =============================================================
def fig_zero_trust_flow():
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_TALL))
    clean_axes(ax)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 8.5)

    # Top: input from ensemble
    box(ax, 4.7, 7.3, 5.1, 0.85,
        "From Ensemble: $(\\hat{y},\\, c,\\, \\mathbf{x})$",
        color="#EFEFEF", edge="#666",
        title_size=9.5, lw=0.9)

    # Risk Scorer (left)
    box(ax, 0.5, 5.3, 4.1, 1.5,
        "1. Risk Scorer",
        color="#D5E4FA", edge=COLORS["primary"],
        sub="LightGBM (independent)\n$P_{\\mathrm{attack}}\\in[0,1]$",
        title_size=10, sub_size=8.5, lw=1.0,
        title_color=COLORS["primary"])

    # Context Analyzer (right)
    box(ax, 9.9, 5.3, 4.1, 1.5,
        "2. Context Analyser",
        color="#FFD5D5", edge=COLORS["secondary"],
        sub="per-segment sliding window\n$r_{\\mathrm{ctx}}\\in[0,1]$",
        title_size=10, sub_size=8.5, lw=1.0,
        title_color=COLORS["secondary"])

    arrow(ax, (5.6, 7.3), (3.0, 6.8), lw=1.0, mutation=10)
    arrow(ax, (8.9, 7.3), (11.5, 6.8), lw=1.0, mutation=10)

    # Policy Engine
    box(ax, 3.0, 3.0, 8.5, 1.5,
        "3. Policy Engine",
        color="#F5D5E4", edge=COLORS["accent1"],
        sub=("$\\mathrm{action} = f(\\hat{y}, c, P_{\\mathrm{attack}}, "
              "r_{\\mathrm{ctx}})$"),
        title_size=10.5, sub_size=9.0, lw=1.1,
        title_color=COLORS["accent1"])

    arrow(ax, (3.0, 5.3), (5.5, 4.5), lw=1.0, mutation=10)
    arrow(ax, (11.5, 5.3), (9.0, 4.5), lw=1.0, mutation=10)

    # Five actions
    actions = [
        ("ALLOW",      COLORS["tertiary"]),
        ("MONITOR",    COLORS["accent3"]),
        ("CHALLENGE",  COLORS["accent1"]),
        ("QUARANTINE", COLORS["accent2"]),
        ("BLOCK",      COLORS["secondary"]),
    ]
    base_x = 0.5
    aw = 2.7
    gap = 0.05
    for i, (a, c) in enumerate(actions):
        x_a = base_x + i * (aw + gap)
        box(ax, x_a, 1.0, aw, 0.85, a, color="#FFFFFF", edge=c,
            title_size=10, lw=1.1, title_color=c)
        arrow(ax, (7.25, 3.0), (x_a + aw/2, 1.85),
              lw=0.6, mutation=8, color="#666", alpha=0.7)

    # Highlight CHALLENGE — the key contribution
    ax.annotate(("captures\n``confidently-wrong''\nensemble predictions"),
                xy=(7.25, 1.4), xytext=(7.25, 0.05),
                fontsize=8, color="#C00000", ha="center",
                fontweight="bold", style="italic",
                arrowprops=dict(arrowstyle="-",
                                 color="#C00000", lw=0.6))

    ax.set_title("Innovation #3: Zero-Trust Validation Layer",
                 pad=9, fontsize=11.5, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return save_fig(fig, OUT, "fig_zero_trust_flow")


# =============================================================
#  5. GAN feeding ensemble (Q1)
# =============================================================
def fig_gan_ensemble_combined():
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_MED*1.3))
    clean_axes(ax)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.2)

    # Real minority samples
    box(ax, 0.3, 3.6, 2.7, 1.05,
        "Minority classes",
        color="#FFD5D5", edge=COLORS["secondary"],
        sub="R2L (995)  |  U2R (52)",
        title_size=10, sub_size=8.5, lw=1.0,
        title_color=COLORS["secondary"])

    # CGAN
    box(ax, 3.7, 4.35, 3.0, 1.0, "CGAN",
        color="#D5E4FA", edge=COLORS["primary"],
        sub="BCE + sigmoid", title_size=10, sub_size=8.5, lw=1.0,
        title_color=COLORS["primary"])
    # WGAN-GP
    box(ax, 3.7, 2.85, 3.0, 1.0, "WGAN-GP",
        color="#FFD5D5", edge=COLORS["secondary"],
        sub="Wasserstein + GP", title_size=10, sub_size=8.5, lw=1.0,
        title_color=COLORS["secondary"])
    arrow(ax, (3.0, 4.4), (3.7, 4.85), lw=1.0, mutation=10)
    arrow(ax, (3.0, 3.9), (3.7, 3.35), lw=1.0, mutation=10)

    # Synthetic samples
    box(ax, 7.5, 3.6, 2.5, 1.05,
        "Synthetic samples",
        color="#FFE6B5", edge=COLORS["accent2"],
        sub="210,742 generated", title_size=9.5, sub_size=8.5, lw=1.0,
        title_color=COLORS["accent2"])
    arrow(ax, (6.7, 4.85), (7.5, 4.2), lw=1.0, mutation=10)
    arrow(ax, (6.7, 3.35), (7.5, 4.0), lw=1.0, mutation=10)

    # Balanced set
    box(ax, 10.8, 3.6, 3.4, 1.05,
        "Balanced training set",
        color="#D5F0D5", edge=COLORS["tertiary"],
        sub="$5 \\times 67{,}343 = 336{,}715$",
        title_size=9.5, sub_size=8.5, lw=1.0,
        title_color=COLORS["tertiary"])
    arrow(ax, (10.0, 4.1), (10.8, 4.1), lw=1.0, mutation=10)

    # Down to ensemble
    box(ax, 5.8, 0.7, 2.9, 1.0, "Ensemble training",
        color="#F5D5E4", edge=COLORS["accent1"],
        title_size=10, lw=1.0, title_color=COLORS["accent1"])
    arrow(ax, (12.5, 3.6), (7.25, 1.7), lw=1.0, mutation=10,
          color="#666")

    ax.set_title("GAN-based augmentation feeding the Ensemble (Innovation #1)",
                 pad=9, fontsize=11.5, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return save_fig(fig, OUT, "fig_gan_ensemble_combined")


# =============================================================
#  6. Blockchain Threat Intelligence flow (Q1) — NEW
# =============================================================
def fig_blockchain_flow():
    """Dedicated blockchain diagram showing:
       (a) block anatomy (PoA-signed)
       (b) hash chain linking
       (c) two ingestion paths (FL audit + IoC harvest)
       (d) cross-tenant query path
    """
    fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_FULL*1.4))
    clean_axes(ax)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 11.5)

    # ----- Title regions (visual grouping) -----
    # subtle tinted band for "INPUTS"
    ax.add_patch(Rectangle((0.2, 9.0), 14.1, 2.0,
                            facecolor="#F7F3EE", edgecolor="none",
                            zorder=0))
    ax.text(7.25, 10.85, "Input streams to the blockchain",
            ha="center", va="top",
            fontsize=9.5, fontweight="bold",
            color="#664400", style="italic")

    # ----- (a) FL aggregator → audit hash IoC -----
    box(ax, 0.5, 9.4, 5.5, 1.2,
        "FL Aggregator (after each round)",
        color="#D5E4FA", edge=COLORS["primary"],
        sub=("publishes  $\\mathrm{SHA256}(\\theta_{\\mathrm{global}}^{(r)})$ "
              "as audit IoC"),
        title_size=9.5, sub_size=8.0, lw=1.0,
        title_color=COLORS["primary"])

    # ----- (b) Clients → DP-noised IoC prototypes -----
    box(ax, 8.5, 9.4, 5.5, 1.2,
        "Clients (after training)",
        color="#FFD5D5", edge=COLORS["secondary"],
        sub=("submit  class prototypes  $\\widetilde{\\mu}_c$  with  "
              "$(\\varepsilon,\\delta)$-DP"),
        title_size=9.5, sub_size=8.0, lw=1.0,
        title_color=COLORS["secondary"])

    # Arrows down from inputs into the chain
    arrow(ax, (3.25, 9.4), (3.25, 8.0), lw=1.1, mutation=11, color="#444")
    arrow(ax, (11.25, 9.4), (11.25, 8.0), lw=1.1, mutation=11, color="#444")

    # ----- (c) Hash chain (centre, 3 blocks visible) -----
    chain_y = 5.1
    chain_h = 2.7
    block_w = 3.6
    gap = 0.6
    block_xs = [0.6 + i*(block_w + gap) for i in range(3)]

    # Block cards (label moved to top via header band)
    for i, x in enumerate(block_xs):
        block_label = f"Block $N{'-2' if i == 0 else '-1' if i == 1 else ''}$"

        # Outer empty card
        box(ax, x, chain_y, block_w, chain_h, "",
            color="#FFFBEE", edge="#996600", lw=1.1)

        # Header band (top stripe with block index)
        ax.add_patch(Rectangle((x + 0.04, chain_y + chain_h - 0.42),
                                block_w - 0.08, 0.36,
                                facecolor="#F4D5A0",
                                edgecolor="#996600", linewidth=0.6))
        ax.text(x + block_w/2, chain_y + chain_h - 0.24, block_label,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="#664400")

        # Inner anatomy lines (lower in the box)
        ax.text(x + block_w/2, chain_y + chain_h*0.65,
                "header  $|$  index, ts, prev_hash",
                ha="center", va="center",
                fontsize=7.8, color="#333")

        ax.text(x + block_w/2, chain_y + chain_h*0.49,
                "payload  $|$  IoCs $\\times$ $K$",
                ha="center", va="center",
                fontsize=7.8, color="#333")

        ax.text(x + block_w/2, chain_y + chain_h*0.33,
                "PoA  $|$  HMAC-SHA256(secret, payload)",
                ha="center", va="center",
                fontsize=7.8, color="#333")

        ax.text(x + block_w/2, chain_y + chain_h*0.14,
                "hash  $|$  SHA256(everything above)",
                ha="center", va="center",
                fontsize=7.8, fontstyle="italic",
                color=COLORS["primary"])

    # Three trailing dots after chain
    last_x_end = block_xs[-1] + block_w
    ax.text(last_x_end + 0.6, chain_y + chain_h/2, r"$\cdots$",
            fontsize=18, ha="left", va="center", color="#996600")

    # Hash chain links (curly arrows from prev_hash to next block.header)
    for i in range(2):
        x_from = block_xs[i] + block_w
        x_to   = block_xs[i+1]
        arrow(ax, (x_from + 0.02, chain_y + chain_h*0.93),
                  (x_to - 0.02,   chain_y + chain_h*0.93),
              lw=1.1, mutation=10, color=COLORS["primary"])
        ax.text((x_from + x_to)/2, chain_y + chain_h*0.99,
                "prev\\_hash",
                fontsize=6.8, ha="center", va="bottom",
                color=COLORS["primary"], style="italic")

    # ----- Far-right: chain growth indicator -----
    box(ax, last_x_end + 1.4, chain_y + 0.3, 1.7, chain_h - 0.6,
        "30 blocks\n420 IoCs",
        color="#D5F0D5", edge=COLORS["tertiary"],
        title_size=9.0, lw=0.9,
        title_color=COLORS["tertiary"])

    # ----- (d) Query / verification region (bottom) -----
    box(ax, 0.5, 1.5, 6.0, 2.6,
        "QUERY: cross-tenant similarity match",
        color="#F5D5E4", edge=COLORS["accent1"],
        sub=("lookup  $\\,\\widehat{\\mu}_c$  via 64-bit LSH index\n"
              "Hamming  $\\geq$  threshold  $\\Rightarrow$  enrich prediction"),
        title_size=9.5, sub_size=8.5, lw=1.1,
        title_color=COLORS["accent1"])

    box(ax, 8.0, 1.5, 6.0, 2.6,
        "VERIFY: tamper detection",
        color="#D5E4FA", edge=COLORS["primary"],
        sub=("recompute every block hash  +  HMAC sig\n"
              "$O(B)$ time  $|$  $<\\!50$\\,ms for $B=30$"),
        title_size=9.5, sub_size=8.5, lw=1.1,
        title_color=COLORS["primary"])

    # Down arrows from chain to query/verify
    arrow(ax, (3.5, chain_y), (3.5, 4.1), lw=1.0, mutation=10,
          color=COLORS["accent1"])
    arrow(ax, (11.0, chain_y), (11.0, 4.1), lw=1.0, mutation=10,
          color=COLORS["primary"])

    # ----- Two outputs at very bottom -----
    box(ax, 1.5, 0.1, 4.0, 1.0,
        "$(\\hat{y}', c')$  to Layer 4",
        color="#FFFFFF", edge=COLORS["accent1"],
        title_size=9.5, lw=1.0, title_color=COLORS["accent1"])
    box(ax, 9.0, 0.1, 4.0, 1.0,
        "audit report (SOC)",
        color="#FFFFFF", edge=COLORS["primary"],
        title_size=9.5, lw=1.0, title_color=COLORS["primary"])

    arrow(ax, (3.5, 1.5), (3.5, 1.1), lw=1.0, mutation=10,
          color=COLORS["accent1"])
    arrow(ax, (11.0, 1.5), (11.0, 1.1), lw=1.0, mutation=10,
          color=COLORS["primary"])

    # Title
    ax.set_title("Blockchain Threat Intelligence (Layer 3b): "
                 "block anatomy, hash chain, and two purposes",
                 pad=10, fontsize=11.5, fontweight="bold")
    fig.tight_layout(pad=0.4)
    return save_fig(fig, OUT, "fig_blockchain_flow")


# =============================================================
#  Main
# =============================================================
def main():
    print("=" * 60)
    print("  Q1-grade architecture diagrams")
    print("=" * 60)
    print("\n[1/6] 7-layer architecture ...");  fig_arch_7layers()
    print("\n[2/6] Methodology pipeline ...");  fig_methodology_pipeline()
    print("\n[3/6] Ensemble + Self-Learning ..."); fig_ensemble_selflearn()
    print("\n[4/6] Zero-Trust flow ...");       fig_zero_trust_flow()
    print("\n[5/6] GAN -> Ensemble ...");        fig_gan_ensemble_combined()
    print("\n[6/6] Blockchain flow (NEW) ...");  fig_blockchain_flow()
    print(f"\n[OK] saved to {OUT}")


if __name__ == "__main__":
    main()
