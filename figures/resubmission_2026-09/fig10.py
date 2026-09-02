"""Fig. 10: Accuracy-FPR trade-off, HCAPT-Detect vs. literature baselines (regenerated)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, r"E:\Strudin\Hussein2026\05-Implementation\src")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plotting.ieee_style import COLORS, WIDTH_WIDE, HEIGHT_MED

OUT = Path(__file__).resolve().parent / "fig10"
OUT.mkdir(exist_ok=True)

# Published baselines (accuracy, FPR); None = FPR not reported
refs = [
    ("Jia et al. [27]",    0.9990, 0.0003),
    ("Arafah et al. [28]", 0.9962, None),
    ("Radha & Kavitha [29]", 0.9960, 0.0400),
    ("Adnan et al. [30]",  0.9900, None),
]
# This work, official KDDTrain+/KDDTest+ split, binary metrics as in Table 8 / Section 5.1
ours = [
    ("Ens",             0.7748, 0.0280, "D", COLORS["primary"]),
    ("Ens + WGAN-GP",   0.7757, 0.0283, "D", COLORS["secondary"]),
    ("Ens + CGAN",      0.7881, 0.0296, "D", COLORS["tertiary"]),
    ("Ens + CGAN + ZT", 0.7904, 0.0298, "*", COLORS["accent1"]),
]
random_split = ("This work, random 90:10 split", 0.9965, 0.0022)

fig, ax = plt.subplots(figsize=(WIDTH_WIDE, HEIGHT_MED + 0.3))

# target zone
ax.add_patch(Rectangle((0, 0.97), 0.015, 0.03, alpha=0.10, color=COLORS["tertiary"], zorder=1))
ax.text(0.0148, 0.9712, "target zone", ha="right", va="bottom", fontsize=6.5,
        color=COLORS["tertiary"], fontweight="bold")

# baselines: labels in a column to the right of the cluster, with leader lines
arrow = dict(arrowstyle="-", color="0.55", linewidth=0.5, shrinkA=0, shrinkB=3)
label_col = [(0.0185, 0.9905), (0.0185, 0.9775), (0.0185, 0.9645), (0.0185, 0.9515)]
for (name, acc, fpr), (tx, ty) in zip(refs, label_col):
    x = fpr if fpr is not None else 0.0
    filled = fpr is not None
    ax.scatter(x, acc, s=70, facecolor=COLORS["gray"] if filled else "white",
               edgecolor=COLORS["gray"], linewidth=1.0, marker="o", zorder=3)
    if x > 0.03:   # Radha: label next to the point
        ax.annotate(name, (x, acc), textcoords="offset points", xytext=(6, -3), fontsize=6.5, color=COLORS["gray"])
    else:
        ax.annotate(name + ("" if filled else " (FPR not reported)"), (x, acc), xytext=(tx, ty),
                    textcoords="data", fontsize=6.5, color=COLORS["gray"], va="center", arrowprops=arrow)

# random-split point of this work (easy protocol)
ax.scatter(random_split[2], random_split[1], s=80, facecolor="white",
           edgecolor=COLORS["primary"], linewidth=1.2, marker="D", zorder=3)
ax.annotate(random_split[0], (random_split[2], random_split[1]), xytext=(0.0185, 1.0035),
            textcoords="data", fontsize=6.5, color=COLORS["primary"], fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=COLORS["primary"], linewidth=0.5, shrinkA=0, shrinkB=4))

# our official-split points, labels staggered
lab_off = [(-6, -12), (7, -14), (-6, 9), (8, 6)]
for (name, acc, fpr, mk, col), (dx, dy) in zip(ours, lab_off):
    ax.scatter(fpr, acc, s=130 if mk == "*" else 80, color=col, edgecolor="black",
               linewidth=0.8, marker=mk, zorder=4)
    ax.annotate(name, (fpr, acc), textcoords="offset points", xytext=(dx, dy),
                ha="right" if dx < 0 else "left", fontsize=6.5, color=col, fontweight="bold")

ax.annotate("", xy=(0.0335, 0.795), xytext=(0.0335, 0.965),
            arrowprops=dict(arrowstyle="<->", color="0.4", linewidth=0.7, linestyle="--"))
ax.text(0.0341, 0.87, "protocol gap:\nofficial KDDTest+\nvs. random split", fontsize=6, color="0.35", va="center")

ax.set_xlabel("False Positive Rate (lower is better)")
ax.set_ylabel("Accuracy (higher is better)")
ax.set_title("Accuracy-FPR trade-off: HCAPT-Detect vs. literature baselines")
ax.set_xlim(-0.004, 0.06)
ax.set_ylim(0.74, 1.015)

handles = [
    plt.Line2D([], [], marker="o", linestyle="", color=COLORS["gray"], label="Published baselines (filled: FPR reported)"),
    plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", color=COLORS["gray"], label="Published baselines (open: FPR not reported)"),
    plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="white", color=COLORS["primary"], label="This work, random split"),
    plt.Line2D([], [], marker="D", linestyle="", color=COLORS["primary"], label="This work, official split (Ens variants)"),
    plt.Line2D([], [], marker="*", linestyle="", color=COLORS["accent1"], label="This work, official split (+ Zero-Trust)", markersize=10),
]
ax.legend(handles=handles, loc="center right", fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "fig10.png", dpi=300)
fig.savefig(OUT / "fig10.pdf")
print("saved", OUT)
