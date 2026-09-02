"""Generate PNG figures for the deep ablation studies."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RES = STRATEGY_ROOT / "results" / "_deep_ablation"


def main():
    with open(RES / "deep_ablation.json") as f:
        data = json.load(f)

    # ----- A: ensemble composition -----
    A = data["A_ensemble_composition"]
    names = [r["variant"] for r in A]
    f1s   = [r["f1"]      for r in A]
    accs  = [r["accuracy"]for r in A]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = range(len(names))
    bars1 = ax.bar([i-0.2 for i in x], accs, width=0.4, label="Accuracy", color="#4C6EF5", edgecolor="black")
    bars2 = ax.bar([i+0.2 for i in x], f1s,  width=0.4, label="F1",       color="#E8590C", edgecolor="black")
    # Highlight best
    best_idx = f1s.index(max(f1s))
    bars1[best_idx].set_edgecolor("green"); bars1[best_idx].set_linewidth(2.5)
    bars2[best_idx].set_edgecolor("green"); bars2[best_idx].set_linewidth(2.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Score (binary, NSL-KDD KDDTest+)")
    ax.set_ylim(0.74, 0.83)
    ax.set_title("Study A: Ensemble Composition Ablation")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RES / "study_A_ensemble.png", dpi=160)
    plt.close(fig)

    # ----- B: SAGE vs GAT -----
    B = data["B_gnn_layer"]
    names = [r["layer_type"].upper() for r in B]
    f1s = [r["f1"] for r in B]
    accs = [r["accuracy"] for r in B]
    params = [r["params"] for r in B]
    times = [r["train_time_s"] for r in B]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, vals, title, ylab, fmt in [
        (axes[0], f1s,    "F1 (binary)",      "F1",          "{:.3f}"),
        (axes[1], params, "Parameter count",   "params",      "{:,}"),
        (axes[2], times,  "Training time (s)", "seconds",     "{:.1f}"),
    ]:
        colors = ["#4C6EF5", "#E8590C"]
        ax.bar(names, vals, color=colors, edgecolor="black")
        ax.set_title(title); ax.set_ylabel(ylab)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Study B: SAGE vs GAT layer", y=1.02)
    plt.tight_layout()
    plt.savefig(RES / "study_B_sage_vs_gat.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ----- C: ZT context analyzer -----
    C = data["C_zt_context"]
    names = [r["variant"] for r in C]
    f1s = [r["f1"] for r in C]
    accs = [r["accuracy"] for r in C]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(names))
    ax.bar([i-0.2 for i in x], accs, width=0.4, label="Accuracy", color="#4C6EF5", edgecolor="black")
    ax.bar([i+0.2 for i in x], f1s,  width=0.4, label="F1",       color="#E8590C", edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=10, fontsize=9)
    ax.set_ylim(0.795, 0.810)
    ax.set_ylabel("Score")
    ax.set_title("Study C: Zero-Trust Context Analyzer ablation\n(differences ≤ 0.01pp = within noise)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RES / "study_C_zt_context.png", dpi=160)
    plt.close(fig)

    print("Generated 3 PNGs in:", RES)
    for f in sorted(RES.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
