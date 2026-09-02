"""
Strategy A — Cross-dataset Comparator

Aggregates per-dataset ablation tables into a single comparison view:
  - results/_cross_dataset/comparison_matrix.csv
  - results/_cross_dataset/comparison_heatmap.png
  - results/_cross_dataset/component_contribution.png
  - results/_cross_dataset/CROSS_DATASET_REPORT.md

Run after at least one dataset's ablation has completed; gracefully
handles the case where only one or two datasets have results.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluator import CANONICAL_CONFIGS, ABLATION_METRICS

STRATEGY_ROOT     = Path(__file__).resolve().parent.parent
RESULTS_ROOT      = STRATEGY_ROOT / "results"
CROSS_DATASET_DIR = RESULTS_ROOT / "_cross_dataset"


def _load_dataset_table(dataset_name: str) -> pd.DataFrame:
    csv_path = RESULTS_ROOT / dataset_name / "ablation_table.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["dataset"] = dataset_name
    return df


def _discover_datasets() -> List[str]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted([d.name for d in RESULTS_ROOT.iterdir()
                    if d.is_dir() and not d.name.startswith("_")])


def build_comparison_matrix() -> pd.DataFrame:
    """Stack per-dataset ablation tables. Rows = (dataset, config)."""
    datasets = _discover_datasets()
    frames = [_load_dataset_table(d) for d in datasets]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_comparison_heatmap(df: pd.DataFrame, out_dir: Path):
    """One row per (dataset, config), columns = metrics. Single big heatmap."""
    if df.empty:
        return
    pivot = df.set_index(["dataset", "config"])[ABLATION_METRICS]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pivot) + 1.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ABLATION_METRICS)))
    ax.set_xticklabels(ABLATION_METRICS, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"{d} | {c}" for (d, c) in pivot.index])
    ax.set_title("Cross-dataset Ablation Comparison")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt = "—" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.4 < (v if not np.isnan(v) else 0.5) < 0.85 else "white")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    out_path = out_dir / "comparison_heatmap.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[compare] wrote {out_path.relative_to(STRATEGY_ROOT)}")


def plot_component_contribution(df: pd.DataFrame, out_dir: Path):
    """Bar chart: F1 of each ablation config side-by-side per dataset.

    Reveals which components contribute most to which dataset.
    """
    if df.empty:
        return
    pivot = df.pivot(index="dataset", columns="config", values="f1")
    pivot = pivot[[c for c in CANONICAL_CONFIGS if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("F1-Score (binary)")
    ax.set_xlabel("")
    ax.set_title("Component contribution across datasets (HCAPT-Defense Strategy A)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = out_dir / "component_contribution.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[compare] wrote {out_path.relative_to(STRATEGY_ROOT)}")


def build_cross_dataset_report(df: pd.DataFrame, out_dir: Path):
    md = ["# Strategy A — Cross-dataset Comparison\n"]
    if df.empty:
        md.append("_No dataset results yet. Run the pipeline on at least one dataset first._\n")
    else:
        datasets = sorted(df["dataset"].unique().tolist())
        md.append(f"**Datasets:** {', '.join(datasets)} ({len(datasets)} total)\n")
        md.append("## Comparison Matrix\n")
        from .evaluator import _df_to_md_table
        md.append(_df_to_md_table(df[["dataset","config"] + ABLATION_METRICS]))
        # Best per dataset
        md.append("\n## Best configuration per dataset\n")
        for ds in datasets:
            sub = df[df["dataset"] == ds]
            if sub["f1"].isna().all():
                continue
            best = sub.loc[sub["f1"].idxmax()]
            md.append(f"- **{ds}:** `{best['config']}` (F1 = {best['f1']:.4f})")
        # Effect of full vs no-gan baseline
        md.append("\n## Effect of full HCAPT stack vs ensemble baseline\n")
        for ds in datasets:
            sub = df[df["dataset"] == ds]
            try:
                base = sub[sub["config"]=="ens_no_gan"]["f1"].iloc[0]
                full = sub[sub["config"]=="full"]["f1"].iloc[0]
                if np.isnan(base) or np.isnan(full):
                    continue
                delta = full - base
                md.append(f"- **{ds}:** F1 base={base:.4f} → full={full:.4f} (Δ {delta:+.4f})")
            except (IndexError, KeyError):
                continue
    out_path = out_dir / "CROSS_DATASET_REPORT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[compare] wrote {out_path.relative_to(STRATEGY_ROOT)}")


def compare_all() -> pd.DataFrame:
    CROSS_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df = build_comparison_matrix()
    if df.empty:
        print("[compare] No datasets have ablation_table.csv yet — nothing to compare.")
    csv_path = CROSS_DATASET_DIR / "comparison_matrix.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[compare] wrote {csv_path.relative_to(STRATEGY_ROOT)}")
    plot_comparison_heatmap(df, CROSS_DATASET_DIR)
    plot_component_contribution(df, CROSS_DATASET_DIR)
    build_cross_dataset_report(df, CROSS_DATASET_DIR)
    return df


if __name__ == "__main__":
    compare_all()
