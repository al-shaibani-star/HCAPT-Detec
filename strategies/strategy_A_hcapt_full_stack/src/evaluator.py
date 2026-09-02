"""
Strategy A — Per-dataset Evaluator

Reads the ablation results produced by the pipeline (JSON/CSV files
collected into `results/<dataset>/`) and produces:
  - ablation_table.csv          (6 configs × 5 metrics)
  - ablation_heatmap.png        (visual summary)
  - per_config_summary.json     (machine-readable)
  - REPORT.md                   (markdown report draft)

If the pipeline didn't produce the expected files yet, this evaluator
prints a helpful message and exits cleanly (idempotent, safe to re-run).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT  = STRATEGY_ROOT / "results"

# Canonical config order for plotting / table rows
CANONICAL_CONFIGS = [
    "ens_no_gan",
    "ens_gan",
    "ens_gan_zt",
    "ens_gan_fl",
    "ens_gan_fl_bc",
    "full",
]

# Metrics to extract for the ablation table
ABLATION_METRICS = ["accuracy", "precision", "recall", "f1", "fpr"]


# =============================================================================
# Result discovery
# =============================================================================
# Map the human-readable names from the project pipeline -> canonical config IDs
NAME_TO_CONFIG = {
    "Ens (no GAN)":                    "ens_no_gan",
    "Ens + GAN (central)":             "ens_gan",
    "Ens + GAN":                       "ens_gan",
    "Ens + GAN + ZT (central)":        "ens_gan_zt",
    "Ens + GAN + ZT":                  "ens_gan_zt",
    "Ens + GAN + FL":                  "ens_gan_fl",
    "Ens + GAN + FL + BC":             "ens_gan_fl_bc",
    "FULL: Ens + GAN + FL + BC + ZT":  "full",
    "FULL":                            "full",
}


def _normalise_name(name: str) -> str:
    """Map any input config name to a canonical id (lowercase + underscored)."""
    if name in NAME_TO_CONFIG:
        return NAME_TO_CONFIG[name]
    if name in CANONICAL_CONFIGS:
        return name
    # heuristic fallback: lowercase + underscore
    n = name.lower().replace("(", "").replace(")", "")
    n = n.replace(" + ", "_").replace(" ", "_").replace(":", "")
    n = n.replace("ens_no_gan", "ens_no_gan")
    return n


def _find_ablation_results(dataset_dir: Path) -> Optional[Dict[str, Dict]]:
    """Search dataset results dir for an ablation JSON / CSV.

    Returns dict mapping CANONICAL config_id -> metrics dict, or None.
    """
    candidates = [
        dataset_dir / "ablation_results.json",
        dataset_dir / "ablation_matrix.json",
        dataset_dir / "ablation_evaluation.json",
        dataset_dir / "evaluation_results.json",
        dataset_dir / "all_configurations.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _normalise_ablation_data(data)
    # Fallback: scan for any JSON whose top-level looks like config names
    for j in dataset_dir.glob("*.json"):
        try:
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and any(k in data for k in CANONICAL_CONFIGS):
            return _normalise_ablation_data(data)
        if isinstance(data, list) and data and "name" in data[0]:
            return _normalise_ablation_data(data)
    return None


def _normalise_ablation_data(data) -> Dict[str, Dict]:
    """Make the input structure look like {canonical_config_id: {metrics...}}."""
    if isinstance(data, list):
        out = {}
        for x in data:
            if "name" in x:
                cid = _normalise_name(x["name"])
                out[cid] = x
        return out
    if isinstance(data, dict):
        # Maybe already keyed by config id
        out = {}
        for k, v in data.items():
            cid = _normalise_name(k)
            out[cid] = v
        return out
    return {}


# =============================================================================
# Ablation table
# =============================================================================
def build_ablation_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Produce the canonical 6×5 ablation table."""
    rows = []
    for config_id in CANONICAL_CONFIGS:
        if config_id not in results:
            rows.append({"config": config_id, **{m: np.nan for m in ABLATION_METRICS}})
            continue
        r = results[config_id]
        # Accept either flat metrics or nested under "binary"
        m = r.get("binary", r)
        rows.append({
            "config": config_id,
            **{metric: float(m.get(metric, np.nan)) for metric in ABLATION_METRICS},
        })
    df = pd.DataFrame(rows)
    return df


def save_ablation_table(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_table.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[eval] wrote {csv_path.relative_to(STRATEGY_ROOT)}")
    return csv_path


def plot_ablation_heatmap(df: pd.DataFrame, out_dir: Path,
                           dataset_name: str):
    """Heatmap of the ablation table (configs × metrics)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    data = df.set_index("config")[ABLATION_METRICS]
    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ABLATION_METRICS)))
    ax.set_xticklabels(ABLATION_METRICS, rotation=20, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_title(f"Ablation matrix — {dataset_name}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if 0.4 < v < 0.85 else "white")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    out_path = out_dir / "ablation_heatmap.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[eval] wrote {out_path.relative_to(STRATEGY_ROOT)}")


# =============================================================================
# Markdown report
# =============================================================================
def _df_to_md_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to a Markdown table without depending on tabulate."""
    if df.empty:
        return "_(empty table)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if not (v != v) else "—")  # NaN check
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(dataset_name: str, df: pd.DataFrame, out_dir: Path):
    md = []
    md.append(f"# Strategy A — HCAPT-Defense Full Stack")
    md.append(f"## Dataset: `{dataset_name}`\n")
    md.append("Auto-generated by `evaluator.py`. See `ablation_table.csv` for raw numbers.\n")
    md.append("## Ablation matrix\n")
    if df.empty:
        md.append("_No ablation results found. Re-run the pipeline._\n")
    else:
        md.append(_df_to_md_table(df))
        md.append("")
        # Best config
        if "f1" in df and not df["f1"].isna().all():
            best = df.loc[df["f1"].idxmax()]
            md.append(f"**Best F1:** `{best['config']}` ({best['f1']:.4f})\n")
    md.append("\n## Files in this folder\n")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            md.append(f"- `{f.name}` ({size_kb:.1f} KB)")

    out_path = out_dir / "REPORT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[eval] wrote {out_path.relative_to(STRATEGY_ROOT)}")


# =============================================================================
# Public entry point
# =============================================================================
def evaluate_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """Build ablation table + heatmap + report for one dataset."""
    out_dir = RESULTS_ROOT / dataset_name
    if not out_dir.exists():
        print(f"[eval] No results dir for {dataset_name}")
        return None

    print(f"\n[eval] === {dataset_name} ===")
    results = _find_ablation_results(out_dir)
    if results is None:
        print(f"[eval] No ablation JSON found in {out_dir}")
        print(f"[eval]   (pipeline may have not finished yet)")
        return None

    df = build_ablation_table(results)
    save_ablation_table(df, out_dir)
    plot_ablation_heatmap(df, out_dir, dataset_name)
    build_report(dataset_name, df, out_dir)
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate_dataset(sys.argv[1])
    else:
        for ds_dir in sorted(RESULTS_ROOT.iterdir()):
            if ds_dir.is_dir() and not ds_dir.name.startswith("_"):
                evaluate_dataset(ds_dir.name)
