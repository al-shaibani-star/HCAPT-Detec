"""
IEEE Q1-quality matplotlib configuration for HCAPT-Defense paper.

Strict IEEEtran specifications:
  - 600 DPI minimum (IEEE submission requirement)
  - Times New Roman 10pt body, 11pt bold title, 10pt bold labels
  - STIX math fonts (Times-compatible)
  - 7.16 x 4.5 in default figure size (double-column)
  - Inward ticks, framed legends, partial spines
  - 0.05 in padding, 0.5pt 0.25-alpha dashed grid

This module is imported by all plotting/*.py modules to enforce
consistent Q1 styling across all 43 figures.
"""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# --- Force vector-friendly backend ---
matplotlib.use("Agg")

# =============================================================
#  IEEE Q1 rcParams
# =============================================================
IEEE_RC = {
    # ---- Fonts (Times-family, IEEEtran-compatible) ----
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "Nimbus Roman",
                          "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex":      False,         # avoid system LaTeX dependency
    "font.size":        10,            # IEEEtran body 10pt

    # ---- Title / labels ----
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.labelweight":   "bold",
    "figure.titlesize":   12,
    "figure.titleweight": "bold",

    # ---- Tick labels ----
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # ---- Legend ----
    "legend.fontsize":       9,
    "legend.title_fontsize": 9,
    "legend.frameon":        True,
    "legend.framealpha":     0.9,
    "legend.fancybox":       False,
    "legend.edgecolor":      "0.5",
    "legend.borderpad":      0.4,
    "legend.handlelength":   1.5,

    # ---- Lines, markers, axes spines ----
    "lines.linewidth":  1.2,
    "lines.markersize": 4.0,
    "axes.linewidth":   0.6,           # Spine thickness
    "axes.spines.top":   False,        # IEEE: partial spines
    "axes.spines.right": False,
    "axes.edgecolor":    "0.2",

    # ---- Ticks (inward, IEEE convention) ----
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.minor.size":   1.5,
    "ytick.minor.size":   1.5,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,

    # ---- Grid (dashed, low alpha) ----
    "axes.grid":      True,
    "axes.axisbelow": True,
    "grid.color":     "0.5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha":     0.25,

    # ---- Output (Q1 submission grade) ----
    "savefig.dpi":         600,        # IEEE minimum
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "savefig.format":      "pdf",
    "pdf.fonttype":        42,         # TrueType (editable in
                                        # Illustrator / Inkscape)
    "ps.fonttype":         42,
    "pdf.compression":     6,

    # ---- Figure ----
    "figure.dpi":         100,         # screen preview
    "figure.facecolor":   "white",
    "figure.edgecolor":   "white",
    "figure.autolayout":  False,
}

plt.rcParams.update(IEEE_RC)

# =============================================================
#  Standard sizes (IEEE double-column, inches)
# =============================================================
WIDTH_COL    = 3.50    # single-column width
WIDTH_WIDE   = 7.16    # double-column / full text width
HEIGHT_SHORT = 2.20
HEIGHT_MED   = 2.80
HEIGHT_TALL  = 3.50
HEIGHT_FULL  = 4.50    # IEEE Q1 default (per spec)

# =============================================================
#  Color palette (Okabe-Ito, colorblind-safe)
# =============================================================
COLORS = {
    "primary":   "#0072B2",   # blue
    "secondary": "#D55E00",   # vermillion
    "tertiary":  "#009E73",   # green
    "accent1":   "#CC79A7",   # purple
    "accent2":   "#E69F00",   # orange
    "accent3":   "#56B4E9",   # sky blue
    "gray":      "#555555",
    "light":     "#BBBBBB",
}

CLASS_COLORS = {
    "Normal": "#0072B2",
    "DoS":    "#D55E00",
    "Probe":  "#009E73",
    "R2L":    "#CC79A7",
    "U2R":    "#E69F00",
}

MODEL_COLORS = {
    "No GAN":         "#555555",
    "CGAN":           "#0072B2",
    "WGAN-GP":        "#D55E00",
    "RF":             "#009E73",
    "XGBoost":        "#E69F00",
    "IF":             "#56B4E9",
    "Ensemble":       "#0072B2",
    "Ensemble + ZT":  "#CC79A7",
}


# =============================================================
#  Save helper (Q1: PDF + 600-DPI PNG preview)
# =============================================================
def save_fig(fig, out_dir: Path, name: str):
    """Save figure as PDF (vector, 600 DPI default) plus a
    300-DPI PNG preview. Returns (pdf_path, png_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"

    # IEEEtran wants minimum 600 DPI on raster elements
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=300)        # smaller preview
    plt.close(fig)
    print(f"  -> {pdf_path.name}  +  {png_path.name}")
    return pdf_path, png_path


# =============================================================
#  Helper: clean an axis (used by architecture diagrams)
# =============================================================
def clean_axes(ax, keep_box: bool = False):
    """Remove ticks/labels/spines for diagram axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(keep_box)


# =============================================================
#  Helper: Q1-style legend with frame
# =============================================================
def style_legend(ax, loc="best", ncol=1, **kwargs):
    """Apply IEEE-style legend: thin border, 0.9 alpha, no fancy."""
    leg = ax.legend(loc=loc, ncol=ncol, **kwargs)
    if leg is not None:
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_edgecolor("0.5")
        leg.get_frame().set_alpha(0.9)
    return leg
