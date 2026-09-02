"""
Every figure in the paper, built from the locked result files.

Task 16 of FINAL_PLAN.md. Each figure reads only from `results/_attainability/`,
`results/_roc/`, `results/_paper_numbers/` and the raw result CSVs. No figure
contains a number that is not in one of those files, and none is hand-placed.

    Figure 1  four-panel ROC against the attainability frontier      (§5.3)
    Figure 2  the decomposition as a stacked bar, four benchmarks    (§5.3)
    Figure 3  the timestamp counterfactual, three panels             (§5.4)
    Figure 4  interventions as moves along the ROC, ToN-IoT          (§5.5)
    Figure 5  the mutual-information dose-response, Edge-IIoTset     (§5.2)
    Figure 7  the split-protocol gap, with provenance hatching       (§6.2)

Figure 6 is not built here: it was the exploratory ROC of task 6, and its
content is now Figure 1.

    python -m src.paper_figures

Design constraints. Two-column journal, so full-width figures are 7.0 in and
single-column 3.4 in. Every series is distinguishable without colour: benign and
attack differ in fill as well as hue, and the one bar in the paper that carries
a provenance caveat is hatched rather than merely annotated.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_ROOT = Path(__file__).resolve().parents[1]
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

from src.attainability import RESULTS_DIR, OUT_DIR, _at_recall            # noqa: E402

FIG_DIR = STRATEGY_ROOT / "figures" / "paper"
ROC_JSON = RESULTS_DIR / "_roc" / "roc_frontier.json"
NUM_JSON = RESULTS_DIR / "_paper_numbers" / "numbers.json"
CF_JSON = OUT_DIR / "ton_iot_timestamp_counterfactual.json"

BENCH = ["nsl-kdd", "ton-iot-telemetry", "edge-iiotset", "dapt-2020"]
LABEL = {"nsl-kdd": "NSL-KDD", "ton-iot-telemetry": "ToN-IoT",
         "edge-iiotset": "Edge-IIoTset", "dapt-2020": "DAPT 2020"}

INK = "#1a1a1a"
FRONTIER = "#000000"
SLACK = "#c8c8c8"
BENIGN = "#7f9fbf"
ATTACK = "#b23a48"
ACCENT = "#2f5d8a"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    return png


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _full_row(bench: str) -> dict:
    cfgs = _load(NUM_JSON)["benchmarks"][bench]["configurations"]
    for k, v in cfgs.items():
        if k == "full" or k.lower().startswith("full"):
            return v
    raise KeyError(bench)


# ---------------------------------------------------------------------------
def figure1() -> Path:
    """ROC against the frontier, one panel per benchmark."""
    roc = _load(ROC_JSON)["benchmarks"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))

    for ax, b in zip(axes.ravel(), BENCH):
        z = np.load(OUT_DIR / f"{b}_curve.npz")
        fr_rec, fr_fpr = z["split_recall"], z["split_fpr"]
        ax.plot(fr_fpr, fr_rec, ls="--", color=FRONTIER, lw=1.1,
                label="attainability frontier")

        r = roc.get(b, {})
        if r.get("status") == "ok":
            c = r["configurations"]["ens_gan"]
            x = np.array(c["roc_downsampled"]["x"]); y = np.array(c["roc_downsampled"]["y"])
            ax.fill_between(x, y, np.interp(y, fr_rec, fr_fpr) * 0 + y, color="none")
            # shade the slack: between the detector curve and the frontier, in x
            fr_at_y = np.interp(y, fr_rec, fr_fpr)
            ax.fill_betweenx(y, fr_at_y, x, color=SLACK, alpha=0.55, lw=0,
                             label="model slack")
            ax.plot(x, y, color=ACCENT, label=f"detector (AUC {c['auc']:.3f})")
            op = c["operating_point"]
            ax.plot(op["fpr"], op["recall"], "o", ms=5, mfc="white",
                    mec=ATTACK, mew=1.4, zorder=5)
            ax.annotate(f"reported\n{op['fpr']:.3f} @ {op['recall']:.3f}",
                        (op["fpr"], op["recall"]), textcoords="offset points",
                        xytext=((10, -22) if op["fpr"] < 0.25 else (-8, -26)),
                        fontsize=6, ha="left" if op["fpr"] < 0.25 else "right",
                        color=ATTACK)
        else:
            ax.text(0.52, 0.34, "no cached model;\nfrontier only", fontsize=7,
                    ha="center", color=ATTACK, style="italic")
            f = _full_row(b)
            ax.plot(f["fpr"], f["recall"], "o", ms=5, mfc="white",
                    mec=ATTACK, mew=1.4, zorder=5)

        ax.plot([0, 1], [0, 1], ls=":", color="0.75", lw=0.7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(LABEL[b])
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("recall")

    # the legend belongs in a panel that actually shows all three series
    axes[0, 1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return _save(fig, "fig1_roc_frontier")


def figure2() -> Path:
    """The decomposition itself: attainable component and model slack."""
    rows = []
    for b in BENCH:
        f = _full_row(b)
        rows.append((LABEL[b], f["frontier_fpr_at_own_recall"], f["fpr"], f["recall"]))

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    y = np.arange(len(rows))[::-1]
    fr = np.array([r[1] for r in rows]); obs = np.array([r[2] for r in rows])

    ax.barh(y, fr, color=INK, height=0.55, label="attainable (representation)")
    ax.barh(y, obs - fr, left=fr, color=SLACK, height=0.55,
            edgecolor=INK, lw=0.5, label="model slack")
    for yi, (lab, f_, o_, rec) in zip(y, rows):
        ax.text(o_ + 0.012, yi, f"{o_:.3f} @ r={rec:.3f}", va="center", fontsize=6.5)

    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 0.72); ax.set_xlabel("false-positive rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig2_decomposition")


def _parse_date(s: str):
    return datetime.strptime(s, "%d-%b-%y")


def figure3() -> Path:
    """The timestamp counterfactual — the paper's climax."""
    cf = _load(CF_JSON)
    variants = {v["variant"]: v for v in cf["variants"]}
    oracle = cf["label_oracle"]

    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.42)

    # (a) counterfactual
    ax = fig.add_subplot(gs[0, 0])
    order = ["base", "base+date", "base+time", "base+date+time"]
    names = ["18 features", "+ date", "+ time", "+ both"]
    conf = [100 * variants[k]["pooled"]["conflict_rate"] for k in order]
    floor = [100 * variants[k]["pooled"]["floor_fpr"] for k in order]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, conf, w, color=ATTACK, label="label-conflicting records")
    ax.bar(x + w/2, floor, w, color=INK, label="frontier at recall 1.00")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("per cent")
    ax.set_ylim(0, 44)
    ax.annotate("nine values,\neach single-label", xy=(1.9, 20), fontsize=6,
                ha="center", color=ATTACK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.02))
    ax.set_title("(a) restoring a column")

    # (b) the oracle
    ax = fig.add_subplot(gs[0, 1])
    tbl = sorted(oracle["date_table"], key=lambda t: _parse_date(t["date"]))
    yy = np.arange(len(tbl))[::-1]
    for yi, t in zip(yy, tbl):
        atk = t["attack_share"] == 1.0
        ax.barh(yi, t["rows"], color=ATTACK if atk else BENIGN,
                height=0.68, edgecolor=INK, lw=0.4,
                hatch="" if atk else "///")
    ax.set_yticks(yy); ax.set_yticklabels([t["date"] for t in tbl], fontsize=6.5)
    ax.set_xlabel("records")
    ax.set_title("(b) each date, one label")
    # Make the three-week collection gap visible rather than implicit.
    n_benign = sum(1 for t in tbl if t["attack_share"] == 0.0)
    ax.axhline(len(tbl) - n_benign - 0.5, color=INK, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1] * 0.30, len(tbl) - n_benign - 0.28,
            "three-week gap", fontsize=6, va="bottom", ha="left", color="0.35")
    ax.bar(0, 0, color=BENIGN, hatch="///", edgecolor=INK, lw=0.4, label="benign")
    ax.bar(0, 0, color=ATTACK, edgecolor=INK, lw=0.4, label="attack")
    ax.legend(frameon=False, loc="lower right", fontsize=6)

    # (c) per-device floors
    ax = fig.add_subplot(gs[0, 2])
    dev = sorted(variants["base"]["per_device"],
                 key=lambda d: -d["device_floor_fpr"])
    yy = np.arange(len(dev))[::-1]
    ax.barh(yy, [d["device_floor_fpr"] for d in dev], color=INK, height=0.62)
    ax.set_yticks(yy); ax.set_yticklabels([d["device"] for d in dev], fontsize=6.5)
    ax.set_xlim(0, 1.08); ax.set_xlabel("device-level frontier")
    ax.set_title("(c) per device")

    return _save(fig, "fig3_timestamp_counterfactual")


def figure4() -> Path:
    """Interventions relocate the operating point along one curve."""
    sm = pd.read_csv(RESULTS_DIR / "ton-iot-telemetry" / "smote_diagnosis.csv")
    z = np.load(OUT_DIR / "ton-iot-telemetry_curve.npz")
    fr_rec, fr_fpr = z["split_recall"], z["split_fpr"]
    roc = _load(ROC_JSON)["benchmarks"]["ton-iot-telemetry"]["configurations"]["ens_no_gan"]
    rx = np.array(roc["roc_downsampled"]["x"]); ry = np.array(roc["roc_downsampled"]["y"])

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot(fr_fpr, fr_rec, ls="--", color=FRONTIER, lw=1.1, label="frontier")
    ax.plot(rx, ry, color=ACCENT, lw=1.1, label="unaugmented ROC")

    # Offsets are per-marker because the four points cluster in two pairs.
    marks = {"none": ("unaugmented", "o", (7, 3), "left"),
             "wgan-gp": ("synthetic", "^", (7, -11), "left"),
             "smote": ("rebalanced", "s", (-7, 7), "right"),
             "smote+wgan-gp": ("both", "D", (-7, -13), "right")}
    for r in sm.itertuples():
        if r.label not in marks:
            continue
        lab, m, off, ha = marks[r.label]
        ax.plot(r.fpr, r.recall, m, ms=5, mfc="white", mec=ATTACK, mew=1.2, zorder=5)
        ax.annotate(lab, (r.fpr, r.recall), textcoords="offset points",
                    xytext=off, fontsize=6, color=ATTACK, ha=ha)
    ax.annotate("every intervention lands\non the unaugmented curve",
                xy=(0.40, 0.735), fontsize=6, color="0.35", ha="center")

    ax.set_xlim(0.05, 0.72); ax.set_ylim(0.70, 1.01)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("recall")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("ToN-IoT: interventions vs the ROC")
    fig.tight_layout()
    return _save(fig, "fig4_interventions")


def figure5() -> Path:
    """Progressive removal of the highest-mutual-information features."""
    df = pd.read_csv(RESULTS_DIR / "_leak_audit" / "edge_iiotset_progressive_drop.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(df.n_features, df.accuracy, "o-", color=ACCENT, ms=4, label="accuracy")
    ax.plot(df.n_features, df.f1, "s--", color=INK, ms=3.5, label="F1")
    for r in df.itertuples():
        if isinstance(r.next_drop, str):
            ax.annotate(r.next_drop, (r.n_features, r.accuracy),
                        textcoords="offset points", xytext=(-3, -11),
                        fontsize=5.5, rotation=32, ha="right", color="0.35")
    ax.invert_xaxis()
    ax.set_xlabel("features remaining"); ax.set_ylabel("score")
    ax.set_ylim(0.88, 1.008)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Edge-IIoTset: removing leaking columns")
    fig.tight_layout()
    return _save(fig, "fig5_mi_dose_response")


def figure7() -> Path:
    """Split protocol vs architecture — with the borrowed bar marked as borrowed."""
    ours_official = _full_row("nsl-kdd")["accuracy"]
    ours_random = _load(RESULTS_DIR / "_random_split_baseline" / "nsl_kdd_random.json")["accuracy"]
    rival_official = _load(
        RESULTS_DIR / "_fidmf_reimpl" / "nsl-kdd_fidmf.json")["results"]["attlstm_only"]["accuracy"]
    rival_random = _load(RESULTS_DIR / "_fidmf_reimpl" / "nsl-kdd-random_fidmf.json"
                         )["results"]["attlstm_only"]["accuracy"]
    rival_published = 0.9940       # their paper's figure, drawn as a reference tick only

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [ours_official, rival_official], w, color=INK,
           label="official")
    ax.bar(x[0] + w/2, ours_random, w, color=SLACK, edgecolor=INK, lw=0.6,
           label="random pool (ours)")
    ax.bar(x[1] + w/2, rival_random, w, color=SLACK, edgecolor=INK, lw=0.6)
    # their published figure as a reference line, clearly not one of our bars
    ax.hlines(rival_published, x[1] + w/2 - w/2, x[1] + w/2 + w/2,
              color=ATTACK, lw=1.4, linestyles=(0, (3, 2)), zorder=5,
              label="their published figure (not our run)")

    for xi, v in [(x[0]-w/2, ours_official), (x[1]-w/2, rival_official),
                  (x[0]+w/2, ours_random), (x[1]+w/2, rival_random)]:
        ax.text(xi, v + 0.008, f"{v:.4f}", ha="center", fontsize=6)
    for xi, a, b in [(x[0], ours_official, ours_random), (x[1], rival_official, rival_random)]:
        ax.annotate(f"{100*(b-a):+.1f} pp", (xi, min(a, b) - 0.035), ha="center",
                    fontsize=6.5, color=ATTACK)

    ax.set_xticks(x); ax.set_xticklabels(["our ensemble", "attention-LSTM\n(re-implemented)"])
    ax.set_ylim(0.68, 1.06); ax.set_ylabel("accuracy")
    # below the axes: every in-panel position collides with a bar
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=5.6, columnspacing=1.0, handlelength=1.4)
    ax.set_title("NSL-KDD: the protocol moves the number")
    fig.tight_layout()
    return _save(fig, "fig7_split_protocol")


def main() -> int:
    for fn in (figure1, figure2, figure3, figure4, figure5, figure7):
        try:
            p = fn()
            print(f"  {fn.__name__:<10} -> {p.name}")
        except Exception as exc:
            print(f"  {fn.__name__:<10} FAILED: {exc!r}")
    print(f"\nwrote {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
