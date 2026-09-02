"""
The graphical abstract.

Elsevier asks for at least 531 x 1328 pixels (height x width) — a wide banner —
and wants a reader to grasp the paper from it alone. This one carries the single
claim the paper is about: two benchmarks reporting almost the same false-positive
rate are opposite cases once the rate is split into what the corpus forces and
what the model adds.

Every value is read from the locked result files, like every other figure.

    python -m src.graphical_abstract
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRATEGY_ROOT = Path(__file__).resolve().parents[1]
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

from src.attainability import RESULTS_DIR                      # noqa: E402

OUT = STRATEGY_ROOT / "figures" / "paper" / "graphical_abstract"
NUM = RESULTS_DIR / "_paper_numbers" / "numbers.json"

W_PX, H_PX, DPI = 1328, 531, 200
LABEL = {"nsl-kdd": "NSL-KDD", "ton-iot-telemetry": "ToN-IoT",
         "edge-iiotset": "Edge-IIoTset", "dapt-2020": "DAPT 2020"}
# Drawn bottom-to-top, so the pair the paper is about sits together at the foot.
ORDER = ["edge-iiotset", "ton-iot-telemetry", "dapt-2020", "nsl-kdd"]

INK, COST, SLACK, ACCENT = "#1a1a1a", "#2f2f2f", "#c9c9c9", "#b23a48"


def rows():
    nums = json.loads(NUM.read_text(encoding="utf-8"))["benchmarks"]
    out = []
    for b in ORDER:
        cfgs = nums[b]["configurations"]
        d = next(v for k, v in cfgs.items() if k == "full" or k.lower().startswith("full"))
        out.append((LABEL[b], d["frontier_fpr_at_own_recall"], d["fpr"],
                    d["recall"], d["slack_share"]))
    return out


def main() -> int:
    data = rows()
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")

    # ---- left column: the question, the identity, the setting
    axt = fig.add_axes([0.022, 0.0, 0.25, 1.0])
    axt.axis("off")
    axt.text(0, 0.94, "How much of a\nfalse-positive rate\nbelongs to the\nbenchmark?",
             fontsize=10.5, color=INK, va="top", weight="bold", linespacing=1.35)
    axt.text(0, 0.555,
             "Records sharing a feature\nvector but carrying different\n"
             "labels are an error no model\ncan avoid. Counting them\n"
             "splits a reported rate exactly:",
             fontsize=6.9, color="#333333", va="top", linespacing=1.55)
    axt.text(0, 0.225, "observed =\nbenchmark cost + model slack",
             fontsize=7.3, color=INK, va="top", style="italic", linespacing=1.5)
    axt.text(0, 0.085, "Four public corpora, one detector.",
             fontsize=6.9, color="#333333", va="top")

    # ---- right: the decomposition
    ax = fig.add_axes([0.345, 0.20, 0.545, 0.70])
    y = np.arange(len(data))
    cost = np.array([r[1] for r in data])
    obs = np.array([r[2] for r in data])

    ax.barh(y, cost, color=COST, height=0.58, label="benchmark cost (unavoidable)")
    ax.barh(y, obs - cost, left=cost, color=SLACK, height=0.58,
            edgecolor=INK, lw=0.6, label="model slack (avoidable)")

    for yi, (name, c, o, rec, share) in zip(y, data):
        ax.text(o + 0.015, yi + 0.16, f"{o:.3f}", va="center",
                fontsize=8.2, color=INK, weight="bold")
        ax.text(o + 0.015, yi - 0.22, f"recall {rec:.2f}", va="center",
                fontsize=6.4, color="#777777")
        if o - c > 0.25:            # only where the band is wide enough to hold text
            ax.text(c + (o - c) / 2, yi, f"{100*share:.0f}% avoidable",
                    va="center", ha="center", fontsize=7.2, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in data], fontsize=8.5)
    ax.set_xlim(0, 0.80)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlabel("false-positive rate", fontsize=7.6, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.8, pad=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center",
              bbox_to_anchor=(0.42, -0.14), ncol=2, columnspacing=1.4, handlelength=1.3)

    # The pair the paper exists to separate: y = 0 Edge-IIoTset, y = 1 ToN-IoT.
    ax.annotate("", xy=(0.665, 0.0), xytext=(0.665, 1.0),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0))
    ax.text(0.678, 0.5, "same rate,\nopposite causes", fontsize=6.9, color=ACCENT,
            va="center", ha="left", style="italic", linespacing=1.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)

    from PIL import Image
    w, h = Image.open(f"{OUT}.png").size
    print(f"  wrote {OUT}.png  {w} x {h} px "
          f"({'meets' if w >= 1328 and h >= 531 else 'BELOW'} the 1328 x 531 minimum)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
