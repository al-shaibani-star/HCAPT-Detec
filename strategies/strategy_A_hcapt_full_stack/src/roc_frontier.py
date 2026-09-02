"""
ROC and precision-recall curves with the attainability frontier on the same axes.

Task 6 of FINAL_PLAN.md. The manuscript reports one operating point per
benchmark and no ROC anywhere, which is the reviewer objection that "you chose
a threshold and showed us only that". This script answers it from stored
artefacts alone: it reloads the cached ensembles, scores the stored test split,
and plots the full curve.

The overlay is the point. A ROC curve lives in (FPR, TPR) space, and the
attainability frontier of `src/attainability.py` is a curve in exactly the same
space — the best (FPR, recall) pair obtainable from the feature representation
at each recall. Plotting them together turns two abstractions into one picture:
the vertical distance between the detector's curve and the frontier IS the model
slack, drawn at every operating point rather than asserted at one.

INFERENCE ONLY. Nothing is retrained. The models are the ones that produced the
published numbers.

Coverage is limited by what was cached: `cache/nsl-kdd/` holds no ensemble, so
NSL-KDD gets no curve and the §8 disclosure covers it instead.

    python -m src.roc_frontier
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STRATEGY_ROOT = Path(__file__).resolve().parents[1]
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

from src.attainability import (                                    # noqa: E402
    OUT_DIR, RESULTS_DIR, SPEC_VERSION, _sha256, _at_recall, load_via_loader)

HERE = Path(__file__).resolve()
CACHE_DIR = STRATEGY_ROOT / "cache"
ROC_DIR = RESULTS_DIR / "_roc"
CONFIGS = ["ens_no_gan", "ens_gan"]
BENCHMARKS = ["nsl-kdd", "ton-iot-telemetry", "edge-iiotset", "dapt-2020"]


def load_ensemble(bench: str, config: str, feature_dim: int, num_classes: int):
    """Reload a cached ensemble. Returns None if this benchmark was never cached."""
    d = CACHE_DIR / bench / "ensemble" / config
    needed = ["rf.joblib", "xgb.json", "nn.pt", "weights.npy"]
    missing = [f for f in needed if not (d / f).exists()]
    if missing:
        return None, f"missing {', '.join(missing)} in {d}"
    from src.phase2_pipeline import EnsembleAdHoc
    ens = EnsembleAdHoc(feature_dim=feature_dim, num_classes=num_classes)
    try:
        ens.load(d)
    except Exception as exc:
        return None, f"load failed: {exc!r}"
    return ens, None


def attack_scores(ens, X: np.ndarray) -> np.ndarray:
    """P(attack) under the same soft vote that produced the published labels."""
    p_rf = ens.rf.predict_proba(X)
    p_xgb = ens.xgb.predict_proba(X)
    p_nn = ens._nn_predict_proba(X)
    w = ens.weights
    p = w[0] * p_rf + w[1] * p_xgb + w[2] * p_nn
    return p[:, 1]


def roc_pr(y: np.ndarray, s: np.ndarray) -> dict:
    """ROC, PR, AUC and average precision, computed directly from the scores."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fpr, tpr, thr = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": thr},
        "pr": {"precision": prec, "recall": rec},
        "auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
    }


def frontier_curve(bench: str):
    z = OUT_DIR / f"{bench}_curve.npz"
    if not z.exists():
        return None
    a = np.load(z)
    return a["split_recall"], a["split_fpr"]


def _downsample(x: np.ndarray, y: np.ndarray, k: int = 600) -> dict:
    idx = np.unique(np.linspace(0, len(x) - 1, min(k, len(x))).astype(int))
    return {"x": [float(v) for v in x[idx]], "y": [float(v) for v in y[idx]]}


def plot(bench: str, curves: dict, front, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for name, c in curves.items():
        ax1.plot(c["roc"]["fpr"], c["roc"]["tpr"], lw=1.6,
                 label=f"{name} (AUC {c['auc']:.4f})")
        ax1.plot(c["operating_point"]["fpr"], c["operating_point"]["recall"],
                 "o", ms=6, mfc="none", mew=1.6)
    if front is not None:
        rec, fpr = front
        ax1.plot(fpr, rec, "k--", lw=1.3, label="attainability frontier")
        ax1.fill_betweenx(rec, fpr, 0, color="k", alpha=0.06)
    ax1.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":")
    ax1.set_xlabel("false-positive rate")
    ax1.set_ylabel("recall")
    ax1.set_title(f"{bench} — ROC against the frontier")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_xlim(-0.01, 1.01); ax1.set_ylim(-0.01, 1.01)

    for name, c in curves.items():
        ax2.plot(c["pr"]["recall"], c["pr"]["precision"], lw=1.6,
                 label=f"{name} (AP {c['average_precision']:.4f})")
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision")
    ax2.set_title(f"{bench} — precision–recall")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_xlim(-0.01, 1.01)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run(bench: str) -> dict:
    ds, _ = load_via_loader(bench)
    X, y = ds.X_test, ds.y_test.astype(int)
    front = frontier_curve(bench)

    curves, failures = {}, {}
    for cfg in CONFIGS:
        ens, err = load_ensemble(bench, cfg, ds.X_train.shape[1], len(np.unique(ds.y_train)))
        if ens is None:
            failures[cfg] = err
            continue
        s = attack_scores(ens, X)
        c = roc_pr(y, s)

        # The published operating point is the argmax rule, i.e. threshold 0.5.
        pred = s >= 0.5
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        c["operating_point"] = {
            "threshold": 0.5, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr": fp / (fp + tn), "recall": tp / (tp + fn),
            "precision": tp / (tp + fp) if tp + fp else None,
        }
        if front is not None:
            f = _at_recall(front[0], front[1], c["operating_point"]["recall"])
            if f:
                c["frontier_at_operating_recall"] = f["frontier_fpr"]
                c["slack_pp"] = 100 * (c["operating_point"]["fpr"] - f["frontier_fpr"])

        # The question a single operating point cannot answer: what does the
        # detector cost at the recall where the frontier is cheapest to meet?
        c["threshold_sweep"] = []
        for target in (0.80, 0.90, 0.95, 0.99):
            i = int(np.searchsorted(c["roc"]["tpr"], target, side="left"))
            if i < len(c["roc"]["tpr"]):
                fr = _at_recall(front[0], front[1], target)[  "frontier_fpr"] if front is not None else None
                c["threshold_sweep"].append({
                    "target_recall": target,
                    "attained_recall": float(c["roc"]["tpr"][i]),
                    "fpr": float(c["roc"]["fpr"][i]),
                    "threshold": float(c["roc"]["thresholds"][i]),
                    "frontier_fpr": fr,
                    "slack_pp": 100 * (float(c["roc"]["fpr"][i]) - fr) if fr is not None else None,
                })
        curves[cfg] = c

    if not curves:
        return {"benchmark": bench, "status": "no cached models", "failures": failures}

    ROC_DIR.mkdir(parents=True, exist_ok=True)
    png = ROC_DIR / f"{bench}_roc_frontier.png"
    plot(bench, curves, front, png)

    return {
        "benchmark": bench,
        "status": "ok",
        "figure": str(png),
        "failures": failures,
        "configurations": {
            k: {"auc": c["auc"], "average_precision": c["average_precision"],
                "operating_point": c["operating_point"],
                "frontier_at_operating_recall": c.get("frontier_at_operating_recall"),
                "slack_pp": c.get("slack_pp"),
                "threshold_sweep": c["threshold_sweep"],
                "roc_downsampled": _downsample(c["roc"]["fpr"], c["roc"]["tpr"]),
                "pr_downsampled": _downsample(c["pr"]["recall"], c["pr"]["precision"])}
            for k, c in curves.items()},
    }


def main() -> int:
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    report = {"spec_version": SPEC_VERSION,
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "script": {"path": str(HERE), "sha256": _sha256(HERE)},
              "method": ("inference only; cached ensembles reloaded and scored over the "
                         "stored test split; P(attack) is the same weighted soft vote "
                         "that produced the published labels"),
              "benchmarks": {}}

    for b in BENCHMARKS:
        print(f"[{b}] ...", flush=True)
        try:
            r = run(b)
        except Exception as exc:
            r = {"benchmark": b, "status": "failed", "error": repr(exc)}
        report["benchmarks"][b] = r

        if r["status"] != "ok":
            print(f"  {r['status']}"
                  + (f" — {r.get('error') or list(r.get('failures', {}).values())}"
                     if r.get("error") or r.get("failures") else ""))
            continue
        for cfg, c in r["configurations"].items():
            op = c["operating_point"]
            print(f"  {cfg:<12} AUC {c['auc']:.4f}  AP {c['average_precision']:.4f}   "
                  f"@0.5: FPR {op['fpr']:.4f} recall {op['recall']:.4f}"
                  + (f"  frontier {c['frontier_at_operating_recall']:.4f} "
                     f"slack {c['slack_pp']:+.2f} pp" if c.get("slack_pp") is not None else ""))
            for s in c["threshold_sweep"]:
                print(f"      recall {s['target_recall']:.2f} -> FPR {s['fpr']:.4f}"
                      + (f"  (frontier {s['frontier_fpr']:.4f}, "
                         f"slack {s['slack_pp']:+.2f} pp)" if s["slack_pp"] is not None else ""))

    (ROC_DIR / "roc_frontier.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {ROC_DIR/'roc_frontier.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
