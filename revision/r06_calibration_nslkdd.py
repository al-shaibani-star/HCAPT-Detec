"""
Step 4 of the IJIES revision (Reviewer 2, comment 3; Reviewer 1, comment 2).

Question: is the gain attributed to "calibration + Zero-Trust" obtainable by
plain threshold tuning, and does calibration change decisions at all?

Works only on the frozen predictions from r01/r03 (no retraining).
Every calibrator and every threshold is fitted on VAL; KDDTest+ is scored once.

Decision rules compared (binary: Normal vs attack), per model and seed:
  argmax          predicted class != Normal (uncalibrated, the paper's default)
  temp-argmax     temperature scaling fitted on VAL (NLL), then argmax
                  -> argmax-invariant by construction; reported to show that
                     calibration alone cannot change detections
  thr@b           attack score s = 1 - p(Normal); threshold set on VAL so that
                  VAL FPR = b  (no extra model)
  iso-thr@b       isotonic-calibrated s, same budget rule
  risk-thr@b      Zero-Trust risk scorer alone, same budget rule
  max(s,risk)@b   ensemble OR validator (score = max), same budget rule
  ZT-policy       the paper's rule (conf >= 0.85 & risk >= 0.70 -> CHALLENGE)
Budgets b in {1 %, 2 %, 3 %, 5 %} of VAL normals.

Also reported: ECE on VAL and TEST before/after temperature scaling, threshold-
free ROC-AUC on TEST for each score, analyst workload (alerts and challenges),
and R2L detection rate (the class Zero-Trust was designed for).

Outputs: revision/out/tables/calibration_summary.csv, calibration_per_seed.json,
         revision/out/figures/fig_reliability.png
Usage:   python revision\\r06_calibration_nslkdd.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import rev_common as rc
from rev_common import OUT_DIR, PRED_DIR

MODELS = {"hybrid": "Ensemble + WGAN-GP/SMOTE (proposed)",
          "none": "Ensemble + class weighting only",
          "nn_w": "DNN alone (class-weighted)",
          "lr": "Logistic regression (class-weighted)"}
DERIVED = {"nn_w": "nn"}
BUDGETS = [0.01, 0.02, 0.03, 0.05]
ZT = (0.85, 0.70)


def load(seed, m):
    if m in DERIVED:
        z = np.load(PRED_DIR / f"nslkdd_s{seed}_none.npz")
        return z[f"pv_{DERIVED[m]}"], z[f"pt_{DERIVED[m]}"]
    z = np.load(PRED_DIR / f"nslkdd_s{seed}_{m}.npz")
    return z["p_val"], z["p_test"]


def fit_temperature(p, y):
    logit = np.log(np.clip(p, 1e-12, 1))

    def nll(t):
        z = logit / t; z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(p, t):
    z = np.log(np.clip(p, 1e-12, 1)) / t; z -= z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(p, y, bins=15):
    conf = p.max(1); ok = p.argmax(1) == y; e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def thr_for_budget(score_val, y_val, b):
    """Smallest threshold whose VAL FPR <= b (flag = score > thr)."""
    return float(np.quantile(score_val[y_val == 0], 1 - b))


def evaluate(flag, y):
    m = rc.binary_metrics(y, flag.astype(int))
    m["alerts"] = int(flag.sum())
    m["r2l_detected"] = float(flag[y == 3].mean())
    m["u2r_detected"] = float(flag[y == 4].mean())
    return m


def main():
    d = np.load(OUT_DIR / "nslkdd_data.npz"); y_val, y_te = d["y_val"], d["y_test"]
    seeds = sorted({int(f.name.split("_s")[1].split("_")[0]) for f in PRED_DIR.glob("nslkdd_s*_risk.npz")})
    rows, detail = [], {}
    for s in seeds:
        R = np.load(PRED_DIR / f"nslkdd_s{s}_risk.npz"); r_val, r_te = R["r_val"], R["r_test"]
        for m, mname in MODELS.items():
            pv, pt = load(s, m)
            sv, st = 1 - pv[:, 0], 1 - pt[:, 0]
            T = fit_temperature(pv, y_val)
            pv_T, pt_T = apply_temperature(pv, T), apply_temperature(pt, T)
            iso = IsotonicRegression(out_of_bounds="clip").fit(sv, (y_val != 0).astype(int))
            sv_i, st_i = iso.predict(sv), iso.predict(st)
            comb_v, comb_t = np.maximum(sv, r_val), np.maximum(st, r_te)
            detail[f"s{s}_{m}"] = {
                "temperature": T,
                "ece_val_raw": ece(pv, y_val), "ece_val_T": ece(pv_T, y_val),
                "ece_test_raw": ece(pt, y_te), "ece_test_T": ece(pt_T, y_te),
                "argmax_changed_by_T": int((pt.argmax(1) != pt_T.argmax(1)).sum()),
                "auc_test": {"s": roc_auc_score(y_te != 0, st), "iso_s": roc_auc_score(y_te != 0, st_i),
                             "risk": roc_auc_score(y_te != 0, r_te),
                             "max(s,risk)": roc_auc_score(y_te != 0, comb_t)}}

            def add(rule, budget, flag, extra=None):
                row = {"seed": s, "model": mname, "rule": rule, "val_fpr_budget": budget,
                       **evaluate(flag, y_te), **(extra or {})}
                rows.append(row)
            add("argmax", None, pt.argmax(1) != 0)
            add("temp-argmax", None, pt_T.argmax(1) != 0)
            pred, conf = pt.argmax(1), pt.max(1)
            ch = (pred == 0) & (conf >= ZT[0]) & (r_te >= ZT[1])
            zt_flag = (pred != 0) | ch
            add("ZT-policy (0.85/0.70)", None, zt_flag, {"challenges": int(ch.sum())})
            # Descriptive only (uses TEST labels to place the threshold): recall of a
            # single-score threshold at exactly the TEST FPR the ZT policy produced.
            zt_fpr = rc.binary_metrics(y_te, zt_flag.astype(int))["fpr"]
            for nm, sc in (("s", st), ("risk", r_te)):
                t = float(np.quantile(sc[y_te == 0], 1 - zt_fpr))
                add(f"{nm}@ZT-test-FPR (oracle, descriptive)", None, sc > t)
            for b in BUDGETS:
                add("thr", b, st > thr_for_budget(sv, y_val, b))
                add("iso-thr", b, st_i > thr_for_budget(sv_i, y_val, b))
                add("risk-thr", b, r_te > thr_for_budget(r_val, y_val, b))
                add("max(s,risk)", b, comb_t > thr_for_budget(comb_v, y_val, b))

    df = pd.DataFrame(rows)
    tdir = OUT_DIR / "tables"; tdir.mkdir(exist_ok=True)
    keys = ["model", "rule", "val_fpr_budget"]
    agg = df.groupby(keys, dropna=False).agg(
        recall=("recall", "mean"), recall_sd=("recall", "std"), fpr=("fpr", "mean"),
        precision=("precision", "mean"), f1=("f1", "mean"), accuracy=("accuracy", "mean"),
        alerts=("alerts", "mean"), r2l=("r2l_detected", "mean"), u2r=("u2r_detected", "mean"),
        challenges=("challenges", "mean")).reset_index()
    agg.to_csv(tdir / "calibration_summary.csv", index=False)
    rc.dump_json(detail, tdir / "calibration_per_seed.json")

    # summary of calibration diagnostics (mean over seeds)
    diag = pd.DataFrame([{"model": k.split("_", 1)[1], **{kk: v for kk, v in d_.items() if kk != "auc_test"},
                          **{f"auc_{a}": b for a, b in d_["auc_test"].items()}} for k, d_ in detail.items()])
    diag = diag.groupby("model").mean(numeric_only=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nCalibration diagnostics (mean over seeds)")
    print(diag.round(4).to_string())
    print("\nDecision rules on KDDTest+ (thresholds fixed on VAL)")
    print(agg.drop(columns=["recall_sd"]).round(4).to_string(index=False))

    # reliability diagram: proposed model, seed 0, raw vs temperature-scaled
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv, pt = load(seeds[0], "hybrid"); T = fit_temperature(pv, y_val)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (name, p, y) in zip(axes, [("VAL (in-distribution)", pv, y_val), ("KDDTest+ (shifted)", pt, y_te)]):
        for lab, q, col in [("uncalibrated", p, "#d62728"), (f"temperature T={T:.2f}", apply_temperature(p, T), "#1f77b4")]:
            conf = q.max(1); ok = q.argmax(1) == y
            edges = np.linspace(0, 1, 11); xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mm = (conf > lo) & (conf <= hi)
                if mm.sum() > 20:
                    xs.append(conf[mm].mean()); ys.append(ok[mm].mean())
            ax.plot(xs, ys, marker="o", ms=3, color=col, label=f"{lab} (ECE {ece(q, y):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_title(name, fontsize=9); ax.set_xlabel("confidence", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8); ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fdir = OUT_DIR / "figures"; fdir.mkdir(exist_ok=True)
    fig.savefig(fdir / "fig_reliability.png", dpi=300)
    print(f"\n[r06] tables -> {tdir}, figure -> {fdir / 'fig_reliability.png'}")


if __name__ == "__main__":
    main()
