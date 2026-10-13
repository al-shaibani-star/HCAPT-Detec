"""
Step 1 (b): build every NSL-KDD table/figure of the revised manuscript from the
frozen predictions written by r01_train_nslkdd.py.  Nothing here is fitted on TEST.

Covers: R1-1 (Fig 7 from the same predictions as the +ZT row), R1-2 (recovery
recall with explicit denominator, challenge outcome), R1-3 (per-class recall from
one prediction set), R1-5 (unambiguous configuration names), R1-6 (validator
error correlation), R1-8 (named metric, seeds, McNemar discordant counts, Holm),
R2-4 (Zero-Trust thresholds chosen on VAL only).

Usage:  python revision\\r02_tables_nslkdd.py            (full run)
        python revision\\r02_tables_nslkdd.py --smoke    (smoke outputs)
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import rev_common as rc
from rev_common import CLASS_NAMES, NUM_CLASSES, OUT_DIR

LABEL = {
    # single learners (step 2 baselines)
    "lr": "Logistic regression (class-weighted)",
    "rf_unw": "RF alone (no weighting)", "rf_bal": "RF alone (class-weighted)",
    "xgb_unw": "XGBoost alone (no weighting)", "xgb_w": "XGBoost alone (class-weighted)",
    "lgbm": "LightGBM alone (class-weighted)", "nn_w": "DNN alone (class-weighted)",
    "ens_unw": "Plain ensemble (no weighting, equal vote)",
    # the paper's ensemble with different imbalance treatments
    "none": "Ensemble + class weighting only",
    "ros": "Ensemble + random oversampling", "smote": "Ensemble + SMOTE",
    "bsmote": "Ensemble + Borderline-SMOTE", "adasyn": "Ensemble + ADASYN",
    "ctgan": "Ensemble + CTGAN", "cgan": "Ensemble + CGAN", "wgan": "Ensemble + WGAN-GP",
    "hybrid": "Ensemble + WGAN-GP/SMOTE (proposed)"}
DERIVED = {"rf_bal": "rf", "xgb_w": "xgb", "nn_w": "nn"}   # learners read from the 'none' run
DEFAULT_ZT = {"high_conf": 0.85, "high_risk": 0.70}   # pre-declared values of the original paper
MID_CONF, MID_RISK = 0.55, 0.40


# ------------------------------------------------------------------ Zero-Trust
def zt_policy(p, risk, high_conf, high_risk):
    """Vectorised PolicyEngine.decide without the stateful context term.
    The context term only moves samples between BLOCK/QUARANTINE/MONITOR and never
    changes a final label, so labels and CHALLENGE counts are exact."""
    pred = p.argmax(1); conf = p.max(1)
    att = pred != 0
    act = np.full(len(p), "MONITOR", dtype=object)
    chal = (~att) & (conf >= high_conf) & (risk >= high_risk)
    block = att & (conf >= high_conf) & (risk >= high_risk) & ~chal
    quar = att & (conf >= MID_CONF) & (risk >= MID_RISK) & ~chal & ~block
    low = (conf < MID_CONF) & ~chal & ~block & ~quar
    allow = (~att) & (conf >= high_conf) & (risk < MID_RISK) & ~chal & ~block & ~quar & ~low
    act[chal] = "CHALLENGE"; act[block] = "BLOCK"; act[quar] = "QUARANTINE"; act[allow] = "ALLOW"
    final = pred.copy(); final[chal] = 1   # a challenge is counted as an attack alert
    return act, final, pred, conf


def tune_zt_on_val(p_val, r_val, y_val, min_precision=0.80, max_false_rate=0.005):
    """Grid search on VAL only.  Criterion fixed in advance: maximise correct
    challenges subject to challenge precision >= 0.80 and false challenges
    <= 0.5 % of VAL normals; fall back to the pre-declared values if nothing qualifies."""
    best, best_key = dict(DEFAULT_ZT, source="pre-declared (no VAL point qualified)"), None
    n_norm = int((y_val == 0).sum())
    for hc, hr in itertools.product(np.arange(0.70, 0.99, 0.05), np.arange(0.50, 0.99, 0.05)):
        act, _, _, _ = zt_policy(p_val, r_val, hc, hr)
        ch = act == "CHALLENGE"
        good = int((ch & (y_val != 0)).sum()); bad = int((ch & (y_val == 0)).sum())
        if good == 0 or good / (good + bad) < min_precision or bad > max_false_rate * n_norm:
            continue
        key = (good, -bad, hc)
        if best_key is None or key > best_key:
            best_key = key
            best = {"high_conf": round(float(hc), 2), "high_risk": round(float(hr), 2),
                    "source": "VAL-tuned", "val_correct": good, "val_false": bad}
    return best


def zt_report(p, risk, y, high_conf, high_risk):
    hc, hr = high_conf, high_risk
    act, final, pred, conf = zt_policy(p, risk, hc, hr)
    ch = act == "CHALLENGE"
    eligible = (pred == 0) & (conf >= hc) & (y != 0)        # high-confidence benign errors
    all_hc_err = (conf >= hc) & (pred != y)                  # every high-confidence error (Fig 9)
    good = int((ch & (y != 0)).sum()); bad = int((ch & (y == 0)).sum())
    by_action = {}
    for a in ["ALLOW", "MONITOR", "CHALLENGE", "QUARANTINE", "BLOCK"]:
        m = act == a
        by_action[a] = {"count": int(m.sum()), "pct": float(m.mean() * 100),
                        "ensemble_says_attack": int((m & (pred != 0)).sum()),
                        "truly_attack": int((m & (y != 0)).sum())}
    auto_only = pred.copy()      # CHALLENGE treated as no automatic detection
    return {
        "thresholds": {"high_conf": hc, "high_risk": hr},
        "binary_with_challenge_as_alert": rc.binary_metrics(y, final),
        "binary_automatic_only": rc.binary_metrics(y, auto_only),
        "multiclass_accuracy": float((final == y).mean()),
        "challenges": int(ch.sum()), "correct_challenges": good, "false_challenges": bad,
        "challenge_precision": good / max(good + bad, 1),
        "eligible_high_conf_benign_errors": int(eligible.sum()),
        "recovery_recall": good / max(int(eligible.sum()), 1),
        "all_high_conf_errors": int(all_hc_err.sum()),
        "actions": by_action,
    }


# ------------------------------------------------------------------ statistics
def bootstrap_ci(y, pred, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); vals.append(fn(y[i], pred[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar(y, a, b):
    """Exact McNemar on binary correctness; returns discordant counts and p."""
    ca = (a != 0) == (y != 0); cb = (b != 0) == (y != 0)
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_exact": float(p)}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(run, 1.0)
    return adj.tolist()


def ece(p, y, bins=15):
    conf = p.max(1); correct = p.argmax(1) == y; e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    pdir = OUT_DIR / ("preds_smoke" if a.smoke else "preds")
    tdir = OUT_DIR / ("tables_smoke" if a.smoke else "tables"); tdir.mkdir(exist_ok=True)
    d = np.load(OUT_DIR / "nslkdd_data.npz")
    y_val, y_te = d["y_val"], d["y_test"]

    seeds = sorted({int(f.name.split("_s")[1].split("_")[0]) for f in pdir.glob("nslkdd_s*_risk.npz")})
    def have(c):
        f = "none" if c in DERIVED else c
        return all((pdir / f"nslkdd_s{s}_{f}.npz").exists() for s in seeds)
    cfgs = [c for c in LABEL if have(c)]
    print(f"[tables] seeds {seeds} | configs {cfgs}")

    per_seed = {c: [] for c in cfgs}; zt_rows = {"default": [], "tuned": []}
    seed_detail = {}
    for s in seeds:
        R = np.load(pdir / f"nslkdd_s{s}_risk.npz")
        P = {c: np.load(pdir / f"nslkdd_s{s}_{c}.npz") for c in cfgs if c not in DERIVED}
        for c, learner in DERIVED.items():
            if c in cfgs:
                P[c] = {"p_val": P["none"][f"pv_{learner}"], "p_test": P["none"][f"pt_{learner}"]}
        rows = {}
        for c in cfgs:
            pr = P[c]["p_test"].argmax(1)
            m = rc.binary_metrics(y_te, pr)
            m["multiclass_accuracy"] = float((pr == y_te).mean())
            m["per_class_recall"] = {CLASS_NAMES[k]: float((pr[y_te == k] == k).mean())
                                     for k in range(NUM_CLASSES)}
            m["ece"] = ece(P[c]["p_test"], y_te)
            rows[c] = m; per_seed[c].append(m)
        # Zero-Trust on the proposed configuration (or the best available)
        base = "hybrid" if "hybrid" in cfgs else cfgs[-1]
        tuned = tune_zt_on_val(P[base]["p_val"], R["r_val"], y_val)
        z_def = zt_report(P[base]["p_test"], R["r_test"], y_te, **DEFAULT_ZT)
        z_tun = zt_report(P[base]["p_test"], R["r_test"], y_te, tuned["high_conf"], tuned["high_risk"])
        z_tun["tuning"] = tuned
        zt_rows["default"].append(z_def); zt_rows["tuned"].append(z_tun)

        # McNemar of every configuration (+ZT) against RF alone (from the 'none' run)
        tests = {}
        if "none" in cfgs:
            rf = P["none"]["pt_rf"].argmax(1)
            comps = {c: P[c]["p_test"].argmax(1) for c in cfgs}
            comps[f"{base}+ZT(tuned)"] = zt_policy(P[base]["p_test"], R["r_test"],
                                                   tuned["high_conf"], tuned["high_risk"])[1]
            names = list(comps)
            res = [mcnemar(y_te, comps[n], rf) for n in names]
            adj = holm([r["p_exact"] for r in res])
            for n, r, q in zip(names, res, adj):
                r["p_holm"] = q; tests[n] = r
        # Validator / ensemble error correlation (R1-6)
        ens_err = (P[base]["p_test"].argmax(1) != 0) != (y_te != 0)
        risk_err = (R["r_test"] >= 0.5) != (y_te != 0)
        phi = float(np.corrcoef(ens_err, risk_err)[0, 1])
        corr = {"P(risk wrong)": float(risk_err.mean()),
                "P(risk wrong | ensemble wrong)": float(risk_err[ens_err].mean()),
                "joint_errors": int((ens_err & risk_err).sum()),
                "ensemble_errors": int(ens_err.sum()), "risk_errors": int(risk_err.sum()),
                "phi_error_correlation": phi}
        seed_detail[s] = {"zt_default": z_def, "zt_tuned": z_tun, "mcnemar_vs_rf": tests,
                          "error_correlation": corr, "rows": rows}
        if s == seeds[0]:
            # 5x5 confusion matrix of the +ZT(tuned) decisions for Fig 7
            fin = zt_policy(P[base]["p_test"], R["r_test"], tuned["high_conf"], tuned["high_risk"])[1]
            cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
            np.add.at(cm, (y_te, fin), 1)
            seed_detail[s]["confusion_zt_tuned"] = cm
            # bootstrap CIs on the proposed + ZT
            seed_detail[s]["bootstrap_ci_zt_tuned"] = {
                "accuracy": bootstrap_ci(y_te, fin, lambda y, p: rc.binary_metrics(y, p)["accuracy"]),
                "f1": bootstrap_ci(y_te, fin, lambda y, p: rc.binary_metrics(y, p)["f1"])}

    def ms(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    summary = {"seeds": seeds, "metric_note": "binary metrics = Normal vs any attack on KDDTest+ (22,544)",
               "ablation": {}, "zt": {}}
    for c in cfgs:
        rs = per_seed[c]
        summary["ablation"][LABEL[c]] = {
            k: ms([r[k] for r in rs]) for k in ("accuracy", "precision", "recall", "f1", "fpr",
                                                "multiclass_accuracy", "ece")}
        summary["ablation"][LABEL[c]]["per_class_recall"] = {
            n: ms([r["per_class_recall"][n] for r in rs]) for n in CLASS_NAMES}
    for k in ("default", "tuned"):
        zs = zt_rows[k]
        summary["zt"][k] = {
            "accuracy": ms([z["binary_with_challenge_as_alert"]["accuracy"] for z in zs]),
            "f1": ms([z["binary_with_challenge_as_alert"]["f1"] for z in zs]),
            "fpr": ms([z["binary_with_challenge_as_alert"]["fpr"] for z in zs]),
            "challenges": ms([z["challenges"] for z in zs]),
            "challenge_precision": ms([z["challenge_precision"] for z in zs]),
            "recovery_recall": ms([z["recovery_recall"] for z in zs]),
            "eligible": ms([z["eligible_high_conf_benign_errors"] for z in zs]),
        }
    rc.dump_json(summary, tdir / "nslkdd_summary.json")
    rc.dump_json(seed_detail, tdir / "nslkdd_per_seed.json")

    # ---- console tables
    print("\nTable 8 (revised) - KDDTest+, binary, mean ± sd over seeds")
    print(f"{'Configuration':<42}{'Acc':>14}{'Rec':>14}{'F1':>14}{'FPR':>14}{'ECE':>14}")
    for name, v in summary["ablation"].items():
        print(f"{name:<42}" + "".join(f"{v[k]['mean']:>8.4f}±{v[k]['std']:.3f}"
                                      for k in ("accuracy", "recall", "f1", "fpr", "ece")))
    for k in ("default", "tuned"):
        v = summary["zt"][k]
        print(f"{'  + Zero-Trust (' + k + ')':<42}" + "".join(f"{v[m]['mean']:>8.4f}±{v[m]['std']:.3f}"
                                                         for m in ("accuracy", "f1", "fpr")))
        print(f"      challenges {v['challenges']['mean']:.1f} | precision {v['challenge_precision']['mean']:.3f}"
              f" | recovery {v['recovery_recall']['mean']:.4f} of {v['eligible']['mean']:.0f} eligible")
    print("\nPer-class recall (multiclass argmax)")
    for name, v in summary["ablation"].items():
        print(f"{name:<42}" + "".join(f"{n}:{v['per_class_recall'][n]['mean']:.3f} " for n in CLASS_NAMES))
    s0 = seeds[0]
    print(f"\nSeed {s0} detail: ZT tuning -> {seed_detail[s0]['zt_tuned']['tuning']}")
    print("Actions (tuned):", {a: (v['count'], v['ensemble_says_attack'], v['truly_attack'])
                               for a, v in seed_detail[s0]['zt_tuned']['actions'].items()})
    print("Error correlation:", seed_detail[s0]["error_correlation"])
    print("McNemar vs RF:", {n: (r['a_right_b_wrong'], r['a_wrong_b_right'], f"{r['p_holm']:.2g}")
                             for n, r in seed_detail[s0]["mcnemar_vs_rf"].items()})
    print(f"\n[tables] written to {tdir}")


if __name__ == "__main__":
    main()
