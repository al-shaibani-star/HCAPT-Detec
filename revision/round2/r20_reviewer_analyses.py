"""
r20 - analyses added for the second review round (IJIES paper 20267615).

Reads the frozen artefacts of the first revision and writes ONLY to --out.
Nothing in the implementation folder is modified.

  A  learning curve: seed-level GAN - SMOTE recall differences, paired 95 % t-intervals,
     and the stricter changeover rule (Reviewer 1, comments 1 and 4)
  B  Zero-Trust vs single-score thresholds selected on VAL only, matched to the policy's
     VAL false-positive rate and to its VAL alert count, scored once on KDDTest+, with a
     test-flow bootstrap interval of the 5-seed mean difference (Reviewer 1, comment 3)
  C  analyst-referral cost of the CHALLENGE action and its precision under other attack
     prevalences (Reviewer 2, comment 3)
  D  effect sizes (accuracy difference with bootstrap CI) for the McNemar comparisons
     of Table 8 (Reviewer 2, comment 5)
  E  DAPT 2020 leave-one-day-out re-run with per-stage recall inside each held-out day,
     training support of every stage, Wilson and Flow-ID cluster-bootstrap intervals
     (Reviewer 1, comment 8; Reviewer 2, comment 5)  [needs the GPU and the raw DAPT csv]

Usage:
  python r20_reviewer_analyses.py --impl E:\\...\\05-Implementation --out .\\out [--skip-dapt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ZT_CONF, ZT_RISK = 0.85, 0.70
B = 1000


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


# ---------------------------------------------------------------- A
def part_a(impl, out):
    df = pd.read_csv(impl / "revision/out/learning_curve.csv")
    rows = []
    for cls, g in df.groupby("class_name"):
        for gan in ("cgan", "wgan"):
            for sp in ("recall_val", "recall_test"):
                p = g[g.method.isin([gan, "smote"])].pivot_table(
                    index=["seed", "n_real"], columns="method", values=sp)
                d = (p[gan] - p["smote"]).unstack("n_real")
                levels = list(d.columns)
                for n in levels:
                    x = d[n].values
                    m, lo, hi = tci(x)
                    rows.append({"class": cls, "gan": gan, "split": sp.replace("recall_", ""),
                                 "n": int(n), **{f"seed{i}": float(v) for i, v in enumerate(x)},
                                 "mean": m, "ci_lo": lo, "ci_hi": hi,
                                 "seeds_positive": int((x > 0).sum()), "terminal_level": n == levels[-1]})
    a = pd.DataFrame(rows)
    # stricter rule: accept n* only if CI lower bound > 0 at every tested level >= n*
    rule = []
    for (cls, gan, sp), g in a.groupby(["class", "gan", "split"]):
        g = g.sort_values("n")
        ok = (g.ci_lo > 0).values
        acc = None
        for i in range(len(g)):
            if ok[i:].all():
                acc = int(g.n.values[i]); break
        rule.append({"class": cls, "gan": gan, "split": sp,
                     "first_level_with_sustained_ci_above_0": acc,
                     "only_terminal": acc is not None and acc == int(g.n.values[-1]),
                     "max_abs_mean_delta_n_ge_1000": float(g[g.n >= 1000]["mean"].abs().max())
                     if (g.n >= 1000).any() else np.nan})
    a.to_csv(out / "A_learning_curve_paired_deltas.csv", index=False)
    pd.DataFrame(rule).to_csv(out / "A_changeover_strict_rule.csv", index=False)
    return a, pd.DataFrame(rule)


# ---------------------------------------------------------------- helpers for B-D
def bin_metrics(y, flag):
    t = y != 0
    tp = (t & flag).sum(); fp = (~t & flag).sum(); fn = (t & ~flag).sum(); tn = (~t & ~flag).sum()
    rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    return {"recall": rec, "fpr": fpr, "precision": prec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "alerts": int(flag.sum()), "accuracy": ((flag == t).mean())}


def zt(p, r):
    pred, conf = p.argmax(1), p.max(1)
    ch = (pred == 0) & (conf >= ZT_CONF) & (r >= ZT_RISK)
    return (pred != 0) | ch, ch


def load_preds(impl):
    P = impl / "revision/out/preds"
    d = np.load(impl / "revision/out/nslkdd_data.npz")
    seeds = sorted({int(f.name.split("_s")[1].split("_")[0]) for f in P.glob("nslkdd_s*_risk.npz")})
    return P, d["y_val"], d["y_test"], seeds


# ---------------------------------------------------------------- B
def part_b(impl, out):
    P, yv, yt, seeds = load_preds(impl)
    rng = np.random.default_rng(20)
    idx = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
    per_seed, flags = [], {}
    for s in seeds:
        z = np.load(P / f"nslkdd_s{s}_hybrid.npz"); R = np.load(P / f"nslkdd_s{s}_risk.npz")
        pv, pt, rv, rt = z["p_val"], z["p_test"], R["r_val"], R["r_test"]
        sv, st = 1 - pv[:, 0], 1 - pt[:, 0]
        zt_v, _ = zt(pv, rv); zt_t, _ = zt(pt, rt)
        mv = bin_metrics(yv, zt_v)
        nv = yv == 0
        cand = {"ZT policy": zt_t}
        for nm, scv, sct in (("score s", sv, st), ("validator r", rv, rt),
                             ("max(s, r)", np.maximum(sv, rv), np.maximum(st, rt))):
            thr_fpr = np.quantile(scv[nv], 1 - mv["fpr"])            # VAL FPR matched to the policy
            thr_alert = np.quantile(scv, 1 - mv["alerts"] / len(yv))  # VAL alert count matched
            cand[f"{nm}, VAL-matched FPR"] = sct > thr_fpr
            cand[f"{nm}, VAL-matched alerts"] = sct > thr_alert
        for k, f in cand.items():
            per_seed.append({"seed": s, "rule": k, "zt_val_fpr": mv["fpr"], "zt_val_alerts": mv["alerts"],
                             "r2l_detected": float(f[yt == 3].mean()), **bin_metrics(yt, f)})
            flags.setdefault(k, []).append(f)
    ps = pd.DataFrame(per_seed)
    att = yt != 0
    rows = []
    zt_flags = flags["ZT policy"]
    for k, fl in flags.items():
        g = ps[ps.rule == k]
        d_rec, d_fpr = [], []
        for ii in idx:
            yb = att[ii]
            r_c = np.mean([f[ii][yb].mean() for f in fl]); r_z = np.mean([f[ii][yb].mean() for f in zt_flags])
            f_c = np.mean([f[ii][~yb].mean() for f in fl]); f_z = np.mean([f[ii][~yb].mean() for f in zt_flags])
            d_rec.append(r_c - r_z); d_fpr.append(f_c - f_z)
        rows.append({"rule": k, "recall": g.recall.mean(), "recall_sd": g.recall.std(), "fpr": g.fpr.mean(),
                     "f1": g.f1.mean(), "alerts": g.alerts.mean(), "r2l_detected": g.r2l_detected.mean(),
                     "d_recall_vs_zt": g.recall.mean() - ps[ps.rule == "ZT policy"].recall.mean(),
                     "d_recall_ci": [float(np.percentile(d_rec, 2.5)), float(np.percentile(d_rec, 97.5))],
                     "d_fpr_vs_zt": g.fpr.mean() - ps[ps.rule == "ZT policy"].fpr.mean(),
                     "d_fpr_ci": [float(np.percentile(d_fpr, 2.5)), float(np.percentile(d_fpr, 97.5))],
                     "zt_val_fpr_mean": g.zt_val_fpr.mean(), "zt_val_alerts_mean": g.zt_val_alerts.mean()})
    res = pd.DataFrame(rows)
    ps.to_csv(out / "B_zt_val_matched_per_seed.csv", index=False)
    res.to_csv(out / "B_zt_val_matched_summary.csv", index=False)
    return res


# ---------------------------------------------------------------- C
def part_c(impl, out):
    P, yv, yt, seeds = load_preds(impl)
    rows = []
    for s in seeds:
        z = np.load(P / f"nslkdd_s{s}_hybrid.npz"); R = np.load(P / f"nslkdd_s{s}_risk.npz")
        pt, rt = z["p_test"], R["r_test"]
        _, ch = zt(pt, rt)
        pred, conf = pt.argmax(1), pt.max(1)
        elig = (yt != 0) & (pred == 0) & (conf >= ZT_CONF)
        rows.append({"seed": s, "challenges": int(ch.sum()), "true_referrals": int((ch & (yt != 0)).sum()),
                     "false_referrals": int((ch & (yt == 0)).sum()), "eligible": int(elig.sum()),
                     "recovered": int((ch & elig).sum()), "normals": int((yt == 0).sum()),
                     "attacks": int((yt != 0).sum())})
    c = pd.DataFrame(rows)
    a = c.true_referrals.sum() / c.attacks.sum()      # referral rate per attack flow
    b = c.false_referrals.sum() / c.normals.sum()     # referral rate per normal flow
    proj = []
    for prev in (c.attacks[0] / (c.attacks[0] + c.normals[0]), 0.10, 0.01, 0.001):
        prec = a * prev / (a * prev + b * (1 - prev))
        per_m = 1e6 * (a * prev + b * (1 - prev))
        proj.append({"attack_prevalence": prev, "challenge_precision": prec,
                     "referrals_per_million_flows": per_m,
                     "false_referrals_per_million_flows": 1e6 * b * (1 - prev),
                     "attacks_recovered_per_million_flows": 1e6 * a * prev})
    c.to_csv(out / "C_referral_counts_per_seed.csv", index=False)
    pd.DataFrame(proj).to_csv(out / "C_referral_prevalence_projection.csv", index=False)
    return c, pd.DataFrame(proj), a, b


# ---------------------------------------------------------------- D
def part_d(impl, out):
    P, yv, yt, seeds = load_preds(impl)
    rng = np.random.default_rng(21)
    idx = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
    att = yt != 0
    comps = {"DNN alone (class-weighted)": ("none", "pt_nn"), "Ensemble + SMOTE": ("smote", "p_test"),
             "Ensemble + WGAN-GP/SMOTE (no ZT)": ("hybrid", "p_test"),
             "Ensemble + class weighting only": ("none", "p_test"),
             "Logistic regression (class-weighted)": ("lr", "p_test")}
    ok_zt, ok = [], {k: [] for k in comps}
    for s in seeds:
        z = np.load(P / f"nslkdd_s{s}_hybrid.npz"); R = np.load(P / f"nslkdd_s{s}_risk.npz")
        f, _ = zt(z["p_test"], R["r_test"]); ok_zt.append(f == att)
        for k, (m, key) in comps.items():
            pp = np.load(P / f"nslkdd_s{s}_{m}.npz")[key]
            ok[k].append((pp.argmax(1) != 0) == att)
    rows = []
    for k in comps:
        d = np.mean([c.mean() - zz.mean() for c, zz in zip(ok[k], ok_zt)])
        bs = [np.mean([c[ii].mean() - zz[ii].mean() for c, zz in zip(ok[k], ok_zt)]) for ii in idx]
        rows.append({"comparator": k, "acc_comparator_minus_proposed_zt": d,
                     "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                     "flows_per_22544": d * len(yt)})
    r = pd.DataFrame(rows)
    r.to_csv(out / "D_effect_sizes_table8.csv", index=False)
    return r


# ---------------------------------------------------------------- E
def part_e(impl, out):
    sys.path.insert(0, str(impl / "revision"))
    import r08_dapt_stages as r8
    big, X, _ = r8.load()
    y = big["_y"].values; days = big["_day"].values; fid = big["Flow ID"].astype(str).values
    yb = (y != 0).astype(int)
    rng = np.random.default_rng(22)
    rows, dayrows, preds = [], [], {}
    for day in ("Tue", "Wed", "Thu", "Fri"):
        tr, te = days != day, days == day
        for model in ("lr", "xgb", "ens"):
            p = r8.fit_predict(model, X[tr], yb[tr], X[te], 2, r8.SEED)
            pr = p.argmax(1)
            preds[f"{day}_{model}"] = pr
            yt, ft = y[te], fid[te]
            att = yt != 0
            k, n = int((pr[att] == 1).sum()), int(att.sum())
            fp, nn = int((pr[~att] == 1).sum()), int((~att).sum())
            dayrows.append({"held_out_day": day, "model": model, "attack_flows": n,
                            "recall": k / n, "recall_wilson": r8.wilson(k, n)[1:],
                            "fpr": fp / nn, "fpr_wilson": r8.wilson(fp, nn)[1:]})
            for c in np.unique(yt[att]):
                m = yt == c
                kk, nm = int((pr[m] == 1).sum()), int(m.sum())
                # Flow-ID cluster bootstrap of the stage recall
                g = pd.DataFrame({"f": ft[m], "hit": pr[m] == 1}).groupby("f")["hit"].agg(["sum", "count"])
                hs, cs = g["sum"].values, g["count"].values
                bs = []
                for _ in range(2000):
                    j = rng.integers(0, len(hs), len(hs))
                    bs.append(hs[j].sum() / cs[j].sum())
                train_support = int(((y == c) & tr).sum())
                rows.append({"held_out_day": day, "stage": r8.STAGES[c], "model": model, "flows": nm,
                             "flow_ids": len(hs), "train_flows_of_stage": train_support,
                             "status": "unseen" if train_support == 0 else
                                       ("near-unseen (<100 training flows)" if train_support < 100 else "seen"),
                             "recall": kk / nm, "wilson_lo": r8.wilson(kk, nm)[1], "wilson_hi": r8.wilson(kk, nm)[2],
                             "cluster_lo": float(np.percentile(bs, 2.5)), "cluster_hi": float(np.percentile(bs, 97.5))})
            print(f"[r20-E] {day} {model} done", flush=True)
    pd.DataFrame(rows).to_csv(out / "E_dapt_dayout_per_stage.csv", index=False)
    pd.DataFrame(dayrows).to_csv(out / "E_dapt_dayout_day_level_rerun.csv", index=False)
    np.savez_compressed(out / "E_dapt_dayout_predictions.npz", **preds)
    return pd.DataFrame(rows), pd.DataFrame(dayrows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-dapt", action="store_true")
    ap.add_argument("--only", default="ABCDE")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if "A" in a.only:
        aa, rule = part_a(a.impl, a.out); print(rule.to_string())
    if "B" in a.only:
        print(part_b(a.impl, a.out).to_string())
    if "C" in a.only:
        c, proj, ra, rb = part_c(a.impl, a.out); print(c.to_string()); print(proj.to_string()); print(ra, rb)
    if "D" in a.only:
        print(part_d(a.impl, a.out).to_string())
    if "E" in a.only and not a.skip_dapt:
        st, dl = part_e(a.impl, a.out); print(dl.to_string()); print(st.to_string())


if __name__ == "__main__":
    main()
