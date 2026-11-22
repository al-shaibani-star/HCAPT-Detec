"""
r21 - second review round, remaining analyses (IJIES paper 20267615).

  D  DAPT 2020 protocol P2 (Flow-ID-grouped 5-fold) re-run with Flow-ID cluster-bootstrap intervals for the
     per-stage recall of Table 13                                   (Reviewer 2, comment 5)
  L  end-to-end latency of the proposed pipeline from raw NSL-KDD records to a Zero-Trust action:
     preprocessing, cold start vs. warm calls, GPU vs. CPU placement, batch-size curve, and a simulated stream
     of Poisson-arriving flows served with dynamic micro-batching   (Reviewer 1, comment 9)
  S  sensitivity of the NSL-KDD conclusions to the FIT/VAL split: three additional, independently drawn
     stratified splits, each with its own preprocessing, validator, WGAN-GP generator and ensembles
     (class weighting, SMOTE, WGAN-GP/SMOTE hybrid), scored once on KDDTest+   (Reviewer 2, comment 5)

Reads the implementation package (--impl) and writes ONLY to --out. Usage:
  python r21_latency_and_sensitivity.py --impl <05-Implementation> --out <folder> [--parts DLS]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

A = None  # parsed args


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def cluster_ci(hit, groups, rng, B=2000):
    g = pd.DataFrame({"g": groups, "h": hit}).groupby("g")["h"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    bs = [s[j].sum() / c[j].sum() for j in (rng.integers(0, len(s), len(s)) for _ in range(B))]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(s)


# =============================================================================== D
def part_d(impl, out):
    sys.path.insert(0, str(impl / "revision"))
    import r08_dapt_stages as r8
    from sklearn.model_selection import StratifiedGroupKFold
    big, X, _ = r8.load()
    y = big["_y"].values; groups = big["Flow ID"].astype(str).values
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=r8.SEED).split(X, y, groups))
    rng = np.random.default_rng(31); rows = []
    for model in ("lr", "xgb", "ens"):
        oof = np.zeros((len(y), 5), np.float32)
        for f, (tr, te) in enumerate(splits):
            oof[te] = r8.fit_predict(model, X[tr], y[tr], X[te], 5, r8.SEED + f)
        pred = oof.argmax(1)
        for c in range(1, 5):
            m = y == c
            k = int((pred[m] == c).sum()); n = int(m.sum())
            p, lo, hi = wilson(k, n)
            clo, chi, ng = cluster_ci(pred[m] == c, groups[m], rng)
            rows.append({"model": model, "stage": r8.STAGES[c], "flows": n, "flow_ids": ng, "stage_recall": p,
                         "wilson_lo": lo, "wilson_hi": hi, "cluster_lo": clo, "cluster_hi": chi,
                         "flagged_any_attack": float((pred[m] != 0).mean())})
        print(f"[r21-D] P2 {model} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "D_dapt_p2_cluster.csv", index=False)
    print(df.round(3).to_string())


# =============================================================================== shared NSL-KDD pieces
def setup_impl(impl, out):
    sys.path.insert(0, str(impl / "revision"))
    import rev_common as rc
    import r01_train_nslkdd as r1
    import ensemble_selflearn as es
    import wgan_gp_augment as wg
    import gan_augment as cg
    from sklearn.ensemble import RandomForestClassifier
    tmp = out / "models_tmp"; tmp.mkdir(exist_ok=True)
    for m in (cg, wg):                                   # never touch the published checkpoints
        m.save_checkpoint = lambda *a, **k: None; m.MODELS_DIR = tmp
    es.RandomForestClassifier = lambda **kw: RandomForestClassifier(**{**kw, "n_estimators": 200})
    return rc, r1, es, wg


class Preprocessor:
    """Exactly the transform of rev_common.load_nslkdd, fitted on the FIT rows of a given split."""

    def __init__(self, rc, df_fit, feature_cols):
        from sklearn.preprocessing import StandardScaler
        self.cols = feature_cols; self.maps = {}
        for col in rc.CATEGORICAL_COLS:
            vals = sorted(df_fit[col].unique()); self.maps[col] = ({v: i for i, v in enumerate(vals)}, len(vals))
        self.scaler = StandardScaler().fit(self.encode(df_fit))

    def encode(self, df):
        df = df[self.cols].copy()
        for col, (mp, unk) in self.maps.items():
            df[col] = df[col].map(lambda v: mp.get(v, unk))
        return df.values.astype(np.float32)

    def __call__(self, df):
        return self.scaler.transform(self.encode(df)).astype(np.float32)


def raw_frames(rc):
    df_tr, df_te = rc.load_raw()
    df_tr, df_te = rc.map_attack_category(df_tr), rc.map_attack_category(df_te)
    cols = [c for c in df_tr.columns if c not in ("label", "attack_category", "y")]
    return df_tr, df_te, cols


def make_split(rc, df_tr, split_seed):
    from sklearn.model_selection import train_test_split
    if split_seed == 42:
        s = np.load(rc.OUT_DIR / "split_nslkdd.npz"); return s["idx_fit"], s["idx_val"]
    idx = np.arange(len(df_tr))
    a, b = train_test_split(idx, test_size=0.2, random_state=split_seed, stratify=df_tr["y"].values)
    return np.sort(a), np.sort(b)


def train_pipeline(rc, r1, es, wg, df_tr, df_te, cols, split_seed, seed=0, configs=("none", "smote", "hybrid")):
    import zero_trust as zt
    i_fit, i_val = make_split(rc, df_tr, split_seed)
    pp = Preprocessor(rc, df_tr.iloc[i_fit], cols)
    Xf, Xv, Xt = pp(df_tr.iloc[i_fit]), pp(df_tr.iloc[i_val]), pp(df_te)
    yf, yv, yt = (df_tr["y"].values[i_fit].astype(np.int64), df_tr["y"].values[i_val].astype(np.int64),
                  df_te["y"].values.astype(np.int64))
    r1.set_seed(seed)
    risk = zt.RiskScorer(feature_indices=rc.RISK_FEATURES); risk.fit(Xf, yf, Xv, yv)
    gens = {}
    if "hybrid" in configs:
        r1.set_seed(seed); t0 = time.time()
        G, _, _ = wg.train_wgan_gp(Xf, yf, epochs=100, resume=False); gens["wgan"] = (G, wg.generate_synthetic)
        print(f"[r21] split {split_seed}: WGAN-GP {time.time() - t0:.0f}s", flush=True)
    models = {}
    for cfg in configs:
        r1.set_seed(seed)
        Xs, ys, _ = r1.augment(cfg, Xf, yf, gens, seed)
        Xtr = np.concatenate([Xf, Xs]) if len(Xs) else Xf; ytr = np.concatenate([yf, ys]) if len(ys) else yf
        perm = np.random.default_rng(seed).permutation(len(ytr))
        ens = es.EnsembleDetector(feature_dim=Xf.shape[1])
        ens.fit(Xtr[perm], ytr[perm], Xv, yv, nn_epochs=30, verbose=False)
        models[cfg] = ens
        print(f"[r21] split {split_seed}: {cfg} trained", flush=True)
    return dict(pp=pp, risk=risk, models=models, Xv=Xv, Xt=Xt, yv=yv, yt=yt, i_fit=i_fit)


def ens_proba(ens, X):
    w = ens.weights
    return sum(wi * p for wi, p in zip(w, (ens.rf.predict_proba(X), ens.xgb.predict_proba(X), ens.nn.predict_proba(X))))


def zt_flags(p, r):
    pred, conf = p.argmax(1), p.max(1)
    ch = (pred == 0) & (conf >= 0.85) & (r >= 0.70)
    return (pred != 0) | ch, ch


def policy_actions(p, r):
    """Algorithm 2 (context risk q = 0): 0 ALLOW, 1 MONITOR, 2 CHALLENGE, 3 QUARANTINE, 4 BLOCK."""
    pred, c = p.argmax(1), p.max(1); a = np.full(len(p), 1)
    normal = pred == 0
    a[normal & (c >= 0.85) & (r >= 0.70)] = 2
    a[~normal & (c >= 0.85) & (r >= 0.70)] = 4
    m = ~normal & (c >= 0.55) & (r >= 0.40) & (a == 1); a[m] = 3
    a[(c >= 0.55) & normal & (c >= 0.85) & (r < 0.40)] = 0
    return a


# =============================================================================== L
def part_l(impl, out, P):
    import torch
    import zero_trust  # noqa: F401
    rc, r1, es, wg = P["mods"]; df_te = P["df_te"]; m = P["pipe"]
    ens, risk, pp = m["models"]["hybrid"], m["risk"], m["pp"]
    raw = df_te[pp.cols].reset_index(drop=True)

    def run(df):
        X = pp(df); p = ens_proba(ens, X); r = risk.predict_risk(X); return policy_actions(p, r)

    def place(where):
        dev = torch.device("cuda" if where == "gpu" else "cpu")
        es.DEVICE = dev; ens.nn.to(dev); ens.xgb.set_params(device="cuda" if where == "gpu" else "cpu")

    def timed(f, *a):
        t0 = time.perf_counter(); f(*a); return time.perf_counter() - t0

    res = {"placements": {}}
    rng = np.random.default_rng(5)
    sizes = [1, 4, 16, 64, 256, 1024, 4096, len(raw)]
    for where in ("gpu", "cpu"):
        place(where)
        if where == "gpu":
            torch.cuda.synchronize()
        cold = [timed(run, raw.iloc[[i]]) * 1e3 for i in range(5)]          # first calls after (re)placement
        idx = rng.integers(0, len(raw), 500)
        warm = np.array([timed(run, raw.iloc[[i]]) for i in idx]) * 1e3
        # per-component breakdown, single flow, warm
        comp = {k: [] for k in ("preprocess", "rf", "xgb", "dnn", "vote", "validator", "policy")}
        for i in idx[:200]:
            d1 = raw.iloc[[i]]
            t = time.perf_counter(); X = pp(d1); comp["preprocess"].append(time.perf_counter() - t)
            t = time.perf_counter(); a1 = ens.rf.predict_proba(X); comp["rf"].append(time.perf_counter() - t)
            t = time.perf_counter(); a2 = ens.xgb.predict_proba(X); comp["xgb"].append(time.perf_counter() - t)
            t = time.perf_counter(); a3 = ens.nn.predict_proba(X); comp["dnn"].append(time.perf_counter() - t)
            t = time.perf_counter(); p = sum(w * q for w, q in zip(ens.weights, (a1, a2, a3))); comp["vote"].append(time.perf_counter() - t)
            t = time.perf_counter(); r = risk.predict_risk(X); comp["validator"].append(time.perf_counter() - t)
            t = time.perf_counter(); policy_actions(p, r); comp["policy"].append(time.perf_counter() - t)
        curve = {}
        for b in sizes:
            reps = 30 if b <= 1024 else (5 if b < len(raw) else 3)
            ts = []
            for _ in range(reps):
                st = rng.integers(0, len(raw) - b + 1); ts.append(timed(run, raw.iloc[st:st + b]))
            curve[b] = float(np.median(ts))
        res["placements"][where] = {
            "cold_first_calls_ms": cold, "warm_single_median_ms": float(np.median(warm)),
            "warm_single_p95_ms": float(np.percentile(warm, 95)),
            "components_median_ms": {k: float(np.median(v) * 1e3) for k, v in comp.items()},
            "batch_seconds": curve, "per_flow_us": {b: curve[b] / b * 1e6 for b in sizes}}
        print(f"[r21-L] {where}: warm single {np.median(warm):.1f} ms; full batch {curve[len(raw)] / len(raw) * 1e6:.1f} us/flow", flush=True)
    # streaming simulation with measured service times (dynamic micro-batching, Bmax = 1024)
    sim = []
    for where in ("gpu", "cpu"):
        bs = np.array(sizes[:-1], float); st_ = np.array([res["placements"][where]["batch_seconds"][b] for b in sizes[:-1]])

        def service(n):
            return float(np.interp(np.log(n), np.log(bs), st_))
        for lam in (100, 1000, 5000, 20000):
            for policy, bmax in (("one flow per call", 1), ("micro-batching", 1024)):
                cap = bmax / service(bmax)
                arr = np.cumsum(np.random.default_rng(lam).exponential(1 / lam, 100000))
                t, i, lat = 0.0, 0, []
                while i < len(arr):
                    t = max(t, arr[i]); j = i
                    while j < len(arr) and arr[j] <= t and j - i < bmax:
                        j += 1
                    t += service(j - i); lat.extend(t - arr[i:j]); i = j
                lat = np.array(lat) * 1e3
                stable = lam < cap
                sim.append({"placement": where, "arrival_rate_per_s": lam, "serving": policy,
                            "capacity_flows_per_s": cap, "stable": bool(stable),
                            "p50_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                            "p99_ms": float(np.percentile(lat, 99))})
    place("gpu")                               # leave every model on the device it was trained on
    res["streaming"] = sim
    (out / "L_latency.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(sim).to_csv(out / "L_streaming_simulation.csv", index=False)
    print(pd.DataFrame(sim).round(2).to_string())


# =============================================================================== S
def evaluate_split(m):
    probs = {cfg: (ens_proba(e, m["Xv"]), ens_proba(e, m["Xt"])) for cfg, e in m["models"].items()}
    return evaluate_probs(probs, m["risk"].predict_risk(m["Xv"]), m["risk"].predict_risk(m["Xt"]), m["yv"], m["yt"])


def frozen_split42(impl, seed=0):
    P = impl / "revision/out/preds"; d = np.load(impl / "revision/out/nslkdd_data.npz")
    probs = {}
    for cfg in ("none", "smote", "hybrid"):
        z = np.load(P / f"nslkdd_s{seed}_{cfg}.npz"); probs[cfg] = (z["p_val"], z["p_test"])
    R = np.load(P / f"nslkdd_s{seed}_risk.npz")
    return evaluate_probs(probs, R["r_val"], R["r_test"], d["y_val"], d["y_test"])


def evaluate_probs(probs, rv, rt, yv, yt):
    att = yt != 0; row = {}
    for cfg, (pv, pt) in probs.items():
        fl = pt.argmax(1) != 0
        tp, fp = (fl & att).sum(), (fl & ~att).sum(); rec = tp / att.sum(); prec = tp / max(fl.sum(), 1)
        row[f"{cfg}_accuracy"] = float((fl == att).mean()); row[f"{cfg}_f1"] = float(2 * prec * rec / (prec + rec))
        row[f"{cfg}_fpr"] = float(fp / (~att).sum()); row[f"{cfg}_r2l_recall"] = float((pt.argmax(1)[yt == 3] == 3).mean())
        if cfg == "hybrid":
            ztf, ch = zt_flags(pt, rt)
            pred, conf = pt.argmax(1), pt.max(1); elig = att & (pred == 0) & (conf >= 0.85)
            row.update(zt_accuracy=float((ztf == att).mean()), zt_recall=float((ztf & att).sum() / att.sum()),
                       zt_fpr=float((ztf & ~att).sum() / (~att).sum()), zt_challenges=int(ch.sum()),
                       zt_challenge_precision=float((ch & att).sum() / max(ch.sum(), 1)),
                       zt_recovery=float((ch & elig).sum() / max(elig.sum(), 1)), zt_eligible=int(elig.sum()))
            zv, _ = zt_flags(pv, rv); fpr_v = (zv & (yv == 0)).sum() / (yv == 0).sum()
            sv, st = 1 - pv[:, 0], 1 - pt[:, 0]; thr = np.quantile(sv[yv == 0], 1 - fpr_v); fs = st > thr
            row.update(valmatched_recall=float((fs & att).sum() / att.sum()), valmatched_fpr=float((fs & ~att).sum() / (~att).sum()))
            ev = ((pt.argmax(1) != 0) != att); ev2 = ((rt >= 0.5) != att)   # as in r07
            row["phi_validator_ensemble"] = float(np.corrcoef(ev, ev2)[0, 1])
    return row


def part_s(impl, out, P, splits=(1, 2, 3)):
    rows = []
    for s in (42,) + tuple(splits):
        if s == 42:   # original split: the authors' frozen seed-0 predictions (identical models)
            r = {"split_seed": s, "training_seed": 0, "source": "frozen predictions", **frozen_split42(impl)}
        else:
            m = train_pipeline(*P["mods"], P["df_tr"], P["df_te"], P["cols"], s)
            r = {"split_seed": s, "training_seed": 0, "source": "retrained", **evaluate_split(m)}
        rows.append(r)
        pd.DataFrame(rows).to_csv(out / "S_split_sensitivity.csv", index=False)
        print(f"[r21-S] split {s}: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)


def main():
    global A
    ap = argparse.ArgumentParser(); ap.add_argument("--impl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--parts", default="DLS")
    A = ap.parse_args(); A.out.mkdir(parents=True, exist_ok=True)
    if "D" in A.parts:
        part_d(A.impl, A.out)
    if "L" in A.parts or "S" in A.parts:
        mods = setup_impl(A.impl, A.out); rc = mods[0]
        df_tr, df_te, cols = raw_frames(rc)
        pipe = None
        if "L" in A.parts:
            pipe = train_pipeline(*mods, df_tr, df_te, cols, 42)
            # check: the preprocessing reproduces the frozen KDDTest+ features exactly
            frozen = np.load(A.impl / "revision/out/nslkdd_data.npz")["X_test"]
            print(f"[r21] max |X_test - frozen| = {np.abs(pipe['Xt'] - frozen).max():.2e}", flush=True)
        P = dict(mods=mods, df_tr=df_tr, df_te=df_te, cols=cols, pipe=pipe)
        if "L" in A.parts:
            part_l(A.impl, A.out, P)
        if "S" in A.parts:
            part_s(A.impl, A.out, P)
    print("[r21] done")


if __name__ == "__main__":
    main()
