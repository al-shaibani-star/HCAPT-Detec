"""
Confirmatory evaluation on UNSW-NB15, a corpus that was never used while the
revision was designed.  The configurations, metrics and hypotheses are frozen in
revision/out/unsw/PREREGISTRATION.json (written, with a timestamp and SHA-256 hashes
of this script and the data files, BEFORE any model is trained).

Data
  Official partitions.  In this copy the file names are swapped relative to the
  dataset documentation (the 175,341-row file is named "testing-set"); the script
  assigns the partitions by row count and records it.  The official training
  partition is split once into FIT (80 %) and VAL (20 %); the official testing
  partition (82,332 flows) is scored once.  5.9 % of test rows have an exact feature
  duplicate in the training partition, so results are also reported on the
  deduplicated test set (sensitivity analysis).
  Ten classes: Normal + nine attack categories.  'id' is dropped; proto, service and
  state use the FIT vocabulary (unseen -> extra code); scaling is fitted on FIT.

Configurations (fixed in advance; no tuning on UNSW-NB15)
  lr        logistic regression, class-weighted
  xgb       XGBoost, sample-weighted (300 trees, depth 8, lr 0.1)
  dnn       the paper's DNN, class-weighted, 30 epochs
  ens_cw    RF + XGBoost + DNN, class-weighted, validation-accuracy soft vote (paper)
  ens_smote ensemble + SMOTE           (each minority class topped up to the FIT majority)
  ens_cgan  ensemble + CGAN            (paper architecture, 50 epochs, batch 256)
  ens_wgan  ensemble + WGAN-GP         (paper architecture, 50 epochs, batch 256)
  ens_hyb   ensemble + WGAN-GP for classes with >= 1,000 FIT rows, SMOTE otherwise (proposed)
  + Zero-Trust policy (LightGBM validator on odd-indexed features; 0.85 / 0.70) on ens_cw, ens_hyb
  RF trees use bootstrap samples of at most 50,000 rows (memory), identical for all ensembles.
Seeds 0-4.  Each (seed, configuration) runs in its own process.

Usage (PowerShell, from 05-Implementation):
  python -u revision\\r17_unsw_confirmatory.py --smoke        # pipeline check (minutes)
  python -u revision\\r17_unsw_confirmatory.py --run          # all seeds and configurations (hours)
  python -u revision\\r17_unsw_confirmatory.py --analyze      # tables + hypothesis tests
"""
import argparse
import datetime
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import rev_common as rc
import ensemble_selflearn as es
import gan_augment as cg
import wgan_gp_augment as wg

DATA = rc.SRC_DIR.parent / "data" / "UNSW-NB15"
OUT = rc.OUT_DIR / "unsw"; PRED = OUT / "preds"
CONFIGS = ["lr", "xgb", "dnn", "ens_cw", "ens_smote", "ens_cgan", "ens_wgan", "ens_hyb"]
SEEDS = [0, 1, 2, 3, 4]
GAN_EPOCHS, GAN_BATCH, RF_MAX_SAMPLES = 50, 256, 50000
HC, HR, MIDC, MIDR = 0.85, 0.70, 0.55, 0.40
HYPOTHESES = {
    "H1": "No GAN-based ensemble (ens_cgan, ens_wgan, ens_hyb) has a higher mean test macro-F1 than BOTH ens_cw and ens_smote.",
    "H2": "ens_hyb + Zero-Trust is not significantly more often correct than ens_smote (exact McNemar on binary correctness, "
          "Holm over the comparison family) in a majority of seeds.",
    "H3": "Zero-Trust recovers fewer than 10% of eligible high-confidence benign errors (attacks labelled Normal with confidence >= 0.85).",
    "H4": "The binary-error correlation (phi) between the validator and ens_hyb exceeds 0.5 on the test partition.",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ data
def load(smoke=False):
    files = [DATA / "UNSW_NB15_training-set.csv", DATA / "UNSW_NB15_testing-set.csv"]
    dfs = sorted((pd.read_csv(f) for f in files), key=len, reverse=True)
    tr, te = dfs                                   # larger file = official training partition
    assert len(tr) == 175341 and len(te) == 82332, (len(tr), len(te))
    for d in (tr, te):
        d["attack_cat"] = d["attack_cat"].astype(str).str.strip()
    classes = ["Normal"] + sorted(c for c in tr.attack_cat.unique() if c != "Normal")
    cmap = {c: i for i, c in enumerate(classes)}
    y_tr, y_te = tr.attack_cat.map(cmap).values, te.attack_cat.map(cmap).values
    feats = [c for c in tr.columns if c not in ("id", "attack_cat", "label")]
    idx_fit, idx_val = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42, stratify=y_tr)
    if smoke:
        r = np.random.default_rng(0)
        idx_fit = r.choice(idx_fit, 15000, replace=False); idx_val = r.choice(idx_val, 4000, replace=False)
        te = te.sample(8000, random_state=0); y_te = te.attack_cat.map(cmap).values
    F, V, T = tr.iloc[idx_fit][feats].copy(), tr.iloc[idx_val][feats].copy(), te[feats].copy()
    for c in ("proto", "service", "state"):
        vocab = {v: i for i, v in enumerate(pd.unique(F[c].astype(str)))}
        for d in (F, V, T):
            d[c] = d[c].astype(str).map(vocab).fillna(len(vocab))
    sc = StandardScaler().fit(F.values.astype(np.float64))
    f = lambda d: np.clip(sc.transform(d.values.astype(np.float64)), -50, 50).astype(np.float32)
    seen = set(map(tuple, tr[feats].astype(str).values))
    novel = np.array([tuple(r) not in seen for r in te[feats].astype(str).values])
    return dict(X_fit=f(F), y_fit=y_tr[idx_fit], X_val=f(V), y_val=y_tr[idx_val], X_test=f(T), y_test=y_te,
                classes=classes, features=feats, test_novel=novel)


# ------------------------------------------------------------------ models
def class_weights(y, k):
    c = np.bincount(y, minlength=k); return c.sum() / (k * np.maximum(c, 1))


def train_rf(X, y, k, seed):
    m = RandomForestClassifier(n_estimators=200, min_samples_split=5, class_weight="balanced", random_state=seed,
                               n_jobs=4, max_samples=RF_MAX_SAMPLES if len(y) > RF_MAX_SAMPLES else None).fit(X, y)
    return lambda Z: full(m.predict_proba(Z), m.classes_, k)


def train_xgb(X, y, k, seed):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, objective="multi:softprob",
                          num_class=k, tree_method="hist", device="cuda", random_state=seed)
    m.fit(X, y, sample_weight=class_weights(y, k)[y]); m.set_params(device="cpu")
    return m.predict_proba


def train_dnn(X, y, k, seed):
    torch.manual_seed(seed)
    nn = es.DeepDetector(X.shape[1], num_classes=k).to(es.DEVICE)
    nn.fit(X, y, epochs=30, class_weights=class_weights(y, k), verbose=False)
    return nn.predict_proba


def full(p, cls, k):
    out = np.zeros((len(p), k), np.float32); out[:, cls] = p; return out


def topup(kind, X, y, k, seed, smoke):
    counts = np.bincount(y, minlength=k); target = int(counts.max())
    need = {c: target - int(counts[c]) for c in range(k) if counts[c] < target}
    if kind == "hyb":
        gan_need = {c: n for c, n in need.items() if counts[c] >= rc.SUPPORT_THRESHOLD}
        smo_need = {c: n for c, n in need.items() if counts[c] < rc.SUPPORT_THRESHOLD}
        X, y = smote(X, y, smo_need, k, seed)
        return gan_topup("wgan", X, y, k, seed, gan_need, smoke)
    if kind == "smote":
        return smote(X, y, need, k, seed)
    return gan_topup(kind, X, y, k, seed, need, smoke)


def smote(X, y, need, k, seed):
    from imblearn.over_sampling import SMOTE
    if not need:
        return X, y
    counts = np.bincount(y, minlength=k)
    strat = {c: int(counts[c] + n) for c, n in need.items()}
    kk = min(5, min(counts[c] for c in strat) - 1)
    return SMOTE(sampling_strategy=strat, k_neighbors=kk, random_state=seed).fit_resample(X, y)


def gan_topup(kind, X, y, k, seed, need, smoke):
    if not need:
        return X, y
    torch.manual_seed(seed); np.random.seed(seed)
    epochs = 1 if smoke else GAN_EPOCHS
    for mod in (cg, wg):
        mod.NUM_CLASSES = k; mod.save_checkpoint = lambda *a, **kw: None; mod.MODELS_DIR = OUT
    if kind == "cgan":
        G, D, _ = cg.train_cgan(X, y, epochs=epochs, batch_size=GAN_BATCH, resume=False, save_path=OUT)
        Xs, ys = cg.generate_synthetic(G, need)
    else:
        G, D, _ = wg.train_wgan_gp(X, y, epochs=epochs, batch_size=GAN_BATCH, resume=False)
        Xs, ys = wg.generate_synthetic(G, need)
    del G, D; release()
    return np.vstack([X, Xs.astype(np.float32)]), np.r_[y, ys.astype(np.int64)]


def worker(cfg, seed, smoke):
    D = load(smoke); k = len(D["classes"])
    pdir = PRED if not smoke else OUT / "preds_smoke"; pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / f"s{seed}_{cfg}.npz"
    if out.exists():
        return
    np.random.seed(seed); torch.manual_seed(seed)
    Xf, yf, Xv, yv, Xt = D["X_fit"], D["y_fit"], D["X_val"], D["y_val"], D["X_test"]
    t0 = time.time()
    if cfg == "lr":
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xf, yf)
        np.savez_compressed(out, p_val=full(m.predict_proba(Xv), m.classes_, k), p_test=full(m.predict_proba(Xt), m.classes_, k))
    elif cfg in ("xgb", "dnn"):
        f = (train_xgb if cfg == "xgb" else train_dnn)(Xf, yf, k, seed)
        np.savez_compressed(out, p_val=f(Xv), p_test=f(Xt))
    else:
        kind = cfg.split("_")[1]
        X, y = (Xf, yf) if kind == "cw" else topup(kind, Xf, yf, k, seed, smoke)
        perm = np.random.default_rng(seed).permutation(len(y)); X, y = X[perm], y[perm]
        fs = [train_rf(X, y, k, seed), train_xgb(X, y, k, seed), train_dnn(X, y, k, seed)]
        pv = [f(Xv) for f in fs]; pt = [f(Xt) for f in fs]
        accs = np.array([(p.argmax(1) == yv).mean() for p in pv]); w = np.exp(10 * accs); w /= w.sum()
        np.savez_compressed(out, p_val=sum(a * b for a, b in zip(w, pv)), p_test=sum(a * b for a, b in zip(w, pt)),
                            weights=w, n_train=len(y))
    rp = pdir / f"s{seed}_risk.npz"
    if not rp.exists():
        odd = list(range(1, Xf.shape[1], 2))
        v = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, class_weight="balanced", random_state=seed + 777, verbose=-1)
        v.fit(Xf[:, odd], (yf != 0).astype(int))
        np.savez_compressed(rp, r_val=v.predict_proba(Xv[:, odd])[:, 1], r_test=v.predict_proba(Xt[:, odd])[:, 1])
    print(f"[unsw] seed {seed} {cfg} saved ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ orchestration
def preregister(smoke):
    path = OUT / ("PREREGISTRATION_smoke.json" if smoke else "PREREGISTRATION.json")
    if path.exists():
        return json.loads(path.read_text())
    reg = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "status": "frozen before any UNSW-NB15 model was trained",
           "configurations": CONFIGS, "seeds": SEEDS,
           "zero_trust": {"high_conf": HC, "high_risk": HR, "mid_conf": MIDC, "mid_risk": MIDR,
                          "validator": "LightGBM on odd-indexed features, trained on FIT"},
           "primary_metrics": ["test macro-F1 (10 classes)", "test binary accuracy / F1 / FPR"],
           "hypotheses": HYPOTHESES,
           "data_sha256": {f.name: sha(f) for f in sorted(DATA.glob("UNSW_NB15_*-set.csv"))},
           "script_sha256": sha(__file__)}
    OUT.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(reg, indent=2))
    print(f"[unsw] pre-registration written: {path}")
    return reg


def run(smoke):
    preregister(smoke)
    seeds = [0] if smoke else SEEDS
    for s in seeds:
        for c in CONFIGS:
            pdir = PRED if not smoke else OUT / "preds_smoke"
            if (pdir / f"s{s}_{c}.npz").exists():
                continue
            cmd = [sys.executable, "-u", __file__, "--worker", c, str(s)] + (["--smoke"] if smoke else [])
            for attempt in (1, 2):
                if subprocess.run(cmd).returncode == 0:
                    break
                print(f"[unsw] seed {s} {c}: attempt {attempt} failed")
            else:
                raise SystemExit(f"seed {s} {c} failed twice; run the same command again to resume")
    print("[unsw] run complete")


# ------------------------------------------------------------------ analysis
def zt(p, r, y):
    pred, conf = p.argmax(1), p.max(1)
    ch = (pred == 0) & (conf >= HC) & (r >= HR)
    final = pred.copy(); final[ch] = 1
    elig = (pred == 0) & (conf >= HC) & (y != 0)
    good = int((ch & (y != 0)).sum()); bad = int((ch & (y == 0)).sum())
    return final, {"challenges": int(ch.sum()), "challenge_precision": good / max(good + bad, 1),
                   "eligible": int(elig.sum()), "recovery_recall": (good / int(elig.sum())) if elig.any() else float("nan")}


def mcnemar(y, a, b):
    ca, cb = (a != 0) == (y != 0), (b != 0) == (y != 0)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, (m - r) * ps[i]); adj[i] = min(run_, 1.0)
    return adj


def analyze(smoke):
    D = load(smoke); y = D["y_test"]; k = len(D["classes"]); novel = D["test_novel"]
    pdir = PRED if not smoke else OUT / "preds_smoke"
    seeds = sorted({int(f.name[1:].split("_")[0]) for f in pdir.glob("s*_risk.npz")})
    rows, ztrows, mcrows, phis = [], [], [], []
    for s in seeds:
        P = {c: np.load(pdir / f"s{s}_{c}.npz")["p_test"] for c in CONFIGS if (pdir / f"s{s}_{c}.npz").exists()}
        R = np.load(pdir / f"s{s}_risk.npz")["r_test"]
        preds = {c: p.argmax(1) for c, p in P.items()}
        for base in ("ens_cw", "ens_hyb"):
            if base in P:
                fin, info = zt(P[base], R, y); preds[base + "+ZT"] = fin; ztrows.append({"seed": s, "base": base, **info})
        for c, pr in preds.items():
            for subset, m in (("full test", np.ones(len(y), bool)), ("deduplicated test", novel)):
                b = rc.binary_metrics(y[m], pr[m])
                rows.append({"seed": s, "config": c, "subset": subset, "macro_f1": f1_score(y[m], pr[m], average="macro"),
                             "accuracy": b["accuracy"], "f1": b["f1"], "fpr": b["fpr"],
                             **{f"recall_{D['classes'][j]}": float((pr[m][y[m] == j] == j).mean()) for j in range(1, k)}})
        if "ens_hyb+ZT" in preds:
            comps = [c for c in ("ens_smote", "ens_cw", "lr", "dnn", "ens_hyb") if c in preds]
            res = [mcnemar(y, preds["ens_hyb+ZT"], preds[c]) for c in comps]
            adj = holm(np.array([r[2] for r in res]))
            for c, r_, q in zip(comps, res, adj):
                mcrows.append({"seed": s, "comparator": c, "proposed_right": r_[0], "comparator_right": r_[1], "p_holm": q})
            e = (preds["ens_hyb"] != 0) != (y != 0); v = (R >= 0.5) != (y != 0)
            phis.append(float(np.corrcoef(e, v)[0, 1]))
    res = pd.DataFrame(rows); zdf = pd.DataFrame(ztrows); mdf = pd.DataFrame(mcrows)
    tdir = rc.OUT_DIR / "tables"; sfx = "_smoke" if smoke else ""
    res.to_csv(tdir / f"unsw_results{sfx}.csv", index=False); zdf.to_csv(tdir / f"unsw_zt{sfx}.csv", index=False)
    mdf.to_csv(tdir / f"unsw_mcnemar{sfx}.csv", index=False)
    full_ = res[res.subset == "full test"].groupby("config")
    mf = full_["macro_f1"].mean()
    verdict = {}
    if {"ens_cw", "ens_smote"} <= set(mf.index):
        verdict["H1"] = bool(all(mf.get(g, -1) <= max(mf["ens_cw"], mf["ens_smote"]) for g in ("ens_cgan", "ens_wgan", "ens_hyb")))
    if len(mdf):
        sm = mdf[mdf.comparator == "ens_smote"]
        sig_better = ((sm.p_holm < 0.05) & (sm.proposed_right > sm.comparator_right)).sum()
        verdict["H2"] = bool(sig_better <= len(sm) / 2)
    if len(zdf):
        rr = zdf[zdf.base == "ens_hyb"].recovery_recall.dropna()
        verdict["H3"] = bool(rr.mean() < 0.10) if len(rr) else "undeterminable (no eligible errors)"
    if phis:
        verdict["H4"] = bool(np.mean(phis) > 0.5)
    summary = {"seeds": seeds, "hypotheses": HYPOTHESES, "supported": verdict, "phi_mean": float(np.mean(phis)) if phis else None,
               "test_rows": int(len(y)), "deduplicated_test_rows": int(novel.sum()),
               "zt": zdf.groupby("base").mean(numeric_only=True).drop(columns="seed").to_dict("index") if len(zdf) else {},
               "mcnemar": mdf.groupby("comparator").agg(proposed_right=("proposed_right", "mean"),
                                                       comparator_right=("comparator_right", "mean"),
                                                       p_holm_max=("p_holm", "max")).to_dict("index") if len(mdf) else {}}
    rc.dump_json(summary, tdir / f"unsw_summary{sfx}.json")
    pd.set_option("display.width", 250)
    for subset in ("full test", "deduplicated test"):
        g = res[res.subset == subset].groupby("config")[["macro_f1", "accuracy", "f1", "fpr"]].agg(["mean", "std"]).round(4)
        print(f"\n=== UNSW-NB15 {subset}\n{g.sort_values(('macro_f1', 'mean'), ascending=False).to_string()}")
    print("\nZero-Trust:", summary["zt"]); print("McNemar (proposed+ZT vs):", summary["mcnemar"])
    print("phi(validator, ens_hyb):", summary["phi_mean"]); print("PRE-REGISTERED HYPOTHESES SUPPORTED:", verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", nargs=2, metavar=("CONFIG", "SEED"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker(a.worker[0], int(a.worker[1]), a.smoke)
    elif a.smoke and not a.analyze:
        run(True); analyze(True)
    elif a.run:
        run(False)
    elif a.analyze:
        analyze(a.smoke)
    else:
        ap.error("use --smoke, --run or --analyze")


if __name__ == "__main__":
    main()
