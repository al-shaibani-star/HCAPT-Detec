"""
Step 2 of the IJIES revision (Reviewer 2, comment 1): common-protocol baselines.

Same FIT/VAL/TEST split as r01, same seeds, and no per-method tuning: every
resampler feeds the identical ensemble with identical fixed hyper-parameters
(equal tuning budget = zero for all methods).

Single learners on real FIT data (no resampling):
  rf_unw   Random Forest, no class weighting
  xgb_unw  XGBoost, no class weighting
  lgbm     LightGBM, class_weight="balanced"
  lr       logistic regression, class_weight="balanced" (linear reference)
  ens_unw  plain ensemble: unweighted RF + XGB + DNN, equal-weight soft vote
           (the ensemble "without the added framework")
  (RF/XGB/DNN with class weighting alone are read from r01's "none" run.)

Alternative resamplers feeding the paper's ensemble:
  ros      random oversampling
  bsmote   Borderline-SMOTE
  adasyn   ADASYN
  ctgan    CTGAN (Xu et al. 2019), class-conditional, trained on FIT only

Outputs: revision/out/preds/nslkdd_s{seed}_{name}.npz  (p_val, p_test, ...)

Usage (PowerShell, from 05-Implementation):
  python -u revision\\r03_baselines_nslkdd.py --smoke
  python -u revision\\r03_baselines_nslkdd.py --seeds 0 1 2 3 4
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import rev_common as rc
from rev_common import NUM_CLASSES, OUT_DIR, PRED_DIR
import ensemble_selflearn as es

SINGLE = ["rf_unw", "xgb_unw", "lgbm", "lr", "ens_unw"]
RESAMPLE = ["ros", "bsmote", "adasyn", "ctgan"]


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    es.SEED = seed


def resample(name, X, y, seed, ctgan_epochs):
    from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, ADASYN
    counts = np.bincount(y, minlength=NUM_CLASSES); target = int(counts.max())
    strat = {c: target for c in range(NUM_CLASSES) if counts[c] < target}
    k = min(5, int(min(counts[c] for c in strat)) - 1)
    if name == "ros":
        return RandomOverSampler(sampling_strategy=strat, random_state=seed).fit_resample(X, y)
    if name == "bsmote":
        return BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed).fit_resample(X, y)
    if name == "adasyn":
        try:
            return ADASYN(sampling_strategy=strat, n_neighbors=k, random_state=seed).fit_resample(X, y)
        except (ValueError, RuntimeError) as e:   # ADASYN refuses classes with no hard neighbours
            print(f"[adasyn] {e} -> per-class fallback")
            Xo, yo = [X], [y]
            for c in strat:
                try:
                    Xr, yr = ADASYN(sampling_strategy={c: target}, n_neighbors=k,
                                    random_state=seed).fit_resample(X, y)
                    Xo.append(Xr[len(X):]); yo.append(yr[len(y):])
                except (ValueError, RuntimeError):
                    Xr, yr = RandomOverSampler(sampling_strategy={c: target},
                                               random_state=seed).fit_resample(X, y)
                    Xo.append(Xr[len(X):]); yo.append(yr[len(y):])
            return np.concatenate(Xo), np.concatenate(yo)
    if name == "ctgan":
        from ctgan import CTGAN
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["cls"] = y.astype(int)
        # features are already standard-scaled -> modelled as continuous; only the
        # class label is discrete (the old src/ctgan_augment.py rounded scaled codes)
        try:
            model = CTGAN(epochs=ctgan_epochs, batch_size=500, verbose=False,
                          enable_gpu=torch.cuda.is_available())
        except TypeError:   # older ctgan
            model = CTGAN(epochs=ctgan_epochs, batch_size=500, verbose=False,
                          cuda=torch.cuda.is_available())
        model.set_random_state(seed)
        # CTGAN's transformer runs Parallel(n_jobs=-1); with process workers every
        # worker re-imports torch+CUDA DLLs and Windows runs out of paging file
        # (WinError 1455).  Threads share the already-loaded DLLs.
        from joblib import parallel_config
        t0 = time.time()
        with parallel_config(backend="threading"):
            model.fit(df, discrete_columns=["cls"])
        print(f"[ctgan] fit {time.time() - t0:.0f}s")
        Xs, ys = [X], [y]
        for c, tgt in strat.items():
            n = tgt - counts[c]
            s = model.sample(n, condition_column="cls", condition_value=int(c))
            Xs.append(s[[f"f{i}" for i in range(X.shape[1])]].values.astype(np.float32))
            ys.append(np.full(n, c, np.int64))
        return np.concatenate(Xs), np.concatenate(ys)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="+", default=SINGLE + RESAMPLE)
    ap.add_argument("--nn-epochs", type=int, default=30)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--ctgan-epochs", type=int, default=30)
    ap.add_argument("--rf-jobs", type=int, default=6, help="leave cores for a parallel r01 run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pdir = PRED_DIR
    if a.smoke:
        a.nn_epochs, a.rf_trees, a.ctgan_epochs = 1, 10, 1
        pdir = OUT_DIR / "preds_smoke"; pdir.mkdir(exist_ok=True)
    trees, jobs = a.rf_trees, a.rf_jobs
    es.RandomForestClassifier = lambda **kw: RandomForestClassifier(
        **{**kw, "n_estimators": trees, "n_jobs": jobs})

    D = rc.load_nslkdd()
    X_fit, y_fit, X_val, y_val, X_te = D["X_fit"], D["y_fit"], D["X_val"], D["y_val"], D["X_test"]
    log = {}
    for seed in a.seeds:
        for name in a.only:
            out = pdir / f"nslkdd_s{seed}_{name}.npz"
            if out.exists() and not a.force:
                print(f"[r03] seed {seed} {name}: exists"); continue
            set_seed(seed); t0 = time.time(); print(f"\n[r03] seed {seed} | {name}")
            if name == "rf_unw":
                m = RandomForestClassifier(n_estimators=trees, min_samples_split=5, n_jobs=jobs,
                                           random_state=seed).fit(X_fit, y_fit)
                np.savez_compressed(out, p_val=m.predict_proba(X_val), p_test=m.predict_proba(X_te))
            elif name == "xgb_unw":
                m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                      objective="multi:softprob", tree_method="hist", device="cuda",
                                      random_state=seed).fit(X_fit, y_fit)
                np.savez_compressed(out, p_val=m.predict_proba(X_val), p_test=m.predict_proba(X_te))
            elif name == "lgbm":
                m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                       class_weight="balanced", random_state=seed,
                                       verbose=-1).fit(X_fit, y_fit)
                np.savez_compressed(out, p_val=m.predict_proba(X_val), p_test=m.predict_proba(X_te))
            elif name == "lr":
                m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_fit, y_fit)
                np.savez_compressed(out, p_val=m.predict_proba(X_val), p_test=m.predict_proba(X_te))
            elif name == "ens_unw":
                rf = RandomForestClassifier(n_estimators=trees, min_samples_split=5, n_jobs=jobs,
                                            random_state=seed).fit(X_fit, y_fit)
                xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                       objective="multi:softprob", tree_method="hist", device="cuda",
                                       random_state=seed).fit(X_fit, y_fit)
                nn = es.DeepDetector(X_fit.shape[1]).to(es.DEVICE)
                nn.fit(X_fit, y_fit, epochs=a.nn_epochs, class_weights=None, verbose=False)
                pv = [rf.predict_proba(X_val), xg.predict_proba(X_val), nn.predict_proba(X_val)]
                pt = [rf.predict_proba(X_te), xg.predict_proba(X_te), nn.predict_proba(X_te)]
                np.savez_compressed(out, p_val=sum(pv) / 3, p_test=sum(pt) / 3,
                                    pv_rf=pv[0], pv_xgb=pv[1], pv_nn=pv[2],
                                    pt_rf=pt[0], pt_xgb=pt[1], pt_nn=pt[2])
            else:   # resampler + the paper's ensemble
                Xr, yr = resample(name, X_fit, y_fit, seed, a.ctgan_epochs)
                perm = np.random.default_rng(seed).permutation(len(yr))
                Xr, yr = Xr[perm].astype(np.float32), yr[perm].astype(np.int64)
                print(f"[r03] train counts {np.bincount(yr, minlength=NUM_CLASSES).tolist()}")
                ens = es.EnsembleDetector(feature_dim=X_fit.shape[1])
                ens.fit(Xr, yr, X_val, y_val, nn_epochs=a.nn_epochs, verbose=False)
                pv = [ens.rf.predict_proba(X_val), ens.xgb.predict_proba(X_val), ens.nn.predict_proba(X_val)]
                pt = [ens.rf.predict_proba(X_te), ens.xgb.predict_proba(X_te), ens.nn.predict_proba(X_te)]
                w = ens.weights
                np.savez_compressed(out, p_val=sum(wi * p for wi, p in zip(w, pv)),
                                    p_test=sum(wi * p for wi, p in zip(w, pt)),
                                    pv_rf=pv[0], pv_xgb=pv[1], pv_nn=pv[2],
                                    pt_rf=pt[0], pt_xgb=pt[1], pt_nn=pt[2], weights=w)
            log[f"s{seed}_{name}"] = {"seconds": time.time() - t0}
            print(f"[r03] saved {out.name} ({time.time() - t0:.0f}s)")
            logf = pdir / "r03_log.json"
            prev = json.loads(logf.read_text()) if logf.exists() else {}
            prev.update(log); logf.write_text(json.dumps(prev, indent=2))
    print("\n[r03] done")


if __name__ == "__main__":
    main()
