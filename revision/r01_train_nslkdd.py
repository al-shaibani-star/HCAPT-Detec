"""
Step 1 of the IJIES revision: train every NSL-KDD configuration under the clean
protocol in rev_common.py and freeze per-sample predictions.

Configurations (augmentation applied to FIT only, every class topped up to the
majority count):
  none    - no augmentation, class-weighted learners
  smote   - SMOTE for every minority class
  cgan    - conditional GAN for every minority class
  wgan    - WGAN-GP for every minority class
  hybrid  - WGAN-GP where a class has >= SUPPORT_THRESHOLD real FIT samples,
            SMOTE otherwise (the method the paper proposes)

Outputs (revision/out/preds/):
  nslkdd_s{seed}_{config}.npz  ensemble + per-learner probabilities on VAL and TEST
  nslkdd_s{seed}_risk.npz      Zero-Trust risk scores on VAL and TEST
  nslkdd_s{seed}_meta.json     counts, weights, timings, hardware

Usage (PowerShell, from 05-Implementation):
  python revision\\r01_train_nslkdd.py --smoke            # 2-minute pipeline check
  python revision\\r01_train_nslkdd.py --seeds 0 1 2 3 4  # full run
"""
import argparse
import time

import numpy as np
import torch

import rev_common as rc
from rev_common import NUM_CLASSES, PRED_DIR, OUT_DIR, SUPPORT_THRESHOLD, RISK_FEATURES

import ensemble_selflearn as es
import gan_augment as cg
import wgan_gp_augment as wg
import zero_trust as zt
from sklearn.ensemble import RandomForestClassifier

CONFIGS = ["none", "smote", "cgan", "wgan", "hybrid"]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # library modules read SEED as a module global at call time
    es.SEED = seed
    zt.SEED = seed
    cg.SEED = seed
    wg.SEED = seed


def smote_to(X, y, classes, target, seed):
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    counts = np.bincount(y, minlength=NUM_CLASSES)
    strat = {c: target for c in classes if counts[c] < target}
    if not strat:
        return np.empty((0, X.shape[1]), np.float32), np.empty((0,), np.int64)
    k = min(5, min(counts[c] for c in strat) - 1)
    sampler = SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed) if k >= 1 \
        else RandomOverSampler(sampling_strategy=strat, random_state=seed)
    Xr, yr = sampler.fit_resample(X, y)
    return Xr[len(X):].astype(np.float32), yr[len(y):].astype(np.int64)


def augment(cfg, X, y, gens, seed):
    """Return synthetic (X_syn, y_syn) for FIT, and how each class was handled."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = int(counts.max())
    need = {c: target - int(counts[c]) for c in range(NUM_CLASSES) if counts[c] < target}
    how = {}
    if cfg == "none":
        return np.empty((0, X.shape[1]), np.float32), np.empty((0,), np.int64), how
    if cfg == "smote":
        how = {c: "smote" for c in need}
        Xs, ys = smote_to(X, y, list(need), target, seed)
        return Xs, ys, how
    if cfg in ("cgan", "wgan"):
        G, gen_fn = gens[cfg]
        how = {c: cfg for c in need}
        Xs, ys = gen_fn(G, need)
        return Xs.astype(np.float32), ys.astype(np.int64), how
    if cfg == "hybrid":
        gan_cls = {c: n for c, n in need.items() if counts[c] >= SUPPORT_THRESHOLD}
        smo_cls = [c for c in need if counts[c] < SUPPORT_THRESHOLD]
        how = {**{c: "wgan" for c in gan_cls}, **{c: "smote" for c in smo_cls}}
        parts_X, parts_y = [], []
        if gan_cls:
            G, gen_fn = gens["wgan"]
            Xg, yg = gen_fn(G, gan_cls)
            parts_X.append(Xg.astype(np.float32)); parts_y.append(yg.astype(np.int64))
        if smo_cls:
            Xm, ym = smote_to(X, y, smo_cls, target, seed)
            parts_X.append(Xm); parts_y.append(ym)
        return np.concatenate(parts_X), np.concatenate(parts_y), how
    raise ValueError(cfg)


def latency(fn, X, n=500):
    """Median single-sample latency (ms) and batch throughput (ms/sample)."""
    singles = []
    for i in range(n):
        t0 = time.perf_counter(); fn(X[i:i + 1]); singles.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); fn(X); batch = time.perf_counter() - t0
    return {"single_sample_median_ms": float(np.median(singles) * 1e3),
            "single_sample_p95_ms": float(np.percentile(singles, 95) * 1e3),
            "batch_ms_per_sample": float(batch / len(X) * 1e3),
            "batch_size": int(len(X)), "n_single": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--gan-epochs", type=int, default=100)
    ap.add_argument("--nn-epochs", type=int, default=30)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="tiny settings, separate output dir")
    ap.add_argument("--force", action="store_true", help="recompute existing outputs")
    a = ap.parse_args()

    pred_dir = PRED_DIR
    if a.smoke:
        a.gan_epochs, a.nn_epochs, a.rf_trees = 1, 1, 10
        pred_dir = OUT_DIR / "preds_smoke"
        pred_dir.mkdir(exist_ok=True)

    # Library code must not overwrite the published models or checkpoints.
    tmp_models = OUT_DIR / "models_tmp"; tmp_models.mkdir(exist_ok=True)
    cg.save_checkpoint = lambda *args, **kw: None
    wg.save_checkpoint = lambda *args, **kw: None
    cg.MODELS_DIR = tmp_models
    wg.MODELS_DIR = tmp_models
    rf_trees = a.rf_trees
    es.RandomForestClassifier = lambda **kw: RandomForestClassifier(**{**kw, "n_estimators": rf_trees})

    D = rc.load_nslkdd()
    X_fit, y_fit, X_val, y_val, X_te, y_te = (D[k] for k in
        ("X_fit", "y_fit", "X_val", "y_val", "X_test", "y_test"))
    np.savez_compressed(OUT_DIR / "nslkdd_data.npz", y_val=y_val, y_test=y_te,
                        X_val=X_val, X_test=X_te, feature_names=np.array(D["feature_names"]))
    rc.dump_json({"provenance": D["provenance"], "hardware": rc.hardware_info(),
                  "fit_counts": np.bincount(y_fit, minlength=NUM_CLASSES),
                  "val_counts": np.bincount(y_val, minlength=NUM_CLASSES),
                  "test_counts": np.bincount(y_te, minlength=NUM_CLASSES),
                  "support_threshold": SUPPORT_THRESHOLD,
                  "risk_features": [D["feature_names"][i] for i in RISK_FEATURES],
                  "ensemble_features": D["feature_names"],
                  "settings": vars(a)}, OUT_DIR / "nslkdd_protocol.json")
    print(f"[rev] FIT counts {np.bincount(y_fit).tolist()} | VAL {np.bincount(y_val).tolist()}"
          f" | TEST {np.bincount(y_te).tolist()}")

    for seed in a.seeds:
        meta_path = pred_dir / f"nslkdd_s{seed}_meta.json"
        meta = {"seed": seed, "configs": {}}
        if meta_path.exists() and not a.force:
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        set_seed(seed)

        # --- Zero-Trust risk scorer: real FIT data, odd-indexed features ---
        rpath = pred_dir / f"nslkdd_s{seed}_risk.npz"
        risk = zt.RiskScorer(feature_indices=RISK_FEATURES)
        risk.fit(X_fit, y_fit, X_val, y_val)
        np.savez_compressed(rpath, r_val=risk.predict_risk(X_val), r_test=risk.predict_risk(X_te))
        meta["risk_latency"] = latency(risk.predict_risk, X_te)

        # --- Generators, trained on FIT only, once per seed ---
        gens = {}
        need_gan = {c for c in a.configs if c in ("cgan", "wgan", "hybrid")}
        if "cgan" in need_gan:
            set_seed(seed); t0 = time.time()
            G, _, _ = cg.train_cgan(X_fit, y_fit, epochs=a.gan_epochs, resume=False,
                                    save_path=tmp_models)
            gens["cgan"] = (G, cg.generate_synthetic)
            meta["cgan_train_s"] = time.time() - t0
        if need_gan & {"wgan", "hybrid"}:
            set_seed(seed); t0 = time.time()
            G, _, _ = wg.train_wgan_gp(X_fit, y_fit, epochs=a.gan_epochs, resume=False)
            gens["wgan"] = (G, wg.generate_synthetic)
            meta["wgan_train_s"] = time.time() - t0

        for cfg in a.configs:
            out = pred_dir / f"nslkdd_s{seed}_{cfg}.npz"
            if out.exists() and not a.force:
                print(f"[rev] seed {seed} {cfg}: exists, skipping"); continue
            set_seed(seed)
            Xs, ys, how = augment(cfg, X_fit, y_fit, gens, seed)
            X_tr = np.concatenate([X_fit, Xs]) if len(Xs) else X_fit
            y_tr = np.concatenate([y_fit, ys]) if len(ys) else y_fit
            perm = np.random.default_rng(seed).permutation(len(y_tr))
            X_tr, y_tr = X_tr[perm], y_tr[perm]
            print(f"\n[rev] seed {seed} | {cfg} | train {np.bincount(y_tr, minlength=NUM_CLASSES).tolist()}")

            t0 = time.time()
            ens = es.EnsembleDetector(feature_dim=X_fit.shape[1])
            info = ens.fit(X_tr, y_tr, X_val, y_val, nn_epochs=a.nn_epochs, verbose=False)
            train_s = time.time() - t0

            def learners(X):
                return (ens.rf.predict_proba(X), ens.xgb.predict_proba(X), ens.nn.predict_proba(X))
            pv = learners(X_val); pt = learners(X_te)
            w = ens.weights
            np.savez_compressed(
                out,
                p_val=sum(wi * p for wi, p in zip(w, pv)), p_test=sum(wi * p for wi, p in zip(w, pt)),
                pv_rf=pv[0], pv_xgb=pv[1], pv_nn=pv[2], pt_rf=pt[0], pt_xgb=pt[1], pt_nn=pt[2],
                weights=w)
            meta["configs"][cfg] = {
                "class_handling": {rc.CLASS_NAMES[c]: h for c, h in how.items()},
                "train_counts": np.bincount(y_tr, minlength=NUM_CLASSES),
                "n_synthetic": int(len(ys)), "learner_val_acc": info, "train_s": train_s,
            }
            if cfg == "hybrid":
                meta["configs"][cfg]["ensemble_latency"] = latency(ens.predict_proba, X_te)
            rc.dump_json(meta, meta_path)
            print(f"[rev] saved {out.name}")
        rc.dump_json(meta, meta_path)
    print("\n[rev] done")


if __name__ == "__main__":
    main()
