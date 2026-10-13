"""
Equal-budget hyper-parameter tuning on NSL-KDD (closes the "fixed hyper-parameters"
threat to validity, Section 6.3).

Every treatment receives the SAME budget: 40 Optuna TPE trials (sampler seed 0),
objective = five-class macro-F1 on VAL.  KDDTest+ is never touched while tuning.
Afterwards the best configuration of each treatment is retrained with seeds 0-4 on
FIT and scored once on KDDTest+.

Treatments (downstream learner for the resampling treatments = XGBoost, the strongest
single tree learner, so that 40 trials per treatment fit in a few hours):
  T1 xgb_cw      XGBoost + class weighting
  T2 xgb_smote   XGBoost + SMOTE
  T3 xgb_cgan    XGBoost + CGAN
  T4 xgb_wgan    XGBoost + WGAN-GP
  T5 xgb_hybrid  XGBoost + WGAN-GP for classes with >= 1,000 FIT rows, SMOTE otherwise
  T6 lr          logistic regression
  T7 dnn         the paper's DNN
Search spaces (shared parts identical across treatments):
  XGBoost   n_estimators 100-600, max_depth 3-10, learning_rate 0.02-0.3 (log),
            min_child_weight 1-10, subsample 0.6-1, colsample_bytree 0.5-1
  weights   class-weight power a in [0, 1]: w_c = (N / (K n_c))^a
  resample  top-up ratio r in {0.05, 0.1, 0.25, 0.5, 1.0}: each minority class is topped up
            to max(n_c, r * n_majority); SMOTE k in {3, 5, 10}
  GAN       epochs in {20, 50, 100}, learning rate in {1e-4, 2e-4, 5e-4}; generators are
            cached per (kind, epochs, lr) so a setting is trained once
  LR        C in 1e-3-1e2 (log), weight power a
  DNN       lr 1e-4-3e-3 (log), epochs {10, 20, 30, 50}, batch {256, 1024}, weight power a
Optuna studies are stored in revision/out/tuning/optuna.db, so an interrupted run resumes.

Usage (PowerShell, from 05-Implementation; one treatment per command):
  python -u revision\\r16_equal_budget_tuning.py --treatment xgb_cw --smoke
  python -u revision\\r16_equal_budget_tuning.py --treatment xgb_cw
  ... (xgb_smote, xgb_cgan, xgb_wgan, xgb_hybrid, lr, dnn)
  python -u revision\\r16_equal_budget_tuning.py --final        # retrain best configs, 5 seeds
"""
import argparse
import gc
import json
import time

import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import rev_common as rc
from rev_common import NUM_CLASSES, OUT_DIR, SUPPORT_THRESHOLD
import ensemble_selflearn as es
import gan_augment as cg
import wgan_gp_augment as wg

TREATMENTS = ["xgb_cw", "xgb_smote", "xgb_cgan", "xgb_wgan", "xgb_hybrid", "lr", "dnn"]
N_TRIALS = 40
TUNE = OUT_DIR / "tuning"; TUNE.mkdir(parents=True, exist_ok=True)
GEN_CACHE = TUNE / "generators"; GEN_CACHE.mkdir(exist_ok=True)
cg.save_checkpoint = lambda *a, **k: None; wg.save_checkpoint = lambda *a, **k: None
cg.MODELS_DIR = TUNE; wg.MODELS_DIR = TUNE


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def class_weights(y, power):
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    return (counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))) ** power


# ------------------------------------------------------------------ resampling
def topup_counts(y, ratio):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = int(ratio * counts.max())
    return {c: target - int(counts[c]) for c in range(NUM_CLASSES) if counts[c] < target}


def smote(X, y, need, k, seed):
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    if not need:
        return X, y
    counts = np.bincount(y, minlength=NUM_CLASSES)
    strat = {c: int(counts[c] + n) for c, n in need.items()}
    kk = min(k, min(counts[c] for c in strat) - 1)
    s = (SMOTE(sampling_strategy=strat, k_neighbors=kk, random_state=seed) if kk >= 1
         else RandomOverSampler(sampling_strategy=strat, random_state=seed))
    return s.fit_resample(X, y)


def generator(kind, epochs, lr, X, y, seed):
    """Train (or load from cache) a class-conditional generator on FIT."""
    path = GEN_CACHE / f"{kind}_e{epochs}_lr{lr:g}_s{seed}.pt"
    torch.manual_seed(seed); np.random.seed(seed)
    if kind == "cgan":
        G = cg.Generator(cg.GAN_LATENT_DIM, NUM_CLASSES, X.shape[1]).to(es.DEVICE)
        if path.exists():
            G.load_state_dict(torch.load(path, map_location=es.DEVICE)); return G, cg.generate_synthetic
        cg.GAN_LR = lr
        G, D, _ = cg.train_cgan(X, y, epochs=epochs, resume=False, save_path=TUNE)
        torch.save(G.state_dict(), path); del D; release()
        return G, cg.generate_synthetic
    G = wg.Generator(wg.GAN_LATENT_DIM, NUM_CLASSES, X.shape[1]).to(es.DEVICE)
    if path.exists():
        G.load_state_dict(torch.load(path, map_location=es.DEVICE)); return G, wg.generate_synthetic
    wg.WGAN_LR = lr
    G, C, _ = wg.train_wgan_gp(X, y, epochs=epochs, resume=False)
    torch.save(G.state_dict(), path); del C; release()
    return G, wg.generate_synthetic


def augment(treatment, p, X, y, seed):
    """Return the training set for one trial/seed."""
    if treatment in ("xgb_cw", "lr", "dnn"):
        return X, y
    need = topup_counts(y, p["ratio"])
    if treatment == "xgb_smote":
        return smote(X, y, need, p["k"], seed)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    if treatment == "xgb_hybrid":
        gan_need = {c: n for c, n in need.items() if counts[c] >= SUPPORT_THRESHOLD}
        smo_need = {c: n for c, n in need.items() if counts[c] < SUPPORT_THRESHOLD}
        X, y = smote(X, y, smo_need, 5, seed)
        need, kind = gan_need, "wgan"
    else:
        kind = "cgan" if treatment == "xgb_cgan" else "wgan"
    if not need:
        return X, y
    G, gen = generator(kind, p["gan_epochs"], p["gan_lr"], *FIT, seed)
    Xs, ys = gen(G, need); del G; release()
    return np.vstack([X, Xs.astype(np.float32)]), np.r_[y, ys.astype(np.int64)]


# ------------------------------------------------------------------ learners
def fit_predict(treatment, p, Xtr, ytr, Xev_list, seed):
    """Fit the treatment's learner; return class predictions for each evaluation set."""
    w = class_weights(ytr, p["w_power"])
    if treatment == "lr":
        m = LogisticRegression(C=p["C"], max_iter=3000, class_weight={c: w[c] for c in range(NUM_CLASSES)})
        m.fit(Xtr, ytr)
        return [m.predict(X) for X in Xev_list]
    if treatment == "dnn":
        torch.manual_seed(seed)
        nn = es.DeepDetector(Xtr.shape[1]).to(es.DEVICE)
        nn.fit(Xtr, ytr, epochs=p["epochs"], batch_size=p["batch"], lr=p["lr"], class_weights=w, verbose=False)
        out = [nn.predict_proba(X).argmax(1) for X in Xev_list]
        del nn; release(); return out
    m = xgb.XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                          min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                          colsample_bytree=p["colsample_bytree"], objective="multi:softprob",
                          num_class=NUM_CLASSES, tree_method="hist", device="cuda", random_state=seed)
    m.fit(Xtr, ytr, sample_weight=w[ytr])
    m.set_params(device="cpu")
    out = [m.predict(X) for X in Xev_list]
    del m; release(); return out


def suggest(trial, treatment):
    p = {"w_power": trial.suggest_float("w_power", 0.0, 1.0)}
    if treatment == "lr":
        p["C"] = trial.suggest_float("C", 1e-3, 1e2, log=True); return p
    if treatment == "dnn":
        p.update(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                 epochs=trial.suggest_categorical("epochs", [10, 20, 30, 50]),
                 batch=trial.suggest_categorical("batch", [256, 1024]))
        return p
    p.update(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
             max_depth=trial.suggest_int("max_depth", 3, 10),
             learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
             min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
             subsample=trial.suggest_float("subsample", 0.6, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0))
    if treatment != "xgb_cw":
        p["ratio"] = trial.suggest_categorical("ratio", [0.05, 0.1, 0.25, 0.5, 1.0])
    if treatment == "xgb_smote":
        p["k"] = trial.suggest_categorical("k", [3, 5, 10])
    if treatment in ("xgb_cgan", "xgb_wgan", "xgb_hybrid"):
        p["gan_epochs"] = trial.suggest_categorical("gan_epochs", [20, 50, 100])
        p["gan_lr"] = trial.suggest_categorical("gan_lr", [1e-4, 2e-4, 5e-4])
    return p


def tune(treatment, n_trials):
    storage = f"sqlite:///{(TUNE / 'optuna.db').as_posix()}"
    study = optuna.create_study(study_name=treatment, storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def objective(trial):
        t0 = time.time()
        p = suggest(trial, treatment)
        Xtr, ytr = augment(treatment, p, *FIT, seed=0)
        pv, = fit_predict(treatment, p, Xtr, ytr, [VAL[0]], seed=0)
        trial.set_user_attr("seconds", time.time() - t0)
        return f1_score(VAL[1], pv, average="macro")

    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, gc_after_trial=True)
    best = study.best_trial
    secs = sum(t.user_attrs.get("seconds", 0) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"\n[{treatment}] best VAL macro-F1 {best.value:.4f} | params {best.params} | total {secs/60:.1f} min")
    return study


def final(sfx=""):
    storage = f"sqlite:///{(TUNE / 'optuna.db').as_posix()}"
    rows, summary = [], []
    out = OUT_DIR / "tables" / f"tuning_final{sfx}.csv"
    done = set(zip(*[pd.read_csv(out)[c] for c in ("treatment", "seed")])) if out.exists() else set()
    for tr in TREATMENTS:
        try:
            study = optuna.load_study(study_name=tr, storage=storage)
        except KeyError:
            print(f"[final] {tr}: not tuned yet, skipped"); continue
        n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        best = study.best_trial
        p = {**best.params}
        secs = sum(t.user_attrs.get("seconds", 0) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        summary.append({"treatment": tr, "trials": n, "best_val_macro_f1": best.value,
                        "tuning_minutes": secs / 60, "best_params": json.dumps(p)})
        for seed in range(5):
            if (tr, seed) in done:
                continue
            Xtr, ytr = augment(tr, p, *FIT, seed=seed)
            pv, pt = fit_predict(tr, p, Xtr, ytr, [VAL[0], TEST[0]], seed=seed)
            yt = TEST[1]; b = rc.binary_metrics(yt, pt)
            row = {"treatment": tr, "seed": seed, "val_macro_f1": f1_score(VAL[1], pv, average="macro"),
                   "test_macro_f1": f1_score(yt, pt, average="macro"), "test_binary_accuracy": b["accuracy"],
                   "test_binary_f1": b["f1"], "test_binary_fpr": b["fpr"],
                   **{f"recall_{rc.CLASS_NAMES[c]}": float((pt[yt == c] == c).mean()) for c in range(NUM_CLASSES)}}
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"[final] {tr} seed {seed}: test acc {b['accuracy']:.4f} F1 {b['f1']:.4f} "
                  f"macro-F1 {row['test_macro_f1']:.4f}")
    pd.DataFrame(summary).to_csv(OUT_DIR / "tables" / f"tuning_summary{sfx}.csv", index=False)
    d = pd.read_csv(out)
    pd.set_option("display.width", 220)
    print(d.groupby("treatment")[["val_macro_f1", "test_binary_accuracy", "test_binary_f1", "test_binary_fpr",
                                  "test_macro_f1", "recall_R2L", "recall_U2R"]].agg("mean").round(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", choices=TREATMENTS)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 trials into a separate study database")
    a = ap.parse_args()
    global FIT, VAL, TEST, TUNE, GEN_CACHE
    if a.smoke:
        TUNE = OUT_DIR / "tuning_smoke"; TUNE.mkdir(exist_ok=True)
        GEN_CACHE = TUNE / "generators"; GEN_CACHE.mkdir(exist_ok=True)
        cg.MODELS_DIR = TUNE; wg.MODELS_DIR = TUNE
    D = rc.load_nslkdd()
    FIT, VAL, TEST = (D["X_fit"], D["y_fit"]), (D["X_val"], D["y_val"]), (D["X_test"], D["y_test"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.smoke:
        # tiny GANs for the pipeline check only
        cg_train, wg_train = cg.train_cgan, wg.train_wgan_gp
        cg.train_cgan = lambda X, y, epochs=1, **k: cg_train(X, y, epochs=1, **k)
        wg.train_wgan_gp = lambda X, y, epochs=1, **k: wg_train(X, y, epochs=1, **k)
    if a.final:
        final("_smoke" if a.smoke else ""); return
    if not a.treatment:
        ap.error("--treatment or --final is required")
    study = tune(a.treatment, 3 if a.smoke else N_TRIALS)
    print(study.trials_dataframe(attrs=("number", "value", "params", "user_attrs")).round(4).tail(5).to_string())


if __name__ == "__main__":
    main()
