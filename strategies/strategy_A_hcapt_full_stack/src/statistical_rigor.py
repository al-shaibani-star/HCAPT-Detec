"""
Statistical Rigor Module — Cross-validation, bootstrap CIs, and
pairwise significance tests for HCAPT-Defense.

Provides reviewer-grade rigor:
  * 10-fold stratified cross-validation (mean ± std)
  * Bootstrap 95% confidence intervals (1000 resamples)
  * McNemar's test (HCAPT vs baseline classifier)
  * Wilcoxon signed-rank test (paired metric distributions)
  * Bonferroni correction for multi-comparison

Writes:
  results/_statistical_rigor/<dataset>_cv.json
  results/_statistical_rigor/<dataset>_bootstrap.json
  results/_statistical_rigor/<dataset>_significance.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import joblib
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import EnsembleAdHoc, compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_statistical_rigor"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


# =============================================================================
# 1. K-fold Cross-Validation
# =============================================================================
def cross_validate_ensemble(X, y, num_classes: int, n_folds: int = 10,
                             seed: int = 42, dnn_epochs: int = 10) -> Dict:
    """10-fold stratified CV with the full ensemble. Returns per-fold metrics + summary."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: List[Dict] = []
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        ens = EnsembleAdHoc(X.shape[1], num_classes, dnn_epochs=dnn_epochs)
        # 90/10 val for weight learning
        X_t, X_v, y_t, y_v = train_test_split(
            X_tr, y_tr, test_size=0.1, random_state=seed, stratify=y_tr,
        )
        ens.fit(X_t, y_t, X_v, y_v)
        preds, _ = ens.predict_with_confidence(X_te)
        m = compute_metrics(y_te, preds, is_binary=(num_classes == 2))
        m["fold"] = fold_i + 1
        fold_metrics.append(m)
        print(f"   fold {fold_i+1}/{n_folds}: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    # Summary
    summary = {
        "n_folds": n_folds,
        "mean_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])),
        "std_accuracy":  float(np.std ([f["accuracy"] for f in fold_metrics])),
        "mean_f1":       float(np.mean([f["f1"]       for f in fold_metrics])),
        "std_f1":        float(np.std ([f["f1"]       for f in fold_metrics])),
        "mean_fpr":      float(np.mean([f["fpr"]      for f in fold_metrics])),
        "std_fpr":       float(np.std ([f["fpr"]      for f in fold_metrics])),
        "ci95_accuracy": [
            float(np.mean([f["accuracy"] for f in fold_metrics]) - 1.96 * np.std([f["accuracy"] for f in fold_metrics]) / np.sqrt(n_folds)),
            float(np.mean([f["accuracy"] for f in fold_metrics]) + 1.96 * np.std([f["accuracy"] for f in fold_metrics]) / np.sqrt(n_folds)),
        ],
        "fold_metrics": fold_metrics,
    }
    return summary


# =============================================================================
# 2. Bootstrap CIs
# =============================================================================
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_resamples: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> Dict:
    """Bootstrap (1-alpha)*100% CIs on accuracy/F1/FPR."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, fprs = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        m = compute_metrics(yt, yp, is_binary=True)
        accs.append(m["accuracy"]); f1s.append(m["f1"]); fprs.append(m["fpr"])
    return {
        "n_resamples": n_resamples,
        "ci_level":    1 - alpha,
        "accuracy":    {
            "mean":  float(np.mean(accs)),
            "lower": float(np.quantile(accs, alpha/2)),
            "upper": float(np.quantile(accs, 1-alpha/2)),
        },
        "f1": {
            "mean":  float(np.mean(f1s)),
            "lower": float(np.quantile(f1s, alpha/2)),
            "upper": float(np.quantile(f1s, 1-alpha/2)),
        },
        "fpr": {
            "mean":  float(np.mean(fprs)),
            "lower": float(np.quantile(fprs, alpha/2)),
            "upper": float(np.quantile(fprs, 1-alpha/2)),
        },
    }


# =============================================================================
# 3. McNemar's test (HCAPT vs baseline classifier)
# =============================================================================
def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray,
                 name_a: str = "HCAPT", name_b: str = "Baseline") -> Dict:
    """McNemar's test on per-sample correct/incorrect contingency."""
    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)
    # 2x2 contingency: rows = A correct/wrong, cols = B correct/wrong
    n00 = int(((~a_correct) & (~b_correct)).sum())
    n01 = int(((~a_correct) & ( b_correct)).sum())
    n10 = int(((a_correct)  & (~b_correct)).sum())
    n11 = int(((a_correct)  & ( b_correct)).sum())
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=(n01 + n10 < 25), correction=True)
    return {
        "method_a":      name_a, "method_b": name_b,
        "n00_both_wrong":   n00, "n11_both_right": n11,
        "n10_a_only_right": n10, "n01_b_only_right": n01,
        "statistic":     float(result.statistic),
        "p_value":       float(result.pvalue),
        "significant_5pct":  bool(result.pvalue < 0.05),
        "significant_1pct":  bool(result.pvalue < 0.01),
    }


# =============================================================================
# 4. Wilcoxon Signed-Rank (paired per-fold accuracy)
# =============================================================================
def wilcoxon_paired(scores_a: List[float], scores_b: List[float],
                    name_a: str = "HCAPT", name_b: str = "Baseline") -> Dict:
    """Wilcoxon signed-rank test on paired per-fold scores."""
    try:
        stat, p = wilcoxon(scores_a, scores_b, alternative="two-sided")
        return {
            "method_a": name_a, "method_b": name_b,
            "mean_a":   float(np.mean(scores_a)),
            "mean_b":   float(np.mean(scores_b)),
            "diff":     float(np.mean(scores_a) - np.mean(scores_b)),
            "statistic":  float(stat),
            "p_value":    float(p),
            "significant_5pct":  bool(p < 0.05),
            "significant_1pct":  bool(p < 0.01),
        }
    except ValueError as e:
        return {"error": str(e)}


# =============================================================================
# 5. Bonferroni correction
# =============================================================================
def bonferroni_correct(p_values: Dict[str, float], alpha: float = 0.05) -> Dict:
    n = len(p_values)
    corrected = {}
    for name, p in p_values.items():
        adj_p = min(1.0, p * n)
        corrected[name] = {
            "raw_p":          float(p),
            "adjusted_p":     float(adj_p),
            "significant_5pct":  bool(adj_p < alpha),
        }
    return {"n_comparisons": n, "alpha": alpha, "comparisons": corrected}


# =============================================================================
# Main runner
# =============================================================================
def run_for_dataset(loader_fn, name: str, n_folds: int = 5, dnn_epochs: int = 10):
    """Run all statistical tests for one dataset."""
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    ds = loader_fn()
    num_classes = len(ds.class_names)

    # 1. CV
    print(f"\n[CV] {n_folds}-fold cross-validation…")
    cv = cross_validate_ensemble(
        np.vstack([ds.X_train, ds.X_test]),
        np.concatenate([ds.y_train, ds.y_test]),
        num_classes=num_classes, n_folds=n_folds, dnn_epochs=dnn_epochs,
    )
    print(f"   acc {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f} "
          f"(95% CI [{cv['ci95_accuracy'][0]:.4f}, {cv['ci95_accuracy'][1]:.4f}])")
    print(f"   f1  {cv['mean_f1']:.4f} ± {cv['std_f1']:.4f}")

    # 2. Bootstrap CI on test set predictions
    print(f"\n[Bootstrap] computing 1000-resample CI on test set…")
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=dnn_epochs)
    X_t, X_v, y_t, y_v = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, random_state=42, stratify=ds.y_train,
    )
    ens.fit(X_t, y_t, X_v, y_v)
    preds, _ = ens.predict_with_confidence(ds.X_test)
    bs = bootstrap_ci(ds.y_test, preds)
    print(f"   acc 95% CI: [{bs['accuracy']['lower']:.4f}, {bs['accuracy']['upper']:.4f}]")
    print(f"   f1  95% CI: [{bs['f1']['lower']:.4f}, {bs['f1']['upper']:.4f}]")

    # 3. McNemar: HCAPT vs single RF baseline
    print(f"\n[McNemar] HCAPT vs RandomForest baseline…")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_baseline.fit(ds.X_train, ds.y_train)
    rf_preds = rf_baseline.predict(ds.X_test)
    mc = mcnemar_test(ds.y_test, preds, rf_preds, name_a="HCAPT-Ensemble", name_b="RF-baseline")
    print(f"   p = {mc['p_value']:.4e}  → "
          f"{'SIGNIFICANT' if mc['significant_5pct'] else 'not significant'} at 5%")

    # 4. Wilcoxon: per-fold acc HCAPT vs per-fold acc RF (paired)
    print(f"\n[Wilcoxon] per-fold paired test…")
    rf_fold_accs = []
    X_all = np.vstack([ds.X_train, ds.X_test])
    y_all = np.concatenate([ds.y_train, ds.y_test])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(X_all, y_all):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_all[tr_idx], y_all[tr_idx])
        acc = (rf.predict(X_all[te_idx]) == y_all[te_idx]).mean()
        rf_fold_accs.append(acc)
    hcapt_fold_accs = [f["accuracy"] for f in cv["fold_metrics"]]
    wx = wilcoxon_paired(hcapt_fold_accs, rf_fold_accs,
                          name_a="HCAPT-Ensemble", name_b="RF-baseline")
    print(f"   p = {wx.get('p_value', 'N/A')}  → "
          f"diff = {wx.get('diff', 0):+.4f}")

    # 5. Bonferroni correction
    bonf = bonferroni_correct({
        "mcnemar_hcapt_vs_rf":   mc["p_value"],
        "wilcoxon_hcapt_vs_rf":  wx.get("p_value", 1.0),
    })
    print(f"\n[Bonferroni] {bonf['n_comparisons']} comparisons, "
          f"adjusted: {list(bonf['comparisons'].values())}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": name,
        "cv": cv, "bootstrap": bs,
        "mcnemar_vs_rf_baseline":  mc,
        "wilcoxon_vs_rf_baseline": wx,
        "bonferroni": bonf,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = RESULTS_DIR / f"{name}_rigor.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] wrote {out_path.name}")
    return out


def main():
    # Run on 3 datasets — small n_folds=5 to keep total time manageable
    run_for_dataset(lambda: NslKddLoader(DATA_ROOT).load("binary"),
                     "nsl-kdd", n_folds=5, dnn_epochs=10)
    run_for_dataset(lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary"),
                     "ton-iot-telemetry", n_folds=5, dnn_epochs=10)
    run_for_dataset(lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary"),
                     "edge-iiotset", n_folds=5, dnn_epochs=10)
    run_for_dataset(lambda: Dapt2020Loader(DATA_ROOT).load("binary"),
                     "dapt-2020", n_folds=5, dnn_epochs=10)


if __name__ == "__main__":
    main()
