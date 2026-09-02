"""
HCAPT-Defense — Baseline ML Models
Layer 2 (partial): Classical ML detectors for comparison

Models:
  - Random Forest
  - XGBoost
  - Isolation Forest (anomaly detection baseline)

These form the base learners for the Ensemble (Innovation #2).
"""
import time
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
import xgboost as xgb

from config import MODELS_DIR, RESULTS_DIR, SEED, LABEL_MAP, NUM_CLASSES
from data_loader import load_raw, map_attack_category, encode_and_scale


# ============================================================
#  Metrics helpers
# ============================================================
def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    # Per-class metrics
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    # False Positive Rate (binary: attack vs normal)
    # class 0 = Normal; everything else = attack
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred != 0)).sum()
    fpr = fp / max(tn + fp, 1)

    return {
        "model": model_name,
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "fpr": float(fpr),
        "per_class": {
            label: {
                "precision": float(prec_pc[idx]),
                "recall": float(rec_pc[idx]),
                "f1": float(f1_pc[idx]),
                "support": int(sup_pc[idx]),
            }
            for label, idx in LABEL_MAP.items()
        },
    }


def print_metrics(m: dict):
    print(f"\n{'='*60}")
    print(f"  {m['model']}")
    print(f"{'='*60}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1 (wgt) : {m['f1_weighted']:.4f}")
    print(f"  Precision: {m['precision_weighted']:.4f}")
    print(f"  Recall   : {m['recall_weighted']:.4f}")
    print(f"  FPR      : {m['fpr']:.4f}")
    print(f"  Per-class:")
    for cls, s in m["per_class"].items():
        print(f"    {cls:<7} | P={s['precision']:.3f} "
              f"R={s['recall']:.3f} F1={s['f1']:.3f} n={s['support']}")


# ============================================================
#  Models
# ============================================================
def train_random_forest(X_tr, y_tr, X_te, y_te):
    print("\n[RF] Training Random Forest...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    infer_time = (time.time() - t0) / len(X_te)

    m = compute_metrics(y_te, y_pred, "RandomForest")
    m["train_time_s"] = train_time
    m["inference_time_per_sample_ms"] = infer_time * 1000

    joblib.dump(model, MODELS_DIR / "baseline_rf.joblib")
    print_metrics(m)
    print(f"  Train time: {train_time:.1f}s | "
          f"Inference: {infer_time*1000:.3f} ms/sample")
    return model, m


def train_xgboost(X_tr, y_tr, X_te, y_te):
    print("\n[XGB] Training XGBoost (GPU)...")
    t0 = time.time()
    # Compute class weights for imbalance
    counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    sample_weight = weights[y_tr]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    infer_time = (time.time() - t0) / len(X_te)

    m = compute_metrics(y_te, y_pred, "XGBoost")
    m["train_time_s"] = train_time
    m["inference_time_per_sample_ms"] = infer_time * 1000

    model.save_model(str(MODELS_DIR / "baseline_xgb.json"))
    print_metrics(m)
    print(f"  Train time: {train_time:.1f}s | "
          f"Inference: {infer_time*1000:.3f} ms/sample")
    return model, m


def train_isolation_forest(X_tr, y_tr, X_te, y_te):
    """Binary anomaly detection baseline: Normal vs. everything else."""
    print("\n[IF] Training Isolation Forest (anomaly detection)...")
    # Train only on Normal class (as classical anomaly detection does)
    X_normal = X_tr[y_tr == 0]
    print(f"  Training on {len(X_normal)} Normal samples only")

    t0 = time.time()
    model = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_normal)
    train_time = time.time() - t0

    # predict: 1 = normal, -1 = anomaly → map to 0 (Normal) or 1 (Attack)
    t0 = time.time()
    y_pred_if = model.predict(X_te)
    infer_time = (time.time() - t0) / len(X_te)
    y_pred_binary = (y_pred_if == -1).astype(int)  # 0=normal, 1=attack
    y_true_binary = (y_te != 0).astype(int)

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0,
    )
    tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
    fpr = fp / max(tn + fp, 1)

    m = {
        "model": "IsolationForest (binary)",
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "fpr": float(fpr),
        "train_time_s": train_time,
        "inference_time_per_sample_ms": infer_time * 1000,
        "note": "Binary (Normal vs Attack) — trained on Normal only",
    }
    joblib.dump(model, MODELS_DIR / "baseline_if.joblib")
    print(f"\n{'='*60}\n  {m['model']}\n{'='*60}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1       : {m['f1_weighted']:.4f}")
    print(f"  FPR      : {m['fpr']:.4f}")
    print(f"  Train time: {train_time:.1f}s | "
          f"Inference: {infer_time*1000:.3f} ms/sample")
    return model, m


# ============================================================
#  Main — train all baselines
# ============================================================
def main():
    print("="*60)
    print("  HCAPT-Defense — Baseline Models Training")
    print("="*60)

    # Load data
    df_train, df_test = load_raw()
    df_train = map_attack_category(df_train)
    df_test = map_attack_category(df_test)
    X_tr, X_te, y_tr, y_te = encode_and_scale(df_train, df_test)

    results = []

    # 1. Random Forest
    _, m_rf = train_random_forest(X_tr, y_tr, X_te, y_te)
    results.append(m_rf)

    # 2. XGBoost
    _, m_xgb = train_xgboost(X_tr, y_tr, X_te, y_te)
    results.append(m_xgb)

    # 3. Isolation Forest
    _, m_if = train_isolation_forest(X_tr, y_tr, X_te, y_te)
    results.append(m_if)

    # Save results
    out_file = RESULTS_DIR / "baselines_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    # Summary table
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"{'Model':<28}{'Acc':>8}{'F1':>8}{'FPR':>8}")
    print("-"*60)
    for m in results:
        print(f"{m['model']:<28}{m['accuracy']:>8.4f}"
              f"{m['f1_weighted']:>8.4f}{m['fpr']:>8.4f}")


if __name__ == "__main__":
    main()
