"""
HCAPT-Defense - Final Evaluation Script
Compares our model against the 4 primary baseline references
highlighted in Referenc1.pdf:

  [3]  Jia et al. (2019)      - DNN/NDNN/ReLU/Entropy on NSL-KDD, KDD99
  [16] Arafah et al. (2024)   - BiGAN + WGAN + DL on CICIDS-2017, NSL-KDD
  [41] Radha & Kavitha (2025) - Fused IGAN-IDS on IIoT
  [43] Adnan et al. (2025)    - EFL-Adam (FL + Gen-AI) on CICIDS2018

Since these papers report results on varied datasets, we:
  1. Compare directly on NSL-KDD where they overlap (Jia, Arafah)
  2. Report their published metrics as reference points
  3. Produce a unified table + HCAPT-Defense targets
"""
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, NUM_CLASSES, LABEL_MAP,
    TARGET_ACCURACY, TARGET_FPR, TARGET_RESPONSE_TIME_S,
)
from data_loader import get_preprocessed
from ensemble_selflearn import EnsembleDetector, DeepDetector, evaluate_ensemble


INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  Published baseline metrics (from Referenc1.pdf)
# ============================================================
# These are numbers reported in the original papers - included here as
# scholarly reference points for the comparison table.
PUBLISHED_BASELINES = [
    {
        "ref": "[3]",
        "authors": "Jia et al. (2019)",
        "model": "DNN + NDNN + ReLU + Entropy",
        "dataset": "NSL-KDD + KDD99",
        "accuracy": 0.999,
        "precision": None,
        "recall": 0.9997,
        "f1": 0.9998,
        "fpr": 0.0003,
        "note": "Reported on NSL-KDD; parameter tuning difficulty; "
                "high computational cost; lack of real-world testing",
    },
    {
        "ref": "[16]",
        "authors": "Arafah et al. (2024)",
        "model": "BiGAN + WGAN + DL",
        "dataset": "CICIDS-2017 / NSL-KDD",
        "accuracy": 0.9962,          # NSL-KDD row
        "precision": 0.97,
        "recall": 0.95,
        "f1": 0.9101,
        "fpr": None,
        "note": "WGAN showed highest stability; ANOVA feature selection; "
                "GAN-based improved rare/unseen attack detection",
    },
    {
        "ref": "[41]",
        "authors": "Radha & Kavitha (2025)",
        "model": "Fused IGAN-IDS (GAN + RT + Sparse SAE)",
        "dataset": "IIoT (augmented with GAN samples)",
        "accuracy": 0.996,
        "precision": 0.984,
        "recall": 0.995,
        "f1": None,
        "fpr": 0.04,
        "note": "Jaccard similarity 99.7%; not validated in live IIoT",
    },
    {
        "ref": "[43]",
        "authors": "Adnan et al. (2025)",
        "model": "EFL-Adam (Federated Learning + Generative AI)",
        "dataset": "CICIDS2018",
        "accuracy": 0.99,
        "precision": 0.98,
        "recall": 0.98,
        "f1": 0.98,
        "fpr": None,
        "note": "Strong privacy compliance (GDPR/HIPAA); outperforms RF/SVM/DL",
    },
]


# ============================================================
#  Load our trained models
# ============================================================
def load_trained_ensemble(feature_dim: int) -> EnsembleDetector:
    ens = EnsembleDetector(feature_dim=feature_dim)
    try:
        ens.load()
        print("[eval] Loaded trained ensemble from disk")
        return ens
    except Exception as e:
        print(f"[eval] Could not load ensemble: {e}")
        return None


def load_baselines_results() -> List[Dict]:
    f = RESULTS_DIR / "baselines_results.json"
    if not f.exists():
        return []
    with open(f, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_ensemble_results() -> Dict:
    f = RESULTS_DIR / "ensemble_selflearn_results.json"
    if not f.exists():
        return {}
    with open(f, "r", encoding="utf-8") as fp:
        return json.load(fp)


# ============================================================
#  Build unified comparison table
# ============================================================
def build_comparison_table() -> pd.DataFrame:
    rows = []

    # 1. Published baselines (literature)
    for b in PUBLISHED_BASELINES:
        rows.append({
            "Source": "Literature",
            "Ref": b["ref"],
            "Model": f"{b['authors']} - {b['model']}",
            "Dataset": b["dataset"],
            "Accuracy": b["accuracy"],
            "Precision": b["precision"],
            "Recall": b["recall"],
            "F1": b["f1"],
            "FPR": b["fpr"],
        })

    # 2. Our baselines
    for m in load_baselines_results():
        rows.append({
            "Source": "Ours",
            "Ref": "-",
            "Model": f"Baseline - {m['model']}",
            "Dataset": "NSL-KDD",
            "Accuracy": m.get("accuracy"),
            "Precision": m.get("precision_weighted"),
            "Recall": m.get("recall_weighted"),
            "F1": m.get("f1_weighted"),
            "FPR": m.get("fpr"),
        })

    # 3. Our ensemble
    ens_results = load_ensemble_results()
    for key, label in [("before_selflearning", "HCAPT-Ensemble"),
                       ("after_selflearning", "HCAPT-Ensemble + SelfLearn")]:
        if key in ens_results:
            m = ens_results[key]
            rows.append({
                "Source": "Ours",
                "Ref": "-",
                "Model": label,
                "Dataset": "NSL-KDD",
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision_weighted"),
                "Recall": m.get("recall_weighted"),
                "F1": m.get("f1_weighted"),
                "FPR": m.get("fpr"),
            })

    # 4. Targets
    rows.append({
        "Source": "Target",
        "Ref": "-",
        "Model": "HCAPT-Defense (design target)",
        "Dataset": "-",
        "Accuracy": TARGET_ACCURACY,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "FPR": TARGET_FPR,
    })

    df = pd.DataFrame(rows)
    return df


def format_table(df: pd.DataFrame) -> str:
    """Pretty-print comparison table."""
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  -  "
        return f"{v:.4f}"

    df_fmt = df.copy()
    for col in ("Accuracy", "Precision", "Recall", "F1", "FPR"):
        df_fmt[col] = df[col].apply(fmt)

    return df_fmt.to_string(index=False, max_colwidth=50)


# ============================================================
#  Per-class deep dive (our model only, on NSL-KDD test)
# ============================================================
def per_class_breakdown(ens: EnsembleDetector, X_te: np.ndarray,
                        y_te: np.ndarray) -> pd.DataFrame:
    t0 = time.time()
    preds, conf = ens.predict_with_confidence(X_te)
    infer_time_ms = (time.time() - t0) / len(X_te) * 1000

    rows = []
    for i in range(NUM_CLASSES):
        cls = INV_LABEL_MAP[i]
        mask = y_te == i
        n = mask.sum()
        if n == 0:
            continue
        acc_cls = (preds[mask] == i).mean()
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te == i, preds == i, average="binary", zero_division=0,
        )
        rows.append({
            "Class": cls,
            "Support": int(n),
            "Accuracy": float(acc_cls),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "MeanConfidence": float(conf[mask].mean()),
        })
    df = pd.DataFrame(rows)
    return df, infer_time_ms


# ============================================================
#  Target check
# ============================================================
def check_targets(ens_metrics: Dict, infer_time_ms: float) -> Dict:
    acc = ens_metrics.get("accuracy", 0)
    fpr = ens_metrics.get("fpr", 1)
    response_time_s = infer_time_ms / 1000

    return {
        "accuracy": {
            "value": acc, "target": TARGET_ACCURACY,
            "passed": acc >= TARGET_ACCURACY,
        },
        "fpr": {
            "value": fpr, "target": TARGET_FPR,
            "passed": fpr <= TARGET_FPR,
        },
        "response_time_s": {
            "value": response_time_s, "target": TARGET_RESPONSE_TIME_S,
            "passed": response_time_s <= TARGET_RESPONSE_TIME_S,
        },
    }


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 70)
    print("  HCAPT-Defense - Final Evaluation & Comparison")
    print("=" * 70)

    # Load data
    X_tr, X_te, y_tr, y_te = get_preprocessed()

    # Load trained ensemble
    ens = load_trained_ensemble(feature_dim=X_tr.shape[1])
    if ens is None:
        print("\n[!] No trained ensemble found. "
              "Run ensemble_selflearn.py first.")
        return

    # Per-class breakdown
    print("\n[eval] Per-class performance (NSL-KDD test set):")
    per_class_df, infer_time_ms = per_class_breakdown(ens, X_te, y_te)
    print(per_class_df.to_string(index=False, float_format="%.4f"))
    print(f"\n[eval] Inference: {infer_time_ms:.3f} ms/sample "
          f"(target <= {TARGET_RESPONSE_TIME_S*1000:.0f} ms)")

    # Full metrics
    m_ens = evaluate_ensemble(ens, X_te, y_te, name="HCAPT-Ensemble")

    # Target check
    targets = check_targets(m_ens, infer_time_ms)
    print("\n" + "=" * 70)
    print("  Performance Targets Check")
    print("=" * 70)
    for key, t in targets.items():
        status = "PASS" if t["passed"] else "FAIL"
        arrow = "<=" if key != "accuracy" else ">="
        print(f"  {key:<18} | value={t['value']:.6f} "
              f"{arrow} target={t['target']:.6f}  [{status}]")

    # Full comparison table
    print("\n" + "=" * 70)
    print("  Comparison Table - Literature vs. HCAPT-Defense")
    print("=" * 70)
    df = build_comparison_table()
    print(format_table(df))

    # Save outputs
    out_csv = RESULTS_DIR / "final_comparison.csv"
    df.to_csv(out_csv, index=False)
    per_class_df.to_csv(RESULTS_DIR / "per_class_breakdown.csv", index=False)

    final = {
        "ensemble_metrics": m_ens,
        "per_class": per_class_df.to_dict(orient="records"),
        "targets": targets,
        "comparison_table": df.to_dict(orient="records"),
        "published_baselines": PUBLISHED_BASELINES,
    }
    with open(RESULTS_DIR / "final_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\n[OK] Saved:")
    print(f"  - {out_csv}")
    print(f"  - {RESULTS_DIR / 'per_class_breakdown.csv'}")
    print(f"  - {RESULTS_DIR / 'final_evaluation.json'}")


if __name__ == "__main__":
    main()
