"""
xgb+nn ensemble variant — drop RF, keep XGBoost + DNN soft-vote.

Loads the trained ensemble artefacts from `cache/<dataset>/ensemble/`
(produced by Phase 2's standard run) and recomputes binary metrics on
each dataset using only XGBoost + DNN with equal-weight soft voting.

This makes the "RF hurts" finding from the deep-ablation reproducible
across all 3 datasets, not just NSL-KDD.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict
import numpy as np
import joblib
import torch
import xgboost as xgb

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .cache import DatasetCache
from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .phase2_pipeline import compute_metrics, EnsembleAdHoc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_xgb_nn_variant"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


def _load_ensemble(cache: DatasetCache, tag: str, feature_dim: int, num_classes: int):
    sub = cache.ensemble_dir / tag
    if not (sub / "rf.joblib").exists():
        return None
    return EnsembleAdHoc(feature_dim, num_classes).load(sub)


def _xgb_nn_predict(ens: EnsembleAdHoc, X):
    """Soft-vote prediction using XGBoost + DNN only (RF dropped)."""
    p_xgb = ens.xgb.predict_proba(X)
    p_nn  = ens._nn_predict_proba(X)
    p     = 0.5 * p_xgb + 0.5 * p_nn   # equal weights
    return p.argmax(axis=1), p.max(axis=1)


def _full_3_predict(ens: EnsembleAdHoc, X):
    """Full RF + XGBoost + DNN soft-vote (uses ensemble's learnt weights)."""
    return ens.predict_with_confidence(X)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loaders = {
        "nsl-kdd":           lambda: NslKddLoader(DATA_ROOT).load("binary"),
        "ton-iot-telemetry": lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary"),
        "edge-iiotset":      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary"),
    }
    rows = []
    for ds_name, loader in loaders.items():
        print(f"\n=== {ds_name} ===")
        cache = DatasetCache(ds_name)
        # NSL-KDD uses the project pipeline, not Phase 2 — skip
        if ds_name == "nsl-kdd":
            # try to load from project models/
            from config import MODELS_DIR
            from ensemble_selflearn import EnsembleDetector, DeepDetector
            ds = loader()
            feature_dim = ds.X_train.shape[1]
            try:
                ens = EnsembleDetector(feature_dim=feature_dim)
                ens.rf = joblib.load(MODELS_DIR / "master_ens_rf.joblib")
                ens.xgb = xgb.XGBClassifier()
                ens.xgb.load_model(str(MODELS_DIR / "master_ens_xgb.json"))
                ens.nn = DeepDetector(feature_dim).to(DEVICE)
                ens.nn.load_state_dict(torch.load(
                    MODELS_DIR / "master_ens_nn.pt", map_location=DEVICE,
                ))
                ens.weights = np.load(MODELS_DIR / "master_ens_weights.npy")
            except Exception as e:
                print(f"   ! cannot load master ensemble: {e}")
                continue
            # 2-learner variant
            from ensemble_selflearn import EnsembleDetector
            p_xgb = ens.xgb.predict_proba(ds.X_test)
            ens.nn.eval()
            X_t = torch.tensor(ds.X_test, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                logits = ens.nn(X_t)
                p_nn = torch.softmax(logits, dim=1).cpu().numpy()
            p = 0.5 * p_xgb + 0.5 * p_nn
            xn_preds = p.argmax(axis=1)
            f3_preds, _ = ens.predict_with_confidence(ds.X_test)
        else:
            ds = loader()
            feature_dim = ds.X_train.shape[1]
            num_classes = len(ds.class_names)
            # try ens_gan first (the augmented one), fall back to ens_no_gan
            ens = _load_ensemble(cache, "ens_gan", feature_dim, num_classes)
            if ens is None:
                ens = _load_ensemble(cache, "ens_no_gan", feature_dim, num_classes)
            if ens is None:
                print("   ! no cached ensemble — skip"); continue
            xn_preds, _ = _xgb_nn_predict(ens, ds.X_test)
            f3_preds, _ = _full_3_predict(ens, ds.X_test)

        m_xn = compute_metrics(ds.y_test, xn_preds, is_binary=True)
        m_f3 = compute_metrics(ds.y_test, f3_preds, is_binary=True)
        delta_f1  = m_xn["f1"]       - m_f3["f1"]
        delta_acc = m_xn["accuracy"] - m_f3["accuracy"]
        print(f"   xgb+nn:        acc={m_xn['accuracy']:.4f}  f1={m_xn['f1']:.4f}")
        print(f"   rf+xgb+nn:     acc={m_f3['accuracy']:.4f}  f1={m_f3['f1']:.4f}")
        print(f"   Δ (xn - full): acc {delta_acc:+.4f}  f1 {delta_f1:+.4f}")
        rows.append({
            "dataset":      ds_name,
            "xgb_nn_acc":    m_xn["accuracy"],
            "xgb_nn_f1":     m_xn["f1"],
            "xgb_nn_fpr":    m_xn["fpr"],
            "full_3_acc":    m_f3["accuracy"],
            "full_3_f1":     m_f3["f1"],
            "full_3_fpr":    m_f3["fpr"],
            "delta_acc":     delta_acc,
            "delta_f1":      delta_f1,
            "rf_helps":      delta_f1 < 0,   # if 2-learner > 3-learner, RF hurts
        })

    # Save consolidated CSV
    import csv
    csv_path = RESULTS_DIR / "xgb_nn_vs_full.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\n[xgb_nn] wrote {csv_path}")

        with open(RESULTS_DIR / "xgb_nn_vs_full.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)

        print("\n" + "="*72)
        print("VERDICT — does dropping RF help?")
        print("="*72)
        for r in rows:
            verdict = "✓ xgb+nn WINS" if r["delta_f1"] > 0 else "✗ full 3 wins"
            print(f"  {r['dataset']:<22}  Δ F1 = {r['delta_f1']:+.4f}   "
                  f"FPR Δ = {r['xgb_nn_fpr'] - r['full_3_fpr']:+.4f}   {verdict}")
    return rows


if __name__ == "__main__":
    main()
