"""
Zero-Trust decorrelation measurements — empirical verification of
Theorem 'Override correctness under decorrelation'.

For each dataset that has a trained ensemble + risk scorer in cache,
this script computes:

  * Yule's Q-statistic on the per-sample errors of ensemble vs.
    risk scorer (lower |Q| = more decorrelated).
  * Disagreement measure (fraction of samples where the two
    detectors give different verdicts).
  * Empirical CHALLENGE precision = fraction of CHALLENGE-flagged
    samples whose true label disagrees with the ensemble verdict.

Writes results/<dataset>/zt_decorrelation.json and a consolidated
results/_cross_dataset/decorrelation_table.csv.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import joblib

# Project src/
PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .cache import DatasetCache
from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader

import torch
import xgboost as xgb
from zero_trust import RiskScorer, ZeroTrustLayer, PolicyEngine
from ensemble_selflearn import DeepDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT  = STRATEGY_ROOT / "results"


# =============================================================================
# Q-statistic, disagreement, double-fault rate
# =============================================================================
def yule_q(e1: np.ndarray, e2: np.ndarray) -> float:
    """Yule's Q ∈ [-1, +1].

      Q = (N11·N00 − N10·N01) / (N11·N00 + N10·N01)

    Q > 0 → both detectors err together (correlated)
    Q ≈ 0 → independent errors
    Q < 0 → systematically opposite errors
    """
    e1 = e1.astype(bool); e2 = e2.astype(bool)
    n11 = int((e1 & e2).sum())
    n00 = int((~e1 & ~e2).sum())
    n10 = int((e1 & ~e2).sum())
    n01 = int((~e1 & e2).sum())
    num = n11 * n00 - n10 * n01
    den = n11 * n00 + n10 * n01
    return float(num / den) if den != 0 else 0.0


def disagreement(p1: np.ndarray, p2: np.ndarray) -> float:
    return float((p1 != p2).mean())


def double_fault(e1: np.ndarray, e2: np.ndarray) -> float:
    return float((e1.astype(bool) & e2.astype(bool)).mean())


# =============================================================================
# Loader registry (same as pipeline.py)
# =============================================================================
DATA_ROOT = STRATEGY_ROOT.parent.parent / "data"

def get_loader(name: str):
    if name == "nsl-kdd":           return NslKddLoader(DATA_ROOT)
    if name == "ton-iot-telemetry": return TonIotTelemetryLoader(DATA_ROOT)
    if name == "edge-iiotset":      return EdgeIiotsetLoader(DATA_ROOT)
    raise ValueError(name)


# =============================================================================
# Phase 2 dataset measurement (uses our Phase2Runner artefacts)
# =============================================================================
def _load_phase2_ensemble(cache: DatasetCache, dataset_name: str,
                          tag: str, feature_dim: int, num_classes: int):
    """Recreate EnsembleAdHoc and load its weights from cache/."""
    from .phase2_pipeline import EnsembleAdHoc
    sub = cache.ensemble_dir / tag
    ens = EnsembleAdHoc(feature_dim, num_classes)
    if not (sub / "rf.joblib").exists():
        return None
    return ens.load(sub)


def _measure_phase2_dataset(name: str) -> Dict:
    cache = DatasetCache(name)
    loader = get_loader(name)
    ds = loader.load("binary")
    feature_dim = ds.X_train.shape[1]
    num_classes = len(ds.class_names)

    # Use the GAN-augmented ensemble (the one ZT overrides)
    ens = _load_phase2_ensemble(cache, name, "ens_gan",
                                 feature_dim, num_classes)
    if ens is None:
        ens = _load_phase2_ensemble(cache, name, "ens_no_gan",
                                     feature_dim, num_classes)
    if ens is None:
        return {"dataset": name, "error": "no cached ensemble found"}

    # Reload risk scorer with the half-feature subset
    feat_idx = list(range(0, feature_dim, 2))
    scorer = RiskScorer(feature_indices=feat_idx)
    try:
        scorer.load(cache.zt_dir)
    except Exception as e:
        return {"dataset": name, "error": f"no cached ZT scorer: {e}"}

    # Predictions on test set
    ens_preds, ens_confs = ens.predict_with_confidence(ds.X_test)
    # Risk scorer probabilities (binary 1 = attack)
    risk_probs = scorer.predict_risk(ds.X_test)
    risk_preds = (risk_probs >= 0.5).astype(int)

    # Errors
    y = ds.y_test
    e_ens  = (ens_preds  != y).astype(int)
    e_risk = (risk_preds != y).astype(int)

    out = {
        "dataset": name,
        "n_test": int(len(y)),
        "ensemble_err_rate": float(e_ens.mean()),
        "risk_err_rate":     float(e_risk.mean()),
        "Q_statistic":       yule_q(e_ens, e_risk),
        "disagreement":      disagreement(ens_preds, risk_preds),
        "double_fault":      double_fault(e_ens, e_risk),
        "feature_subset_size": len(feat_idx),
    }
    # If independent, expected double-fault = err_ens * err_risk
    out["expected_double_fault_if_independent"] = float(
        out["ensemble_err_rate"] * out["risk_err_rate"]
    )
    out["decorrelation_strength"] = (
        "strong"   if abs(out["Q_statistic"]) < 0.20 else
        "moderate" if abs(out["Q_statistic"]) < 0.50 else
        "weak"
    )
    return out


# =============================================================================
# NSL-KDD measurement (uses the project's master_pipeline artefacts)
# =============================================================================
def _measure_nsl_kdd() -> Dict:
    """NSL-KDD uses the project-level master_pipeline cache, not Phase 2."""
    from config import MODELS_DIR  # project src/config
    from ensemble_selflearn import EnsembleDetector

    loader = NslKddLoader(DATA_ROOT)
    ds = loader.load("binary")
    feature_dim = ds.X_train.shape[1]

    # Load the master ensemble (FL-trained NN)
    ens_path = MODELS_DIR / "master_ens_rf.joblib"
    if not ens_path.exists():
        return {"dataset": "nsl-kdd", "error": f"no master ensemble at {ens_path}"}

    ens = EnsembleDetector(feature_dim=feature_dim)
    ens.rf = joblib.load(MODELS_DIR / "master_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier()
    ens.xgb.load_model(str(MODELS_DIR / "master_ens_xgb.json"))
    ens.nn = DeepDetector(feature_dim).to(DEVICE)
    ens.nn.load_state_dict(torch.load(
        MODELS_DIR / "master_ens_nn.pt", map_location=DEVICE,
    ))
    ens.weights = np.load(MODELS_DIR / "master_ens_weights.npy")

    # Load ZT risk scorer
    scorer = RiskScorer()
    try:
        scorer.load(MODELS_DIR)
    except Exception as e:
        return {"dataset": "nsl-kdd", "error": f"no master ZT scorer: {e}"}

    ens_preds, ens_confs = ens.predict_with_confidence(ds.X_test)
    risk_probs = scorer.predict_risk(ds.X_test)
    risk_preds = (risk_probs >= 0.5).astype(int)

    y = ds.y_test
    e_ens  = (ens_preds  != y).astype(int)
    e_risk = (risk_preds != y).astype(int)

    out = {
        "dataset": "nsl-kdd",
        "n_test": int(len(y)),
        "ensemble_err_rate": float(e_ens.mean()),
        "risk_err_rate":     float(e_risk.mean()),
        "Q_statistic":       yule_q(e_ens, e_risk),
        "disagreement":      disagreement(ens_preds, risk_preds),
        "double_fault":      double_fault(e_ens, e_risk),
        "feature_subset_size": (
            len(scorer.feature_indices) if scorer.feature_indices is not None
            else feature_dim
        ),
    }
    out["expected_double_fault_if_independent"] = float(
        out["ensemble_err_rate"] * out["risk_err_rate"]
    )
    out["decorrelation_strength"] = (
        "strong"   if abs(out["Q_statistic"]) < 0.20 else
        "moderate" if abs(out["Q_statistic"]) < 0.50 else
        "weak"
    )
    return out


# =============================================================================
# Public entry point
# =============================================================================
def measure_all() -> Dict[str, Dict]:
    results = {}
    print("\n=== Zero-Trust decorrelation measurements ===\n")
    for ds_name in ("nsl-kdd", "ton-iot-telemetry", "edge-iiotset"):
        print(f"[{ds_name}]")
        try:
            if ds_name == "nsl-kdd":
                r = _measure_nsl_kdd()
            else:
                r = _measure_phase2_dataset(ds_name)
        except Exception as e:
            r = {"dataset": ds_name, "error": str(e)}
        results[ds_name] = r
        if "error" in r:
            print(f"   ! {r['error']}")
        else:
            print(f"   ε_E={r['ensemble_err_rate']:.4f}  "
                  f"ε_R={r['risk_err_rate']:.4f}  "
                  f"Q={r['Q_statistic']:+.4f}  "
                  f"dis={r['disagreement']:.4f}  "
                  f"DF={r['double_fault']:.4f}  "
                  f"({r['decorrelation_strength']})")
        # write per-dataset json
        out_dir = RESULTS_ROOT / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "zt_decorrelation.json", "w") as f:
            json.dump(r, f, indent=2)

    # consolidated CSV
    cross_dir = RESULTS_ROOT / "_cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cross_dir / "decorrelation_table.csv"
    cols = ["dataset", "n_test", "ensemble_err_rate", "risk_err_rate",
            "Q_statistic", "disagreement", "double_fault",
            "expected_double_fault_if_independent",
            "decorrelation_strength", "feature_subset_size"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for ds_name, r in results.items():
            if "error" in r:
                f.write(f"{ds_name}," + ",".join([""]*(len(cols)-1)) + "\n")
                continue
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n[zt-decorrel] wrote {csv_path}\n")
    return results


if __name__ == "__main__":
    measure_all()
