"""
HCAPT-Defense - Compare CGAN vs WGAN-GP augmentation.

Trains a fresh ensemble on WGAN-GP balanced data, evaluates on NSL-KDD
test, and compares side-by-side with:
  - Non-augmented ensemble
  - CGAN-augmented ensemble
  - WGAN-GP-augmented ensemble
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split

from config import MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP
from data_loader import get_preprocessed
from ensemble_selflearn import (
    EnsembleDetector, evaluate_ensemble, print_metrics,
)

CACHE_DIR = RESULTS_DIR.parent / "cache"


def load_balanced_wgangp() -> tuple:
    f = CACHE_DIR / "wgangp_balanced_data.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"WGAN-GP balanced data not found at {f}. "
            f"Run wgan_gp_augment.py first."
        )
    d = np.load(f)
    return d["X_bal"], d["y_bal"]


def main():
    print("=" * 70)
    print("  HCAPT-Defense - CGAN vs WGAN-GP comparison")
    print("=" * 70)

    # 1. Load test data
    X_full, X_te, y_full, y_te = get_preprocessed()
    feature_dim = X_te.shape[1]

    # 2. Load WGAN-GP balanced data
    X_bal, y_bal = load_balanced_wgangp()
    print(f"[compare] WGAN-GP balanced data: {X_bal.shape}")

    # 3. Train ensemble on WGAN-GP augmented
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_bal, y_bal, test_size=0.1, random_state=SEED, stratify=y_bal,
    )

    ens = EnsembleDetector(feature_dim=feature_dim)
    train_info = ens.fit(X_tr, y_tr, X_val, y_val, nn_epochs=30)

    # Save under 'wgangp_' prefix
    joblib.dump(ens.rf, MODELS_DIR / "wgangp_ens_rf.joblib")
    ens.xgb.save_model(str(MODELS_DIR / "wgangp_ens_xgb.json"))
    torch.save(ens.nn.state_dict(), MODELS_DIR / "wgangp_ens_nn.pt")
    np.save(MODELS_DIR / "wgangp_ens_weights.npy", ens.weights)

    # 4. Evaluate on real test set
    m_wgangp = evaluate_ensemble(
        ens, X_te, y_te, name="HCAPT-Ensemble (WGAN-GP Augmented)",
    )
    print_metrics(m_wgangp)

    # 5. Load other results for comparison
    with open(RESULTS_DIR / "ensemble_selflearn_results.json",
              "r", encoding="utf-8") as f:
        non_gan = json.load(f).get("before_selflearning", {})
    with open(RESULTS_DIR / "gan_augmented_ensemble.json",
              "r", encoding="utf-8") as f:
        cgan = json.load(f).get("gan_aug_before_selflearn", {})

    # 6. Side-by-side comparison
    print("\n" + "=" * 75)
    print("  Ensemble Augmentation Comparison (NSL-KDD test)")
    print("=" * 75)
    print(f"{'Metric':<20}{'No GAN':>14}{'CGAN':>14}{'WGAN-GP':>14}"
          f"{'Best Delta':>14}")
    print("-" * 75)

    metrics_to_show = [
        ("Accuracy", "accuracy"),
        ("Precision (w)", "precision_weighted"),
        ("Recall (w)", "recall_weighted"),
        ("F1 (w)", "f1_weighted"),
        ("FPR", "fpr"),
        ("Mean Confidence", "mean_confidence"),
    ]
    for label, key in metrics_to_show:
        v_none = non_gan.get(key, 0.0)
        v_cgan = cgan.get(key, 0.0)
        v_wgan = m_wgangp.get(key, 0.0)
        best_delta = v_wgan - v_none
        print(f"  {label:<18}{v_none:>14.4f}{v_cgan:>14.4f}"
              f"{v_wgan:>14.4f}{best_delta:>+14.4f}")

    # 7. Per-class comparison (minority classes matter most!)
    print("\n" + "=" * 75)
    print("  Per-Class Recall Comparison (catching each attack type)")
    print("=" * 75)
    print(f"{'Class':<10}{'Support':>10}{'No GAN':>14}{'CGAN':>14}{'WGAN-GP':>14}")
    print("-" * 75)
    for cls in ("Normal", "DoS", "Probe", "R2L", "U2R"):
        s = m_wgangp["per_class"][cls]["support"]
        r_none = non_gan.get("per_class", {}).get(cls, {}).get("recall", 0.0)
        r_cgan = cgan.get("per_class", {}).get(cls, {}).get("recall", 0.0)
        r_wgan = m_wgangp["per_class"][cls]["recall"]
        print(f"  {cls:<10}{s:>10}{r_none:>14.4f}{r_cgan:>14.4f}"
              f"{r_wgan:>14.4f}")

    # 8. Save
    out = {
        "non_gan": non_gan,
        "cgan": cgan,
        "wgangp": m_wgangp,
        "wgangp_train_info": train_info,
    }
    with open(RESULTS_DIR / "gan_comparison.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[OK] Saved: {RESULTS_DIR / 'gan_comparison.json'}")


if __name__ == "__main__":
    main()
