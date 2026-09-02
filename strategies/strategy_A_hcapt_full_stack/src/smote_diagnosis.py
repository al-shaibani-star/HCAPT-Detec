"""
ToN-IoT GAN issue diagnosis — SMOTE alternatives.

Runs four augmentation regimes on ToN-IoT IoT Telemetry and reports
the binary F1 / FPR / accuracy of an Ensemble classifier for each:

    1. NO augment        (baseline)
    2. SMOTE only        (interpolation)
    3. WGAN-GP only      (what we use in Strategy A)
    4. SMOTE + WGAN-GP   (FIDMF-style staged)

Validates the "Mode Collapse on low-dim telemetry" hypothesis: if
SMOTE alone OUTPERFORMS WGAN-GP, then the GAN is the bottleneck
on ToN-IoT, not class imbalance per se.

Writes  results/ton-iot-telemetry/smote_diagnosis.json
        results/ton-iot-telemetry/smote_diagnosis.csv
        results/ton-iot-telemetry/smote_diagnosis.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# Project src/
PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .phase2_pipeline import (
    EnsembleAdHoc, train_wgan_gp_explicit, gan_balance, compute_metrics,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "ton-iot-telemetry"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


# =============================================================================
# Augmentation strategies
# =============================================================================
def augment_none(X_train, y_train, **_):
    return X_train, y_train, "none"


def augment_smote(X_train, y_train, **_):
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=5)
    Xb, yb = sm.fit_resample(X_train, y_train)
    return Xb.astype(np.float32), yb.astype(int), "smote"


def augment_gan(X_train, y_train, feature_dim, num_classes, **_):
    G = train_wgan_gp_explicit(
        X_train, y_train,
        feature_dim=feature_dim, num_classes=num_classes,
        epochs=30,
    )
    Xb, yb = gan_balance(X_train, y_train, G, num_classes)
    return Xb, yb, "wgan-gp"


def augment_smote_then_gan(X_train, y_train, feature_dim, num_classes, **_):
    Xs, ys, _ = augment_smote(X_train, y_train)
    G = train_wgan_gp_explicit(
        Xs, ys, feature_dim=feature_dim, num_classes=num_classes,
        epochs=30,
    )
    Xb, yb = gan_balance(Xs, ys, G, num_classes)
    return Xb, yb, "smote+wgan-gp"


# =============================================================================
# One-shot evaluation
# =============================================================================
def evaluate_strategy(X_train, y_train, X_test, y_test,
                       feature_dim, num_classes,
                       augment_fn, label: str) -> Dict:
    print(f"\n[smote-diag] === {label} ===")
    t0 = time.time()
    Xa, ya, _ = augment_fn(X_train, y_train,
                            feature_dim=feature_dim,
                            num_classes=num_classes)
    aug_time = time.time() - t0
    print(f"   augmentation: {X_train.shape} -> {Xa.shape}  ({aug_time:.1f}s)")

    t0 = time.time()
    ens = EnsembleAdHoc(feature_dim, num_classes, dnn_epochs=15)
    # 90/10 val split inside augmented data
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xa, ya, test_size=0.1, random_state=42, stratify=ya,
    )
    ens.fit(X_tr, y_tr, X_va, y_va)
    train_time = time.time() - t0
    print(f"   train time: {train_time:.1f}s")

    preds, _ = ens.predict_with_confidence(X_test)
    metrics = compute_metrics(y_test, preds, is_binary=True)
    metrics["label"] = label
    metrics["augment_time_s"] = round(aug_time, 1)
    metrics["train_time_s"]  = round(train_time, 1)
    metrics["augmented_n"]   = int(len(ya))
    print(f"   acc={metrics['accuracy']:.4f}  "
          f"f1={metrics['f1']:.4f}  fpr={metrics['fpr']:.4f}")
    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n" + "#"*72)
    print("#  ToN-IoT GAN Diagnosis — SMOTE vs WGAN-GP vs Combined")
    print("#"*72)

    # Load dataset
    loader = TonIotTelemetryLoader(DATA_ROOT)
    ds = loader.load("binary")
    print(ds.summary())
    feature_dim = ds.X_train.shape[1]
    num_classes = len(ds.class_names)

    strategies = [
        ("none",            augment_none),
        ("smote",           augment_smote),
        ("wgan-gp",         augment_gan),
        ("smote+wgan-gp",   augment_smote_then_gan),
    ]

    results = []
    for label, fn in strategies:
        try:
            m = evaluate_strategy(
                ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                feature_dim, num_classes, fn, label,
            )
            results.append(m)
        except Exception as e:
            print(f"   FAILED: {e}")
            results.append({"label": label, "error": str(e)})

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "smote_diagnosis.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    cols = ["label", "accuracy", "precision", "recall", "f1", "fpr",
             "augmented_n", "augment_time_s", "train_time_s"]
    with open(RESULTS_DIR / "smote_diagnosis.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # Plot — bar chart of F1 / FPR per strategy
    valid = [r for r in results if "error" not in r]
    if valid:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [r["label"] for r in valid]
        axes[0].bar(labels, [r["f1"] for r in valid], color="#4C6EF5",
                     edgecolor="black")
        axes[0].set_ylabel("F1 (binary)"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("F1 by augmentation strategy")
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)
        axes[1].bar(labels, [r["fpr"] for r in valid], color="#E8590C",
                     edgecolor="black")
        axes[1].set_ylabel("FPR"); axes[1].set_ylim(0, 1.05)
        axes[1].set_title("FPR by augmentation strategy")
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "smote_diagnosis.png", dpi=160)
        plt.close(fig)

    # Final summary
    print("\n" + "="*72)
    print("DIAGNOSIS SUMMARY (ToN-IoT IoT Telemetry, binary)")
    print("="*72)
    print(f"{'strategy':<18}  {'acc':>7}  {'f1':>7}  {'fpr':>7}  {'aug_n':>10}")
    for r in valid:
        print(f"{r['label']:<18}  {r['accuracy']:>7.4f}  "
              f"{r['f1']:>7.4f}  {r['fpr']:>7.4f}  {r['augmented_n']:>10}")

    # Verdict
    if len(valid) >= 3:
        f_none = next((r['f1'] for r in valid if r['label']=='none'),    None)
        f_smot = next((r['f1'] for r in valid if r['label']=='smote'),   None)
        f_gan  = next((r['f1'] for r in valid if r['label']=='wgan-gp'), None)
        if f_none and f_smot and f_gan:
            print(f"\nVERDICT:")
            print(f"  Baseline (none):  F1 = {f_none:.4f}")
            print(f"  SMOTE only:       F1 = {f_smot:.4f}  (Δ = {f_smot-f_none:+.4f})")
            print(f"  WGAN-GP only:     F1 = {f_gan:.4f}  (Δ = {f_gan-f_none:+.4f})")
            if f_smot > f_gan + 0.01:
                print(f"  → SMOTE OUTPERFORMS WGAN-GP by {f_smot-f_gan:.4f}: "
                      f"the GAN is the bottleneck on this dataset.")
                print(f"    Mode collapse hypothesis CONFIRMED.")
            elif f_gan > f_smot + 0.01:
                print(f"  → WGAN-GP wins by {f_gan-f_smot:.4f}: GAN is fine; "
                      f"mode collapse hypothesis REJECTED.")
            else:
                print(f"  → SMOTE and WGAN-GP are within ±1pp: no clear winner.")
    return results


if __name__ == "__main__":
    main()
