"""
HCAPT-Defense - Full Pipeline Orchestrator
Runs the complete training pipeline end-to-end, using cache so each
stage is only done once. Safe to re-run after a crash - resumes where
it left off.

Pipeline stages:
  1. Preprocess NSL-KDD            (cached)
  2. Train GAN                     (checkpoints every 10 epochs)
  3. Generate synthetic samples    (balance minority classes)
  4. Train Ensemble on augmented  (RF + XGBoost + Deep NN)
  5. Run Self-Learning simulation
  6. Final evaluation vs baselines
"""
import json
import time
import numpy as np
import torch
from pathlib import Path

from config import DEVICE, MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES
from data_loader import get_preprocessed
from cache import (
    mark_phase_done, is_phase_done, load_checkpoint,
    CHECKPOINT_DIR, CACHE_DIR,
)
from gan_augment import Generator, train_cgan, balance_dataset
from ensemble_selflearn import (
    EnsembleDetector, SelfLearningLoop, evaluate_ensemble, print_metrics,
)
from sklearn.model_selection import train_test_split

np.random.seed(SEED)
torch.manual_seed(SEED)


BALANCED_DATA_FILE = CACHE_DIR / "gan_balanced_data.npz"
GAN_AUGMENTED_ENSEMBLE_FILE = RESULTS_DIR / "gan_augmented_ensemble.json"


# ============================================================
#  Stage 1: Preprocess
# ============================================================
def stage_preprocess():
    print("\n" + "=" * 60)
    print("  STAGE 1 - Preprocess NSL-KDD")
    print("=" * 60)
    X_tr, X_te, y_tr, y_te = get_preprocessed(use_cache=True)
    mark_phase_done("preprocess", {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "features": int(X_tr.shape[1]),
    })
    return X_tr, X_te, y_tr, y_te


# ============================================================
#  Stage 2: Train GAN
# ============================================================
def stage_train_gan(X_tr, y_tr, epochs: int = 100):
    print("\n" + "=" * 60)
    print("  STAGE 2 - Train Conditional GAN")
    print("=" * 60)

    # train_cgan resumes from latest checkpoint automatically
    G, D, hist = train_cgan(X_tr, y_tr, epochs=epochs,
                            checkpoint_every=10, resume=True)

    # Save final
    torch.save(G.state_dict(), MODELS_DIR / "gan_generator.pt")
    torch.save(D.state_dict(), MODELS_DIR / "gan_discriminator.pt")

    mark_phase_done("gan_training", {
        "epochs": epochs,
        "final_d_loss": float(hist["d_loss"][-1]),
        "final_g_loss": float(hist["g_loss"][-1]),
    })
    return G


# ============================================================
#  Stage 3: Generate balanced dataset
# ============================================================
def stage_balance_data(G, X_tr, y_tr, X_te, y_te):
    print("\n" + "=" * 60)
    print("  STAGE 3 - Generate Balanced Dataset via GAN")
    print("=" * 60)

    if BALANCED_DATA_FILE.exists() and is_phase_done("balance_data"):
        print(f"[cache] HIT: loading balanced data from "
              f"{BALANCED_DATA_FILE.name}")
        d = np.load(BALANCED_DATA_FILE)
        return d["X_bal"], d["y_bal"]

    X_bal, y_bal = balance_dataset(X_tr, y_tr, G)
    np.savez(BALANCED_DATA_FILE, X_bal=X_bal, y_bal=y_bal)
    counts = np.bincount(y_bal, minlength=NUM_CLASSES).tolist()
    print(f"[balance] Final counts: {counts}")
    mark_phase_done("balance_data", {
        "n_total": int(len(y_bal)),
        "counts": counts,
    })
    return X_bal, y_bal


# ============================================================
#  Stage 4: Train Ensemble on GAN-augmented data
# ============================================================
def stage_train_ensemble(X_bal, y_bal, X_te, y_te, feature_dim):
    print("\n" + "=" * 60)
    print("  STAGE 4 - Train Ensemble on GAN-Augmented Data")
    print("=" * 60)

    # Split the augmented training set into train/val
    X_tr_aug, X_val, y_tr_aug, y_val = train_test_split(
        X_bal, y_bal, test_size=0.1, random_state=SEED, stratify=y_bal,
    )

    ens = EnsembleDetector(feature_dim=feature_dim)
    train_info = ens.fit(X_tr_aug, y_tr_aug, X_val, y_val, nn_epochs=30)

    # Save under distinct name so we don't overwrite the non-augmented version
    import joblib
    joblib.dump(ens.rf, MODELS_DIR / "gan_ens_rf.joblib")
    ens.xgb.save_model(str(MODELS_DIR / "gan_ens_xgb.json"))
    torch.save(ens.nn.state_dict(), MODELS_DIR / "gan_ens_nn.pt")
    np.save(MODELS_DIR / "gan_ens_weights.npy", ens.weights)

    mark_phase_done("gan_ensemble_training", train_info)
    return ens, train_info


# ============================================================
#  Stage 5: Self-Learning simulation
# ============================================================
def stage_self_learning(ens, X_te, y_te, baseline_acc: float):
    print("\n" + "=" * 60)
    print("  STAGE 5 - Self-Learning Simulation")
    print("=" * 60)

    loop = SelfLearningLoop(ensemble=ens, baseline_acc=baseline_acc)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_te))
    X_stream, y_stream = X_te[perm], y_te[perm]

    chunk_size = 500
    for i in range(0, len(X_stream), chunk_size):
        Xb = X_stream[i:i+chunk_size]
        yb = y_stream[i:i+chunk_size]
        loop.process_batch(Xb, y_true=yb)

    print(f"\n[SelfLearn] Total retrains: {loop.retrain_count}")
    mark_phase_done("self_learning", {
        "retrains": loop.retrain_count,
        "n_events": len(loop.events),
    })
    return loop


# ============================================================
#  Stage 6: Evaluate on NSL-KDD test
# ============================================================
def stage_evaluate(ens, X_te, y_te, tag: str = "GAN-Augmented"):
    print("\n" + "=" * 60)
    print(f"  STAGE 6 - Evaluate: {tag}")
    print("=" * 60)
    m = evaluate_ensemble(ens, X_te, y_te,
                          name=f"HCAPT-Ensemble ({tag})")
    print_metrics(m)
    return m


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("  HCAPT-Defense - Full Pipeline")
    print("=" * 60)

    t_start = time.time()
    feature_dim = 41

    # 1. Preprocess
    X_tr, X_te, y_tr, y_te = stage_preprocess()
    feature_dim = X_tr.shape[1]

    # 2. GAN
    G = stage_train_gan(X_tr, y_tr, epochs=100)

    # 3. Balance
    X_bal, y_bal = stage_balance_data(G, X_tr, y_tr, X_te, y_te)

    # 4. Train augmented ensemble
    ens, train_info = stage_train_ensemble(X_bal, y_bal, X_te, y_te,
                                            feature_dim)

    # 5. Evaluate BEFORE self-learning (on real test)
    m_before = stage_evaluate(ens, X_te, y_te, tag="GAN-Aug, pre-SelfLearn")

    # 6. Self-Learning
    loop = stage_self_learning(ens, X_te, y_te,
                                baseline_acc=train_info["nn_acc"])

    # 7. Evaluate AFTER self-learning
    m_after = stage_evaluate(ens, X_te, y_te, tag="GAN-Aug, post-SelfLearn")

    # Save combined results
    combined = {
        "pipeline_time_s": time.time() - t_start,
        "ensemble_training": train_info,
        "gan_aug_before_selflearn": m_before,
        "gan_aug_after_selflearn": m_after,
        "retrains": loop.retrain_count,
    }
    with open(GAN_AUGMENTED_ENSEMBLE_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n[OK] Pipeline complete in {combined['pipeline_time_s']:.1f}s")
    print(f"[OK] Results: {GAN_AUGMENTED_ENSEMBLE_FILE}")


if __name__ == "__main__":
    main()
