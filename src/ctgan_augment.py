"""
CTGAN-based conditional tabular data augmentation.

CTGAN (Xu et al. 2019) is specifically designed for tabular data and
handles:
  - Mixed continuous + discrete features via mode-specific normalization
  - Class imbalance via training-by-sampling
  - Mode collapse via conditional architecture

Reference: "Modeling Tabular Data using Conditional GAN"
           https://arxiv.org/abs/1907.00503

Used as an alternative to our custom CGAN / WGAN-GP when extreme
minority classes (<1000 samples) are the issue.
"""
import time
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR, RESULTS_DIR, SEED, NUM_CLASSES, LABEL_MAP,
    NSL_KDD_COLUMNS, CATEGORICAL_COLS,
)
from cache import CACHE_DIR

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _pre_scaled_to_df(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Convert scaled numpy array + labels to DataFrame with column names."""
    feat_names = [c for c in NSL_KDD_COLUMNS
                  if c not in ("label", "difficulty")]
    df = pd.DataFrame(X, columns=feat_names)
    df["label_class"] = y
    return df


def train_ctgan(X: np.ndarray, y: np.ndarray,
                epochs: int = 100, batch_size: int = 500,
                resume: bool = True):
    """Train CTGAN on (X, y). Returns fitted CTGAN instance.

    CTGAN sees the label as a 'discrete' column it can condition on.
    """
    from ctgan import CTGAN

    model_path = MODELS_DIR / "ctgan_model.pkl"
    if resume and model_path.exists():
        print(f"[CTGAN] cache hit: loading from {model_path}")
        import joblib
        return joblib.load(model_path)

    df = _pre_scaled_to_df(X, y)
    print(f"[CTGAN] training on {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[CTGAN] epochs={epochs}, batch={batch_size}")

    # Since the 3 categorical features (protocol_type, service, flag)
    # were label-encoded in our pipeline (integer values), treat them
    # as discrete. Plus label_class.
    discrete_cols = [c for c in CATEGORICAL_COLS if c in df.columns] + \
                    ["label_class"]
    # Quantize any integer-like numerical features to treat as discrete
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].round().astype(int)
    df["label_class"] = df["label_class"].astype(int)

    ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True)
    t0 = time.time()
    ctgan.fit(df, discrete_columns=discrete_cols)
    print(f"[CTGAN] training done in {time.time()-t0:.1f}s")

    import joblib
    joblib.dump(ctgan, model_path)
    print(f"[CTGAN] saved to {model_path}")
    return ctgan


def generate_synthetic(ctgan, num_per_class: Dict[int, int]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples per class. CTGAN's sample() supports
    conditional generation via condition_column / condition_value.
    """
    X_syn, y_syn = [], []
    for cls, n in num_per_class.items():
        if n <= 0:
            continue
        print(f"[CTGAN] generating {n} samples for class "
              f"{INV_LABEL_MAP[cls]} (cls {cls}) ...")
        df_s = ctgan.sample(
            n,
            condition_column="label_class",
            condition_value=cls,
        )
        # Drop the label column from features
        feat_cols = [c for c in df_s.columns if c != "label_class"]
        X_s = df_s[feat_cols].values.astype(np.float32)
        X_syn.append(X_s)
        y_syn.append(np.full(n, cls, dtype=np.int64))

    if not X_syn:
        return np.empty((0,)), np.empty((0,))
    return (np.concatenate(X_syn, axis=0),
            np.concatenate(y_syn, axis=0))


def balance_dataset(X: np.ndarray, y: np.ndarray,
                     ctgan, target: int = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Use CTGAN to rebalance classes to `target` per class."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = target or int(counts.max())
    need = {c: max(0, target - int(counts[c])) for c in range(NUM_CLASSES)}
    print(f"[CTGAN] current: {dict(enumerate(counts.tolist()))}")
    print(f"[CTGAN] target:  {target} per class")
    print(f"[CTGAN] need:    {need}")

    X_s, y_s = generate_synthetic(ctgan, need)
    if len(X_s) == 0:
        return X, y
    X_bal = np.concatenate([X, X_s], axis=0)
    y_bal = np.concatenate([y, y_s], axis=0)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def main():
    from data_loader import get_preprocessed
    print("=" * 70)
    print("  HCAPT-Defense - CTGAN training")
    print("=" * 70)

    X_tr, X_te, y_tr, y_te = get_preprocessed()

    ctgan = train_ctgan(X_tr, y_tr, epochs=30, batch_size=500)

    X_bal, y_bal = balance_dataset(X_tr, y_tr, ctgan)
    np.savez(CACHE_DIR / "ctgan_balanced_data.npz",
             X_bal=X_bal, y_bal=y_bal)
    counts = np.bincount(y_bal, minlength=NUM_CLASSES).tolist()
    print(f"\n[CTGAN] Balanced counts: {counts}")
    print(f"[OK] saved")


if __name__ == "__main__":
    main()
