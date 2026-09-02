"""
SMOTE-based minority-class oversampling.

Unlike GANs, SMOTE interpolates between real neighbors - so for
extreme-minority classes (< 1000 samples) it produces faithful
synthetic samples without mode collapse.

Strategy:
  - Apply SMOTE only to R2L (995) and U2R (52)
  - Keep majority classes untouched
  - Output: balanced dataset (up to k_samples per class)
"""
from typing import Dict, Tuple
import numpy as np


def smote_minority(
    X: np.ndarray, y: np.ndarray,
    targets: Dict[int, int] = None,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes using SMOTE.

    targets : {class_id: target_sample_count}
              default: R2L (3) -> 5000, U2R (4) -> 1000
    """
    if targets is None:
        targets = {3: 5000, 4: 1000}

    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError:
        raise ImportError(
            "pip install imbalanced-learn to use SMOTE"
        )

    # Use SMOTE but fall back to RandomOverSampler if a class
    # has fewer than k_neighbors+1 samples (SMOTE requirement)
    from collections import Counter
    counts = Counter(y.tolist())
    sampling_strategy = {}
    for cls, target_n in targets.items():
        current = counts.get(cls, 0)
        if current < target_n:
            # Only upsample; never downsample
            sampling_strategy[cls] = target_n

    if not sampling_strategy:
        print("[smote] no minority class needs oversampling")
        return X, y

    # Choose strategy: SMOTE if class has enough samples, else random
    min_count = min(counts[c] for c in sampling_strategy.keys())
    if min_count >= k_neighbors + 1:
        print(f"[smote] applying SMOTE (k={k_neighbors}): {sampling_strategy}")
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(k_neighbors, min_count - 1),
            random_state=random_state,
        )
    else:
        # For classes with < k_neighbors+1 samples, use simple random
        print(f"[smote] class too small for SMOTE, using RandomOverSampler")
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    X_bal, y_bal = sampler.fit_resample(X, y)
    new_counts = Counter(y_bal.tolist())
    print(f"[smote] class counts: {dict(sorted(new_counts.items()))}")
    return X_bal, y_bal


def hybrid_smote_gan(
    X: np.ndarray, y: np.ndarray,
    gan_synthetic: Tuple[np.ndarray, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid: GAN generates for majority-ish (DoS), SMOTE for
    extreme-minority (R2L/U2R).

    Best of both worlds:
      - GAN: more diverse for classes with enough real samples
      - SMOTE: reliable for <1000-sample classes
    """
    # Apply SMOTE to R2L, U2R only
    X_s, y_s = smote_minority(X, y, targets={3: 3000, 4: 500})

    # Append GAN samples for DoS/Probe if provided
    if gan_synthetic is not None:
        Xg, yg = gan_synthetic
        # Keep only DoS/Probe from GAN
        mask = np.isin(yg, [1, 2])
        Xg, yg = Xg[mask], yg[mask]
        X_s = np.concatenate([X_s, Xg], axis=0)
        y_s = np.concatenate([y_s, yg], axis=0)

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_s))
    return X_s[perm], y_s[perm]
