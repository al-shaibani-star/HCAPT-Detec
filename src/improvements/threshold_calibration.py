"""
Per-class threshold calibration.

Default softmax-argmax decision uses threshold 0.5 (implicitly), which
is sub-optimal for imbalanced classes. For R2L/U2R, lowering the
class-specific threshold can recover recall at a modest FPR cost.

Given probabilities P (N×C), the argmax rule is:
    pred = argmax_c P[:, c]

Per-class thresholds introduce a bias:
    pred = argmax_c (P[:, c] / threshold_c)
    or equivalently:
    pred = argmax_c (log P[:, c] - log threshold_c)
"""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def find_class_thresholds(
    y_true: np.ndarray, probs: np.ndarray,
    target_metric: str = "f1",
    num_classes: int = 5,
) -> Dict[int, float]:
    """Find per-class threshold multipliers that maximize `target_metric`.

    Returns {class_id: threshold} where threshold is interpreted as
    a DIVISOR (smaller threshold = more aggressive prediction of that
    class).
    """
    def _score(preds):
        if target_metric == "recall":
            return recall_score(y_true, preds, average="macro",
                                 zero_division=0)
        elif target_metric == "f1":
            return f1_score(y_true, preds, average="macro",
                             zero_division=0)
        raise ValueError(target_metric)

    # Candidate thresholds (smaller = more aggressive)
    candidates = np.linspace(0.05, 1.0, 20)

    # Greedy per-class search
    best_t = {c: 1.0 for c in range(num_classes)}
    best_score = _score(probs.argmax(1))

    for _ in range(3):     # couple of passes
        for c in range(num_classes):
            for t in candidates:
                trial = best_t.copy()
                trial[c] = t
                t_vec = np.array([trial[i] for i in range(num_classes)])
                preds = (probs / t_vec).argmax(1)
                s = _score(preds)
                if s > best_score:
                    best_score = s
                    best_t[c] = t

    return best_t


def apply_thresholds(probs: np.ndarray, thresholds: Dict[int, float]
                     ) -> np.ndarray:
    """Apply per-class thresholds to probability array."""
    num_classes = probs.shape[1]
    t_vec = np.array([thresholds.get(c, 1.0)
                       for c in range(num_classes)])
    return (probs / t_vec).argmax(1)


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    N = 1000
    y = rng.choice(5, N, p=[0.5, 0.3, 0.1, 0.08, 0.02])
    probs = rng.dirichlet([1]*5, size=N)
    # Bias probs roughly toward correct class
    for i in range(N):
        probs[i, y[i]] += 0.4
        probs[i] = probs[i] / probs[i].sum()

    before = f1_score(y, probs.argmax(1), average="macro", zero_division=0)
    thresholds = find_class_thresholds(y, probs, target_metric="f1")
    after = f1_score(y, apply_thresholds(probs, thresholds),
                      average="macro", zero_division=0)
    print(f"F1 (argmax):     {before:.4f}")
    print(f"F1 (calibrated): {after:.4f}")
    print(f"Thresholds: {thresholds}")
