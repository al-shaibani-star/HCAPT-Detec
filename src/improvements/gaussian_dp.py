"""
Gaussian differential privacy mechanism - gives better utility than
Laplace in high-dimensional settings (>~20 dims).

Reference: Dwork et al. "The Algorithmic Foundations of Differential
Privacy", Theorem 3.22 (Gaussian mechanism)

For (ε, δ)-DP with L2-sensitivity Δ_2:
    σ = Δ_2 * sqrt(2 * ln(1.25 / δ)) / ε
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GaussianDPConfig:
    epsilon: float = 5.0
    delta: float = 1e-5
    clip_norm: float = 1.0     # L2 norm bound per sample
    seed: Optional[int] = 42


class GaussianDPSanitizer:
    """(ε, δ)-DP Gaussian mechanism with L2 clipping.

    For high-dim vectors, Gaussian gives much better utility than
    Laplace at the same ε.
    """

    def __init__(self, config: GaussianDPConfig = GaussianDPConfig()):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        self.sigma = (
            config.clip_norm
            * np.sqrt(2.0 * np.log(1.25 / config.delta))
            / config.epsilon
        )

    def l2_clip(self, x: np.ndarray) -> np.ndarray:
        """Clip each row to L2 norm ≤ clip_norm."""
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm > self.config.clip_norm:
                return x * (self.config.clip_norm / norm)
            return x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.config.clip_norm / (norms + 1e-9))
        return x * scale

    def sanitize(self, x: np.ndarray) -> np.ndarray:
        x_c = self.l2_clip(x)
        noise = self.rng.normal(0.0, self.sigma, x_c.shape)
        return x_c + noise

    def sanitize_batch(self, X: np.ndarray) -> np.ndarray:
        return self.sanitize(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 41)).astype(np.float32)

    print("Gaussian DP utility sweep (δ=1e-5, clip=1.0):")
    for eps in (0.5, 1.0, 3.0, 5.0, 10.0):
        san = GaussianDPSanitizer(GaussianDPConfig(epsilon=eps))
        X_s = san.sanitize_batch(X)
        diff = X_s - san.l2_clip(X)
        Xn_ = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Nn_ = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-9)
        cos = float((Xn_ * Nn_).sum(axis=1).mean())
        print(f"  epsilon={eps:<5}  sigma={san.sigma:.3f}  cosine={cos:.3f}")
