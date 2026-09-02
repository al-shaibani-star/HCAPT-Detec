"""
Focal Loss - down-weights easy majority-class samples so the model
focuses on hard minority samples (R2L, U2R).

Reference: Lin et al. 2017 "Focal Loss for Dense Object Detection"
          arxiv.org/abs/1708.02002

For class c with probability p_c and true class t:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Where:
    γ = focusing parameter (2.0 is standard)
    α = class-balancing weight (inverse frequency)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha        # per-class weights, shape (num_classes,)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        # Standard cross-entropy
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                              reduction="none")
        p_t = torch.exp(-ce)                        # softmax prob of true class
        focal = ((1 - p_t) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def class_balance_alpha(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = np.bincount(y, minlength=num_classes)
    # Avoid div-by-zero
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.sum() * num_classes   # normalize to sum=NUM
    return torch.from_numpy(weights.astype(np.float32))
