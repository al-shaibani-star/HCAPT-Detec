"""
Test-Time Adaptation (TTA) - update BatchNorm statistics on test
batches without labels. Helps with distribution shift.

Reference: Schneider et al. 2020 "Improving robustness against
common corruptions by covariate shift adaptation"
https://arxiv.org/abs/2006.16971

Approach:
  Set BN layers to training mode (so running_mean / running_var update)
  but disable gradient tracking.
  Forward-pass unlabeled test batches → BN stats adapt to new domain.
  Set back to eval mode and predict.
"""
import torch
import torch.nn as nn
from typing import Optional
import numpy as np


def collect_bn_layers(module: nn.Module):
    """Yield all BatchNorm modules inside `module`."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.BatchNorm3d, nn.SyncBatchNorm)):
            yield m


def adapt_bn_statistics(model: nn.Module, X_test: np.ndarray,
                         batch_size: int = 256,
                         momentum_override: Optional[float] = None,
                         device: str = "cuda") -> dict:
    """Forward unlabeled test samples to update BN running stats.

    momentum_override : if given, override BN momentum (higher = more
                        aggressive adaptation). Default keeps existing.
    """
    bn_layers = list(collect_bn_layers(model))
    if not bn_layers:
        return {"n_bn_layers": 0, "adapted": False}

    # Save original state
    original_momentum = []
    for bn in bn_layers:
        original_momentum.append(bn.momentum)
        if momentum_override is not None:
            bn.momentum = momentum_override
        bn.train()     # enable running-stat updates

    # Forward passes WITHOUT gradients, WITHOUT classifier updates
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            x_batch = torch.from_numpy(X_test[i:i+batch_size]).float().to(device)
            _ = model(x_batch)
            n_batches += 1

    # Restore
    for bn, m in zip(bn_layers, original_momentum):
        bn.momentum = m
        bn.eval()

    return {"n_bn_layers": len(bn_layers),
            "n_batches_adapted": n_batches,
            "adapted": True}


def tta_predict(model: nn.Module, X_test: np.ndarray, y_test=None,
                batch_size: int = 256, device: str = "cuda") -> np.ndarray:
    """One-shot prediction with TTA:
      1. Adapt BN statistics on test data (no labels)
      2. Set eval mode and predict
    """
    # Step 1: BN adaptation
    info = adapt_bn_statistics(model, X_test, batch_size=batch_size,
                                 device=device)
    print(f"[tta] adapted {info['n_bn_layers']} BN layers using "
          f"{info['n_batches_adapted']} test batches")

    # Step 2: final predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            x_batch = torch.from_numpy(X_test[i:i+batch_size]).float().to(device)
            logits = model(x_batch)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)
