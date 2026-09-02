"""
Strategy A — Abstract Dataset Loader

Defines the contract every dataset loader must satisfy so the pipeline
can run unchanged across NSL-KDD, ToN-IoT, Edge-IIoTset.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np


@dataclass
class LoadedDataset:
    """Standard container returned by every loader."""
    name: str                          # canonical dataset id (kebab-case)
    X_train: np.ndarray                # shape (n_train, n_features), float32
    X_test:  np.ndarray                # shape (n_test,  n_features), float32
    y_train: np.ndarray                # shape (n_train,), int (binary or multi-class)
    y_test:  np.ndarray                # shape (n_test,),  int
    feature_names: List[str]           # length n_features
    class_names:   List[str]           # length n_classes
    is_binary: bool                    # True if y in {0, 1}
    metadata: Dict                     # source paths, encoders, scaler info, etc.

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> str:
        return (
            f"{self.name}: train={self.X_train.shape}, test={self.X_test.shape}, "
            f"features={self.n_features}, classes={self.n_classes} "
            f"({'binary' if self.is_binary else 'multi-class'})"
        )


class DatasetLoader(ABC):
    """Every loader implements load() returning a LoadedDataset."""

    name: str = ""                     # set by subclass

    @abstractmethod
    def load(self, classification: str = "binary") -> LoadedDataset:
        """Return a LoadedDataset.

        classification: 'binary' (Normal vs Attack) or 'multiclass'
        """
        ...

    @abstractmethod
    def source_paths(self) -> List[Path]:
        """Return all source files (used for cache fingerprinting)."""
        ...
