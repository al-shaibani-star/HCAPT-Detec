"""
HCAPT-Defense — NSL-KDD Data Loader
Layer 1: Data Collection & Preprocessing

Handles:
  - Loading KDDTrain+ and KDDTest+ files
  - Attack-category mapping (5 classes)
  - Categorical encoding + numerical scaling
  - Train/Val split
  - PyTorch DataLoaders
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from config import (
    DATA_DIR, MODELS_DIR, NSL_KDD_COLUMNS, CATEGORICAL_COLS,
    ATTACK_CATEGORIES, LABEL_MAP, BATCH_SIZE, NUM_WORKERS, SEED,
)


# ============================================================
#  File discovery — supports multiple common naming conventions
# ============================================================
def _find_nsl_kdd_files(data_dir: Path) -> Tuple[Path, Path]:
    """Locate NSL-KDD train and test files. After reorg (May 2026) they live
    at <data_dir>/nsl-kdd/train.txt and test.txt; legacy names still work."""
    # Preferred: new scientific naming under nsl-kdd/
    nsl_dir = data_dir / "nsl-kdd"
    if (nsl_dir / "train.txt").exists() and (nsl_dir / "test.txt").exists():
        return nsl_dir / "train.txt", nsl_dir / "test.txt"

    # Fallback: legacy flat layout (KDDTrain+.txt, KDDTest+.txt) anywhere under data_dir
    candidates_train = {"KDDTrain+.txt", "KDDTrain+.csv", "KDDTrain+",
                        "kddtrain+.txt", "NSL_KDD_Train.csv", "train.txt"}
    candidates_test  = {"KDDTest+.txt",  "KDDTest+.csv",  "KDDTest+",
                        "kddtest+.txt",  "NSL_KDD_Test.csv",  "test.txt"}
    train_path, test_path = None, None
    for p in data_dir.rglob("*"):
        name = p.name
        if train_path is None and name in candidates_train:
            train_path = p
        if test_path is None and name in candidates_test:
            test_path = p
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Could not find NSL-KDD train/test files in {data_dir}\n"
            f"Expected layout (preferred): {data_dir}/nsl-kdd/train.txt and test.txt\n"
            f"Or legacy: KDDTrain+.txt, KDDTest+.txt anywhere under {data_dir}"
        )
    return train_path, test_path


# ============================================================
#  Load + preprocess
# ============================================================
def load_raw(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = _find_nsl_kdd_files(data_dir)
    print(f"[data] Train: {train_path}")
    print(f"[data] Test : {test_path}")

    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    # Drop "difficulty" column (NSL-KDD specific, not used for training)
    df_train = df_train.drop(columns=["difficulty"])
    df_test = df_test.drop(columns=["difficulty"])

    print(f"[data] Train shape: {df_train.shape} | Test shape: {df_test.shape}")
    return df_train, df_test


def map_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """Convert fine-grained attack names to 5 super-categories."""
    df = df.copy()
    df["attack_category"] = df["label"].map(
        lambda x: ATTACK_CATEGORIES.get(x, "Unknown")
    )
    # Filter out unknown (rare in NSL-KDD but safe)
    unknown_count = (df["attack_category"] == "Unknown").sum()
    if unknown_count > 0:
        print(f"[data] Warning: {unknown_count} unknown attack labels — dropping")
        df = df[df["attack_category"] != "Unknown"].reset_index(drop=True)
    df["y"] = df["attack_category"].map(LABEL_MAP)
    return df


def encode_and_scale(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-hot categoricals + standard-scale numericals. Fit on train only."""
    # Label-encode categoricals (one-hot would explode dims for 'service')
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fit on union to handle test-only categories gracefully
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        encoders[col] = le

    feature_cols = [c for c in df_train.columns
                    if c not in ("label", "attack_category", "y")]

    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_train = df_train["y"].values.astype(np.int64)
    y_test = df_test["y"].values.astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if save_artifacts:
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        joblib.dump(encoders, MODELS_DIR / "label_encoders.joblib")
        print(f"[data] Saved scaler + encoders to {MODELS_DIR}")

    print(f"[data] Features: {X_train.shape[1]} | Train: {X_train.shape[0]} "
          f"| Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ============================================================
#  PyTorch Dataset
# ============================================================
class NSLKDDDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_preprocessed(use_cache: bool = True):
    """Single source of truth for preprocessed arrays. Cached across runs."""
    if use_cache:
        from cache import cached_preprocessed_data
        train_path, test_path = _find_nsl_kdd_files(DATA_DIR)

        def _process():
            df_train, df_test = load_raw()
            df_train = map_attack_category(df_train)
            df_test = map_attack_category(df_test)
            return encode_and_scale(df_train, df_test)

        return cached_preprocessed_data(train_path, test_path, _process)

    df_train, df_test = load_raw()
    df_train = map_attack_category(df_train)
    df_test = map_attack_category(df_test)
    return encode_and_scale(df_train, df_test)


def build_dataloaders(
    val_size: float = 0.1,
    batch_size: int = BATCH_SIZE,
    use_cache: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Full pipeline: load -> preprocess -> split -> DataLoaders.

    Returns: (train_loader, val_loader, test_loader, X_test, y_test)
    """
    X_train_full, X_test, y_train_full, y_test = get_preprocessed(use_cache)

    # Train/Val split (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=SEED, stratify=y_train_full
    )

    train_loader = DataLoader(
        NSLKDDDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        NSLKDDDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        NSLKDDDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, X_test, y_test


if __name__ == "__main__":
    # Smoke test — loads data and prints summary
    tr_loader, val_loader, te_loader, X_te, y_te = build_dataloaders()
    for Xb, yb in tr_loader:
        print(f"\n[smoke] Batch X shape: {Xb.shape} | y shape: {yb.shape}")
        print(f"[smoke] Sample y: {yb[:10].tolist()}")
        break
    print(f"\n[smoke] Train batches: {len(tr_loader)} | "
          f"Val: {len(val_loader)} | Test: {len(te_loader)}")
