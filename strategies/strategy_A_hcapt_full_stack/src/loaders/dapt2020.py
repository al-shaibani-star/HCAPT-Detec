"""
DAPT 2020 loader — APT lifecycle benchmark with 5 attack stages.

Loads all ten CSV files (CICFlowMeter output, 85 features each),
normalises labels, drops identifiers and timestamp/IP columns that
would leak ground truth, and returns a standard LoadedDataset.

DAPT 2020 stages (multi-class):
  Benign / BENIGN  → Normal
  Reconnaissance   → APT Stage 1
  Establish Foothold → APT Stage 2
  Lateral Movement → APT Stage 3
  Data Exfiltration → APT Stage 4

Binary mode collapses all stages > Normal into a single Attack class.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .base import DatasetLoader, LoadedDataset


# Stage normalisation (case-insensitive)
STAGE_TO_INT = {
    "benign":              0,  # also: "BENIGN", "Normal"
    "reconnaissance":      1,
    "establish foothold":  2,
    "lateral movement":    3,
    "data exfiltration":   4,
}
STAGE_NAMES = ["Normal", "Reconnaissance", "Establish Foothold",
               "Lateral Movement", "Data Exfiltration"]

# Columns to drop (identifiers / leakage / non-numeric metadata)
DROP_COLS = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Timestamp", "Activity",  # Activity is finer-grained than Stage; keep Stage as label
]
LABEL_COL = "Stage"


class Dapt2020Loader(DatasetLoader):
    name = "dapt-2020"

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.dir = self.data_root / "dapt-2020"
        self.csv_dir = self.dir / "csv"

    def source_paths(self) -> List[Path]:
        if not self.csv_dir.exists():
            return []
        return sorted(self.csv_dir.glob("*.csv"))

    def load(self, classification: str = "binary") -> LoadedDataset:
        if classification not in ("binary", "multiclass"):
            raise ValueError(classification)

        csv_files = [p for p in self.source_paths()
                      if p.name != "README.md" and ".pcap_Flow.csv" in p.name]
        if not csv_files:
            raise FileNotFoundError(
                f"No CICFlowMeter CSVs in {self.csv_dir}\n"
                f"Expected files matching *.pcap_Flow.csv"
            )
        print(f"[dapt-2020] loading {len(csv_files)} files…")

        frames = []
        for p in csv_files:
            df = pd.read_csv(p, low_memory=False)
            # Some files may not have Stage/Activity if extraction was raw
            if LABEL_COL not in df.columns:
                print(f"   skipping {p.name} (no '{LABEL_COL}' column)")
                continue
            frames.append(df)
        if not frames:
            raise RuntimeError("No file had a Stage column")
        big = pd.concat(frames, axis=0, ignore_index=True)
        print(f"[dapt-2020] combined: {big.shape}")

        # Normalise Stage labels (case-insensitive)
        big[LABEL_COL] = big[LABEL_COL].astype(str).str.strip().str.lower()
        big["_stage_int"] = big[LABEL_COL].map(STAGE_TO_INT)
        # Drop rows with unknown stage (rare)
        missing = big["_stage_int"].isna().sum()
        if missing:
            print(f"   dropping {missing} rows with unknown Stage label")
            big = big.dropna(subset=["_stage_int"]).reset_index(drop=True)
        big["_stage_int"] = big["_stage_int"].astype(int)

        # Build targets
        if classification == "binary":
            y = (big["_stage_int"] != 0).astype(int).values
            class_names = ["Normal", "Attack"]
            is_binary = True
        else:
            y = big["_stage_int"].values
            class_names = STAGE_NAMES
            is_binary = False

        # Drop label + identifier columns
        cols_to_drop = DROP_COLS + [LABEL_COL, "_stage_int"]
        feat_df = big.drop(columns=[c for c in cols_to_drop if c in big.columns])

        # Drop any remaining string columns (rare in CICFlowMeter output)
        for c in feat_df.columns:
            if feat_df[c].dtype == "object":
                try:
                    feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
                except Exception:
                    feat_df = feat_df.drop(columns=[c])

        # NaN / inf cleanup
        X = feat_df.astype(np.float32).values
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Drop near-constant columns
        std_per_col = X.std(axis=0)
        keep_idx = std_per_col > 1e-7
        feature_names = [c for c, k in zip(feat_df.columns.tolist(), keep_idx) if k]
        X = X[:, keep_idx]
        print(f"[dapt-2020] features kept: {X.shape[1]} (dropped {int((~keep_idx).sum())} constant)")

        # Stratified 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        # Scale (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        return LoadedDataset(
            name=self.name,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=feature_names,
            class_names=class_names,
            is_binary=is_binary,
            metadata={
                "n_files": len(csv_files),
                "split": "stratified 80/20 (random)",
                "classification": classification,
                "scaler": "StandardScaler",
            },
        )


def get_loader(data_root: Path) -> Dapt2020Loader:
    return Dapt2020Loader(data_root)
