"""
ToN-IoT IoT Telemetry loader for Strategy A.

The ToN-IoT IoT split has 7 device types, each with its own feature
schema (Fridge: temp+condition, Modbus: 4 register reads, Weather:
temp+pressure+humidity, etc.).

This loader pools all 7 devices into a single unified dataset by:
  1. Keeping only the columns that exist in every device file.
  2. One-hot encoding the device type as a categorical feature.
  3. Per-device label/type columns are unified.

Result: ~261K rows, ~10–12 features, 9 attack classes (binary or multi-class).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .base import DatasetLoader, LoadedDataset


# Per-device file -> human-readable device name
DEVICE_FILES = {
    "fridge.csv":       "fridge",
    "garage_door.csv":  "garage_door",
    "gps_tracker.csv":  "gps_tracker",
    "modbus.csv":       "modbus",
    "motion_light.csv": "motion_light",
    "thermostat.csv":   "thermostat",
    "weather.csv":      "weather",
}

# Numeric feature columns we'll harmonise: load the union per device, then
# fill with zeros where a device doesn't have a given feature. This gives
# a single common feature space for HCAPT-Defense to operate on.
FEATURE_COLS_BY_DEVICE = {
    "fridge":       ["fridge_temperature"],
    "garage_door":  ["sphone_signal"],
    "gps_tracker":  ["latitude", "longitude"],
    "modbus":       ["FC1_Read_Input_Register", "FC2_Read_Discrete_Value",
                      "FC3_Read_Holding_Register", "FC4_Read_Coil"],
    "motion_light": [],   # only categorical (motion_status, light_status)
    "thermostat":   ["current_temperature"],
    "weather":      ["temperature", "pressure", "humidity"],
}

# Categorical features (one-hot encoded later)
CAT_COLS_BY_DEVICE = {
    "fridge":       ["temp_condition"],
    "garage_door":  ["door_state"],
    "gps_tracker":  [],
    "modbus":       [],
    "motion_light": ["motion_status", "light_status"],
    "thermostat":   ["thermostat_status"],
    "weather":      [],
}

# Standard 9-class attack mapping (per ToN-IoT documentation)
ATTACK_TYPES_TON_IOT = [
    "normal", "backdoor", "ddos", "dos", "injection", "mitm",
    "password", "ransomware", "scanning", "xss",
]
MULTICLASS_LABEL_MAP = {name: i for i, name in enumerate(ATTACK_TYPES_TON_IOT)}


class TonIotTelemetryLoader(DatasetLoader):
    name = "ton-iot-telemetry"

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.dir = self.data_root / "ton-iot" / "iot-telemetry"

    def source_paths(self) -> List[Path]:
        return [self.dir / f for f in DEVICE_FILES if (self.dir / f).exists()]

    def _load_one_device(self, csv_path: Path, device_name: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        # Strip whitespace inside string cells (some files have padded values)
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
        df["__device"] = device_name
        return df

    def load(self, classification: str = "binary") -> LoadedDataset:
        if classification not in ("binary", "multiclass"):
            raise ValueError(classification)

        # Load all 7 devices
        frames = []
        for fname, dname in DEVICE_FILES.items():
            p = self.dir / fname
            if p.exists():
                frames.append(self._load_one_device(p, dname))
            else:
                print(f"[ton-iot] missing device file: {fname} (skip)")
        if not frames:
            raise FileNotFoundError(f"No device CSVs found in {self.dir}")

        # Build the unified feature matrix
        # 1) Harmonise numeric feature columns: union across devices
        all_num_cols = sorted(set(
            col for cols in FEATURE_COLS_BY_DEVICE.values() for col in cols
        ))
        # 2) Harmonise categorical columns
        all_cat_cols = sorted(set(
            col for cols in CAT_COLS_BY_DEVICE.values() for col in cols
        ))

        rows = []
        for df in frames:
            # Add missing columns with NaN (filled with 0 after scaling)
            for col in all_num_cols:
                if col not in df.columns:
                    df[col] = 0.0
            for col in all_cat_cols:
                if col not in df.columns:
                    df[col] = "missing"
            # Unify label columns
            if "label" not in df.columns:
                raise KeyError(f"label col missing in device {df['__device'].iloc[0]}")
            if "type" not in df.columns:
                raise KeyError(f"type col missing in device {df['__device'].iloc[0]}")
            keep = ["__device", "label", "type"] + all_num_cols + all_cat_cols
            rows.append(df[keep])
        big = pd.concat(rows, ignore_index=True)

        # Clean labels
        big["type"] = big["type"].astype(str).str.lower().str.strip()
        big["label"] = pd.to_numeric(big["label"], errors="coerce").fillna(0).astype(int)

        # Encode categoricals + device
        cat_to_encode = list(all_cat_cols) + ["__device"]
        for col in cat_to_encode:
            big[col] = big[col].astype(str)
            le = LabelEncoder()
            big[col] = le.fit_transform(big[col])

        # Numeric columns may still contain string values (e.g. 'true'/'false');
        # coerce to float and fall back to LabelEncoder for stubborn strings.
        for col in all_num_cols:
            coerced = pd.to_numeric(big[col], errors="coerce")
            if coerced.isna().all():
                # Column was entirely string -> label encode
                big[col] = LabelEncoder().fit_transform(big[col].astype(str))
            elif coerced.isna().any():
                # Mixed string/number -> stringify then label encode
                big[col] = LabelEncoder().fit_transform(big[col].astype(str))
            else:
                big[col] = coerced

        # Build target
        if classification == "binary":
            y = big["label"].astype(int).values
            class_names = ["Normal", "Attack"]
            is_binary = True
        else:
            y = big["type"].map(MULTICLASS_LABEL_MAP).fillna(-1).astype(int).values
            keep_mask = y >= 0
            big = big.loc[keep_mask].reset_index(drop=True)
            y = y[keep_mask]
            class_names = ATTACK_TYPES_TON_IOT
            is_binary = False

        # Build features (numeric + encoded categorical + device id)
        feature_names = list(all_num_cols) + list(all_cat_cols) + ["__device"]
        X = big[feature_names].astype(np.float32).values

        # Per-column NaN -> 0 (for numeric pads)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Stratified 80/20 train/test split
        from sklearn.model_selection import train_test_split
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
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            class_names=class_names,
            is_binary=is_binary,
            metadata={
                "n_devices": len(frames),
                "split": "stratified 80/20 (random)",
                "scaler": "StandardScaler",
                "classification": classification,
                "n_attack_classes_present": int(len(np.unique(y))),
            },
        )


def get_loader(data_root: Path) -> TonIotTelemetryLoader:
    return TonIotTelemetryLoader(data_root)
