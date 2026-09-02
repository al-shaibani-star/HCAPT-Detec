"""
NSL-KDD loader for Strategy A.

Wraps the project-level data_loader (../../../src/data_loader.py) but
returns the standard LoadedDataset container expected by Strategy A.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Make project src/ importable
PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .base import DatasetLoader, LoadedDataset


# Column names + categorical features (lifted from project config)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Standard 5-class mapping
ATTACK_CATEGORIES = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "xterm": "U2R", "ps": "U2R", "sqlattack": "U2R",
}
MULTICLASS_LABELS = {"Normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4}


class NslKddLoader(DatasetLoader):
    name = "nsl-kdd"

    def __init__(self, data_root: Path):
        # data_root points to <project>/data
        self.data_root = Path(data_root)
        self.train_path = self.data_root / "nsl-kdd" / "train.txt"
        self.test_path  = self.data_root / "nsl-kdd" / "test.txt"

    def source_paths(self) -> List[Path]:
        return [self.train_path, self.test_path]

    def load(self, classification: str = "binary") -> LoadedDataset:
        if classification not in ("binary", "multiclass"):
            raise ValueError(f"classification must be 'binary' or 'multiclass', got {classification}")

        if not self.train_path.exists():
            raise FileNotFoundError(f"NSL-KDD train file not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"NSL-KDD test file not found: {self.test_path}")

        df_train = pd.read_csv(self.train_path, header=None, names=NSL_KDD_COLUMNS)
        df_test  = pd.read_csv(self.test_path,  header=None, names=NSL_KDD_COLUMNS)

        # Drop difficulty (NSL-KDD specific)
        df_train = df_train.drop(columns=["difficulty"])
        df_test  = df_test.drop(columns=["difficulty"])

        # Map labels to category, then to int
        df_train["category"] = df_train["label"].map(lambda s: ATTACK_CATEGORIES.get(s, "Unknown"))
        df_test["category"]  = df_test["label"].map(lambda s: ATTACK_CATEGORIES.get(s, "Unknown"))

        if classification == "binary":
            y_train = (df_train["category"] != "Normal").astype(int).values
            y_test  = (df_test["category"]  != "Normal").astype(int).values
            class_names = ["Normal", "Attack"]
            is_binary = True
        else:
            # Drop unknown rows (rare; ~38 in test set)
            mask_tr = df_train["category"] != "Unknown"
            mask_te = df_test["category"]  != "Unknown"
            df_train = df_train[mask_tr].reset_index(drop=True)
            df_test  = df_test[mask_te].reset_index(drop=True)
            y_train = df_train["category"].map(MULTICLASS_LABELS).values.astype(int)
            y_test  = df_test["category"].map(MULTICLASS_LABELS).values.astype(int)
            class_names = list(MULTICLASS_LABELS.keys())
            is_binary = False

        # Drop label/category columns from features
        X_train_df = df_train.drop(columns=["label", "category"])
        X_test_df  = df_test.drop(columns=["label", "category"])

        # Encode categoricals — fit on union of (train + test) values to avoid unseen categories
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            combined = pd.concat([X_train_df[col], X_test_df[col]], axis=0).astype(str)
            le.fit(combined)
            X_train_df[col] = le.transform(X_train_df[col].astype(str))
            X_test_df[col]  = le.transform(X_test_df[col].astype(str))
            encoders[col] = list(le.classes_)

        feature_names = list(X_train_df.columns)
        X_train = X_train_df.to_numpy(dtype=np.float32)
        X_test  = X_test_df.to_numpy(dtype=np.float32)

        # Scale numerical features
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
                "source_train": str(self.train_path),
                "source_test":  str(self.test_path),
                "n_categorical_encoded": len(CATEGORICAL_COLS),
                "scaler": "StandardScaler",
                "classification": classification,
            },
        )


# ---------------------------------------------------------------------------
def get_loader(data_root: Path) -> NslKddLoader:
    return NslKddLoader(data_root)
