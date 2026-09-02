"""
Edge-IIoTset loader for Strategy A.

Uses the pre-extracted ML feature set
(`data/edge-iiotset/ml-dl-selected/ML-EdgeIIoT-dataset.csv`,
~157k rows, 61 features + Attack_label + Attack_type).

Drops obvious identifier columns (frame.time, ip.src_host, ip.dst_host)
and any all-zero / nearly-constant columns. Encodes remaining string
columns with LabelEncoder. Returns a stratified 80/20 train/test split.
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


# Columns to drop (identifiers / leakage).
# The May 2026 audit of our Strategy A run showed 100% F1 on every config —
# a strong indicator of feature leakage. The columns below carry per-attack
# fingerprints (e.g. arp/mqtt fields are non-zero only for specific attack
# families). Dropping them yields rigorous, comparable evaluation metrics.
DROP_COLS = [
    # Pure identifiers
    "frame.time", "ip.src_host", "ip.dst_host",
    # ARP fields — only present in MITM/ARP-spoofing attacks (leak)
    "arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size", "arp.src.proto_ipv4",
    # HTTP free-text + URI fields — encode attack-specific payloads
    "http.file_data", "http.referer", "http.request.full_uri",
    "http.request.uri.query", "http.request.method", "http.request.version",
    # DNS query name — high-cardinality identifier
    "dns.qry.name",
    # MQTT free-text fields — bind to specific MQTT attacks
    "mqtt.topic", "mqtt.msg", "mqtt.msg_decoded_as", "mqtt.protoname",
    # TCP payload + options — encode flow-level identifiers / payload signature
    "tcp.payload", "tcp.options",
    # ----- Added 2026-05-29: identified by MI audit -----
    # Per-attack length fingerprints that act as label proxies
    "dns.qry.name.len",      # MI = 0.44
    "mqtt.conack.flags",     # MI = 0.44 (only set during MQTT attack)
    # TCP-level identifiers that leak attack-source signature
    "tcp.dstport",           # MI = 0.35
    "tcp.srcport",           # MI = 0.32
    "tcp.seq",               # MI = 0.24
    "tcp.ack",               # MI = 0.22
]
LABEL_COL = "Attack_label"
TYPE_COL  = "Attack_type"

# Standard 15-class mapping for Edge-IIoTset (Normal + 14 attack types)
ATTACK_TYPES_EDGE = [
    "Normal", "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP",
    "Fingerprinting", "MITM", "Password", "Port_Scanning", "Ransomware",
    "SQL_injection", "Uploading", "Vulnerability_scanner", "XSS",
]
MULTICLASS_LABEL_MAP_EDGE = {name: i for i, name in enumerate(ATTACK_TYPES_EDGE)}


class EdgeIiotsetLoader(DatasetLoader):
    name = "edge-iiotset"

    def __init__(self, data_root: Path,
                 csv_name: str = "ML-EdgeIIoT-dataset.csv"):
        self.data_root = Path(data_root)
        self.csv_path = self.data_root / "edge-iiotset" / "ml-dl-selected" / csv_name

    def source_paths(self) -> List[Path]:
        return [self.csv_path]

    def load(self, classification: str = "binary") -> LoadedDataset:
        if classification not in ("binary", "multiclass"):
            raise ValueError(classification)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        print(f"[edge-iiotset] reading {self.csv_path.name}…")
        df = pd.read_csv(self.csv_path, low_memory=False)
        print(f"[edge-iiotset] raw shape: {df.shape}")

        # Drop identifier columns
        for c in DROP_COLS:
            if c in df.columns:
                df = df.drop(columns=[c])

        # Build target
        if classification == "binary":
            y = df[LABEL_COL].astype(int).values
            class_names = ["Normal", "Attack"]
            is_binary = True
        else:
            y = df[TYPE_COL].map(MULTICLASS_LABEL_MAP_EDGE).fillna(-1).astype(int).values
            keep = y >= 0
            df = df.loc[keep].reset_index(drop=True)
            y = y[keep]
            class_names = ATTACK_TYPES_EDGE
            is_binary = False

        # Drop label/type from features
        df = df.drop(columns=[c for c in (LABEL_COL, TYPE_COL) if c in df.columns])

        # String columns -> LabelEncoder (some columns have HTTP URIs, hex
        # values, etc.; treating them as categorical is fine for tree models
        # and works for the GAN's embedding via OHE).
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].astype(str).fillna("missing")
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c])

        # Drop near-constant columns (≥99% same value) — pure noise for the model
        nunq = df.nunique()
        keep_cols = [c for c in df.columns if nunq[c] > 1]
        df = df[keep_cols]

        feature_names = list(df.columns)
        X = df.astype(np.float32).values
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[edge-iiotset] features kept: {X.shape[1]} / 61 raw")

        # Stratified 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
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
                "csv": str(self.csv_path),
                "split": "stratified 80/20 (random)",
                "scaler": "StandardScaler",
                "classification": classification,
                "n_classes_present": int(len(np.unique(y))),
            },
        )


def get_loader(data_root: Path) -> EdgeIiotsetLoader:
    return EdgeIiotsetLoader(data_root)
