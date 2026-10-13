"""
IJIES 20266447 revision - shared protocol for every revision experiment.

Protocol (answers Reviewer 2, comment 4):
  * KDDTrain+ is split once, stratified, into FIT (80 %) and VAL (20 %).
    The split indices are written to revision/out/split_nslkdd.npz and never change.
  * Every fitted object - label encoders, scaler, generators, SMOTE,
    ensemble learners, ensemble weights, risk scorer, Zero-Trust thresholds,
    calibrators - is fitted on FIT or VAL only.
  * KDDTest+ is used once, for final scoring.  Nothing is tuned on it.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

REV_DIR = Path(__file__).resolve().parent
SRC_DIR = REV_DIR.parent / "src"
OUT_DIR = REV_DIR / "out"
PRED_DIR = OUT_DIR / "preds"
for d in (OUT_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

from config import CATEGORICAL_COLS, LABEL_MAP, NUM_CLASSES  # noqa: E402
from data_loader import _find_nsl_kdd_files, load_raw, map_attack_category  # noqa: E402

CLASS_NAMES = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]
SPLIT_SEED = 42          # the split is fixed across all training seeds
VAL_FRACTION = 0.20
SUPPORT_THRESHOLD = 1000  # GAN if a class has >= this many real FIT samples, else SMOTE
RISK_FEATURES = list(range(1, 41, 2))  # validator sees odd-indexed features only


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_nslkdd():
    """Return dict with X_fit, y_fit, X_val, y_val, X_test, y_test (float32 scaled)
    plus feature names and provenance.  Preprocessing is fitted on FIT only."""
    train_path, test_path = _find_nsl_kdd_files(SRC_DIR.parent / "data")
    df_tr, df_te = load_raw()
    df_tr = map_attack_category(df_tr)
    df_te = map_attack_category(df_te)

    split_file = OUT_DIR / "split_nslkdd.npz"
    if split_file.exists():
        s = np.load(split_file)
        idx_fit, idx_val = s["idx_fit"], s["idx_val"]
    else:
        idx = np.arange(len(df_tr))
        idx_fit, idx_val = train_test_split(
            idx, test_size=VAL_FRACTION, random_state=SPLIT_SEED,
            stratify=df_tr["y"].values)
        idx_fit, idx_val = np.sort(idx_fit), np.sort(idx_val)
        np.savez(split_file, idx_fit=idx_fit, idx_val=idx_val)

    feature_cols = [c for c in df_tr.columns if c not in ("label", "attack_category", "y")]
    df_fit, df_val = df_tr.iloc[idx_fit].copy(), df_tr.iloc[idx_val].copy()
    df_te = df_te.copy()

    # Categorical vocabulary from FIT only; unseen values get one extra code.
    for col in CATEGORICAL_COLS:
        le = LabelEncoder().fit(df_fit[col])
        known = {v: i for i, v in enumerate(le.classes_)}
        unk = len(known)
        for df in (df_fit, df_val, df_te):
            df[col] = df[col].map(lambda v: known.get(v, unk))

    X_fit = df_fit[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_te = df_te[feature_cols].values.astype(np.float32)
    scaler = StandardScaler().fit(X_fit)

    def tr(X):
        return scaler.transform(X).astype(np.float32)

    return {
        "X_fit": tr(X_fit), "y_fit": df_fit["y"].values.astype(np.int64),
        "X_val": tr(X_val), "y_val": df_val["y"].values.astype(np.int64),
        "X_test": tr(X_te), "y_test": df_te["y"].values.astype(np.int64),
        "feature_names": feature_cols,
        "idx_fit": idx_fit, "idx_val": idx_val,
        "provenance": {
            "train_file": str(train_path), "train_sha256": sha256(train_path),
            "test_file": str(test_path), "test_sha256": sha256(test_path),
            "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
        },
    }


def hardware_info() -> dict:
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": platform.processor(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import sklearn, xgboost, lightgbm, imblearn
        info.update(sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                    lightgbm=lightgbm.__version__, imblearn=imblearn.__version__)
    except ImportError:
        pass
    return info


def binary_counts(y_true, y_pred):
    t = y_true != 0
    p = y_pred != 0
    tp = int((t & p).sum()); fp = int((~t & p).sum())
    tn = int((~t & ~p).sum()); fn = int((t & ~p).sum())
    return tp, fp, tn, fn


def binary_metrics(y_true, y_pred) -> dict:
    tp, fp, tn, fn = binary_counts(y_true, y_pred)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / len(y_true), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "fpr": fp / max(fp + tn, 1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def dump_json(obj, path: Path):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")
