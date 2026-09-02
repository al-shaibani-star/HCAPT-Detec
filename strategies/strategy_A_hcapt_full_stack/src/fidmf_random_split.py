"""
The re-implemented attention-LSTM under the random-split protocol.

This is the run §6.2 of the manuscript names as missing. Without it the
official-versus-random gap is established for one architecture in our harness,
and for the second only by comparing our official-split measurement against
that method's *own published* random-split figure — a cross-harness comparison
of exactly the kind the paper criticises.

The protocol here is byte-identical to `nsl_kdd_random_split.py`, which is what
makes the comparison legitimate: the combined KDDTrain+/KDDTest+ pool,
categorical encoding and scaling fitted on that pool before splitting (the
practice the protocol is meant to reproduce, and which leaks), then a stratified
random 90:10 split at seed 42.

    python -m src.fidmf_random_split

Writes `results/_fidmf_reimpl/nsl-kdd-random_fidmf.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

STRATEGY_ROOT = Path(__file__).resolve().parents[1]
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

from src.loaders.base import LoadedDataset                      # noqa: E402
from src.loaders.nsl_kdd import NSL_KDD_COLUMNS, CATEGORICAL_COLS, ATTACK_CATEGORIES  # noqa: E402
from src.fidmf_reimpl import run_fidmf_on                       # noqa: E402

DATA_ROOT = STRATEGY_ROOT.parents[1] / "data"


def load_random_split() -> LoadedDataset:
    """The FIDMF-style protocol, identical to `nsl_kdd_random_split.py`."""
    tr = pd.read_csv(DATA_ROOT / "nsl-kdd" / "train.txt", header=None,
                     names=NSL_KDD_COLUMNS).drop(columns=["difficulty"])
    te = pd.read_csv(DATA_ROOT / "nsl-kdd" / "test.txt", header=None,
                     names=NSL_KDD_COLUMNS).drop(columns=["difficulty"])
    df = pd.concat([tr, te], axis=0, ignore_index=True)
    df["category"] = df["label"].map(lambda s: ATTACK_CATEGORIES.get(s, "Unknown"))

    y = (df["category"] != "Normal").astype(int).values
    X_df = df.drop(columns=["label", "category"])
    for col in CATEGORICAL_COLS:                     # encoded on the COMBINED pool
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    X = StandardScaler().fit_transform(X).astype(np.float32)   # scaled before splitting

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y)

    print(f"[random-split] pool {X.shape}, train {X_train.shape}, test {X_test.shape}")
    print(f"[random-split] attack ratio  train {y_train.mean():.3f}  test {y_test.mean():.3f}")

    return LoadedDataset(
        name="nsl-kdd-random", X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, feature_names=feature_names,
        class_names=["Normal", "Attack"], is_binary=True,
        metadata={"protocol": "FIDMF-style random 90:10 of combined Train+/Test+",
                  "encoding": "fitted on the combined pool",
                  "scaler": "StandardScaler fitted on the combined pool before splitting",
                  "seed": 42})


def main() -> int:
    print("\n" + "#" * 72)
    print("#  Re-implemented attention-LSTM under the random-split protocol")
    print("#  (the run §6.2 of the manuscript records as missing)")
    print("#" * 72)

    results = run_fidmf_on(load_random_split, "nsl-kdd-random")

    # The comparison the run exists to make.
    official = json.loads(
        (STRATEGY_ROOT / "results" / "_fidmf_reimpl" / "nsl-kdd_fidmf.json")
        .read_text(encoding="utf-8"))["results"]
    ours = json.loads(
        (STRATEGY_ROOT / "results" / "_random_split_baseline" / "nsl_kdd_random.json")
        .read_text(encoding="utf-8"))

    print("\n" + "=" * 72)
    print("  THE GAP, NOW IN ONE HARNESS")
    print("=" * 72)
    for var in ("attlstm_only", "smote_attlstm", "smote_gan_attlstm"):
        o, r = official[var]["accuracy"], results[var]["accuracy"]
        print(f"  attention-LSTM / {var:<18} official {o:.4f}   random {r:.4f}   "
              f"gap {100*(r-o):+.2f} pp")
    print(f"  our ensemble{'':<24} official 0.7792   random {ours['accuracy']:.4f}   "
          f"gap {100*(ours['accuracy']-0.7792):+.2f} pp")
    print(f"\n  the method's own published random-split figure: 0.9940 "
          f"(for reference only, not our measurement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
