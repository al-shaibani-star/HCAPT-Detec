"""Does generative augmentation help a class because of its support?

The dissertation asserts a minimum-viable-support threshold: below some class
support n*, conditional generative augmentation stops improving recall on that
class. The evidence offered is a single five-class snapshot in which one class
gained (DoS, n=45,927, +6.01 pp) and four lost, the worst being the rarest
(U2R, n=52, -17.91 pp). That snapshot cannot separate support from class
identity -- Probe has 11,656 training samples, twelve times R2L's, and is hurt
just as much.

This sweep separates them. One class is held as the subject, its training
support is subsampled to a ladder of levels, and every other class is left at
full support. The test partition never changes, which is what Definition 1 of
the dissertation requires. At each level the ensemble is trained twice from the
same subsampled training set -- once plainly, once after WGAN-GP balancing --
and the subject class's recall on the fixed test set is compared.

If support drives the effect, the gain decays along the ladder and inverts.
If class identity drives it, the gain survives at every level.

    python -m src.support_sweep --class DoS --levels 100,1000,5000,20000,45927 --seeds 3

Writes results/_support_sweep/<class>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STRATEGY_ROOT = HERE.parent
DATA_DIR = STRATEGY_ROOT.parents[1] / "data"
OUT_DIR = STRATEGY_ROOT / "results" / "_support_sweep"


def subsample_class(y: np.ndarray, cls: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices keeping every class whole except `cls`, which is cut to n."""
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if c == cls and len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        keep.append(idx)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def recall_of(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    m = y_true == cls
    return float((y_pred[m] == cls).mean()) if m.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls_name", default="DoS")
    ap.add_argument("--levels", default="100,1000,5000,20000,45927")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gan-epochs", type=int, default=50)
    args = ap.parse_args()

    import torch  # noqa: F401  (imported by the pipeline module too)
    from src.loaders.nsl_kdd import NslKddLoader
    from src.phase2_pipeline import EnsembleAdHoc, gan_balance, train_wgan_gp_explicit

    ds = NslKddLoader(DATA_DIR).load("multiclass")
    X_tr = np.asarray(ds.X_train, dtype=np.float32)
    y_tr = np.asarray(ds.y_train).astype(int).ravel()
    X_te = np.asarray(ds.X_test, dtype=np.float32)
    y_te = np.asarray(ds.y_test).astype(int).ravel()
    names = list(ds.class_names)
    cls = names.index(args.cls_name)
    n_classes, feat = len(names), X_tr.shape[1]
    full_n = int((y_tr == cls).sum())

    levels = [int(v) for v in args.levels.split(",")]
    print(f"subject class {args.cls_name} (index {cls}), full support {full_n:,}")
    print(f"levels {levels}   seeds {args.seeds}   gan epochs {args.gan_epochs}")

    # Resume: a run costs ~50 minutes, so never repeat one that is already
    # banked. The output file is written after every run for this reason.
    out_path = OUT_DIR / f"{args.cls_name}.json"
    rows = []
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8")).get("rows", [])
        print(f"resuming: {len(rows)} run(s) already banked")
    done = {(r["requested"], r["seed"]) for r in rows}

    for n in levels:
        for seed in range(args.seeds):
            if (n, seed) in done:
                print(f"  n={n:>6,} seed={seed}  already banked, skipping")
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 * seed + 7)
            idx = subsample_class(y_tr, cls, n, rng)
            Xs, ys = X_tr[idx], y_tr[idx]
            actual = int((ys == cls).sum())

            # control: no augmentation
            ctrl = EnsembleAdHoc(feat, n_classes, seed=seed).fit(Xs, ys)
            r_ctrl = recall_of(y_te, ctrl.predict_with_confidence(X_te)[0], cls)

            # treatment: WGAN-GP balancing on the same subsampled training set
            G = train_wgan_gp_explicit(Xs, ys, feat, n_classes,
                                       epochs=args.gan_epochs, seed=seed)
            Xb, yb = gan_balance(Xs, ys, G, n_classes)
            aug = EnsembleAdHoc(feat, n_classes, seed=seed).fit(Xb, yb)
            r_aug = recall_of(y_te, aug.predict_with_confidence(X_te)[0], cls)

            row = {"support": actual, "requested": n, "seed": seed,
                   "recall_control": r_ctrl, "recall_augmented": r_aug,
                   "delta_pp": 100.0 * (r_aug - r_ctrl),
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"  n={actual:>6,} seed={seed}  ctrl={r_ctrl:.4f}  aug={r_aug:.4f}"
                  f"  delta={row['delta_pp']:+.2f} pp   ({row['seconds']:.0f}s)")

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{args.cls_name}.json").write_text(json.dumps({
                "subject_class": args.cls_name,
                "full_support": full_n,
                "test_partition": "KDDTest+ (held fixed)",
                "generator": "WGAN-GP, conditional, balanced to the majority count",
                "gan_epochs": args.gan_epochs,
                "rows": rows,
            }, indent=2), encoding="utf-8")

    print(f"\nwritten to {OUT_DIR / (args.cls_name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
