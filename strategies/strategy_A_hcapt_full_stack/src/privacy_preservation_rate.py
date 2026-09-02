"""Privacy Preservation Rate (PPR) via membership-inference attack.

PPR = 1 - 2 * |AUC - 0.5|

The attacker is given access to the class-prototype IoCs that
HCAPT's Layer 3b releases (with Gaussian DP at eps=5, delta=1e-5).
The attacker trains a shadow model to predict whether an
arbitrary candidate sample was part of the client's training data
that produced the IoC.

AUC = 0.5 is a random-guess attacker (perfect privacy, PPR = 1);
AUC = 1.0 is a perfect attacker (PPR = 0). The absolute deviation is
used because an attacker who scores reliably *below* chance is just as
informative as one above it — inverting its output recovers the signal.

Comparison: attack against
  (a) the DP-noised prototypes (released by HCAPT) — AUC should be near 0.5
  (b) the raw per-sample IoCs (counterfactual: no DP)        — AUC should be >>0.5

Improvement gap quantifies how much the DP layer buys.

Output: results/_ppr/ppr_results.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_ppr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED          = 42
DP_EPSILON    = 5.0
DP_DELTA      = 1e-5
CLIP_NORM     = 1.0


def build_prototypes(X: np.ndarray, y: np.ndarray, with_dp: bool) -> tuple[np.ndarray, np.ndarray]:
    """One prototype per attack class. If with_dp, add Gaussian noise
    calibrated to eps=5, delta=1e-5 per release."""
    classes = sorted(set(int(c) for c in y))
    protos = []
    sigma  = (CLIP_NORM / DP_EPSILON) * np.sqrt(2.0 * np.log(1.25 / DP_DELTA))
    rng = np.random.RandomState(SEED)
    for c in classes:
        cls_x = X[y == c]
        # L2-clip per sample before averaging
        norms = np.linalg.norm(cls_x, axis=1, keepdims=True) + 1e-9
        cls_x_clipped = cls_x * np.minimum(1.0, CLIP_NORM / norms)
        mean = cls_x_clipped.mean(axis=0)
        if with_dp:
            mean = mean + rng.normal(0, sigma, size=mean.shape)
        protos.append(mean)
    return np.array(protos), np.array(classes)


def membership_inference_attack(
        X_member: np.ndarray, X_nonmember: np.ndarray,
        prototypes: np.ndarray,
) -> float:
    """Shadow-model attack:
      For each candidate sample, attacker computes
        f(x) = - min_c ||x - proto_c||^2
      i.e. larger f => closer to a prototype => more likely a member.
      Attacker classifies sample as MEMBER if f >= threshold;
      threshold tuned to give max balanced-accuracy.
      Returns attack ROC-AUC.
    """
    def score(X):
        # min squared distance to any prototype
        d = ((X[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
        return -d.min(axis=1)
    s_mem = score(X_member)
    s_non = score(X_nonmember)
    y_attack = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
    scores   = np.concatenate([s_mem, s_non])
    auc = roc_auc_score(y_attack, scores)
    # Attack success rate ≡ AUC (random = 0.5)
    return float(auc)


def run_dataset(loader_fn, ds_name: str):
    print(f"\n{'='*72}\n  PPR: {ds_name}\n{'='*72}", flush=True)
    ds = loader_fn()
    # Partition: 50% used to BUILD prototypes (members), 50% held out
    # as non-members for the attacker to distinguish.
    X = ds.X_train
    y = ds.y_train
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    half = len(X) // 2
    member_idx, nonmember_idx = idx[:half], idx[half:]
    X_member, y_member = X[member_idx], y[member_idx]
    X_nonmember        = X[nonmember_idx]

    # Cap to 5k each so the attack model is tractable
    if len(X_member)    > 5000: X_member    = X_member[:5000];    y_member = y_member[:5000]
    if len(X_nonmember) > 5000: X_nonmember = X_nonmember[:5000]

    # (a) HCAPT release: DP-noised prototypes
    protos_dp, _ = build_prototypes(X_member, y_member, with_dp=True)
    auc_dp = membership_inference_attack(X_member, X_nonmember, protos_dp)
    # (b) Counterfactual: no DP
    protos_raw, _ = build_prototypes(X_member, y_member, with_dp=False)
    auc_raw = membership_inference_attack(X_member, X_nonmember, protos_raw)

    # PPR = 1 - 2*|AUC - 0.5|: 1.0 at chance-level AUC, 0.0 at a perfect
    # attacker. NB: the old form `1 - max(auc, 0.5)` capped PPR at 0.5 and
    # never matched the definition used in the paper.
    ppr = lambda auc: 1.0 - 2.0 * abs(auc - 0.5)
    ppr_dp  = ppr(auc_dp)
    ppr_raw = ppr(auc_raw)

    print(f"  Members:     {len(X_member)},  Non-members: {len(X_nonmember)}", flush=True)
    print(f"  HCAPT (DP):     attack AUC = {auc_dp:.4f}  ->  PPR = {ppr_dp:.4f}",
           flush=True)
    print(f"  Counterfactual: attack AUC = {auc_raw:.4f}  ->  PPR = {ppr_raw:.4f}",
           flush=True)
    print(f"  DP benefit:     PPR gap   = {ppr_dp - ppr_raw:+.4f}",
           flush=True)

    out = {
        "dataset":          ds_name,
        "eps":              DP_EPSILON,
        "delta":            DP_DELTA,
        "n_members":        len(X_member),
        "n_nonmembers":     len(X_nonmember),
        "attack_auc_dp":    auc_dp,
        "attack_auc_raw":   auc_raw,
        "ppr_dp":           ppr_dp,
        "ppr_raw":          ppr_raw,
        "ppr_dp_benefit":   ppr_dp - ppr_raw,
    }
    out_path = RESULTS_DIR / f"{ds_name}_ppr.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    datasets = [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
    ]
    summary = []
    for name, loader in datasets:
        try:
            summary.append(run_dataset(loader, name))
        except Exception as e:
            print(f"  FAIL on {name}: {e}", file=sys.stderr, flush=True)
    print("\n" + "="*72)
    print(f"  {'Dataset':<25s}  {'PPR (DP)':>10s}  {'PPR (raw)':>10s}  {'gain':>8s}")
    print("="*72)
    for r in summary:
        print(f"  {r['dataset']:<25s}  {r['ppr_dp']:>10.4f}  "
               f"{r['ppr_raw']:>10.4f}  {r['ppr_dp_benefit']:>+8.4f}")
    with open(RESULTS_DIR / "ppr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
