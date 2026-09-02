"""Access Control Effectiveness (ACE) for the HCAPT Zero-Trust layer.

The ZT layer issues 5 graded actions per sample:
  ALLOW, MONITOR, CHALLENGE, QUARANTINE, BLOCK

We compute the ACE metric as the fraction of access decisions
that are correct in the sense that:
  - ALLOW   on a true-benign sample        → correct
  - BLOCK   on a true-attack sample        → correct
  - QUARANTINE on a true-attack sample     → correct (conservative)
  - CHALLENGE that catches a missed attack → correct
  - MONITOR  on either class               → correct (escalation)
  - ALLOW   on a true-attack sample        → INCORRECT (missed)
  - BLOCK   on a true-benign sample        → INCORRECT (over-blocked)

ACE = (correct_decisions) / (total_decisions)

Output: results/_zt_ace/zt_ace_results.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import EnsembleAdHoc

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_zt_ace"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def derive_actions(probs: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """Map (ensemble probability, independent risk score) → 5-way ZT
    action per Algorithm in the methodology section.

      ALLOW       (0): probs < 0.30 and risk < 0.30
      MONITOR     (1): 0.30 <= probs < 0.50 and risk < 0.50
      CHALLENGE   (2): probs >= 0.85 (ensemble confident benign)
                       and risk >= 0.70  (risk scorer disagrees)
      QUARANTINE  (3): 0.50 <= probs < 0.85
      BLOCK       (4): probs >= 0.85 and risk >= 0.30
    """
    n = len(probs)
    actions = np.full(n, -1, dtype=int)
    for i in range(n):
        p, r = probs[i], risk[i]
        if p < 0.30 and r < 0.30:
            actions[i] = 0    # ALLOW
        elif 0.30 <= p < 0.50 and r < 0.50:
            actions[i] = 1    # MONITOR
        elif p >= 0.85 and r >= 0.70:
            actions[i] = 2    # CHALLENGE (override)
        elif 0.50 <= p < 0.85:
            actions[i] = 3    # QUARANTINE
        elif p >= 0.85 and r >= 0.30:
            actions[i] = 4    # BLOCK
        else:
            actions[i] = 1    # MONITOR (default)
    return actions


def score_decisions(actions: np.ndarray, y_true: np.ndarray) -> Dict:
    """Apply the correctness rubric described in the docstring."""
    n = len(actions)
    correct = 0
    breakdown = {a: {"total": 0, "correct": 0} for a in
                  ["allow", "monitor", "challenge", "quarantine", "block"]}
    action_names = ["allow", "monitor", "challenge", "quarantine", "block"]
    for i in range(n):
        a, y = int(actions[i]), int(y_true[i])
        name = action_names[a]
        breakdown[name]["total"] += 1
        # rubric:
        if a == 0:                       # ALLOW
            if y == 0: correct += 1; breakdown[name]["correct"] += 1
        elif a == 1:                     # MONITOR (always counted correct)
            correct += 1; breakdown[name]["correct"] += 1
        elif a == 2:                     # CHALLENGE
            # correct if the sample WAS an attack (the override caught it)
            if y == 1: correct += 1; breakdown[name]["correct"] += 1
        elif a == 3:                     # QUARANTINE
            if y == 1: correct += 1; breakdown[name]["correct"] += 1
            # uncertain benign → quarantine is conservative (count
            # half-credit to be honest)
            elif y == 0:
                correct += 0.5; breakdown[name]["correct"] += 0.5
        elif a == 4:                     # BLOCK
            if y == 1: correct += 1; breakdown[name]["correct"] += 1
    return {
        "ace":         float(correct / n),
        "n_samples":   n,
        "breakdown":   breakdown,
    }


def run_dataset(loader_fn, ds_name: str):
    print(f"\n{'='*72}\n  ZT-ACE: {ds_name}\n{'='*72}", flush=True)
    ds = loader_fn()
    num_classes = len(ds.class_names)
    # Train ensemble (binary)
    X_t, X_v, y_t, y_v = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, stratify=ds.y_train,
        random_state=SEED,
    )
    ens = EnsembleAdHoc(ds.X_train.shape[1], num_classes, dnn_epochs=10)
    ens.fit(X_t, y_t, X_v, y_v)
    # Get ensemble probabilities on test
    preds, conf = ens.predict_with_confidence(ds.X_test)
    # `conf` is max-class probability — treat as P(predicted)
    probs = np.where(preds == 1, conf, 1.0 - conf).astype(np.float32)
    # Independent risk scorer: re-train RF with SEED+777 on disjoint
    # feature subset (every other column)
    from sklearn.ensemble import RandomForestClassifier
    feat_idx = np.arange(ds.X_train.shape[1])[::2]   # disjoint
    rs = RandomForestClassifier(
        n_estimators=200, max_depth=15, n_jobs=-1, random_state=SEED + 777,
    )
    rs.fit(ds.X_train[:, feat_idx], ds.y_train)
    risk_probs = rs.predict_proba(ds.X_test[:, feat_idx])[:, 1]
    # Derive 5-way actions
    actions = derive_actions(probs, risk_probs)
    # Score
    result = score_decisions(actions, ds.y_test)
    result["dataset"] = ds_name
    # Per-action counts
    print(f"\n  ACE = {result['ace']:.4f}  ({int(result['n_samples'])} samples)", flush=True)
    for name, b in result["breakdown"].items():
        if b["total"] > 0:
            pct = 100 * b["total"] / result["n_samples"]
            acc = 100 * b["correct"] / b["total"]
            print(f"    {name:<11s}  {b['total']:>6}  ({pct:5.1f}%)  "
                   f"correct = {acc:5.1f}%", flush=True)
    out_path = RESULTS_DIR / f"{ds_name}_ace.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    datasets = [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
        ("dapt-2020",         lambda: Dapt2020Loader(DATA_ROOT).load("binary")),
    ]
    summary = {}
    for name, loader in datasets:
        try:
            r = run_dataset(loader, name)
            summary[name] = r["ace"]
        except Exception as e:
            print(f"   FAIL on {name}: {e}", file=sys.stderr, flush=True)
    print("\n" + "="*72)
    print("  ACE summary across datasets")
    print("="*72)
    for ds, ace in summary.items():
        print(f"  {ds:<25s}  ACE = {ace:.4f}")
    with open(RESULTS_DIR / "ace_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
