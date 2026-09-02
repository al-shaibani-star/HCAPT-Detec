"""
Deeper ablation: tests beyond the standard 6-config matrix.

Variants explored:
  A. Ensemble composition: full (RF+XGB+DNN) vs RF-only vs XGB-only vs DNN-only
  B. Multi-flow GNN layer type: SAGE vs GAT
  C. Zero-Trust context analyzer: with vs without

Runs only on NSL-KDD for speed (reuses cached preprocessing).
Writes results/_deep_ablation/{ensemble_variants,gnn_variants,zt_context}.csv
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict
import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .phase2_pipeline import compute_metrics
from .multi_flow_tracker.tracker import MultiFlowTracker, MultiFlowConfig

import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ensemble_selflearn import DeepDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_deep_ablation"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


# =============================================================================
# A. Ensemble composition ablation
# =============================================================================
def _train_dnn(X_tr, y_tr, feature_dim, num_classes, epochs=20):
    nn_model = DeepDetector(feature_dim, num_classes=num_classes).to(DEVICE)
    optim = torch.optim.Adam(nn_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    nn_model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(y_tr), device=DEVICE)
        for i in range(0, len(y_tr) - 256, 256):
            idx = perm[i:i+256]
            loss = loss_fn(nn_model(X_t[idx]), y_t[idx])
            optim.zero_grad(); loss.backward(); optim.step()
    return nn_model


def _predict_dnn(nn_model, X):
    nn_model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = nn_model(X_t)
        return logits.argmax(dim=1).cpu().numpy(), \
               torch.softmax(logits, dim=1).cpu().numpy()


def ablate_ensemble_composition(ds) -> Dict:
    print("\n=== A. Ensemble composition ablation ===")
    X_tr, X_va, y_tr, y_va = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, random_state=42, stratify=ds.y_train,
    )
    feature_dim = X_tr.shape[1]
    num_classes = len(ds.class_names)

    # Train all three base learners once
    print("  training base learners…")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                tree_method="hist", device="cuda",
                                eval_metric="mlogloss", random_state=42, verbosity=0,
                                objective="binary:logistic")
    xgb_m.fit(X_tr, y_tr)
    nn_m = _train_dnn(X_tr, y_tr, feature_dim, num_classes, epochs=20)

    # Get predictions on test set
    p_rf  = rf.predict_proba(ds.X_test)
    p_xgb = xgb_m.predict_proba(ds.X_test)
    _, p_nn = _predict_dnn(nn_m, ds.X_test)

    variants = {
        "rf_only":        p_rf,
        "xgb_only":       p_xgb,
        "dnn_only":       p_nn,
        "rf+xgb":         (p_rf + p_xgb) / 2,
        "rf+nn":          (p_rf + p_nn) / 2,
        "xgb+nn":         (p_xgb + p_nn) / 2,
        "all_3 (default)":(p_rf + p_xgb + p_nn) / 3,
    }
    results = []
    for name, p in variants.items():
        preds = p.argmax(axis=1)
        m = compute_metrics(ds.y_test, preds, is_binary=True)
        m["variant"] = name
        results.append(m)
        print(f"   {name:>18}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  fpr={m['fpr']:.4f}")
    return results


# =============================================================================
# B. GNN layer-type ablation (SAGE vs GAT)
# =============================================================================
def ablate_gnn_layer(ds) -> Dict:
    print("\n=== B. GNN layer-type ablation (SAGE vs GAT) ===")
    results = []
    for layer_type in ("sage", "gat"):
        print(f"  training MultiFlowTracker with {layer_type.upper()}…")
        cfg = MultiFlowConfig(window_size=256, stride=128, epochs=10,
                               batch_size=32, lr=2e-3, layer_type=layer_type)
        np.random.seed(42); torch.manual_seed(42)
        t0 = time.time()
        tracker = MultiFlowTracker(ds.name, ds.feature_names, cfg).fit(
            ds.X_train, ds.y_train, verbose=False,
        )
        train_time = time.time() - t0
        risks = tracker.predict_flow_risk(ds.X_test)
        # Threshold to binary
        preds = (risks >= 0.5).astype(int)
        m = compute_metrics(ds.y_test, preds, is_binary=True)
        m["layer_type"] = layer_type
        m["params"] = sum(p.numel() for p in tracker.model.parameters())
        m["train_time_s"] = round(train_time, 1)
        m["risk_mean"] = float(risks.mean())
        results.append(m)
        print(f"   {layer_type.upper():>6}  acc={m['accuracy']:.4f}  "
              f"f1={m['f1']:.4f}  params={m['params']}  time={train_time:.1f}s")
    return results


# =============================================================================
# C. Zero-Trust context-analyzer ablation
# =============================================================================
def ablate_zt_context(ds) -> Dict:
    """Compare: ZT with vs without context_analyzer contribution."""
    print("\n=== C. Zero-Trust context-analyzer ablation ===")
    from zero_trust import RiskScorer, PolicyEngine, ZeroTrustLayer

    X_tr, X_va, y_tr, y_va = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, random_state=42, stratify=ds.y_train,
    )
    feature_dim = X_tr.shape[1]

    # Train risk scorer
    feat_idx = list(range(0, feature_dim, 2))
    scorer = RiskScorer(feature_indices=feat_idx)
    scorer.fit(X_tr, y_tr, X_va, y_va)

    # Train a simple ensemble (RF + XGB only — fast) for ensemble preds
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1,
                                  random_state=42)
    rf.fit(X_tr, y_tr)
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                tree_method="hist", device="cuda",
                                objective="binary:logistic", verbosity=0,
                                random_state=42)
    xgb_m.fit(X_tr, y_tr)
    p_ens = (rf.predict_proba(ds.X_test) + xgb_m.predict_proba(ds.X_test)) / 2
    ens_preds = p_ens.argmax(axis=1)
    ens_confs = p_ens.max(axis=1)

    results = []
    # Variant 1: full ZT with context
    zt_full = ZeroTrustLayer(risk_scorer=scorer, policy=PolicyEngine())
    out = zt_full.validate_batch(ds.X_test, ens_preds, ens_confs,
                                   segment_features=(1, 2))
    m_full = compute_metrics(ds.y_test, out["final_labels"], is_binary=True)
    m_full["variant"] = "ZT + context_analyzer"
    results.append(m_full)
    print(f"   {'ZT+context':>22}  acc={m_full['accuracy']:.4f}  f1={m_full['f1']:.4f}")

    # Variant 2: ZT without context (pass segment_features=None to disable)
    out = zt_full.validate_batch(ds.X_test, ens_preds, ens_confs,
                                   segment_features=None)
    m_noctx = compute_metrics(ds.y_test, out["final_labels"], is_binary=True)
    m_noctx["variant"] = "ZT - context_analyzer"
    results.append(m_noctx)
    print(f"   {'ZT-context':>22}  acc={m_noctx['accuracy']:.4f}  f1={m_noctx['f1']:.4f}")

    # Variant 3: ensemble only (no ZT at all)
    m_noozt = compute_metrics(ds.y_test, ens_preds, is_binary=True)
    m_noozt["variant"] = "ensemble only (no ZT)"
    results.append(m_noozt)
    print(f"   {'ensemble only':>22}  acc={m_noozt['accuracy']:.4f}  f1={m_noozt['f1']:.4f}")
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*72)
    print("#  Deep Ablation Studies on NSL-KDD")
    print("#"*72)
    ds = NslKddLoader(DATA_ROOT).load("binary")
    print(ds.summary())

    out = {}
    try:
        out["A_ensemble_composition"] = ablate_ensemble_composition(ds)
    except Exception as e:
        out["A_ensemble_composition"] = {"error": str(e)}
        print(f"  [A] FAILED: {e}")

    try:
        out["B_gnn_layer"] = ablate_gnn_layer(ds)
    except Exception as e:
        out["B_gnn_layer"] = {"error": str(e)}
        print(f"  [B] FAILED: {e}")

    try:
        out["C_zt_context"] = ablate_zt_context(ds)
    except Exception as e:
        out["C_zt_context"] = {"error": str(e)}
        print(f"  [C] FAILED: {e}")

    # Save
    with open(RESULTS_DIR / "deep_ablation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[deep-ablation] wrote {RESULTS_DIR / 'deep_ablation.json'}")

    # CSVs per study
    for study_id, results in out.items():
        if isinstance(results, list) and results:
            cols = sorted({k for r in results for k in r.keys()})
            with open(RESULTS_DIR / f"{study_id}.csv", "w") as f:
                f.write(",".join(cols) + "\n")
                for r in results:
                    f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
            print(f"   wrote {study_id}.csv  ({len(results)} rows)")
    return out


if __name__ == "__main__":
    main()
