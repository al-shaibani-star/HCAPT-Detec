"""Evaluate CGAN- and WGAN-GP-augmented ensembles (+ Zero-Trust) on KDDTest+,
save per-sample outputs for figure generation."""
import sys, json
from pathlib import Path
import numpy as np, joblib, torch, xgboost as xgb
from sklearn.metrics import confusion_matrix

SRC = Path(r"E:\Strudin\Hussein2026\05-Implementation\src")
sys.path.insert(0, str(SRC))
from config import MODELS_DIR, DEVICE, NUM_CLASSES, LABEL_MAP
from data_loader import get_preprocessed
from ensemble_selflearn import EnsembleDetector, DeepDetector
from zero_trust import RiskScorer, PolicyEngine, ZeroTrustLayer

OUT = Path(__file__).resolve().parent
_, X_te, _, y_te = get_preprocessed()
fd = X_te.shape[1]
print("test:", X_te.shape, "classes:", np.bincount(y_te))

def load_ens(prefix):
    ens = EnsembleDetector(feature_dim=fd)
    ens.rf = joblib.load(MODELS_DIR / f"{prefix}_ens_rf.joblib")
    ens.xgb = xgb.XGBClassifier(); ens.xgb.load_model(str(MODELS_DIR / f"{prefix}_ens_xgb.json"))
    ens.nn = DeepDetector(fd).to(DEVICE)
    ens.nn.load_state_dict(torch.load(MODELS_DIR / f"{prefix}_ens_nn.pt", map_location=DEVICE))
    ens.weights = np.load(MODELS_DIR / f"{prefix}_ens_weights.npy")
    return ens

def make_zt():
    risk = RiskScorer(); risk.load(MODELS_DIR)
    return ZeroTrustLayer(risk_scorer=risk, policy=PolicyEngine(high_conf=0.85, mid_conf=0.55, high_risk=0.70, mid_risk=0.40))

def binm(y, p):
    yb, pb = (y != 0), (p != 0)
    tn = int((~yb & ~pb).sum()); fp = int((~yb & pb).sum()); fn = int((yb & ~pb).sum()); tp = int((yb & pb).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(accuracy=float((yb == pb).mean()), precision=prec, recall=rec, f1=2*prec*rec/max(prec+rec,1e-9), fpr=fp/max(tn+fp,1), tp=tp, fp=fp, tn=tn, fn=fn)

def ece(conf, correct, bins=np.linspace(0.4, 1.0, 13)):
    N = len(conf); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum(): e += m.sum()/N * abs(correct[m].mean() - conf[m].mean())
    return float(e)

summary = {}
for prefix, label in [("gan", "CGAN"), ("wgangp", "WGAN-GP")]:
    ens = load_ens(prefix)
    proba = ens.predict_proba(X_te)
    preds, conf = ens.predict_with_confidence(X_te)
    assert np.array_equal(preds, proba.argmax(1))
    zt = make_zt()
    out = zt.validate_batch(X_te, preds.copy(), conf.copy(), return_decisions=False)
    zt_preds = np.asarray(out["final_labels"])
    actions = dict(zt.action_counts)
    cm = confusion_matrix(y_te, preds, labels=list(range(NUM_CLASSES)))
    cm_zt = confusion_matrix(y_te, zt_preds, labels=list(range(NUM_CLASSES)))
    correct = (preds == y_te).astype(int)
    # challenge precision
    changed = zt_preds != preds
    caught = int((changed & (y_te != 0)).sum()); false_ch = int((changed & (y_te == 0)).sum())
    s = {
        "ens_binary": binm(y_te, preds), "ens_multiclass_acc": float((preds == y_te).mean()),
        "zt_binary": binm(y_te, zt_preds), "zt_multiclass_acc": float((zt_preds == y_te).mean()),
        "per_class_recall": {c: float(cm[i, i] / cm[i].sum()) for c, i in LABEL_MAP.items()},
        "ece": ece(conf, correct), "mean_conf": float(conf.mean()),
        "hc_errors_total": int(((conf >= 0.85) & (correct == 0)).sum()),
        "hc_errors_by_class": {c: int(((y_te == i) & (conf >= 0.85) & (correct == 0)).sum()) for c, i in LABEL_MAP.items()},
        "zt_actions": {str(k): int(v) for k, v in actions.items()},
        "zt_changed": int(changed.sum()), "zt_caught_attacks": caught, "zt_false_challenges": false_ch,
        "cm": cm.tolist(), "cm_zt": cm_zt.tolist(),
    }
    summary[label] = s
    np.savez(OUT / f"persample_{prefix}.npz", proba=proba, preds=preds, conf=conf, zt_preds=zt_preds, y=y_te)
    print(f"\n==== {label} ====")
    print(" ens  binary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in s["ens_binary"].items()})
    print(" +ZT  binary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in s["zt_binary"].items()})
    print(" multiclass acc: ens=%.4f  zt=%.4f" % (s["ens_multiclass_acc"], s["zt_multiclass_acc"]))
    print(" per-class recall:", {k: round(v, 4) for k, v in s["per_class_recall"].items()})
    print(" ECE=%.4f mean_conf=%.4f  HC errors=%d %s" % (s["ece"], s["mean_conf"], s["hc_errors_total"], s["hc_errors_by_class"]))
    print(" ZT actions:", s["zt_actions"], "| changed:", s["zt_changed"], "caught:", caught, "false:", false_ch)
    print(" CM:\n", cm)
json.dump(summary, open(OUT / "variant_summary.json", "w"), indent=1)
print("\nsaved")
