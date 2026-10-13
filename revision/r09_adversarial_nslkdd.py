"""
Step 8 of the IJIES revision: corrected adversarial-robustness check (old Table 15).

The old table had PGD weaker than single-step FGSM at the same budget, which is
impossible for a correctly configured PGD (PGD with a random start contains FGSM
as a special case).  This script re-runs the check under an explicit protocol:

  * model: the paper's ensemble with class weighting only (RF + XGBoost + DNN),
    retrained with the r01 protocol (FIT/VAL split, seed 0)
  * threat model: white-box L-infinity attacks on the differentiable DNN, in
    standardized feature units; the three categorical columns (protocol_type,
    service, flag) are never perturbed; adversarial inputs are transferred to RF,
    XGBoost and the full ensemble (grey-box for the trees)
  * attacks: FGSM, PGD-20 (step eps/8, random start) at eps in {0.05, 0.1, 0.25, 0.5}
  * metric: binary accuracy (Normal vs attack) on all 22,544 KDDTest+ flows
  * sanity checks printed: accuracy is non-increasing in eps for the attacked DNN,
    and PGD accuracy <= FGSM accuracy for the attacked DNN at every eps
Feature-space perturbations are not guaranteed to correspond to realisable traffic;
this is a stress test, not a claim of real-world robustness.

Output: revision/out/tables/adversarial_nslkdd.csv
Usage:  python -u revision\\r09_adversarial_nslkdd.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import rev_common as rc
from rev_common import OUT_DIR
import ensemble_selflearn as es

EPS = [0.05, 0.1, 0.25, 0.5]
CAT_IDX = [1, 2, 3]       # protocol_type, service, flag
SEED = 0


def binary_loss(logits, yb):
    """Cross-entropy of the binary decision the metric scores:
    logit(Normal) vs logsumexp(logits of the four attack classes)."""
    z = torch.stack([logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)], dim=1)
    return F.cross_entropy(z, yb)


def attack(model, X, y, eps, steps, mask):
    """L-inf attack on the DNN's binary decision; steps=1 -> FGSM, >1 -> PGD (random start)."""
    model.eval()
    x0 = torch.from_numpy(X).float().to(es.DEVICE)
    yt = torch.from_numpy((y != 0).astype(np.int64)).to(es.DEVICE)
    m = torch.from_numpy(mask).float().to(es.DEVICE)
    if steps == 1:
        x = x0.clone().requires_grad_(True)
        binary_loss(model(x), yt).backward()
        return (x0 + eps * x.grad.sign() * m).detach().cpu().numpy()
    alpha = eps / 8
    x = (x0 + torch.empty_like(x0).uniform_(-eps, eps) * m).detach()
    for _ in range(steps):
        x.requires_grad_(True)
        g, = torch.autograd.grad(binary_loss(model(x), yt), x)
        x = x.detach() + alpha * g.sign() * m
        x = torch.max(torch.min(x, x0 + eps), x0 - eps).detach()
    return x.cpu().numpy()


def main():
    np.random.seed(SEED); torch.manual_seed(SEED); es.SEED = SEED
    D = rc.load_nslkdd()
    ens = es.EnsembleDetector(feature_dim=D["X_fit"].shape[1])
    ens.fit(D["X_fit"], D["y_fit"], D["X_val"], D["y_val"], nn_epochs=30, verbose=False)
    X, y = D["X_test"], D["y_test"]
    mask = np.ones(X.shape[1], np.float32); mask[CAT_IDX] = 0

    def acc_all(Xa):
        prf, pxg, pnn = ens.rf.predict_proba(Xa), ens.xgb.predict_proba(Xa), ens.nn.predict_proba(Xa)
        pen = ens.weights[0] * prf + ens.weights[1] * pxg + ens.weights[2] * pnn
        return {k: rc.binary_metrics(y, p.argmax(1))["accuracy"]
                for k, p in (("DNN (attacked)", pnn), ("RF (transfer)", prf),
                             ("XGBoost (transfer)", pxg), ("Ensemble (transfer)", pen))}

    rows = [{"attack": "clean", "eps": 0.0, **acc_all(X)}]
    # batched attacks to bound GPU memory
    def batched(eps, steps):
        out = np.empty_like(X)
        for i in range(0, len(X), 4096):
            out[i:i + 4096] = attack(ens.nn, X[i:i + 4096], y[i:i + 4096], eps, steps, mask)
        return out
    for eps in EPS:
        for name, steps in (("FGSM", 1), ("PGD-20", 20)):
            rows.append({"attack": name, "eps": eps, **acc_all(batched(eps, steps))})
            print(rows[-1])
    df = pd.DataFrame(rows)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "tables" / "adversarial_nslkdd.csv", index=False)

    dnn = "DNN (attacked)"
    f = df[df.attack == "FGSM"].set_index("eps")[dnn]; p = df[df.attack == "PGD-20"].set_index("eps")[dnn]
    print("\n", df.round(4).to_string(index=False))
    print("\nsanity: PGD <= FGSM on the attacked DNN at every eps:", bool((p <= f + 1e-9).all()))
    print("sanity: DNN accuracy non-increasing in eps (PGD):", bool((np.diff(p.values) <= 1e-9).all()))


if __name__ == "__main__":
    main()
