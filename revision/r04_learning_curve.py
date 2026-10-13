"""
Step 3 of the IJIES revision (Reviewer 2, comment 2): controlled learning curve
for the "minimum viable support" claim.

Design (separates support from class identity, generator, and augmentation amount):
  * Target classes DoS, Probe, R2L.  For each one, its real FIT samples are
    subsampled to n in LEVELS (capped by what exists; R2L has 796 in FIT);
    every other class keeps all of its FIT samples.
  * Augmentation amount is held constant: the target class is topped up to
    TOTAL = 5,000 rows (n real + 5,000 - n synthetic) for every method.
      none   no synthetic rows (n real only)
      smote  SMOTE inside the target class
      cgan   conditional GAN, original architecture (Tanh output), class-specific
      wgan   WGAN-GP, original architecture, class-specific
    Generators are trained on the n real rows only with a FIXED budget of
    GAN_STEPS optimisation steps, so cost and capacity do not vary with n.
  * Downstream learning is held constant: one XGBoost with fixed
    hyper-parameters, no class weighting, no tuning.
  * Outcomes per (seed, class, n, method):
      recall of the target class on VAL (in-distribution) and on KDDTest+,
      macro-F1 on KDDTest+,
      synthetic-sample fidelity: C2ST AUC (0.5 = indistinguishable from real
      held-out rows of the class) and coverage (Naeem et al., 2020).
  * Repeated over seeds (subsample + generator + learner seed).

Output: revision/out/learning_curve.csv (one row per run; resumable).
Analysis: revision/r05_learning_curve_analysis.py

Usage (PowerShell, from 05-Implementation):
  python -u revision\\r04_learning_curve.py --smoke
  python -u revision\\r04_learning_curve.py --seeds 0 1 2 3 4
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import rev_common as rc
from rev_common import NUM_CLASSES, OUT_DIR, CLASS_NAMES
import gan_augment as cg
import wgan_gp_augment as wg
from config import DEVICE, GAN_LATENT_DIM

TARGETS = [1, 2, 3]                       # DoS, Probe, R2L
LEVELS = [25, 50, 100, 250, 500, 1000, 2000, 4000]
TOTAL = 5000
METHODS = ["none", "smote", "cgan", "wgan"]
FIELDS = ["seed", "cls", "class_name", "n_real", "method", "n_syn",
          "recall_val", "recall_test", "macro_f1_test", "c2st_auc", "coverage",
          "gan_seconds", "fit_seconds"]


# ---------------------------------------------------------------- generators
def train_gan(kind, X, steps, seed):
    """Class-specific GAN with the paper's architectures (num_classes=1)."""
    torch.manual_seed(seed)
    d = X.shape[1]
    Xt = torch.from_numpy(X).float().to(DEVICE)
    bs = int(min(64, len(X)))
    lab = torch.zeros(bs, dtype=torch.long, device=DEVICE)
    if kind == "cgan":
        G = cg.Generator(GAN_LATENT_DIM, 1, d).to(DEVICE)
        D = cg.Discriminator(d, 1).to(DEVICE)
        oG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        oD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        one = torch.ones(bs, 1, device=DEVICE); zero = torch.zeros(bs, 1, device=DEVICE)
        for _ in range(steps):
            real = Xt[torch.randint(0, len(Xt), (bs,), device=DEVICE)]
            fake = G(torch.randn(bs, GAN_LATENT_DIM, device=DEVICE), lab)
            oD.zero_grad()
            ld = 0.5 * (bce(D(real, lab), one) + bce(D(fake.detach(), lab), zero))
            ld.backward(); oD.step()
            oG.zero_grad()
            lg = bce(D(fake, lab), one); lg.backward(); oG.step()
    else:
        G = wg.Generator(GAN_LATENT_DIM, 1, d).to(DEVICE)
        C = wg.Critic(d, 1).to(DEVICE)
        oG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        oC = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))
        for step in range(steps):
            real = Xt[torch.randint(0, len(Xt), (bs,), device=DEVICE)]
            fake = G(torch.randn(bs, GAN_LATENT_DIM, device=DEVICE), lab).detach()
            oC.zero_grad()
            a = torch.rand(bs, 1, device=DEVICE)
            inter = (a * real + (1 - a) * fake).requires_grad_(True)
            g = torch.autograd.grad(C(inter, lab).sum(), inter, create_graph=True)[0]
            gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
            lc = C(fake, lab).mean() - C(real, lab).mean() + 10.0 * gp
            lc.backward(); oC.step()
            if (step + 1) % 5 == 0:
                oG.zero_grad()
                lg = -C(G(torch.randn(bs, GAN_LATENT_DIM, device=DEVICE), lab), lab).mean()
                lg.backward(); oG.step()
    G.eval()
    return G


def sample_gan(G, n):
    out = []
    with torch.no_grad():
        for i in range(0, n, 1024):
            b = min(1024, n - i)
            z = torch.randn(b, GAN_LATENT_DIM, device=DEVICE)
            out.append(G(z, torch.zeros(b, dtype=torch.long, device=DEVICE)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def smote_within(X_cls, n_new, seed, k=5):
    """SMOTE interpolation inside one class (same rule as imblearn)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(X_cls) - 1)
    if k < 1:
        return X_cls[rng.integers(0, len(X_cls), n_new)]
    nn_idx = NearestNeighbors(n_neighbors=k + 1).fit(X_cls).kneighbors(X_cls, return_distance=False)[:, 1:]
    base = rng.integers(0, len(X_cls), n_new)
    nb = nn_idx[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1)).astype(np.float32)
    return X_cls[base] + gap * (X_cls[nb] - X_cls[base])


# ---------------------------------------------------------------- fidelity
def c2st_auc(real, syn, seed, n=1000):
    rng = np.random.default_rng(seed)
    r = real[rng.choice(len(real), min(n, len(real)), replace=False)]
    s = syn[rng.choice(len(syn), min(n, len(syn)), replace=False)]
    X = np.vstack([r, s]); y = np.r_[np.zeros(len(r)), np.ones(len(s))]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    return float(max(auc, 1 - auc))


def coverage(real, syn, k=5, seed=0, n=2000):
    rng = np.random.default_rng(seed)
    r = real[rng.choice(len(real), min(n, len(real)), replace=False)]
    s = syn[rng.choice(len(syn), min(n, len(syn)), replace=False)]
    radius = NearestNeighbors(n_neighbors=k + 1).fit(r).kneighbors(r)[0][:, -1]
    dist = NearestNeighbors(n_neighbors=1).fit(s).kneighbors(r)[0][:, 0]
    return float((dist <= radius).mean())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--targets", type=int, nargs="+", default=TARGETS)
    ap.add_argument("--levels", type=int, nargs="+", default=LEVELS)
    ap.add_argument("--gan-steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT_DIR / ("learning_curve_smoke.csv" if a.smoke else "learning_curve.csv")
    if a.smoke:
        a.levels, a.targets, a.gan_steps = [50, 1000], [2, 3], 50

    D = rc.load_nslkdd()
    X_fit, y_fit, X_val, y_val, X_te, y_te = (D[k] for k in
        ("X_fit", "y_fit", "X_val", "y_val", "X_test", "y_test"))

    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(int(r["seed"]), int(r["cls"]), int(r["n_real"]), r["method"]) for r in csv.DictReader(f)}
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    for seed in a.seeds:
        for c in a.targets:
            idx_c = np.flatnonzero(y_fit == c); idx_o = np.flatnonzero(y_fit != c)
            for n in a.levels:
                if n > len(idx_c):
                    print(f"[lc] {CLASS_NAMES[c]}: only {len(idx_c)} real rows, skipping n={n}")
                    continue
                rng = np.random.default_rng(1000 * seed + 10 * c + n)
                pick = rng.choice(idx_c, n, replace=False)
                X_c = X_fit[pick]
                # real rows of this class NOT used for training -> fidelity reference
                held = np.vstack([X_fit[np.setdiff1d(idx_c, pick)], X_val[y_val == c]])
                n_syn = max(0, TOTAL - n)
                for method in METHODS:
                    if (seed, c, n, method) in done:
                        continue
                    t0 = time.time(); g_s = 0.0
                    if method == "none" or n_syn == 0:
                        S = np.empty((0, X_fit.shape[1]), np.float32)
                    elif method == "smote":
                        S = smote_within(X_c, n_syn, seed)
                    else:
                        tg = time.time()
                        G = train_gan(method, X_c, a.gan_steps, seed)
                        S = sample_gan(G, n_syn); g_s = time.time() - tg
                    X_tr = np.vstack([X_fit[idx_o], X_c, S])
                    y_tr = np.r_[y_fit[idx_o], np.full(n + len(S), c)]
                    m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                          objective="multi:softprob", num_class=NUM_CLASSES,
                                          tree_method="hist", device="cuda", random_state=seed)
                    m.fit(X_tr, y_tr)
                    pv, pt = m.predict(X_val), m.predict(X_te)
                    row = {"seed": seed, "cls": c, "class_name": CLASS_NAMES[c], "n_real": n,
                           "method": method, "n_syn": len(S),
                           "recall_val": float((pv[y_val == c] == c).mean()),
                           "recall_test": float((pt[y_te == c] == c).mean()),
                           "macro_f1_test": float(f1_score(y_te, pt, average="macro")),
                           "c2st_auc": c2st_auc(held, S, seed) if len(S) else "",
                           "coverage": coverage(held, S, seed=seed) if len(S) else "",
                           "gan_seconds": round(g_s, 1), "fit_seconds": round(time.time() - t0, 1)}
                    with open(out, "a", newline="") as f:
                        csv.DictWriter(f, FIELDS).writerow(row)
                    print(f"[lc] s{seed} {CLASS_NAMES[c]:<5} n={n:<5} {method:<5} "
                          f"recall val {row['recall_val']:.3f} test {row['recall_test']:.3f} "
                          f"c2st {row['c2st_auc'] if row['c2st_auc'] == '' else round(row['c2st_auc'], 3)}")
    print(f"\n[lc] done -> {out}")


if __name__ == "__main__":
    main()
