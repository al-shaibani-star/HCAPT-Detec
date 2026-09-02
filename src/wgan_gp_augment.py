"""
HCAPT-Defense - Innovation #1 (improved): Conditional WGAN-GP
  Wasserstein GAN with Gradient Penalty for stable attack-sample generation.

Why this exists:
  Our first CGAN hit Discriminator Dominance (D_loss -> 0, G_loss -> 10+).
  WGAN-GP addresses this by:
    1. Replacing BCE with Wasserstein (Earth-Mover) distance
    2. Removing sigmoid - the Critic outputs a raw score
    3. Enforcing 1-Lipschitz via gradient penalty (no weight clipping)
    4. Training Critic n_critic=5 times per Generator step
    5. Using Adam(beta1=0, beta2=0.9) instead of beta1=0.5
    6. Removing BatchNorm from Critic (interferes with per-sample gradients)

Reference: Gulrajani et al. 2017 "Improved Training of Wasserstein GANs"
"""
import time
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED,
    GAN_LATENT_DIM, NUM_CLASSES, LABEL_MAP,
)
from cache import save_checkpoint, load_checkpoint

torch.manual_seed(SEED)
np.random.seed(SEED)

# WGAN-GP specific
N_CRITIC = 5                # critic updates per generator update
GP_LAMBDA = 10.0            # gradient penalty weight
WGAN_LR = 1e-4              # lower than CGAN's 2e-4
WGAN_BETA1 = 0.0            # key difference from CGAN
WGAN_BETA2 = 0.9


# ============================================================
#  Generator (same topology as before, still OK for WGAN-GP)
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, feature_dim: int,
                 embed_dim: int = 32):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, feature_dim),
            # No Tanh - WGAN features are unbounded; features are scaled anyway
        )

    def forward(self, z, labels):
        e = self.label_embed(labels)
        return self.net(torch.cat([z, e], dim=1))


# ============================================================
#  Critic (NOT discriminator) - no BN, no sigmoid
# ============================================================
class Critic(nn.Module):
    """Outputs a scalar 'realness score' - higher = more real."""
    def __init__(self, feature_dim: int, num_classes: int, embed_dim: int = 32):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        # LayerNorm replaces BatchNorm (BN interferes with gradient penalty,
        # LayerNorm works per-sample).
        self.net = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
            # No sigmoid - raw score
        )

    def forward(self, x, labels):
        e = self.label_embed(labels)
        return self.net(torch.cat([x, e], dim=1))


# ============================================================
#  Gradient Penalty (the heart of WGAN-GP)
# ============================================================
def compute_gradient_penalty(critic: Critic, real: torch.Tensor,
                             fake: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
    """Enforce 1-Lipschitz by penalizing gradient norm deviation from 1."""
    bs = real.size(0)
    alpha = torch.rand(bs, 1, device=DEVICE)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)

    d_interp = critic(interp, labels)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp


# ============================================================
#  Training
# ============================================================
def train_wgan_gp(X: np.ndarray, y: np.ndarray,
                  epochs: int = 100, batch_size: int = 128,
                  n_critic: int = N_CRITIC, lambda_gp: float = GP_LAMBDA,
                  checkpoint_every: int = 10, resume: bool = True) -> Tuple:
    feature_dim = X.shape[1]

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size,
        shuffle=True, drop_last=True,
    )

    G = Generator(GAN_LATENT_DIM, NUM_CLASSES, feature_dim).to(DEVICE)
    C = Critic(feature_dim, NUM_CLASSES).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=WGAN_LR,
                              betas=(WGAN_BETA1, WGAN_BETA2))
    opt_C = torch.optim.Adam(C.parameters(), lr=WGAN_LR,
                              betas=(WGAN_BETA1, WGAN_BETA2))

    history = {"c_loss": [], "g_loss": [], "gp": [], "wasserstein": []}
    start_epoch = 0

    # ----- Resume -----
    if resume:
        ckpt = load_checkpoint("wgan_gp", which="latest")
        if ckpt is not None and ckpt.get("feature_dim") == feature_dim:
            G.load_state_dict(ckpt["G"])
            C.load_state_dict(ckpt["C"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_C.load_state_dict(ckpt["opt_C"])
            history = ckpt.get("history", history)
            start_epoch = ckpt["epoch"] + 1
            print(f"[WGAN-GP] Resumed from epoch {start_epoch}")
            if start_epoch >= epochs:
                print(f"[WGAN-GP] Already at target epochs - using existing")
                return G, C, history

    print(f"[WGAN-GP] feature_dim={feature_dim} | epochs={start_epoch}->{epochs}")
    print(f"[WGAN-GP] n_critic={n_critic} | GP lambda={lambda_gp} "
          f"| lr={WGAN_LR} | betas=({WGAN_BETA1}, {WGAN_BETA2})")

    data_iter = iter(loader)

    for epoch in range(start_epoch, epochs):
        c_running, g_running, gp_running, w_running = 0.0, 0.0, 0.0, 0.0
        n_c, n_g = 0, 0

        for step, (real_X, real_y) in enumerate(loader):
            real_X = real_X.to(DEVICE, non_blocking=True)
            real_y = real_y.to(DEVICE, non_blocking=True)
            bs = real_X.size(0)

            # --- 1. Train Critic (n_critic times total; we interleave) ---
            opt_C.zero_grad()
            z = torch.randn(bs, GAN_LATENT_DIM, device=DEVICE)
            fake_y = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            fake_X = G(z, fake_y).detach()

            d_real = C(real_X, real_y).mean()
            d_fake = C(fake_X, fake_y).mean()

            # NOTE: GP must be computed on same-class real/fake pairs, not
            # mismatched ones. Use real_y for both to satisfy this.
            z2 = torch.randn(bs, GAN_LATENT_DIM, device=DEVICE)
            fake_matched = G(z2, real_y).detach()
            gp = compute_gradient_penalty(C, real_X, fake_matched, real_y)

            c_loss = d_fake - d_real + lambda_gp * gp
            c_loss.backward()
            opt_C.step()

            wasserstein = (d_real - d_fake).item()
            c_running += c_loss.item()
            gp_running += gp.item()
            w_running += wasserstein
            n_c += 1

            # --- 2. Train Generator every n_critic steps ---
            if (step + 1) % n_critic == 0:
                opt_G.zero_grad()
                z = torch.randn(bs, GAN_LATENT_DIM, device=DEVICE)
                fake_y = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
                fake_X = G(z, fake_y)
                g_loss = -C(fake_X, fake_y).mean()
                g_loss.backward()
                opt_G.step()

                g_running += g_loss.item()
                n_g += 1

        c_avg = c_running / max(n_c, 1)
        g_avg = g_running / max(n_g, 1)
        gp_avg = gp_running / max(n_c, 1)
        w_avg = w_running / max(n_c, 1)
        history["c_loss"].append(c_avg)
        history["g_loss"].append(g_avg)
        history["gp"].append(gp_avg)
        history["wasserstein"].append(w_avg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | C={c_avg:+.4f} "
                  f"G={g_avg:+.4f} GP={gp_avg:.4f} W_dist={w_avg:+.4f}")

        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            save_checkpoint("wgan_gp", epoch, {
                "G": G.state_dict(), "C": C.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_C": opt_C.state_dict(),
                "history": history, "feature_dim": feature_dim,
            })

    torch.save(G.state_dict(), MODELS_DIR / "wgan_gp_generator.pt")
    torch.save(C.state_dict(), MODELS_DIR / "wgan_gp_critic.pt")
    print(f"[WGAN-GP] Saved final models to {MODELS_DIR}")
    return G, C, history


# ============================================================
#  Generate synthetic samples + balance dataset
# ============================================================
@torch.no_grad()
def generate_synthetic(G: Generator, num_per_class: Dict[int, int],
                       batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    G.eval()
    X_syn, y_syn = [], []
    for cls, n in num_per_class.items():
        if n <= 0:
            continue
        generated = 0
        while generated < n:
            this_batch = min(batch, n - generated)
            z = torch.randn(this_batch, GAN_LATENT_DIM, device=DEVICE)
            labels = torch.full((this_batch,), cls, dtype=torch.long,
                                device=DEVICE)
            fake = G(z, labels).cpu().numpy()
            X_syn.append(fake)
            y_syn.append(np.full(this_batch, cls, dtype=np.int64))
            generated += this_batch
    if not X_syn:
        return np.empty((0,)), np.empty((0,))
    X_out = np.concatenate(X_syn, axis=0)
    y_out = np.concatenate(y_syn, axis=0)
    print(f"[WGAN-GP] Generated {len(X_out)} synthetic samples")
    return X_out, y_out


def balance_dataset(X: np.ndarray, y: np.ndarray, G: Generator,
                    target: int = None) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = target or int(counts.max())
    need = {c: max(0, target - int(counts[c])) for c in range(NUM_CLASSES)}
    print(f"[WGAN-GP] Counts: {dict(enumerate(counts.tolist()))}")
    print(f"[WGAN-GP] Target: {target} per class")
    print(f"[WGAN-GP] Will generate: {need}")
    X_s, y_s = generate_synthetic(G, need)
    if len(X_s) == 0:
        return X, y
    X_bal = np.concatenate([X, X_s], axis=0)
    y_bal = np.concatenate([y, y_s], axis=0)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


# ============================================================
#  Main
# ============================================================
def main():
    from data_loader import get_preprocessed

    print("=" * 60)
    print("  HCAPT-Defense - Innovation #1 (improved): WGAN-GP")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = get_preprocessed()
    print(f"[data] Train: {X_tr.shape} | Test: {X_te.shape}")

    t0 = time.time()
    G, C, hist = train_wgan_gp(X_tr, y_tr, epochs=100)
    print(f"\n[WGAN-GP] Total training time: {time.time()-t0:.1f}s")

    # Print loss comparison summary
    print(f"\n[WGAN-GP] Loss progression (last vs first):")
    print(f"  Critic loss:   {hist['c_loss'][0]:+.4f} -> {hist['c_loss'][-1]:+.4f}")
    print(f"  Generator loss:{hist['g_loss'][0]:+.4f} -> {hist['g_loss'][-1]:+.4f}")
    print(f"  Wasserstein distance: {hist['wasserstein'][0]:+.4f} -> "
          f"{hist['wasserstein'][-1]:+.4f}  (should decrease)")
    print(f"  Gradient penalty:     {hist['gp'][0]:.4f} -> {hist['gp'][-1]:.4f}"
          f"  (should stay near 0.1-1.0)")

    # Generate balanced set
    X_bal, y_bal = balance_dataset(X_tr, y_tr, G)
    np.savez(RESULTS_DIR.parent / "cache" / "wgangp_balanced_data.npz",
             X_bal=X_bal, y_bal=y_bal)

    counts = np.bincount(y_bal, minlength=NUM_CLASSES).tolist()
    print(f"\n[WGAN-GP] Balanced counts: {counts}")
    print(f"[OK] WGAN-GP training + balancing complete")


if __name__ == "__main__":
    main()
