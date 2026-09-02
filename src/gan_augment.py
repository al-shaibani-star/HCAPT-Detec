"""
HCAPT-Defense — Innovation #1: Generative AI Attack Simulator
Layer 2: GAN-based synthetic attack sample generation

Uses a Conditional GAN (CGAN) to generate synthetic samples
for minority attack classes (R2L, U2R) — addressing severe
class imbalance in NSL-KDD.

Architecture:
  Generator: noise + class_label -> synthetic feature vector
  Discriminator: feature vector + class_label -> real/fake score
"""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, SEED,
    GAN_LATENT_DIM, GAN_EPOCHS, GAN_LR, GAN_BETA1,
    NUM_CLASSES, LABEL_MAP,
)
from cache import save_checkpoint, load_checkpoint

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
#  Generator
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
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = self.label_embed(labels)
        x = torch.cat([z, e], dim=1)
        return self.net(x)


# ============================================================
#  Discriminator
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, embed_dim: int = 32):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = self.label_embed(labels)
        x = torch.cat([x, e], dim=1)
        return self.net(x)  # raw logits (BCEWithLogitsLoss)


# ============================================================
#  Training — Conditional GAN
# ============================================================
def train_cgan(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = GAN_EPOCHS,
    batch_size: int = 128,
    save_path: Path = None,
    checkpoint_every: int = 10,
    resume: bool = True,
) -> tuple:
    """Train CGAN on (X, y). Returns (generator, discriminator, history).

    Checkpoints every `checkpoint_every` epochs so that a crash mid-training
    only loses up to `checkpoint_every` epochs of work.
    """
    feature_dim = X.shape[1]
    num_classes = NUM_CLASSES

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size,
        shuffle=True, drop_last=True,
    )

    G = Generator(GAN_LATENT_DIM, num_classes, feature_dim).to(DEVICE)
    D = Discriminator(feature_dim, num_classes).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    history = {"d_loss": [], "g_loss": []}
    start_epoch = 0

    # ----- Resume from checkpoint if present -----
    if resume:
        ckpt = load_checkpoint("gan", which="latest")
        if ckpt is not None and ckpt.get("feature_dim") == feature_dim:
            G.load_state_dict(ckpt["G"])
            D.load_state_dict(ckpt["D"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            history = ckpt.get("history", history)
            start_epoch = ckpt["epoch"] + 1
            print(f"[GAN] Resumed from epoch {start_epoch} "
                  f"(skipping {start_epoch} of {epochs})")
            if start_epoch >= epochs:
                print(f"[GAN] Already trained {epochs} epochs - using existing")
                return G, D, history

    print(f"[GAN] Training on {len(X)} samples | feature_dim={feature_dim}")
    print(f"[GAN] Device: {DEVICE} | Epochs: {start_epoch}->{epochs} "
          f"| Batch: {batch_size} | Checkpoint every: {checkpoint_every}")

    for epoch in range(start_epoch, epochs):
        d_running, g_running, n_batches = 0.0, 0.0, 0

        for real_X, real_y in loader:
            bs = real_X.size(0)
            real_X = real_X.to(DEVICE, non_blocking=True)
            real_y = real_y.to(DEVICE, non_blocking=True)

            real_labels = torch.ones(bs, 1, device=DEVICE)
            fake_labels = torch.zeros(bs, 1, device=DEVICE)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            d_real = D(real_X, real_y)
            loss_d_real = bce(d_real, real_labels)

            z = torch.randn(bs, GAN_LATENT_DIM, device=DEVICE)
            fake_y = torch.randint(0, num_classes, (bs,), device=DEVICE)
            fake_X = G(z, fake_y)
            d_fake = D(fake_X.detach(), fake_y)
            loss_d_fake = bce(d_fake, fake_labels)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            z = torch.randn(bs, GAN_LATENT_DIM, device=DEVICE)
            fake_y = torch.randint(0, num_classes, (bs,), device=DEVICE)
            fake_X = G(z, fake_y)
            d_fake = D(fake_X, fake_y)
            loss_g = bce(d_fake, real_labels)  # fool D
            loss_g.backward()
            opt_G.step()

            d_running += loss_d.item()
            g_running += loss_g.item()
            n_batches += 1

        d_avg = d_running / max(n_batches, 1)
        g_avg = g_running / max(n_batches, 1)
        history["d_loss"].append(d_avg)
        history["g_loss"].append(g_avg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"D_loss={d_avg:.4f} | G_loss={g_avg:.4f}")

        # Checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            save_checkpoint("gan", epoch, {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "history": history,
                "feature_dim": feature_dim,
            })

    if save_path is None:
        save_path = MODELS_DIR
    torch.save(G.state_dict(), save_path / "gan_generator.pt")
    torch.save(D.state_dict(), save_path / "gan_discriminator.pt")
    print(f"[GAN] Saved final models to {save_path}")

    return G, D, history


# ============================================================
#  Generate synthetic samples for minority classes
# ============================================================
@torch.no_grad()
def generate_synthetic(
    G: Generator,
    num_per_class: dict,
    batch: int = 512,
) -> tuple:
    """Generate synthetic samples. num_per_class: {class_idx: count}."""
    G.eval()
    X_syn, y_syn = [], []

    for cls_idx, n in num_per_class.items():
        if n <= 0:
            continue
        generated = 0
        while generated < n:
            this_batch = min(batch, n - generated)
            z = torch.randn(this_batch, GAN_LATENT_DIM, device=DEVICE)
            labels = torch.full((this_batch,), cls_idx,
                                dtype=torch.long, device=DEVICE)
            fake = G(z, labels).cpu().numpy()
            X_syn.append(fake)
            y_syn.append(np.full(this_batch, cls_idx, dtype=np.int64))
            generated += this_batch

    X_syn = np.concatenate(X_syn, axis=0) if X_syn else np.empty((0, 0))
    y_syn = np.concatenate(y_syn, axis=0) if y_syn else np.empty((0,))
    print(f"[GAN] Generated {len(X_syn)} synthetic samples across "
          f"{sum(1 for v in num_per_class.values() if v > 0)} classes")
    return X_syn, y_syn


def balance_dataset(X: np.ndarray, y: np.ndarray, G: Generator,
                    target_size: int = None) -> tuple:
    """Use GAN to balance minority classes to match majority."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    if target_size is None:
        target_size = counts.max()

    to_generate = {cls: max(0, target_size - int(counts[cls]))
                   for cls in range(NUM_CLASSES)}

    print(f"[GAN] Original counts: {dict(enumerate(counts.tolist()))}")
    print(f"[GAN] Target per class: {target_size}")
    print(f"[GAN] Will generate:   {to_generate}")

    X_syn, y_syn = generate_synthetic(G, to_generate)

    if len(X_syn) > 0:
        X_balanced = np.concatenate([X, X_syn], axis=0)
        y_balanced = np.concatenate([y, y_syn], axis=0)
    else:
        X_balanced, y_balanced = X, y

    # Shuffle
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y_balanced))
    return X_balanced[idx], y_balanced[idx]


# ============================================================
#  Main
# ============================================================
def main():
    from data_loader import load_raw, map_attack_category, encode_and_scale

    print("="*60)
    print("  HCAPT-Defense — Innovation #1: GAN Training")
    print("="*60)

    df_train, df_test = load_raw()
    df_train = map_attack_category(df_train)
    df_test = map_attack_category(df_test)
    X_tr, X_te, y_tr, y_te = encode_and_scale(df_train, df_test)

    t0 = time.time()
    G, D, hist = train_cgan(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"\n[GAN] Total training time: {train_time:.1f}s")

    # Demo: balance the dataset
    X_bal, y_bal = balance_dataset(X_tr, y_tr, G)
    print(f"\n[GAN] Balanced dataset shape: {X_bal.shape}")
    print(f"[GAN] Balanced class counts: "
          f"{np.bincount(y_bal, minlength=NUM_CLASSES).tolist()}")

    # Save balanced data for downstream ensemble training
    np.savez(
        RESULTS_DIR / "balanced_data.npz",
        X=X_bal, y=y_bal, X_test=X_te, y_test=y_te,
    )
    print(f"\n[OK] Balanced data saved to {RESULTS_DIR / 'balanced_data.npz'}")


if __name__ == "__main__":
    main()
