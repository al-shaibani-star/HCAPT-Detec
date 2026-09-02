"""
FIDMF Re-implementation (compact, for fair comparison).

A faithful but minimal re-implementation of FIDMF
[AlHayan & Al-Muhtadi, Scientific Reports 2026]:
  * Attention-LSTM detection core
  * SMOTE pre-balancing
  * GAN augmentation (vanilla, not WGAN-GP)
  * NO LLM (DistilBERT/GPT-2) — too heavy, focus on architectural parity

Run on our cleaned datasets to obtain apples-to-apples baseline.

Writes:
  results/_fidmf_reimpl/<dataset>_fidmf_results.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader
from .phase2_pipeline import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR   = STRATEGY_ROOT / "results" / "_fidmf_reimpl"
DATA_ROOT     = STRATEGY_ROOT.parent.parent / "data"


# =============================================================================
# Attention-LSTM core (FIDMF)
# =============================================================================
class AttentionLSTM(nn.Module):
    """Attention-LSTM as described in FIDMF.

    Input: tabular feature vector (B, d)
    Treat features as a length-d sequence with 1-channel each.
    Output: binary or K-class logits.
    """
    def __init__(self, feature_dim: int, hidden: int = 96,
                  num_classes: int = 2, lstm_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        # LSTM expects (B, L, input_size). We use input_size=1, L=feature_dim.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                             num_layers=lstm_layers,
                             batch_first=True, dropout=0.42)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) -> (B, d, 1)
        x = x.unsqueeze(-1)
        h, _ = self.lstm(x)                  # (B, d, hidden)
        # Attention over the d-length sequence
        attn_logits = self.attn(h)            # (B, d, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        context = (attn_weights * h).sum(dim=1)  # (B, hidden)
        return self.head(context)            # (B, num_classes)


# =============================================================================
# Compact GAN (vanilla, matches FIDMF's GAN choice)
# =============================================================================
class CompactGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, feature_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128),              nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim),      nn.Tanh(),
        )

    def forward(self, z, y):
        e = self.emb(y)
        return self.net(torch.cat([z, e], dim=1))


class CompactDiscriminator(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 16, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128),               nn.LeakyReLU(0.2),
            nn.Linear(128, 1),                  nn.Sigmoid(),
        )

    def forward(self, x, y):
        e = self.emb(y)
        return self.net(torch.cat([x, e], dim=1))


def train_compact_gan(X: np.ndarray, y: np.ndarray,
                       num_classes: int, epochs: int = 20,
                       latent_dim: int = 100, batch_size: int = 128) -> CompactGenerator:
    G = CompactGenerator(latent_dim, num_classes, X.shape[1]).to(DEVICE)
    D = CompactDiscriminator(num_classes, X.shape[1]).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    N = X_t.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            real = X_t[idx]; real_y = y_t[idx]
            bs = real.size(0)
            # train D
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_y = torch.randint(0, num_classes, (bs,), device=DEVICE)
            fake = G(z, fake_y).detach()
            ones = torch.ones(bs, 1, device=DEVICE)
            zeros = torch.zeros(bs, 1, device=DEVICE)
            d_loss = bce(D(real, real_y), ones) + bce(D(fake, fake_y), zeros)
            optD.zero_grad(); d_loss.backward(); optD.step()
            # train G
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_y = torch.randint(0, num_classes, (bs,), device=DEVICE)
            fake = G(z, fake_y)
            g_loss = bce(D(fake, fake_y), ones)
            optG.zero_grad(); g_loss.backward(); optG.step()
        if ep % 5 == 0:
            print(f"   [GAN] epoch {ep+1}/{epochs}  d_loss={d_loss.item():.3f}  g_loss={g_loss.item():.3f}")
    return G


def gan_balance_compact(X, y, G, num_classes, latent_dim=100):
    counts = np.bincount(y, minlength=num_classes)
    target = int(counts.max())
    G.eval()
    Xs, ys = [X], [y]
    with torch.no_grad():
        for c in range(num_classes):
            need = max(0, target - int(counts[c]))
            if need <= 0: continue
            z = torch.randn(need, latent_dim, device=DEVICE)
            yc = torch.full((need,), c, dtype=torch.long, device=DEVICE)
            xs = G(z, yc).cpu().numpy()
            Xs.append(xs); ys.append(np.full(need, c, dtype=int))
    Xb = np.concatenate(Xs, axis=0).astype(np.float32)
    yb = np.concatenate(ys, axis=0).astype(int)
    perm = np.random.permutation(len(yb))
    return Xb[perm], yb[perm]


# =============================================================================
# Train Attention-LSTM on (optionally) augmented data
# =============================================================================
def train_attlstm(X_tr, y_tr, X_va, y_va, num_classes,
                   epochs: int = 15, lr: float = 5e-4):
    model = AttentionLSTM(X_tr.shape[1], num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long,    device=DEVICE)
    N, bs = len(y_tr), 256
    best_va_f1 = 0.0; best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N - bs, bs):
            idx = perm[i:i+bs]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        # Val
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_va, dtype=torch.float32, device=DEVICE)).argmax(dim=1).cpu().numpy()
        m_va = compute_metrics(y_va, preds, is_binary=(num_classes == 2))
        if m_va["f1"] > best_va_f1:
            best_va_f1 = m_va["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model


def predict_attlstm(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        # Mini-batch inference for memory
        preds = []
        for i in range(0, len(X_t), 1024):
            preds.append(model(X_t[i:i+1024]).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# =============================================================================
# Main runner
# =============================================================================
def run_fidmf_on(loader_fn, ds_name: str) -> Dict:
    print(f"\n{'='*72}\n  FIDMF re-impl: {ds_name}\n{'='*72}")
    ds = loader_fn()
    num_classes = len(ds.class_names)
    X_tr, X_va, y_tr, y_va = train_test_split(
        ds.X_train, ds.y_train, test_size=0.1, random_state=42, stratify=ds.y_train,
    )

    results = {}

    # 1. Attention-LSTM only (baseline)
    print("\n[1] Attention-LSTM only")
    t0 = time.time()
    model = train_attlstm(X_tr, y_tr, X_va, y_va, num_classes, epochs=12)
    preds = predict_attlstm(model, ds.X_test)
    results["attlstm_only"] = compute_metrics(ds.y_test, preds, is_binary=(num_classes == 2))
    results["attlstm_only"]["train_time_s"] = round(time.time() - t0, 1)
    print(f"   acc={results['attlstm_only']['accuracy']:.4f} f1={results['attlstm_only']['f1']:.4f}")

    # 2. SMOTE + Attention-LSTM
    print("\n[2] SMOTE + Attention-LSTM")
    t0 = time.time()
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_tr, y_tr)
    model2 = train_attlstm(X_smote.astype(np.float32), y_smote.astype(int),
                            X_va, y_va, num_classes, epochs=12)
    preds = predict_attlstm(model2, ds.X_test)
    results["smote_attlstm"] = compute_metrics(ds.y_test, preds, is_binary=(num_classes == 2))
    results["smote_attlstm"]["train_time_s"] = round(time.time() - t0, 1)
    print(f"   acc={results['smote_attlstm']['accuracy']:.4f} f1={results['smote_attlstm']['f1']:.4f}")

    # 3. SMOTE + GAN + Attention-LSTM (FIDMF full minus LLM)
    print("\n[3] SMOTE + GAN + Attention-LSTM (FIDMF, no LLM)")
    t0 = time.time()
    G = train_compact_gan(X_smote.astype(np.float32), y_smote.astype(int),
                           num_classes, epochs=15)
    X_aug, y_aug = gan_balance_compact(X_smote.astype(np.float32),
                                         y_smote.astype(int), G, num_classes)
    model3 = train_attlstm(X_aug, y_aug, X_va, y_va, num_classes, epochs=12)
    preds = predict_attlstm(model3, ds.X_test)
    results["smote_gan_attlstm"] = compute_metrics(ds.y_test, preds, is_binary=(num_classes == 2))
    results["smote_gan_attlstm"]["train_time_s"] = round(time.time() - t0, 1)
    print(f"   acc={results['smote_gan_attlstm']['accuracy']:.4f} f1={results['smote_gan_attlstm']['f1']:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{ds_name}_fidmf.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": ds_name, "results": results}, f, indent=2)
    print(f"\n[done] wrote {out_path.name}")
    return results


def main():
    summary = {}
    for ds_name, loader in [
        ("nsl-kdd",           lambda: NslKddLoader(DATA_ROOT).load("binary")),
        ("ton-iot-telemetry", lambda: TonIotTelemetryLoader(DATA_ROOT).load("binary")),
        ("edge-iiotset",      lambda: EdgeIiotsetLoader(DATA_ROOT).load("binary")),
        ("dapt-2020",         lambda: Dapt2020Loader(DATA_ROOT).load("binary")),
    ]:
        try:
            summary[ds_name] = run_fidmf_on(loader, ds_name)
        except Exception as e:
            print(f"   FAILED: {e}")
            summary[ds_name] = {"error": str(e)}

    print("\n" + "="*72)
    print("FIDMF re-implementation: head-to-head with HCAPT")
    print("="*72)
    print(f"{'dataset':<22}  {'FIDMF (full)':>15}  (acc, f1)")
    for ds_name, r in summary.items():
        if "error" in r: continue
        m = r["smote_gan_attlstm"]
        print(f"{ds_name:<22}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")


if __name__ == "__main__":
    main()
