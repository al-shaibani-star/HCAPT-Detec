"""
HCAPT-Defense - Cache & Checkpoint System
Protects against mid-training crashes / shutdowns.

Features:
  - preprocessed_data(): reuse NSL-KDD preprocessing across runs
  - checkpoint management for GAN / NN training
  - atomic file writes (tmp then rename) so partial files never appear valid
"""
import json
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import numpy as np
import torch

from config import RESULTS_DIR, MODELS_DIR

CACHE_DIR = RESULTS_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  Atomic file ops
# ============================================================
def _atomic_save_npz(path: Path, **arrays):
    # numpy.savez auto-appends .npz if not present - so tmp MUST end with .npz
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _atomic_save_torch(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_save_json(path: Path, data: Dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# ============================================================
#  Preprocessed data cache
# ============================================================
def _data_fingerprint(train_path: Path, test_path: Path) -> str:
    """Hash of file sizes + mtimes - invalidates if source files change."""
    h = hashlib.sha256()
    for p in (train_path, test_path):
        stat = p.stat()
        h.update(f"{p.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:16]


def cached_preprocessed_data(train_path: Path, test_path: Path,
                             process_fn) -> Tuple[np.ndarray, ...]:
    """Load cached preprocessed arrays, or process + cache them.

    process_fn() must return (X_train, X_test, y_train, y_test).
    """
    fp = _data_fingerprint(train_path, test_path)
    cache_file = CACHE_DIR / f"preprocessed_{fp}.npz"

    if cache_file.exists():
        print(f"[cache] HIT: loading preprocessed data from {cache_file.name}")
        data = np.load(cache_file)
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    print(f"[cache] MISS: preprocessing from source files")
    X_tr, X_te, y_tr, y_te = process_fn()
    _atomic_save_npz(
        cache_file,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
    )
    print(f"[cache] SAVED: {cache_file.name} "
          f"({cache_file.stat().st_size / 1e6:.1f} MB)")
    return X_tr, X_te, y_tr, y_te


# ============================================================
#  Training checkpoints
# ============================================================
def save_checkpoint(name: str, epoch: int, state: Dict):
    """Save training state (model weights, optimizer, epoch, metrics).

    Keeps latest + best (if state has 'is_best': True).
    """
    latest = CHECKPOINT_DIR / f"{name}_latest.pt"
    _atomic_save_torch(latest, {"epoch": epoch, **state})

    if state.get("is_best", False):
        best = CHECKPOINT_DIR / f"{name}_best.pt"
        _atomic_save_torch(best, {"epoch": epoch, **state})


def load_checkpoint(name: str, which: str = "latest") -> Optional[Dict]:
    """Return checkpoint dict, or None if not found."""
    path = CHECKPOINT_DIR / f"{name}_{which}.pt"
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[cache] Resumed {name} from epoch {ckpt.get('epoch', '?')}")
        return ckpt
    except Exception as e:
        print(f"[cache] Failed to load {path}: {e}")
        return None


def checkpoint_exists(name: str, which: str = "latest") -> bool:
    return (CHECKPOINT_DIR / f"{name}_{which}.pt").exists()


# ============================================================
#  Phase tracker - records which pipeline stages are done
# ============================================================
PHASES_FILE = CACHE_DIR / "phases.json"


def get_phases() -> Dict:
    if PHASES_FILE.exists():
        try:
            with open(PHASES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def mark_phase_done(phase: str, info: Optional[Dict] = None):
    """Record that a pipeline stage completed successfully."""
    phases = get_phases()
    phases[phase] = {"done": True, "info": info or {}}
    _atomic_save_json(PHASES_FILE, phases)
    print(f"[cache] Phase '{phase}' marked done")


def is_phase_done(phase: str) -> bool:
    return get_phases().get(phase, {}).get("done", False)


def clear_cache(confirm: bool = False):
    """Wipe everything (careful)."""
    if not confirm:
        print("[cache] Pass confirm=True to actually clear")
        return
    for d in (CACHE_DIR, CHECKPOINT_DIR):
        if d.exists():
            shutil.rmtree(d)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("[cache] Cleared all caches and checkpoints")


if __name__ == "__main__":
    # Show current cache state
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"\nCurrent phases:")
    for k, v in get_phases().items():
        print(f"  {k}: {v}")
    print(f"\nCache files:")
    for p in sorted(CACHE_DIR.glob("*")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\nCheckpoints:")
    for p in sorted(CHECKPOINT_DIR.glob("*.pt")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
