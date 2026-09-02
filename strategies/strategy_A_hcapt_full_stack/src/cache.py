"""
Strategy A — Dataset-Aware Cache Manager

Builds on the project-level cache pattern (../../../src/cache.py) but
isolates each dataset under its own root so multiple datasets can be
cached independently and re-runs of one dataset never affect another.

Key features:
  - Per-dataset roots:           cache/<dataset>/{models,artifacts}
  - Per-dataset status:          cache/<dataset>/status.json
  - Atomic writes:               *.tmp + rename
  - Hash-based invalidation:     config_hash + data_fingerprint
  - Coarse-grained resume:       step level (preprocess, gan, fl, ...)
"""
from __future__ import annotations
import json
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import numpy as np

# Strategy root = parent of src/ in this folder
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT    = STRATEGY_ROOT / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Atomic file writers
# =============================================================================
def _atomic_write_json(path: Path, data: Dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def _atomic_save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _atomic_save_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(path)


# =============================================================================
# DatasetCache: one instance per (dataset_name)
# =============================================================================
class DatasetCache:
    """Owns the cache for a single dataset. All paths are dataset-scoped."""

    STATUS_FILE = "status.json"

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.root: Path     = CACHE_ROOT / dataset_name
        self.preproc_dir    = self.root / "preprocessed"
        self.gan_dir        = self.root / "gan"
        self.ensemble_dir   = self.root / "ensemble"
        self.fl_dir         = self.root / "fl"
        self.bc_dir         = self.root / "bc"
        self.dp_dir         = self.root / "dp"
        self.zt_dir         = self.root / "zt"
        self.configs_dir    = self.root / "configs"
        for d in (self.preproc_dir, self.gan_dir, self.ensemble_dir,
                  self.fl_dir, self.bc_dir, self.dp_dir, self.zt_dir,
                  self.configs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- status
    def _status_path(self) -> Path:
        return self.root / self.STATUS_FILE

    def status(self) -> Dict[str, Any]:
        p = self._status_path()
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def is_done(self, step: str) -> bool:
        return bool(self.status().get(step, {}).get("done", False))

    def mark_done(self, step: str, info: Optional[Dict] = None):
        s = self.status()
        s[step] = {
            "done": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "info": info or {},
        }
        _atomic_write_json(self._status_path(), s)
        print(f"[cache:{self.dataset_name}] step '{step}' done")

    def mark_failed(self, step: str, err: str):
        s = self.status()
        s[step] = {
            "done": False,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": err[:500],
        }
        _atomic_write_json(self._status_path(), s)

    # ------------------------------------------------------------- artifacts
    def save_npz(self, subdir: str, name: str, **arrays):
        path = self.root / subdir / f"{name}.npz"
        _atomic_save_npz(path, **arrays)
        return path

    def load_npz(self, subdir: str, name: str) -> Optional[Dict[str, np.ndarray]]:
        path = self.root / subdir / f"{name}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def artifact_path(self, subdir: str, filename: str) -> Path:
        return self.root / subdir / filename

    def has_artifact(self, subdir: str, filename: str) -> bool:
        return self.artifact_path(subdir, filename).exists()

    # ---------------------------------------------------------------- json
    def save_json(self, subdir: str, name: str, data: Dict):
        path = self.root / subdir / f"{name}.json"
        _atomic_write_json(path, data)
        return path

    def load_json(self, subdir: str, name: str) -> Optional[Dict]:
        path = self.root / subdir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------------------- clear
    def clear(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.__init__(self.dataset_name)
        print(f"[cache:{self.dataset_name}] cleared")


# =============================================================================
# Hash helpers
# =============================================================================
def hash_file(path: Path) -> str:
    """SHA-256 hash of a file's size + mtime (cheap, good enough for cache invalidation)."""
    if not path.exists():
        return ""
    st = path.stat()
    return hashlib.sha256(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def hash_files(paths) -> str:
    """Combined hash of a list of files."""
    h = hashlib.sha256()
    for p in paths:
        h.update(hash_file(p).encode())
    return h.hexdigest()[:16]


def hash_dict(d: Dict) -> str:
    """Stable hash of a config dictionary."""
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# =============================================================================
# Cached step decorator
# =============================================================================
def cached_step(cache: DatasetCache, step_name: str, force: bool = False):
    """Decorator: skip a step if already done (unless force=True).

    Usage:
        @cached_step(cache, "preprocess")
        def preprocess(...):
            ...

    The wrapped function is responsible for saving its artifacts. The
    decorator only records "done" status after the function returns
    successfully.
    """
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            if not force and cache.is_done(step_name):
                print(f"[cache:{cache.dataset_name}] SKIP '{step_name}' (already done)")
                return None  # caller should reload from disk
            print(f"[cache:{cache.dataset_name}] RUN  '{step_name}'")
            try:
                t0 = time.time()
                result = fn(*args, **kwargs)
                cache.mark_done(step_name, info={"duration_s": round(time.time()-t0, 1)})
                return result
            except Exception as e:
                cache.mark_failed(step_name, str(e))
                raise
        return wrapper
    return decorator


# =============================================================================
# Quick CLI inspector
# =============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ds = sys.argv[1]
        c = DatasetCache(ds)
        print(f"Cache root: {c.root}")
        print(f"Status:")
        for k, v in c.status().items():
            mark = "✓" if v.get("done") else "✗"
            print(f"  [{mark}] {k}  {v.get('timestamp', v.get('failed_at', ''))}")
    else:
        print(f"Cache root: {CACHE_ROOT}")
        for ds_dir in sorted(CACHE_ROOT.iterdir()):
            if ds_dir.is_dir():
                c = DatasetCache(ds_dir.name)
                done = sum(1 for v in c.status().values() if v.get("done"))
                total = len(c.status())
                print(f"  {ds_dir.name}: {done}/{total} steps done")
