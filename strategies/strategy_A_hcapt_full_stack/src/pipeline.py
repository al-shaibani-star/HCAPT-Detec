"""
Strategy A — Pipeline orchestrator.

Approach: re-use the project's existing, tested HCAPT-Defense modules
(`../../src/master_pipeline.py`, `evaluate_all_configurations.py`) but
inject dataset-specific paths so each dataset's cache and results live
under its own folder.

For NSL-KDD this delegates to the existing master_pipeline (which is
already idempotent / resumable via the project-level cache). For
ToN-IoT and Edge-IIoTset (Phase 2), the dataset adapter will need to
override `data_loader.get_preprocessed` — see the docstring of the
`Phase2Adapter` class at the bottom for migration notes.
"""
from __future__ import annotations
import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
STRATEGY_ROOT  = Path(__file__).resolve().parent.parent     # strategy_A_*/
IMPL_ROOT      = STRATEGY_ROOT.parent.parent                # 05-Implementation/
PROJECT_SRC    = IMPL_ROOT / "src"
DATA_ROOT      = IMPL_ROOT / "data"
PROJECT_MODELS = IMPL_ROOT / "models"
PROJECT_RESULTS= IMPL_ROOT / "results"

# Make project src/ importable
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from .cache import DatasetCache, hash_files, hash_dict
from .loaders.base import LoadedDataset
from .loaders.nsl_kdd import NslKddLoader
from .loaders.ton_iot_telemetry import TonIotTelemetryLoader
from .loaders.edge_iiotset import EdgeIiotsetLoader
from .loaders.dapt2020 import Dapt2020Loader


# -----------------------------------------------------------------------
# Loader registry
# -----------------------------------------------------------------------
def get_loader(dataset_name: str):
    if dataset_name == "nsl-kdd":
        return NslKddLoader(DATA_ROOT)
    elif dataset_name == "ton-iot-telemetry":
        return TonIotTelemetryLoader(DATA_ROOT)
    elif dataset_name == "edge-iiotset":
        return EdgeIiotsetLoader(DATA_ROOT)
    elif dataset_name == "dapt-2020":
        return Dapt2020Loader(DATA_ROOT)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =======================================================================
# Pipeline runner
# =======================================================================
class StrategyARunner:
    """Runs HCAPT-Defense Full Stack on a single dataset.

    Phase 1 (NSL-KDD): delegates to master_pipeline + evaluate_all_configurations.
    Phase 2 (ToN-IoT, Edge-IIoTset): full re-implementation using project src/ helpers
                                     called with explicit data passed in.
    """

    def __init__(self, dataset_name: str, force_rerun: bool = False):
        self.dataset_name = dataset_name
        self.cache = DatasetCache(dataset_name)
        self.force = force_rerun
        self.results_dir = STRATEGY_ROOT / "results" / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- run
    def run(self) -> Dict:
        """End-to-end run. Returns summary dict."""
        print("\n" + "#" * 76)
        print(f"#  Strategy A — HCAPT-Defense Full Stack on {self.dataset_name}")
        print("#" * 76)
        t0 = time.time()

        # Step 1: load dataset (this just verifies sources exist + records fingerprint)
        loader = get_loader(self.dataset_name)
        sources = loader.source_paths()
        fp = hash_files(sources)
        for s in sources:
            if not s.exists():
                raise FileNotFoundError(f"Source not found: {s}")
        print(f"[runner] Source fingerprint: {fp}")

        if self.dataset_name == "nsl-kdd":
            self._run_nsl_kdd_phase1(fp)
        else:
            self._run_phase2(loader, fp)

        # Step 2: copy artefacts into results/<dataset>/
        self._collect_results()

        elapsed = time.time() - t0
        print(f"\n[runner] Total elapsed: {elapsed:.1f}s")
        return {"dataset": self.dataset_name, "elapsed_s": round(elapsed, 1)}

    # ---------------------------------------------------------- NSL-KDD
    def _run_nsl_kdd_phase1(self, fp: str):
        """Delegates to existing master_pipeline + evaluate_all_configurations.

        The project-level cache (../../cache/) tracks individual phases
        (preprocess / GAN / FL / ensemble / IoC harvest / risk scorer),
        so re-running is already resumable. We just record the high-level
        progress in the strategy-level cache for visibility.
        """
        # Delegate to master_pipeline.main()
        if self.force or not self.cache.is_done("master_pipeline"):
            print("\n[runner] Phase A (master_pipeline)…")
            self._invoke("master_pipeline.py")
            self.cache.mark_done("master_pipeline", info={"fingerprint": fp})
        else:
            print("[runner] master_pipeline already done — skipping (use --force to re-run)")

        # Delegate to V2 improvements pipeline (SMOTE+Focal+Calibration)
        if self.force or not self.cache.is_done("master_pipeline_v2"):
            print("\n[runner] Phase A V2 (master_pipeline_v2)…")
            self._invoke("master_pipeline_v2.py")
            self.cache.mark_done("master_pipeline_v2", info={"fingerprint": fp})
        else:
            print("[runner] master_pipeline_v2 already done — skipping")

        # Run all 6 ablation configurations
        if self.force or not self.cache.is_done("evaluate_all"):
            print("\n[runner] Ablation evaluation (evaluate_all_configurations)…")
            self._invoke("evaluate_all_configurations.py")
            self.cache.mark_done("evaluate_all", info={"fingerprint": fp})
        else:
            print("[runner] evaluate_all already done — skipping")

    # -------------------------------------------------------------- Phase 2
    def _run_phase2(self, loader, fp: str):
        """Self-contained pipeline for ToN-IoT, Edge-IIoTset, and any new dataset."""
        from .phase2_pipeline import Phase2Runner
        if self.force or not self.cache.is_done("phase2_loaded"):
            print(f"\n[runner] Loading {self.dataset_name}…")
            ds = loader.load(classification="binary")
            print(f"[runner] {ds.summary()}")
            self.cache.mark_done("phase2_loaded",
                                 info={"shape_train": list(ds.X_train.shape),
                                        "shape_test": list(ds.X_test.shape),
                                        "fingerprint": fp})
        else:
            print("[runner] dataset load already recorded — re-loading anyway (cheap)")
            ds = loader.load(classification="binary")

        runner = Phase2Runner(ds, self.cache, self.results_dir)
        runner.run_all()
        self.cache.mark_done("phase2_pipeline_complete")

    def _invoke(self, script_name: str):
        """Run a project-level pipeline script as a subprocess.

        Subprocess so output streams live (you can Ctrl+C cleanly), and
        any sys.path / env mutations stay isolated.
        """
        script = PROJECT_SRC / script_name
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # Run from the src/ directory so its relative imports work
        proc = subprocess.run(
            [sys.executable, str(script.name)],
            cwd=str(PROJECT_SRC), env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{script_name} exited with code {proc.returncode}"
            )

    # ------------------------------------------------------------- copy
    def _collect_results(self):
        """Copy generated artefacts from project results/ into strategy results/."""
        src = PROJECT_RESULTS
        dst = self.results_dir
        if not src.exists():
            print(f"[runner] WARNING: no project results at {src}")
            return

        copied = 0
        # Copy JSON metrics + plot PNGs + CSVs that were produced by the pipeline
        patterns = ["*.json", "*.csv", "*.png", "*.md"]
        for pat in patterns:
            for f in src.glob(pat):
                shutil.copy2(f, dst / f.name)
                copied += 1

        # Also copy any "ablation_*" or "evaluation_*" subfolder if present
        for sub in src.iterdir():
            if sub.is_dir() and (sub.name.startswith("ablation") or
                                  sub.name.startswith("evaluation") or
                                  sub.name.startswith("phase_")):
                target = dst / sub.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(sub, target)
                copied += 1

        print(f"[runner] Copied {copied} artefacts -> {dst}")
        self.cache.mark_done("results_collected",
                             info={"n_files_copied": copied,
                                    "results_dir": str(dst)})


# =======================================================================
# Public entry points
# =======================================================================
def run_dataset(dataset_name: str, force_rerun: bool = False) -> Dict:
    runner = StrategyARunner(dataset_name, force_rerun=force_rerun)
    return runner.run()


def show_status(dataset_name: str):
    """Print cache status for a dataset."""
    cache = DatasetCache(dataset_name)
    print(f"\nStrategy A — Status for '{dataset_name}'")
    print("=" * 60)
    s = cache.status()
    if not s:
        print("  (no progress yet — run hasn't started)")
        return
    for step, info in sorted(s.items()):
        mark = "✓" if info.get("done") else "✗"
        ts = info.get("timestamp", info.get("failed_at", ""))
        print(f"  [{mark}] {step:25s}  {ts}")
