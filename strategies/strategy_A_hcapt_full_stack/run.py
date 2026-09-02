#!/usr/bin/env python
"""
Strategy A — HCAPT-Defense Full Stack
Single CLI entry point.

Usage:
  python run.py --dataset nsl-kdd               # run + evaluate (resumable)
  python run.py --dataset nsl-kdd --force-rerun # ignore cache, re-run all
  python run.py --status --dataset nsl-kdd      # show cache progress
  python run.py --evaluate-only --dataset nsl-kdd  # rebuild plots/report from existing JSON
  python run.py --compare                       # cross-dataset comparison
  python run.py --clear-cache --dataset nsl-kdd # wipe one dataset's cache
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make the local src/ importable regardless of cwd
HERE = Path(__file__).resolve().parent
SRC  = HERE / "src"
if str(SRC.parent) not in sys.path:
    sys.path.insert(0, str(SRC.parent))

from src.cache import DatasetCache
from src.pipeline import run_dataset, show_status
from src.evaluator import evaluate_dataset
from src.comparator import compare_all


VALID_DATASETS = ["nsl-kdd", "ton-iot-telemetry", "edge-iiotset", "dapt-2020"]


def main():
    parser = argparse.ArgumentParser(
        description="Strategy A — HCAPT-Defense Full Stack runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=VALID_DATASETS,
        help="Dataset to operate on (nsl-kdd, ton-iot-telemetry, edge-iiotset)",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore cache and re-run all phases",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show cache status for the chosen dataset and exit",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Only rebuild ablation table/plots/report from existing JSON results",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Cross-dataset comparison (uses existing per-dataset results)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete the cache for the chosen dataset (results are NOT touched)",
    )
    args = parser.parse_args()

    # Compare-only mode
    if args.compare:
        compare_all()
        return 0

    # All remaining modes need a dataset
    if not args.dataset:
        parser.error("--dataset is required for this mode (use --compare for cross-dataset).")
        return 2

    if args.status:
        show_status(args.dataset)
        return 0

    if args.clear_cache:
        cache = DatasetCache(args.dataset)
        confirm = input(f"Clear cache for '{args.dataset}'? [y/N] ").strip().lower()
        if confirm == "y":
            cache.clear()
        else:
            print("aborted.")
        return 0

    if args.evaluate_only:
        evaluate_dataset(args.dataset)
        compare_all()
        return 0

    # Default: run pipeline + evaluate + compare
    run_dataset(args.dataset, force_rerun=args.force_rerun)
    evaluate_dataset(args.dataset)
    compare_all()
    print("\n[run.py] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
