"""
One-shot generator for all publication figures.

Runs EDA + comparison + architecture diagram scripts in order and
reports a summary of what was produced.
"""
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

FIG_DIR = SRC.parent / "figures"


def main():
    t0 = time.time()
    print("#" * 70)
    print("#  HCAPT-Defense - generating all publication figures")
    print("#" * 70)

    from plotting import (
        eda_plots, comparison_plots, architecture_diagrams,
        advanced_plots, extra_plots,
    )

    eda_plots.main()
    comparison_plots.main()
    architecture_diagrams.main()
    advanced_plots.main()
    extra_plots.main()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total_pdf = 0
    total_png = 0
    for subdir in sorted(FIG_DIR.glob("*")):
        if subdir.is_dir():
            pdfs = list(subdir.glob("*.pdf"))
            pngs = list(subdir.glob("*.png"))
            total_pdf += len(pdfs)
            total_png += len(pngs)
            print(f"  {subdir.name:<15} : {len(pdfs)} PDF + {len(pngs)} PNG")
            for p in sorted(pdfs):
                size_kb = p.stat().st_size / 1024
                print(f"      - {p.name:<40}  {size_kb:6.1f} KB")

    print(f"\nTotal: {total_pdf} PDF + {total_png} PNG")
    print(f"Time : {time.time()-t0:.1f}s")
    print(f"Output: {FIG_DIR}")


if __name__ == "__main__":
    main()
