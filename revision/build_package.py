"""
Build the reproducibility package for IJIES 20266447.

  python revision\\build_package.py            write DATA.md and MANIFEST.sha256, build the zip archive
  python revision\\build_package.py --verify   check every file listed in MANIFEST.sha256

Archive layout (paths relative to the package root):
  README.md, DATA.md, MANIFEST.sha256
  revision/   scripts, run/verify scripts, out/ (split indices, per-sample predictions, tables,
              figures, pre-registration, Optuna study)
  src/        the seven modules the scripts import
Raw data are not included; DATA.md lists sources, expected paths and SHA-256 hashes.
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

REV = Path(__file__).resolve().parent
ROOT = REV.parent                                  # 05-Implementation
ZIP = Path(r"E:\Strudin\Hussein2026\اناس\HCAPT-revision-artifact.zip")
SRC_MODULES = ["config", "data_loader", "cache", "ensemble_selflearn", "gan_augment", "wgan_gp_augment", "zero_trust"]
DATA = [  # (path relative to ROOT, corpus)
    ("data/nsl-kdd/train.txt", "NSL-KDD"), ("data/nsl-kdd/test.txt", "NSL-KDD"),
    ("data/UNSW-NB15/UNSW_NB15_training-set.csv", "UNSW-NB15"), ("data/UNSW-NB15/UNSW_NB15_testing-set.csv", "UNSW-NB15"),
    ("data/ton-iot/network/train_test_network.csv", "ToN-IoT network"),
    ("data/edge-iiotset/ml-dl-selected/ML-EdgeIIoT-dataset.csv", "Edge-IIoTset"),
]
SOURCES = {
    "NSL-KDD": "https://www.unb.ca/cic/datasets/nsl.html (KDDTrain+.txt -> train.txt, KDDTest+.txt -> test.txt)",
    "DAPT 2020": "https://gitlab.com/asu22/dapt2020 (folder csv/)",
    "ToN-IoT network": "https://research.unsw.edu.au/projects/toniot-datasets (Train_Test_Network dataset)",
    "ToN-IoT telemetry": "https://research.unsw.edu.au/projects/toniot-datasets (Train_Test_IoT dataset)",
    "Edge-IIoTset": "https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications (ML-EdgeIIoT-dataset.csv)",
    "UNSW-NB15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset (official training/testing partitions)",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_files():
    files = list(DATA)
    files += [(str(p.relative_to(ROOT)).replace("\\", "/"), "DAPT 2020") for p in sorted((ROOT / "data/dapt-2020/csv").glob("*.pcap_Flow.csv"))]
    files += [(str(p.relative_to(ROOT)).replace("\\", "/"), "ToN-IoT telemetry") for p in sorted((ROOT / "data/ton-iot/iot-telemetry").glob("*.csv"))]
    return files


def write_data_md():
    lines = ["# Data", "", "Raw data are not redistributed. Download each corpus from its source and place the files at",
             "the paths below (relative to the package root). Hashes are those of the files used for every result.", "",
             "| Corpus | Source |", "|---|---|"] + [f"| {k} | {v} |" for k, v in SOURCES.items()]
    lines += ["", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    for rel, _ in data_files():
        p = ROOT / rel
        lines.append(f"| `{rel}` | {p.stat().st_size:,} | `{sha(p)}` |" if p.exists() else f"| `{rel}` | missing | — |")
    lines += ["", "Notes:",
              "* DAPT 2020: `enp0s3-pvt-thursday.pcap_Flow.csv` has no stage labels and is excluded by `r08`.",
              "* UNSW-NB15: in our copy the 175,341-row official *training* partition is named `UNSW_NB15_testing-set.csv`; "
              "`r17` assigns partitions by row count.",
              "* ToN-IoT telemetry is audited (`r15`) but not used for modelling.",
              "* Re-analysis from the shipped predictions (`reproduce_from_predictions.ps1`) needs none of these files."]
    (REV / "DATA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_files():
    """(source path, archive name) pairs."""
    out = []
    skip_names = {"tables_shipped", "tables_smoke", "preds_smoke", "tuning_smoke", "models_tmp", "__pycache__", "generators"}
    for p in sorted(REV.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REV)
        parts = set(rel.parts)
        if parts & skip_names or p.name.startswith("_") or ".bak" in p.name or p.suffix in (".pt", ".log", ".err", ".pyc"):
            continue
        if "smoke" in p.name or p.name == "MANIFEST.sha256":
            continue
        if rel.parts[0] == "out" and p.suffix not in (".npz", ".json", ".csv", ".png", ".db"):
            continue
        out.append((p, "revision/" + "/".join(rel.parts)))
    out += [(ROOT / "src" / f"{m}.py", f"src/{m}.py") for m in SRC_MODULES]
    out += [(REV / "README.md", "README.md"), (REV / "DATA.md", "DATA.md")]
    return out


def build():
    write_data_md()
    files = package_files()
    manifest = "".join(f"{sha(src)}  {arc}\n" for src, arc in files)
    (REV / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, arc)
        z.writestr("MANIFEST.sha256", manifest)
    size = ZIP.stat().st_size / 1e6
    print(f"[package] {len(files)} files, archive {size:.0f} MB -> {ZIP}")
    print(f"[package] archive SHA-256 {sha(ZIP)}")


def verify(root):
    man = (root / "MANIFEST.sha256") if (root / "MANIFEST.sha256").exists() else REV / "MANIFEST.sha256"
    bad = 0
    for line in man.read_text(encoding="utf-8").splitlines():
        h, arc = line.split("  ", 1)
        p = root / arc
        if not p.exists():
            print("missing ", arc); bad += 1
        elif sha(p) != h:
            print("changed ", arc); bad += 1
    print("MANIFEST OK" if not bad else f"{bad} problem(s)"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--root", default=None, help="package root to verify (default: the extracted archive layout around this script)")
    a = ap.parse_args()
    if a.verify:
        verify(Path(a.root) if a.root else REV.parent)
    else:
        build()
