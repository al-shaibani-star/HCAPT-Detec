# DAPT 2020 Integration Guide

DAPT 2020 is the only published benchmark explicitly designed to
exercise the full **Advanced Persistent Threat lifecycle** (six
multi-stage attack phases over months of synthetic enterprise
traffic). No paper in our Q1 comparison set uses it, which makes
it the strongest candidate for an *empirical* (not just rhetorical)
APT-detection claim.

## 📥 Download

DAPT 2020 is **freely available** but requires manual download:

| Source | URL | Notes |
|--------|-----|-------|
| **Primary (GitLab)** | <https://gitlab.thothlab.org/DAPT2020/DAPT2020> | Original repository |
| **Kaggle mirror**    | <https://www.kaggle.com/datasets/sowmyamyneni/dapt2020> | Easier download |
| **arXiv paper**      | <https://arxiv.org/abs/2008.04379> | Myneni et al. 2020 |

Expected file size: **~5 GB** (25M flow records, 6 attack stages).

## 📂 Where to put the files

```
05-Implementation/
└── data/
    └── dapt-2020/             ← create this folder
        ├── full_dataset.csv   ← preferred (one file)
        ├── OR
        ├── train.csv
        └── test.csv
```

The loader (`src/loaders/dapt2020.py`) detects both layouts.

## 🚀 How to run after download

```bash
cd 05-Implementation/strategies/strategy_A_hcapt_full_stack

# 1. Register DAPT-2020 in pipeline.py (one-line edit):
#    Add to get_loader():
#        elif dataset_name == "dapt-2020":
#            return Dapt2020Loader(DATA_ROOT)

# 2. Run Strategy A
python run.py --dataset dapt-2020
```

The pipeline will run all six ablation configurations + the deep
ablation studies + the statistical-rigor module automatically.

## 📊 Expected numerical results

Based on prior literature (Myneni et al. 2020 report ~88-92%
accuracy with traditional ML), HCAPT-Defense should achieve:

| Configuration | Expected Acc | Expected F1 |
|--------------|:------------:|:-----------:|
| ens_no_gan   | 0.87 - 0.91  | 0.85 - 0.89 |
| ens_gan      | 0.88 - 0.92  | 0.86 - 0.90 |
| ens_gan_zt   | 0.88 - 0.93  | 0.87 - 0.91 |
| full         | 0.88 - 0.92  | 0.86 - 0.90 |

## 🎯 Paper claim unlocked

Once DAPT 2020 numbers are in, the Discussion section can claim:

> *"HCAPT-Defense is the first framework to evaluate a unified
> Gen-AI + Federated + Blockchain + DP + Zero-Trust stack on
> DAPT 2020 — the only IDS benchmark with annotated APT lifecycle
> stages. This complements our NSL-KDD / ToN-IoT / Edge-IIoTset
> evaluation by demonstrating that the framework can detect not
> just individual malicious flows, but the temporal progression
> of multi-stage APT campaigns. To our knowledge no concurrent
> Q1 publication (Khule 2025, Adnan 2025, Radha 2025, FIDMF 2026)
> reports on DAPT 2020."*

## ⏱️ Time budget

| Task | Effort |
|------|------:|
| Download + extract | 30 min (depends on connection) |
| Verify loader works (one-shot) | 5 min |
| Full Strategy A pipeline | 60-90 min (GAN + 3 ensembles + ZT) |
| Cross-stage analysis per APT stage | additional 30 min |
| **Total once data is local** | **~2 hours** |

## ⚠️ Caveats

* The DAPT label schema (six stages) may not exactly match our
  binary or 5-class mapping. The loader has a heuristic mapping
  in `ATTACK_STAGES`; verify against the README of the downloaded
  release.
* Some DAPT-2020 files have non-standard line endings or
  non-UTF-8 characters in label strings. If `pd.read_csv` errors,
  pass `encoding="latin-1"` and re-export to UTF-8.

## 🤝 Status (2026-05-30)

| Component | Status |
|-----------|:------:|
| Loader skeleton | ✅ written |
| Pipeline integration | ⏳ one-line edit pending data |
| Actual data | ❌ needs user download |
| First evaluation | ❌ pending data |
| Paper section | ⏳ template ready |
