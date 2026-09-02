# Strategy A — HCAPT-Defense Full Stack

End-to-end evaluation of the complete HCAPT-Defense framework (all 7
innovations) on heterogeneous IDS datasets, with per-dataset isolation
and a single cross-dataset comparison report.

## Components evaluated

| # | Component                | File in `../../src/`              | Innovation # |
|---|--------------------------|-----------------------------------|:------------:|
| 1 | WGAN-GP Augmentation     | `wgan_gp_augment.py`              | 1            |
| 2 | Heterogeneous Ensemble   | `ensemble_selflearn.py`           | 2            |
| 3 | Self-Learning / Drift    | `ensemble_selflearn.py`           | 2            |
| 4 | Federated Learning       | `federated_learning.py`           | Layer 3a     |
| 5 | Blockchain Threat Intel. | `blockchain.py` + `threat_intelligence.py` | Layer 3b |
| 6 | Differential Privacy     | `differential_privacy.py`         | Layer 3b     |
| 7 | Zero-Trust Override      | `zero_trust.py`                   | 3            |

## Ablation matrix (per dataset, 6 configurations)

| # | Configuration              | Components                                               |
|---|----------------------------|----------------------------------------------------------|
| 1 | `ens_no_gan`               | Ensemble only                                            |
| 2 | `ens_gan`                  | Ensemble + WGAN-GP                                       |
| 3 | `ens_gan_zt`               | Ensemble + WGAN-GP + Zero-Trust                          |
| 4 | `ens_gan_fl`               | Ensemble + WGAN-GP + Federated Learning                  |
| 5 | `ens_gan_fl_bc`            | Ensemble + WGAN-GP + FL + Blockchain                     |
| 6 | `full`                     | Ensemble + WGAN-GP + FL + Blockchain + DP + Zero-Trust   |

## Folder structure

```
strategy_A_hcapt_full_stack/
├── README.md
├── config.yaml
├── run.py                           # CLI entry point
├── src/
│   ├── cache.py                     # dataset-aware cache + status
│   ├── pipeline.py                  # 6 ablation configs runner
│   ├── evaluator.py                 # metrics + plots
│   ├── comparator.py                # cross-config + cross-dataset
│   └── loaders/
│       ├── base.py
│       ├── nsl_kdd.py
│       ├── ton_iot_telemetry.py     # (Phase 2)
│       └── edge_iiotset.py          # (Phase 2)
├── cache/<dataset-name>/            # resumable intermediate state
└── results/<dataset-name>/          # publication-ready outputs
    └── _cross_dataset/              # final comparison report
```

## Usage

```bash
# Full run on NSL-KDD (resumable)
python run.py --dataset nsl-kdd

# Resume after interrupt
python run.py --dataset nsl-kdd       # auto-detects cache, skips done steps

# Force re-run (ignore cache)
python run.py --dataset nsl-kdd --force-rerun

# Only the cross-dataset comparison
python run.py --compare

# Inspect cache state
python run.py --status --dataset nsl-kdd
```

## Resume guarantees

* **Atomic writes** — each artifact is written to `*.tmp`, then renamed.
  Interrupting mid-write never produces a half-valid file.
* **Status tracker** — each (dataset, step) records a status entry only
  *after* the artifact is on disk.
* **Hash-based invalidation** — config or input changes invalidate
  affected steps automatically.

## Per-dataset publication potential

Each `results/<dataset-name>/` folder is self-contained and includes the
ablation table, confusion matrices, ROC curves, statistical tests, and a
draft `REPORT.md`. Three datasets → three potential papers.
