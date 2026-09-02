# HCAPT-Detect

Code and result files for the paper

> **HCAPT-Detect: A Calibration-Aware Generative-AI and Zero-Trust Framework for APT Detection in Hybrid-Cloud Networks**
> Hussein Ali Ismael, Jamal Mustafa Abbas, Ziyad Tariq Mustafa Al-Ta'i
> Department of Computer Science, College of Science, University of Diyala, Iraq
> Submitted to the *Journal of Cyber Security Technology* (Taylor & Francis), 2026.

HCAPT-Detect is the detection core of the HCAPT-Defense framework. It combines a
class-conditional GAN augmenter with a SMOTE fallback for under-supported classes, a
soft-voting ensemble of Random Forest, XGBoost and a deep neural network, and a
Zero-Trust validation layer in which an independent LightGBM risk scorer challenges
high-confidence benign decisions. All headline numbers use the official NSL-KDD
KDDTrain+/KDDTest+ split; ToN-IoT, Edge-IIoTset and DAPT 2020 are evaluated under the
strictest protocol available for each dataset.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Core pipeline: data loading, CGAN and WGAN-GP augmentation, ensemble, federated learning, blockchain audit, differential privacy, Zero-Trust layer, evaluation and plotting |
| `strategies/strategy_A_hcapt_full_stack/` | Multi-dataset runner (NSL-KDD, ToN-IoT telemetry, Edge-IIoTset, DAPT 2020), statistical rigor, adversarial robustness, leak audit, attainability analysis |
| `results/` | Result files (JSON/CSV) behind the tables in the paper, NSL-KDD |
| `strategies/.../results/` | Per-dataset result files |
| `figures/` | Generated figures; `figures/resubmission_2026-09/` holds the scripts used to regenerate Fig. 1 and Fig. 10 and to re-evaluate the CGAN and WGAN-GP ensembles |
| `docs/UNIFIED_ARCHITECTURE.md` | Architecture and evaluation-matrix specification |
| `RESULTS_SUMMARY.md`, `INTERPRETATION.md` | Result summaries and interpretation notes |

Datasets, trained models and cached arrays are **not** included (see `.gitignore`).

## Datasets

Download and place under `data/` (folder names as in `src/config.py`):

- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- ToN-IoT: https://research.unsw.edu.au/projects/toniot-datasets
- Edge-IIoTset: https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications
- DAPT 2020: https://gitlab.com/asu22/dapt2020

## Reproducing the NSL-KDD results

```bash
pip install -r requirements.txt
cd src
python train_full_pipeline.py          # preprocessing, baselines, ensemble
python gan_augment.py                  # CGAN augmentation
python wgan_gp_augment.py              # WGAN-GP augmentation (comparison variant)
python compare_gans.py                 # no-GAN vs CGAN vs WGAN-GP ensembles
python zero_trust.py                   # LightGBM risk scorer and Zero-Trust policy
python evaluate_all_configurations.py  # ablation matrix (Table 8)
python generate_all_figures.py         # all figures
```

For the other datasets:

```bash
cd strategies/strategy_A_hcapt_full_stack
python run.py --dataset ton-iot-telemetry
python run.py --dataset edge-iiotset
python run.py --dataset dapt-2020
```

Tested with Python 3.12, PyTorch 2.7 (CUDA), scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6.

## Note on the augmenter

The results reported in the paper (Table 8 and Figures 6, 7 and 10) use the
class-conditional GAN (`src/gan_augment.py`, models with the `gan_` prefix). The
WGAN-GP variant (`src/wgan_gp_augment.py`, `wgangp_` prefix) is evaluated as an
alternative and reached a lower downstream accuracy on KDDTest+; see
`figures/resubmission_2026-09/variant_summary.json`.

## Contact

Corresponding author: Hussein Ali Ismael, Al-Shaibani@uodiyala.edu.iq
