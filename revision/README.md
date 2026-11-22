# Reproducibility package — IJIES 20267615 (first submitted as 20266447)

**When Does Generative Augmentation Help APT Detection under Distribution Shift?
A Leakage-Safe Re-Evaluation of a Calibration-Aware Zero-Trust Pipeline**

This folder holds every script, frozen split, per-sample prediction and result file behind
the tables and figures of the revised manuscript. Two layers are provided:

| Layer | Contents | Where |
|---|---|---|
| Code + results | scripts, tables (`out/tables`), figures (`out/figures`), protocol and pre-registration files | this repository |
| Full archive | the above **plus** all per-sample predictions (`*.npz`, ~410 MB), split indices and the Optuna study | `HCAPT-revision-artifact.zip` (review channel / archive DOI) |

With the full archive, every table and figure can be regenerated **without a GPU and without
retraining** (Section 3 below). Retraining from raw data is described in Section 4.

---

## 1. Protocol in one paragraph

KDDTrain+ is split once (stratified, seed 42) into FIT (80 %) and VAL (20 %); the indices are
frozen in `out/split_nslkdd.npz`. Every fitted object (categorical vocabularies, scaler,
SMOTE/GAN generators, learners, ensemble weights, Zero-Trust thresholds, calibrators) is fitted
on FIT or VAL; KDDTest+ is scored once. Five training seeds (0–4) vary model randomness while the
split stays fixed. Per-sample probabilities are written to disk and every table is computed
from those files. UNSW-NB15 was held back as an untouched corpus: its configurations, metrics
and hypotheses were frozen in `out/unsw/PREREGISTRATION.json` (with SHA-256 hashes of the script
and data) **before** any UNSW-NB15 model was trained.

## 2. Environment

Tested on Windows 11, Python 3.12.10, one NVIDIA RTX 5070 Laptop GPU (8 GB), 32 GB RAM.

```powershell
python -m pip install -r revision\requirements-revision.txt
```

The scripts import seven modules from `src/` (`config`, `data_loader`, `cache`,
`ensemble_selflearn`, `gan_augment`, `wgan_gp_augment`, `zero_trust`); they are included in the
archive. Raw data are **not** redistributed — see `DATA.md` for sources, expected paths and the
SHA-256 hash of every file used.

## 3. Regenerate all tables and figures from the frozen predictions (no GPU, ~10 min)

```powershell
cd <package root>
powershell -ExecutionPolicy Bypass -File revision\reproduce_from_predictions.ps1
```

This runs the analysis scripts only (r02, r05, r06, r07, r11, r10) and then compares the
regenerated tables with the shipped ones (`verify_tables.py`). It needs the full archive
(the `*.npz` prediction files) but no raw data for the NSL-KDD results.

## 4. Re-run everything from raw data (GPU, ~15–20 h in total)

```powershell
powershell -ExecutionPolicy Bypass -File revision\run_full_pipeline.ps1
```

Each step is resumable (finished seeds/configurations are skipped) and runs heavy jobs in
separate processes. Run one pipeline at a time: two CUDA jobs on an 8 GB laptop GPU run out of
memory. Close memory-hungry applications first (the Windows commit limit was the cause of the
only failures we met).

## 5. Script → manuscript map

| Script | What it does | Manuscript |
|---|---|---|
| `rev_common.py` | frozen FIT/VAL split, FIT-only preprocessing, metrics | §3.3, Alg. 1, Table 3 |
| `r01_train_nslkdd.py` | 5 imbalance treatments × 5 seeds, Zero-Trust validator | Tables 6–8, 10 |
| `r02_tables_nslkdd.py` | ablation, per-class recall, Zero-Trust outcome, McNemar | Tables 6–8, 10, Fig. 8 |
| `r03_baselines_nslkdd.py` | 9 common-protocol baselines (single learners, ROS, B-SMOTE, ADASYN, CTGAN) | Tables 6–8 |
| `r04_learning_curve.py`, `r05_…analysis.py` | controlled learning curve, C2ST/coverage, changeover CI | Figs. 4–5, Table 9, Alg. 3 |
| `r06_calibration_nslkdd.py` | calibration, thresholds and validator at matched budgets | Tables 11–12, Figs. 6–7 |
| `r07_validator_independence.py` | feature lists, error dependence (φ, Yule's Q) | Table 12, Table A1 |
| `r08_dapt_stages.py` | DAPT 2020 random / Flow-ID-grouped / leave-one-day-out | Tables 4, 13–14, Fig. 9 |
| `r09_adversarial_nslkdd.py` | FGSM / PGD-20 on the binary decision, transfer to trees | Table 16, Fig. 10 |
| `r14_iot_corpora.py` | ToN-IoT network and Edge-IIoTset: R / G / U protocols, augmentation | Tables 18–20 |
| `r15_toniot_telemetry_audit.py` | ToN-IoT telemetry audit (time separation, label conflicts, format leak) | Table 17 |
| `r16_equal_budget_tuning.py` | 40 Optuna trials per treatment on VAL, 5-seed final scoring | Table 21 |
| `r17_unsw_confirmatory.py` | pre-registered UNSW-NB15 test (`--run`, `--analyze`) | Tables 22–23 |
| `r19_unsw_binary_auc.py` | post-hoc (not pre-registered) UNSW-NB15 AUC, recall, FPR | §5.9 (AUC sentence) |
| `r10_figures.py` | Figs. 1, 3, 5, 6, 8–10 (first-revision drafts; printed versions from `round2/make_figs.py`) | figures |
| `round2/r20_reviewer_analyses.py` | second review round: paired changeover intervals, VAL-matched Zero-Trust comparison, referral cost, effect sizes, DAPT stage split (see `round2/README.md`) | Tables 9, 10, 12, 15, B1; §5.1 |
| `round2/make_figs.py` | Figs. 1–10 at printed width with 10-pt lettering | Figs. 1–10 |
| `r11_paper_numbers.py` | collects every quoted number into `out/tables/paper_numbers.json` | all text numbers |
| `verify_tables.py`, `reproduce_from_predictions.ps1`, `run_full_pipeline.ps1`, `build_package.py` | verification, run order, package and manifest | — |
| `r12_build_manuscript.py`, `r12_flow_edits.py`, `r13_build_response.py`, `r18_flow_report.py` | authoring tools that build the manuscript, response letter and editing report from `paper_numbers.json` (included in the archive only, not in the public repository, because they contain the unpublished manuscript text; they read the authors' original `.docx` and are not needed to reproduce results) | — |

Latency (§5.6) is measured inside `r01_train_nslkdd.py` and stored in `out/preds/nslkdd_s*_meta.json`.

## 6. Notes that affect reproduction

* **UNSW-NB15 file names.** In the copy we used, the 175,341-row official *training* partition
  is named `UNSW_NB15_testing-set.csv`; `r17` assigns partitions by row count and asserts it.
* **Memory caps.** In the IoT augmentation runs and on UNSW-NB15, Random Forest trees are grown
  on bootstrap samples of at most 50,000 rows (identical across compared settings).
* **CTGAN.** `r03` models the standardised features as continuous; the older
  `src/ctgan_augment.py` rounded standardised categorical codes and is not used.
* **Determinism.** Tree learners and the split are deterministic given the seed; GPU training
  of the DNN and GANs is not bit-exact across hardware, so regenerated numbers can differ in the
  third decimal when retraining (not when re-analysing the shipped predictions).
* **Authors' paths.** The authoring tools contain absolute paths to the authors' manuscript files.

## 7. Integrity

`MANIFEST.sha256` lists the SHA-256 of every file in the package. Verify with:

```powershell
python revision\build_package.py --verify
```
