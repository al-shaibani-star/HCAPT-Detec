# Second review round — IJIES paper 20267615

This folder also holds `r22_editor_round3.py`, written for the editor's conditional-acceptance items of 2026-11-26.

Analyses and figures added in response to the second round of reviews. They read the frozen artefacts of
the first revision (`revision/out/…`, including the per-sample predictions of the full archive) and write
only to their own output folder; nothing in `revision/out` is modified.

## Scripts

| Script | What it does | Manuscript |
|---|---|---|
| `r20_reviewer_analyses.py` | A: seed-level GAN − SMOTE recall differences, paired 95 % t-intervals, censoring/acceptance rule for the changeover | §5.2, Table 9, Table B1 |
| | B: Zero-Trust policy vs. single-score thresholds chosen on VAL only (matched VAL FPR and matched VAL alert count), scored once on KDDTest+, test-flow bootstrap CIs | §4.4, §5.3, Table 12 |
| | C: analyst-referral cost of the CHALLENGE action and its precision at other attack prevalences | §5.3, Table 10 |
| | D: effect sizes (accuracy difference with bootstrap CI) for the McNemar comparisons | §5.1 |
| | E: DAPT 2020 leave-one-day-out re-run split by stage, with training support per stage, Wilson and Flow-ID cluster-bootstrap CIs (needs the GPU and the raw DAPT 2020 csv files) | §5.4, Tables 14–15 |
| `r21_latency_and_sensitivity.py` | D: DAPT 2020 protocol P2 re-run with Flow-ID cluster-bootstrap intervals per stage | Table 13 |
| | L: end-to-end latency from a raw NSL-KDD record to a Zero-Trust action (preprocessing, cold vs. warm calls, GPU vs. CPU placement, batch sizes 1–22,544) and a simulated Poisson stream of 100–20,000 flows/s with one-flow-per-call or micro-batched serving | §4.6, §5.6, Tables D1–D2 |
| | S: sensitivity to the FIT/VAL split — three additional, independently drawn stratified splits, each with its own preprocessing, validator, WGAN-GP generator and ensembles; the original split is read from the frozen seed-0 predictions | §6.3, Table D3 |
| `r22_editor_round3.py` | M: exact McNemar p-values and their Holm adjustment computed in log space (no numerical underflow) for Table 8 and UNSW-NB15 hypothesis H2; L: Table D2 streaming simulation re-run with the warm single-flow median of Table D1 as the one-flow service time | Table 8, Table 23, §5.9, §4.6, §5.6, Table D2 |
| `make_figs.py` | Redraws Figs. 1–10 at their printed width with 10-pt lettering from the stored result files (Fig. 2: PCA fitted on FIT) | Figs. 1–10 |

## Reproduce

From the package root (the folder that contains `revision/`, `src/` and `data/`):

```powershell
python revision\round2\r20_reviewer_analyses.py --impl . --out revision\round2\out            # A–E
python revision\round2\r20_reviewer_analyses.py --impl . --out revision\round2\out --skip-dapt  # A–D only, no GPU
python revision\round2\make_figs.py --impl . --out revision\round2\out\figures
python revision\round2\r21_latency_and_sensitivity.py --impl . --out revision\round2\out --parts DLS   # GPU, about 1.5 h
python revision\round2\r22_editor_round3.py --impl . --out revision\round2\out   # seconds, no GPU
```

Parts A–D need only the stored predictions (`revision/out/preds/*.npz`, full archive) and
`revision/out/learning_curve.csv`; part E retrains the DAPT 2020 models with `revision/r08_dapt_stages.py`
(seed 0) and reproduces the published day-level results of `revision/out/tables/dapt_day_out.csv` exactly.

## Outputs (`out/`)

| File | Content |
|---|---|
| `A_learning_curve_paired_deltas.csv`, `A_changeover_strict_rule.csv` | per-seed Δ, mean, paired CI; verdict of the acceptance rule |
| `B_zt_val_matched_per_seed.csv`, `B_zt_val_matched_summary.csv` | VAL-matched decision rules on KDDTest+ |
| `C_referral_counts_per_seed.csv`, `C_referral_prevalence_projection.csv` | referral counts and prevalence projection |
| `D_effect_sizes_table8.csv` | accuracy differences with 95 % CI |
| `E_dapt_dayout_per_stage.csv`, `E_dapt_dayout_day_level_rerun.csv`, `E_log.txt` | DAPT 2020 stage split, day-level re-run, run log (the per-flow predictions `E_dapt_dayout_predictions.npz` are, like all `*.npz` files, in the release archive) |
| `D_dapt_p2_cluster.csv` | DAPT 2020 P2 per-stage recall with Wilson and Flow-ID cluster-bootstrap intervals (r21) |
| `L_latency.json`, `L_streaming_simulation.csv` | end-to-end latency per placement, component and batch size; streaming simulation (r21) |
| `S_split_sensitivity.csv` | NSL-KDD results for the original and three additional FIT/VAL splits (r21) |
| `M_mcnemar_nslkdd_logspace.csv`, `M_mcnemar_nslkdd_table8.csv`, `M_mcnemar_unsw_logspace.csv` | per-seed discordant counts, log10 of the exact and Holm-adjusted p, and the Table 8 ranges (r22) |
| `L2_streaming_simulation.csv` | Table D2 simulation with the one-flow service time equal to the warm single-flow median (r22) |
| `figures/` | Figs. 1–10 as printed, with `figure_info.json` (printed widths) |

## Release archive

The complete package (code, per-sample predictions, frozen split indices, calibration parameters, Optuna study,
pre-registration file and every analysis output, with `MANIFEST.sha256`) is attached to the GitHub release
`ijies-20267615-r3` of this repository and can be downloaded without signing in. After extracting it:

```powershell
python revision\build_package.py --verify --root .
```
