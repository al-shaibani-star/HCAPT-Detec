# NUMBERS — regenerated from result files

`attainability-1.0.0` · generated 2026-08-27T14:13:07+00:00 · `src/paper_numbers.py`

**Rule: no false-positive rate without its recall. If a value is not here, it does not go in the paper.**

## nsl-kdd

Test split: 12,833 attack / 9,711 benign (ratio 1.3215:1). Twin rate 2.95 %. Pooled floor 0.001142, split floor 0.005561.

| config | acc | prec | recall | F1 | FPR | specificity | MCC | frontier @ own recall | slack | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Ens (no GAN)` | 0.7748 | 0.9672 | 0.6257 | 0.7598 | 0.0280 | 0.9720 | 0.6136 | 0.0000 | +2.80 pp (100.0 %) |  |
| `Ens + GAN (central)` | 0.7881 | 0.9667 | 0.6500 | 0.7774 | 0.0296 | 0.9704 | 0.6321 | 0.0000 | +2.96 pp (100.0 %) |  |
| `Ens + GAN + ZT (central)` | 0.7904 | 0.9667 | 0.6543 | 0.7804 | 0.0298 | 0.9702 | 0.6354 | 0.0000 | +2.98 pp (100.0 %) |  |
| `Ens + GAN + FL` | 0.7756 | 0.9666 | 0.6274 | 0.7610 | 0.0286 | 0.9714 | 0.6143 | 0.0000 | +2.86 pp (100.0 %) |  |
| `Ens + GAN + FL + BC` | 0.7756 | 0.9666 | 0.6274 | 0.7610 | 0.0286 | 0.9714 | 0.6143 | 0.0000 | +2.86 pp (100.0 %) | **PLACEHOLDER — identical to `Ens + GAN + FL`** |
| `FULL: Ens + GAN + FL + BC + ZT` | 0.7792 | 0.9668 | 0.6338 | 0.7657 | 0.0287 | 0.9713 | 0.6195 | 0.0000 | +2.87 pp (100.0 %) |  |

Trivial reference — a classifier that always says *attack*: F1 **0.7255**, recall 1.0000, FPR 1.0000, MCC 0.0000. Best configuration F1 is 0.7804, a margin of +0.0549.

> **1 of 6 configurations did not run**: `Ens + GAN + FL + BC` carry counts identical to another row. No claim may rest on them.

## ton-iot-telemetry

Test split: 31,224 attack / 21,000 benign (ratio 1.4869:1). Twin rate 59.17 %. Pooled floor 0.295133, split floor 0.289286.

| config | acc | prec | recall | F1 | FPR | specificity | MCC | frontier @ own recall | slack | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ens_no_gan` | 0.7894 | 0.7494 | 0.9734 | 0.8468 | 0.4840 | 0.5160 | 0.5760 | 0.2567 | +22.73 pp (47.0 %) |  |
| `ens_gan` | 0.7675 | 0.7311 | 0.9667 | 0.8326 | 0.5286 | 0.4714 | 0.5279 | 0.2496 | +27.90 pp (52.8 %) |  |
| `ens_gan_zt` | 0.7674 | 0.7310 | 0.9667 | 0.8325 | 0.5290 | 0.4710 | 0.5276 | 0.2496 | +27.94 pp (52.8 %) |  |
| `ens_gan_fl` | 0.7675 | 0.7311 | 0.9667 | 0.8326 | 0.5286 | 0.4714 | 0.5279 | 0.2496 | +27.90 pp (52.8 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `ens_gan_fl_bc` | 0.7675 | 0.7311 | 0.9667 | 0.8326 | 0.5286 | 0.4714 | 0.5279 | 0.2496 | +27.90 pp (52.8 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `full` | 0.7674 | 0.7310 | 0.9667 | 0.8325 | 0.5290 | 0.4710 | 0.5276 | 0.2496 | +27.94 pp (52.8 %) | **PLACEHOLDER — identical to `ens_gan_zt`** |

Trivial reference — a classifier that always says *attack*: F1 **0.7483**, recall 1.0000, FPR 1.0000, MCC 0.0000. Best configuration F1 is 0.8468, a margin of +0.0985.

> **3 of 6 configurations did not run**: `ens_gan_fl`, `ens_gan_fl_bc`, `full` carry counts identical to another row. No claim may rest on them.

## edge-iiotset

Test split: 26,700 attack / 4,860 benign (ratio 5.4938:1). Twin rate 4.80 %. Pooled floor 0.012427, split floor 0.004733.

| config | acc | prec | recall | F1 | FPR | specificity | MCC | frontier @ own recall | slack | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ens_no_gan` | 0.9096 | 0.9109 | 0.9900 | 0.9488 | 0.5319 | 0.4681 | 0.6076 | 0.0037 | +52.82 pp (99.3 %) |  |
| `ens_gan` | 0.9093 | 0.9091 | 0.9920 | 0.9487 | 0.5453 | 0.4547 | 0.6057 | 0.0037 | +54.16 pp (99.3 %) |  |
| `ens_gan_zt` | 0.9093 | 0.9091 | 0.9920 | 0.9487 | 0.5453 | 0.4547 | 0.6057 | 0.0037 | +54.16 pp (99.3 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `ens_gan_fl` | 0.9093 | 0.9091 | 0.9920 | 0.9487 | 0.5453 | 0.4547 | 0.6057 | 0.0037 | +54.16 pp (99.3 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `ens_gan_fl_bc` | 0.9093 | 0.9091 | 0.9920 | 0.9487 | 0.5453 | 0.4547 | 0.6057 | 0.0037 | +54.16 pp (99.3 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `full` | 0.9093 | 0.9091 | 0.9920 | 0.9487 | 0.5453 | 0.4547 | 0.6057 | 0.0037 | +54.16 pp (99.3 %) | **PLACEHOLDER — identical to `ens_gan`** |

Trivial reference — a classifier that always says *attack*: F1 **0.9166**, recall 1.0000, FPR 1.0000, MCC 0.0000. Best configuration F1 is 0.9488, a margin of +0.0322.

> **4 of 6 configurations did not run**: `ens_gan_zt`, `ens_gan_fl`, `ens_gan_fl_bc`, `full` carry counts identical to another row. No claim may rest on them.

## dapt-2020

Test split: 4,133 attack / 12,383 benign (ratio 0.3338:1). Twin rate 16.54 %. Pooled floor 0.004167, split floor 0.002100.

| config | acc | prec | recall | F1 | FPR | specificity | MCC | frontier @ own recall | slack | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ens_no_gan` | 0.9639 | 0.8998 | 0.9630 | 0.9303 | 0.0358 | 0.9642 | 0.9070 | 0.0000 | +3.58 pp (100.0 %) |  |
| `ens_gan` | 0.9631 | 0.8954 | 0.9654 | 0.9291 | 0.0376 | 0.9624 | 0.9054 | 0.0000 | +3.76 pp (100.0 %) |  |
| `ens_gan_zt` | 0.9630 | 0.8950 | 0.9654 | 0.9289 | 0.0378 | 0.9622 | 0.9051 | 0.0000 | +3.78 pp (100.0 %) |  |
| `ens_gan_fl` | 0.9631 | 0.8954 | 0.9654 | 0.9291 | 0.0376 | 0.9624 | 0.9054 | 0.0000 | +3.76 pp (100.0 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `ens_gan_fl_bc` | 0.9631 | 0.8954 | 0.9654 | 0.9291 | 0.0376 | 0.9624 | 0.9054 | 0.0000 | +3.76 pp (100.0 %) | **PLACEHOLDER — identical to `ens_gan`** |
| `full` | 0.9630 | 0.8950 | 0.9654 | 0.9289 | 0.0378 | 0.9622 | 0.9051 | 0.0000 | +3.78 pp (100.0 %) | **PLACEHOLDER — identical to `ens_gan_zt`** |

Trivial reference — a classifier that always says *attack*: F1 **0.4003**, recall 1.0000, FPR 1.0000, MCC 0.0000. Best configuration F1 is 0.9303, a margin of +0.5300.

> **3 of 6 configurations did not run**: `ens_gan_fl`, `ens_gan_fl_bc`, `full` carry counts identical to another row. No claim may rest on them.

## Corrections

| item | manuscript | correct | source |
|---|---|---|---|
| NSL-KDD Table V, FULL row | 0.7776 / 0.9666 / 0.6311 / 0.7636 / 0.0288 | **0.7792 / 0.9668 / 0.6338 / 0.7657 / 0.0287** | `results/nsl-kdd/ablation_matrix.json (FULL row, .binary)` |
| Headline NSL-KDD accuracy in abstract and conclusion | 77.8 % | **77.92 %** | `same` |
| DAPT 2020 FULL false-positive rate | 0.0374 / 0.0358 / 'under 4 %' | **0.0378 at recall 0.9654** | `results/dapt-2020/ablation_results.json (full.binary)` |
| DAPT 2020 test-split size | 80,000 / 16,000 (sums to 96,000, corpus is 82,577) | **test 16,516; corpus 82,577; train 66,061** | `derived from stored counts + cache/dapt-2020/status.json` |
| Table XXII weighted averages | precision 0.9648, recall 0.9620 | **precision 0.9632, recall 0.9618** | `results/dapt-2020/multiclass_5stage_results.json` |
| Cohen's h, U2R recall 0.1045 -> 0.2239 | 0.34 | **0.3274 -> report 0.33** | `recomputed; the Probe value (0.6952 -> 0.8042) is 0.2530 and stands at 0.25` |
| Bonferroni family size | n = 4 across datasets in the text, n = 2 within-dataset in the JSON | **state the family once, in the methods, and apply it consistently** | `results/_statistical_rigor/*_rigor.json` |

## Cross-architecture operating points (always with recall)

**dapt-2020**

| variant | recall | FPR | F1 |
|---|---:|---:|---:|
| `attlstm_only` | 0.8384 | 0.0608 | 0.8298 |
| `smote_attlstm` | 0.9098 | 0.1326 | 0.7887 |
| `smote_gan_attlstm` | 0.9927 | 0.3708 | 0.6397 |

**edge-iiotset**

| variant | recall | FPR | F1 |
|---|---:|---:|---:|
| `attlstm_only` | 0.9855 | 0.5864 | 0.9421 |
| `smote_attlstm` | 0.7832 | 0.1751 | 0.8630 |
| `smote_gan_attlstm` | 0.8163 | 0.2144 | 0.8800 |

**nsl-kdd**

| variant | recall | FPR | F1 |
|---|---:|---:|---:|
| `attlstm_only` | 0.6906 | 0.0238 | 0.8084 |
| `smote_attlstm` | 0.6354 | 0.0230 | 0.7689 |
| `smote_gan_attlstm` | 0.6736 | 0.0257 | 0.7957 |

**ton-iot-telemetry**

| variant | recall | FPR | F1 |
|---|---:|---:|---:|
| `attlstm_only` | 0.9497 | 0.6282 | 0.8007 |
| `smote_attlstm` | 0.7282 | 0.3438 | 0.7432 |
| `smote_gan_attlstm` | 0.8337 | 0.4814 | 0.7729 |

## Prior shift, ToN-IoT (always with recall)

| variant | recall | FPR | F1 |
|---|---:|---:|---:|
| `none` | 0.9720 | 0.4814 | 0.8468 |
| `smote` | 0.7979 | 0.2969 | 0.7989 |
| `wgan-gp` | 0.9619 | 0.5210 | 0.8320 |
| `smote+wgan-gp` | 0.7971 | 0.2940 | 0.7992 |
