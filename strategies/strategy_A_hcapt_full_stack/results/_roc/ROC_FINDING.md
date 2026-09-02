# ROC against the frontier — what one operating point was hiding

**Task 6 of `FINAL_PLAN.md`. Run 2026-08-27.**
Reproduce: `python -m src.roc_frontier` · Data: `results/_roc/roc_frontier.json`
Figures: `{ton-iot-telemetry,edge-iiotset,dapt-2020}_roc_frontier.png`

Inference only. The cached ensembles were reloaded and scored over the stored test
splits; nothing was retrained.

## 1. The cache is sound

At threshold 0.5 the reloaded models reproduce the published operating points exactly:

| benchmark | reloaded (FPR @ recall) | `ablation_table.csv` |
|---|---|---|
| ToN-IoT `ens_gan` | 0.5286 @ 0.9667 | 0.5286 @ 0.9667 |
| Edge-IIoTset `ens_gan` | 0.5453 @ 0.9920 | 0.5453 @ 0.9920 |
| DAPT 2020 `ens_gan` | 0.0376 @ 0.9654 | 0.0376 @ 0.9654 |

Whatever else is wrong with the manuscript, the stored models are the ones that produced
the reported numbers.

**NSL-KDD has no cached ensemble** — `cache/nsl-kdd/` holds only `status.json`. It gets no
ROC, and §8 must say so rather than leave the omission to be noticed.

## 2. Discrimination is far better than the reported operating points suggest

| benchmark | AUC (`ens_no_gan` / `ens_gan`) | average precision |
|---|---|---|
| ToN-IoT | 0.8446 / 0.8446 | 0.8915 |
| Edge-IIoTset | **0.9305** / 0.9287 | 0.9868 / 0.9863 |
| DAPT 2020 | 0.9937 / 0.9937 | 0.9811 / 0.9810 |

An AUC of 0.93 on Edge-IIoTset is not the profile of a detector that cannot separate the
classes. It is the profile of a detector reporting from a badly chosen threshold.

## 3. The headline false-positive rates are a threshold choice

This is the finding that a single operating point concealed, and a reviewer would have
found it in one afternoon.

| benchmark | recall 0.80 | 0.90 | 0.95 | 0.99 | published point |
|---|---:|---:|---:|---:|---|
| Edge-IIoTset | **0.1146** | 0.3002 | 0.4119 | 0.5319 | 0.5453 @ 0.9920 |
| ToN-IoT | **0.2958** | 0.3932 | 0.4468 | 0.5976 | 0.5286 @ 0.9667 |
| DAPT 2020 | 0.0094 | 0.0215 | 0.0303 | 0.0623 | 0.0376 @ 0.9654 |

On Edge-IIoTset the "54.5 % false-positive rate" becomes **11.5 %** at recall 0.80. The
paper's headline number is what the operating point costs at recall 0.99, and the paper
never said so.

**The manuscript must not report a false-positive rate as a property of the system.** It
must report the curve, mark the operating point on it, and say what recall that point buys.

## 4. The decomposition survives — and gets sharper

Lowering the threshold does not close the gap to the frontier; it narrows it while leaving
the detector far above:

| benchmark | slack at recall 0.80 | 0.90 | 0.95 | 0.99 |
|---|---:|---:|---:|---:|
| Edge-IIoTset | +11.46 pp | +30.02 pp | +41.19 pp | +52.82 pp |
| ToN-IoT | +16.42 pp | +18.54 pp | +21.55 pp | +31.18 pp |
| DAPT 2020 | +0.94 pp | +2.15 pp | +3.03 pp | +6.21 pp |

On Edge-IIoTset the frontier is **0.0000 up to recall 0.95** — every false positive the
detector raises below that recall is avoidable in principle. On ToN-IoT the frontier is
0.1316 at recall 0.80 and 0.2858 at recall 0.99, so a real irreducible component exists
and grows with recall, exactly as the label-conflicting cells predict.

Slack is therefore not a scalar. **It is a function of recall, and the paper should plot
it as one.** Table III's single-column slack becomes a curve, and the claim strengthens:
the gap persists at *every* operating point, not just the one we happened to report.

## 5. Generative augmentation is inert across the whole curve

`ens_no_gan` and `ens_gan` are within 0.002 AUC on all three benchmarks, and their
threshold sweeps differ by under 0.5 pp except at Edge recall 0.90. The earlier ablation
showed this at one point; the curves show it everywhere. This is a clean negative result
and costs nothing to state.

## 6. What this obliges the manuscript to do

- Report ROC/PR with the frontier overlaid for the three benchmarks that have one, and
  disclose plainly that NSL-KDD has no cached model and therefore no curve.
- Replace every bare false-positive rate with a point on a curve plus its recall.
- Recast the slack as a function of recall rather than a single number.
- State that the operating point was inherited from an argmax rule at threshold 0.5 and
  was never tuned — which is *why* it sits at recall 0.99, and is itself a finding about
  how these systems get evaluated.
- Keep the negative result on augmentation.

## Deposited artefacts

- `src/roc_frontier.py` — this analysis, rerunnable, inference only.
- `results/_roc/roc_frontier.json` — curves, AUC/AP, operating points, threshold sweeps.
- `results/_roc/*_roc_frontier.png` — working figures; task 16 rebuilds them for print.
