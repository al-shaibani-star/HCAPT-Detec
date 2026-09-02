# ToN-IoT telemetry: the floor, the oracle, and the two devices

**Task 3 of `FINAL_PLAN.md`. Run 2026-08-27.**
Reproduce: `python -m src.attainability_ton_iot_timestamps`
Data: `results/_attainability/ton_iot_timestamp_counterfactual.json`

The gatekeeper's audit reported that restoring the two discarded columns `date`
and `time` drops the label-conflicting cell rate to 0.00 %, and concluded that the
29.5 % "irreducible" floor is manufactured by the preprocessing recipe. That
measurement reproduces exactly. Its interpretation does not survive contact with the
columns themselves.

## 1. The counterfactual

All four rows run through the loader's own pipeline — same construction, same seed-42
stratified split, `StandardScaler` fitted on train, cast to float32 — so the base row
reproduces `ton-iot-telemetry.json` to the digit.

| Column set | Cells | Duplicates | Conflicting | Pooled floor | Split floor | Twin rate |
|---|---:|---:|---:|---:|---:|---:|
| base (18 features) | 116,424 | 59.25 % | **31.31 %** | **0.295133** | 0.289286 | 59.17 % |
| base + `date` | 117,695 | 59.11 % | **0.00 %** | 0.000000 | 0.000000 | 59.07 % |
| base + `time` | 207,746 | 27.49 % | 0.29 % | 0.001543 | 0.000619 | 25.82 % |
| base + `date` + `time` | 210,453 | 25.86 % | **0.00 %** | 0.000000 | 0.000000 | 24.31 % |

One column — `date` — takes the floor from 29.5 % to exactly zero.

## 2. Why: `date` is a label oracle

`date` takes **nine distinct values across 261,119 records, and every one of them
carries a single label.**

| date | rows | label | devices |
|---|---:|---|---:|
| 31-Mar-19 | 69,777 | benign | 7 |
| 1-Apr-19 | 14,007 | benign | 5 |
| 2-Apr-19 | 21,216 | benign | 4 |
| 23-Apr-19 | 2,198 | **attack** | 5 |
| 25-Apr-19 | 58,950 | **attack** | 7 |
| 26-Apr-19 | 37,825 | **attack** | 7 |
| 27-Apr-19 | 6,116 | **attack** | 7 |
| 28-Apr-19 | 50,116 | **attack** | 7 |
| 29-Apr-19 | 914 | **attack** | 1 |

Benign telemetry was collected over three days at the end of March and the start of
April. Attack telemetry was collected over six days three weeks later. The two
collections do not overlap by a single record. A nine-valued lookup on `date` labels
the entire corpus perfectly. `time` is weaker but the same in kind: 56,130 values,
15.81 % label-conflicting on its own.

**So the loader was right to drop these columns, and the finding inverts.** The floor
is not an artefact to be undone. It is the true cost of a sound protocol on this
corpus, and the collapse measures the size of the oracle that a careless protocol
would have handed the model.

This is a stronger result than the one it replaces, because it is a statement about
the benchmark rather than about our preprocessing.

## 3. Where the floor lives: two devices, completely

Forced false positives — benign records sitting in attack-majority cells, which no
rule on this representation can spare — on the base representation:

| device | benign rows | forced FPs | device floor FPR | share of floor |
|---|---:|---:|---:|---:|
| `fridge` | 15,000 | 15,000 | **1.0000** | 48.4 % |
| `motion_light` | 15,000 | 15,000 | **1.0000** | 48.4 % |
| `thermostat` | 15,000 | 989 | 0.0659 | 3.2 % |
| `garage_door` | 15,000 | 0 | 0.0000 | 0.0 % |
| remaining three | 15,000 each | 0 | 0.0000 | 0.0 % |

`fridge` and `motion_light` do not merely contribute to the floor — **every single
benign record in both devices is a forced false positive.** Once the collection-date
oracle is removed, the 18-feature union-and-zero-fill schema retains nothing that
distinguishes normal from attack behaviour on those two devices. Two of seven devices
together account for 96.8 % of the floor; adding `thermostat` accounts for all of it.

This corrects the "two of seven devices produce 100 % of the floor" claim that
appeared in the earlier drafts: the correct figure is **96.8 %**, with `thermostat`
holding the remaining 3.2 %.

## 4. What this licenses the paper to say

Pooled ToN-IoT telemetry offers a benchmark user exactly two regimes:

1. **Keep the collection timestamps.** A nine-valued lookup separates the classes
   perfectly. Reported accuracy approaches 100 % and measures nothing about detection.
2. **Remove them, as a sound protocol must.** The remaining representation carries a
   29.5 % irreducible false-positive floor, of which 96.8 % is two devices whose benign
   traffic is not distinguishable from their attack traffic at all.

There is no third regime, and the benchmark's own structure — not any modelling
choice — is what forbids it.

**A checkable claim about the literature follows**, and the paper should state it in
this careful form: a published ToN-IoT telemetry result that reports both high accuracy
and a low false-positive rate must be drawing on one of two leakage channels — the
timestamp oracle documented here, or the duplicate-driven overlap of a random split,
which leaves 59.17 % of test records with an exact feature twin in training. The paper
should not assert which channel any particular study used without checking that study's
preprocessing; both channels are documented here, and either suffices.

## 5. Honesty constraints on how this is written

- Do not present the collapse as "the floor is fake". It is real, and correct.
- Do not recommend restoring the timestamps anywhere in the manuscript.
- Our own protocol carries the second leakage channel (59.17 % twins). §6 must
  disclose that before a reviewer finds it. The floor computation is what detected it.
- The per-device figure is 96.8 %, not 100 %.
- Every false-positive rate quoted here is at the detector's own recall of 0.9667 or
  at the floor's recall; none may be quoted bare.

## Deposited artefacts

- `src/attainability_ton_iot_timestamps.py` — this analysis, rerunnable.
- `results/_attainability/ton_iot_timestamp_counterfactual.json` — all four variants,
  per-device attribution, and the label-oracle table.
