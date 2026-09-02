# ToN-IoT floor — resolution of the three circulating values

**Task 2 of `FINAL_PLAN.md`. Settled 2026-08-27.**

Three ToN-IoT floors were in circulation. All three are now reproduced and explained.
They are not a disagreement about arithmetic: they are three different *representations*
of the same corpus, and a floor is a property of a representation.

Reproduce with:

```
python -m src.attainability_ton_iot_variants
```

| # | Representation | Cells | Conflict | Pooled floor | Status |
|---|---|---:|---:|---:|---|
| 1 | raw, loader construction, unscaled | 116,531 | 31.27 % | 0.294619 | context |
| 2 | raw, audit rebuild, unscaled | 116,531 | 31.27 % | 0.294619 | context |
| 3 | **scaled float32 — what the detector consumed** | **116,424** | **31.31 %** | **0.295133** | **authoritative** |
| 4 | scaled, then rounded to 6 dp | 116,086 | 31.44 % | 0.296667 | **rejected** |

Rows 1 and 2 produce an **identical partition** — the audit rebuild and the loader's own
construction differ only by injective re-encodings of individual columns, which cannot
merge or split a cell. The 0.2946 that appeared in the gatekeeper report and the
0.29513 from the frozen script were therefore never in conflict about the data; they
differ only in whether the corpus is measured before or after the loader scales it.

## What separates them

**Rows 1-2 → row 3.** `StandardScaler` (fitted on train, applied to both) followed by
the cast back to `float32` merges **107 cells**, moving the pooled floor by **+0.051 pp**.
Column-wise affine scaling is injective in exact arithmetic, so this is entirely a
float32 precision effect: pairs of records that differ far out in the mantissa become
bit-identical once centred, divided and rounded to 24 bits.

**Row 3 → row 4.** An additional `round(x, 6)` merges a further **338 cells** and moves
the floor another +0.153 pp. That rounding existed only inside the audit script
`dup.py`; it is not part of any pipeline that produced a reported result. It is an
artefact of the measurement, not a property of the corpus. **Rejected.**

## The decision

**The manuscript quotes row 3: pooled floor 0.295133, split floor 0.289286, twin rate
59.17 %.**

The reason is pinned choice **P1** in `src/attainability.py`: cells are computed on
exactly the array the detector consumed. The frontier's claim is that no function of
*the representation the detector reads* can do better. The detector read the scaled
float32 array. Bounding the unscaled frame would bound a representation that no
reported result was ever produced from.

## Disclosure required in the manuscript

The 107-cell merge must be stated, not buried. Suggested wording for §4:

> The floor is a property of a representation, not of a corpus. We compute it on the
> array the detector consumes, after scaling and the cast to single precision. That
> pipeline is not lossless: it merges 107 of 116,531 cells on ToN-IoT and raises the
> pooled floor by 0.05 percentage points. We report the post-pipeline value because it
> bounds the representation from which every reported result was produced, and we note
> the pre-pipeline value (0.2946) so the size of the effect is visible.

## Why the published quantity is nonetheless robust

The manuscript never differences against the pooled floor. It differences the
detector's FPR against the **split-level frontier at the detector's own recall**
(pinned choice P6). On that basis all four representations agree to four decimals:

| Representation | Split frontier @ recall 0.9667 | Slack |
|---|---:|---:|
| 1 raw, loader construction | 0.2496 | +27.94 pp (52.8 %) |
| 2 raw, audit rebuild | 0.2496 | +27.94 pp (52.8 %) |
| 3 scaled float32 (authoritative) | 0.2496 | +27.94 pp (52.8 %) |
| 4 scaled + round(6) (rejected) | 0.2496 | +27.94 pp (52.8 %) |

The pooled frontier does move between constructions (0.2568 / 0.2575 / 0.2586 at the
same recall), which is precisely why the pooled floor is reported as context and never
differenced. **The headline — 52.8 % of ToN-IoT's observed false-positive rate is model
slack — does not depend on any construction choice.**

## Numbers that must be corrected wherever they appear

- `0.2967` — withdraw. It is row 4, an audit-script rounding artefact.
- `0.2946` — demote to context, and only ever with "unscaled representation" attached.
- Twin rate `59.12 %` → **59.17 %** (row 3). The 59.12 % figure is row 2's.
- Any ToN-IoT false-positive rate quoted without its recall — there are none left in
  the frozen outputs, and none may be reintroduced.

## Deposited artefacts

- `src/attainability.py` — the frozen estimator (`attainability-1.0.0`).
- `src/attainability_ton_iot_variants.py` — this adjudication, rerunnable.
- `results/_attainability/ton-iot-telemetry.json` — row 3, authoritative.
- `results/_attainability/ton-iot-telemetry__rebuild_raw.json` — row 2, context.
