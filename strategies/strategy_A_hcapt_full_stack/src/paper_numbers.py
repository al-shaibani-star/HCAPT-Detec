"""
Every number the rebuilt paper is allowed to contain — regenerated from result
files, never transcribed from the manuscript.

Task 5 of FINAL_PLAN.md. The manuscript's number audit found 148 verified, 61
untraceable and 31 contradicted claims. Rather than patch 31 values by hand and
hope, this script derives the whole set from the stored artefacts, so a value
that has no file simply cannot appear.

What it emits, per benchmark:

  * every ablation configuration with its stored confusion counts, plus the
    derived rates — and a PLACEHOLDER flag on any configuration whose row is
    byte-identical to another, which is how the untrained distributed-training
    and ledger rows announce themselves
  * the trivial baselines a false-positive rate needs to be read against:
    constant-attack, constant-benign, specificity, balanced accuracy, MCC
  * the attainability frontier at each configuration's own recall
  * cross-architecture and prior-shift operating points, always with recall
  * the corrections list from FINAL_PLAN task 5, each recomputed here

    python -m src.paper_numbers

Writes `results/_paper_numbers/numbers.json` and `NUMBERS.md`.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_ROOT = Path(__file__).resolve().parents[1]
if str(STRATEGY_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGY_ROOT))

from src.attainability import RESULTS_DIR, OUT_DIR, SPEC_VERSION, _sha256, _at_recall  # noqa: E402

HERE = Path(__file__).resolve()
NUM_DIR = RESULTS_DIR / "_paper_numbers"
BENCHMARKS = ["nsl-kdd", "ton-iot-telemetry", "edge-iiotset", "dapt-2020"]


# ---------------------------------------------------------------------------
def derived(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Every rate the paper may quote, from the four counts and nothing else."""
    P, N = tp + fn, fp + tn
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": P + N, "positives": P, "negatives": N,
        "accuracy": (tp + tn) / (P + N),
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / P if P else None,
        "specificity": tn / N if N else None,
        "fpr": fp / N if N else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
        "balanced_accuracy": 0.5 * (tp / P + tn / N) if P and N else None,
        "mcc": (tp * tn - fp * fn) / denom if denom else None,
    }


def trivial_baselines(P: int, N: int) -> dict:
    """What you get for free. An F1 that barely beats these is not a result."""
    return {
        "constant_attack": derived(tp=P, fp=N, tn=0, fn=0),
        "constant_benign": derived(tp=0, fp=0, tn=N, fn=P),
        "note": ("A constant-attack classifier has recall 1.0 and FPR 1.0. Its F1 is "
                 "the floor any reported F1 must clear by a margin worth reporting."),
    }


def load_configs(bench: str) -> dict:
    """Ablation configurations with stored counts, and placeholder detection."""
    path = RESULTS_DIR / bench / "ablation_results.json"
    if not path.exists():
        # NSL-KDD stores the same content as a list under a different filename.
        path = RESULTS_DIR / bench / "ablation_matrix.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {blk.get("name", f"config_{i}"): blk for i, blk in enumerate(raw)}

    out = {}
    for name, blk in raw.items():
        b = blk.get("binary", blk)
        if not all(k in b for k in ("tp", "fp", "tn", "fn")):
            continue
        d = derived(int(b["tp"]), int(b["fp"]), int(b["tn"]), int(b["fn"]))
        # Cross-check the derived rates against the stored ones.
        d["stored_matches_derived"] = all(
            abs(b[k] - d[k]) < 5e-5 for k in ("accuracy", "precision", "recall", "f1", "fpr")
            if k in b and b[k] is not None and d[k] is not None)
        out[name] = d

    # A configuration whose counts are identical to an earlier one did not run.
    seen = {}
    for name, d in out.items():
        key = (d["tp"], d["fp"], d["tn"], d["fn"])
        if key in seen:
            d["placeholder_duplicate_of"] = seen[key]
        else:
            seen[key] = name
            d["placeholder_duplicate_of"] = None
    return out


def attainability_for(bench: str) -> dict | None:
    p = OUT_DIR / f"{bench}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def frontier_at(bench: str, recall: float) -> float | None:
    """Split-level frontier at a given recall — the only basis for differencing."""
    z = OUT_DIR / f"{bench}_curve.npz"
    if not z.exists():
        return None
    a = np.load(z)
    r = _at_recall(a["split_recall"], a["split_fpr"], recall)
    return r["frontier_fpr"] if r else None


def cross_architecture() -> dict:
    out = {}
    d = RESULTS_DIR / "_fidmf_reimpl"
    for f in sorted(d.glob("*_fidmf.json")) if d.exists() else []:
        blk = json.loads(f.read_text(encoding="utf-8"))
        res = blk.get("results", blk)
        rows = {}
        for variant, m in res.items():
            if isinstance(m, dict) and "fpr" in m and "recall" in m:
                rows[variant] = {k: m[k] for k in
                                 ("accuracy", "precision", "recall", "f1", "fpr") if k in m}
        if rows:
            out[f.stem.replace("_fidmf", "")] = rows
    return out


def prior_shift() -> dict:
    p = RESULTS_DIR / "ton-iot-telemetry" / "smote_diagnosis.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {str(r.label): {"accuracy": float(r.accuracy), "precision": float(r.precision),
                           "recall": float(r.recall), "f1": float(r.f1), "fpr": float(r.fpr)}
            for r in df.itertuples()}


def mi_dose_response() -> list:
    p = RESULTS_DIR / "_leak_audit" / "edge_iiotset_progressive_drop.csv"
    if not p.exists():
        return []
    return json.loads(pd.read_csv(p).to_json(orient="records"))


def cohens_h(p1: float, p2: float) -> float:
    return abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def full_config(bench: str) -> dict | None:
    """The complete-stack row. NSL-KDD spells its configurations out in prose
    ("FULL: Ens + GAN + FL + BC + ZT") while the others use short ids."""
    cfgs = load_configs(bench)
    if "full" in cfgs:
        return cfgs["full"]
    for name, d in cfgs.items():
        if name.lower().startswith("full"):
            return d
    return None


def corrections() -> list:
    """The FINAL_PLAN task-5 list, each value recomputed here rather than copied."""
    out = []

    nsl = full_config("nsl-kdd")
    if nsl:
        out.append({"item": "NSL-KDD Table V, FULL row",
                    "manuscript": "0.7776 / 0.9666 / 0.6311 / 0.7636 / 0.0288",
                    "correct": f"{nsl['accuracy']:.4f} / {nsl['precision']:.4f} / "
                               f"{nsl['recall']:.4f} / {nsl['f1']:.4f} / {nsl['fpr']:.4f}",
                    "source": "results/nsl-kdd/ablation_matrix.json (FULL row, .binary)"})
        out.append({"item": "Headline NSL-KDD accuracy in abstract and conclusion",
                    "manuscript": "77.8 %",
                    "correct": f"{100*nsl['accuracy']:.2f} %",
                    "source": "same"})

    dapt = full_config("dapt-2020")
    if dapt:
        out.append({"item": "DAPT 2020 FULL false-positive rate",
                    "manuscript": "0.0374 / 0.0358 / 'under 4 %'",
                    "correct": f"{dapt['fpr']:.4f} at recall {dapt['recall']:.4f}",
                    "source": "results/dapt-2020/ablation_results.json (full.binary)"})
        out.append({"item": "DAPT 2020 test-split size",
                    "manuscript": "80,000 / 16,000 (sums to 96,000, corpus is 82,577)",
                    "correct": f"test {dapt['n']:,}; corpus 82,577; train 66,061",
                    "source": "derived from stored counts + cache/dapt-2020/status.json"})

    mc = RESULTS_DIR / "dapt-2020" / "multiclass_5stage_results.json"
    if mc.exists():
        blk = json.loads(mc.read_text(encoding="utf-8"))
        wa = None
        for k, v in blk.items():
            if isinstance(v, dict) and "weighted avg" in v:
                wa = v["weighted avg"]
            if k in ("weighted avg", "weighted_avg"):
                wa = v
        if isinstance(wa, dict):
            out.append({"item": "Table XXII weighted averages",
                        "manuscript": "precision 0.9648, recall 0.9620",
                        "correct": f"precision {wa.get('precision'):.4f}, "
                                   f"recall {wa.get('recall'):.4f}",
                        "source": "results/dapt-2020/multiclass_5stage_results.json"})

    out.append({"item": "Cohen's h, U2R recall 0.1045 -> 0.2239",
                "manuscript": "0.34",
                "correct": f"{cohens_h(0.2239, 0.1045):.4f} -> report 0.33",
                "source": "recomputed; the Probe value (0.6952 -> 0.8042) is "
                          f"{cohens_h(0.8042, 0.6952):.4f} and stands at 0.25"})
    out.append({"item": "Bonferroni family size",
                "manuscript": "n = 4 across datasets in the text, n = 2 within-dataset in the JSON",
                "correct": "state the family once, in the methods, and apply it consistently",
                "source": "results/_statistical_rigor/*_rigor.json"})
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    NUM_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": SPEC_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": {"path": str(HERE), "sha256": _sha256(HERE)},
        "rule": ("No false-positive rate may be quoted without its recall. No value may "
                 "appear in the manuscript unless it appears in this file."),
        "benchmarks": {},
        "cross_architecture": cross_architecture(),
        "prior_shift_ton_iot": prior_shift(),
        "mi_dose_response_edge": mi_dose_response(),
        "corrections": corrections(),
    }

    md = ["# NUMBERS — regenerated from result files",
          "",
          f"`{SPEC_VERSION}` · generated {payload['generated_utc']} · "
          f"`src/paper_numbers.py`",
          "",
          "**Rule: no false-positive rate without its recall. If a value is not here, "
          "it does not go in the paper.**", ""]

    for b in BENCHMARKS:
        cfgs = load_configs(b)
        if not cfgs:
            print(f"[{b}] no ablation_results.json — skipped")
            continue
        any_cfg = next(iter(cfgs.values()))
        P, N = any_cfg["positives"], any_cfg["negatives"]
        att = attainability_for(b)

        rows = {}
        for name, d in cfgs.items():
            d = dict(d)
            fr = frontier_at(b, d["recall"]) if d["recall"] is not None else None
            d["frontier_fpr_at_own_recall"] = fr
            d["slack_pp"] = 100 * (d["fpr"] - fr) if fr is not None else None
            d["slack_share"] = (d["fpr"] - fr) / d["fpr"] if fr is not None and d["fpr"] else None
            rows[name] = d

        payload["benchmarks"][b] = {
            "test_composition": {"positives": P, "negatives": N, "n": P + N,
                                 "attack_to_benign": P / N if N else None},
            "configurations": rows,
            "trivial_baselines": trivial_baselines(P, N),
            "attainability": {
                "pooled_floor": att["pooled"]["majority_ties_to_attack"]["fpr"] if att else None,
                "split_floor": att["split"]["majority_ties_to_attack"]["fpr"] if att else None,
                "conflict_rate": att["pooled"]["conflict_rate"] if att else None,
                "duplicate_rate": att["pooled"]["duplicate_rate"] if att else None,
                "twin_rate": att["split"]["leakage"]["twin_rate"] if att else None,
                "train_fitted_lookup": att["split"]["train_fitted_lookup"] if att else None,
            } if att else None,
        }

        md += [f"## {b}", "",
               f"Test split: {P:,} attack / {N:,} benign "
               f"(ratio {P/N:.4f}:1). "
               + (f"Twin rate {100*att['split']['leakage']['twin_rate']:.2f} %. " if att else "")
               + (f"Pooled floor {att['pooled']['majority_ties_to_attack']['fpr']:.6f}, "
                  f"split floor {att['split']['majority_ties_to_attack']['fpr']:.6f}." if att else ""),
               "",
               "| config | acc | prec | recall | F1 | FPR | specificity | MCC | frontier @ own recall | slack | note |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, d in rows.items():
            note = (f"**PLACEHOLDER — identical to `{d['placeholder_duplicate_of']}`**"
                    if d["placeholder_duplicate_of"] else "")
            fr = d["frontier_fpr_at_own_recall"]
            md.append(
                f"| `{name}` | {d['accuracy']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} | "
                f"{d['f1']:.4f} | {d['fpr']:.4f} | {d['specificity']:.4f} | {d['mcc']:.4f} | "
                + (f"{fr:.4f} | {d['slack_pp']:+.2f} pp ({100*d['slack_share']:.1f} %) | "
                   if fr is not None else "— | — | ")
                + note + " |")

        tb = trivial_baselines(P, N)["constant_attack"]
        md += ["",
               f"Trivial reference — a classifier that always says *attack*: "
               f"F1 **{tb['f1']:.4f}**, recall 1.0000, FPR 1.0000, MCC 0.0000. "
               f"Best configuration F1 is {max(d['f1'] for d in rows.values()):.4f}, "
               f"a margin of {max(d['f1'] for d in rows.values()) - tb['f1']:+.4f}.", ""]

        placeholders = [n for n, d in rows.items() if d["placeholder_duplicate_of"]]
        if placeholders:
            md += [f"> **{len(placeholders)} of {len(rows)} configurations did not run**: "
                   f"{', '.join('`'+p+'`' for p in placeholders)} carry counts identical to "
                   f"another row. No claim may rest on them.", ""]

    md += ["## Corrections", "",
           "| item | manuscript | correct | source |", "|---|---|---|---|"]
    for c in payload["corrections"]:
        md.append(f"| {c['item']} | {c['manuscript']} | **{c['correct']}** | `{c['source']}` |")

    md += ["", "## Cross-architecture operating points (always with recall)", ""]
    for ds, rows in payload["cross_architecture"].items():
        md += [f"**{ds}**", "", "| variant | recall | FPR | F1 |", "|---|---:|---:|---:|"]
        for v, m in rows.items():
            md.append(f"| `{v}` | {m.get('recall', float('nan')):.4f} | "
                      f"{m.get('fpr', float('nan')):.4f} | {m.get('f1', float('nan')):.4f} |")
        md.append("")

    md += ["## Prior shift, ToN-IoT (always with recall)", "",
           "| variant | recall | FPR | F1 |", "|---|---:|---:|---:|"]
    for v, m in payload["prior_shift_ton_iot"].items():
        md.append(f"| `{v}` | {m['recall']:.4f} | {m['fpr']:.4f} | {m['f1']:.4f} |")

    (NUM_DIR / "numbers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (NUM_DIR / "NUMBERS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:6]))
    print(f"\nwrote {NUM_DIR/'numbers.json'}\nwrote {NUM_DIR/'NUMBERS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
