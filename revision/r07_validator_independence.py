"""
Step 5 of the IJIES revision (Reviewer 1, comment 6).

The paper claimed the Zero-Trust validator used a feature subset "disjoint" from
the ensemble.  In the code the ensemble sees all 41 NSL-KDD features and the
validator sees the 20 odd-indexed ones, i.e. a strict SUBSET.  This script
writes the exact column lists and measures dependence of the two models' errors
on held-out data (VAL and KDDTest+), as the reviewer asked:

  phi                 Pearson correlation of the binary error indicators
  P(V wrong)          marginal validator error rate
  P(V wrong | E wrong) conditional validator error rate given an ensemble error
  lift                P(V wrong | E wrong) / P(V wrong)   (1 = independent)
  joint / expected    observed joint errors vs. the count expected if independent
  Yule's Q            association of the 2x2 error table (0 = independent)

Errors are binary (Normal vs attack); the validator is thresholded at 0.5.
Output: revision/out/tables/validator_independence.{csv,json}
Usage:  python revision\\r07_validator_independence.py
"""
import json

import numpy as np
import pandas as pd

import rev_common as rc
from rev_common import OUT_DIR, PRED_DIR, RISK_FEATURES

BASES = {"hybrid": "Ensemble + WGAN-GP/SMOTE (proposed)", "none": "Ensemble + class weighting only"}


def dependence(e, v):
    n = len(e); a = int((e & v).sum()); b = int((e & ~v).sum())
    c = int((~e & v).sum()); d = int((~e & ~v).sum())
    pv = v.mean(); pv_e = v[e].mean() if e.any() else np.nan
    expected = e.mean() * pv * n
    q = (a * d - b * c) / (a * d + b * c) if (a * d + b * c) else np.nan
    return {"phi": float(np.corrcoef(e, v)[0, 1]), "P(V wrong)": float(pv),
            "P(V wrong | E wrong)": float(pv_e), "lift": float(pv_e / pv),
            "joint_errors": a, "expected_if_independent": float(expected),
            "ensemble_errors": int(e.sum()), "validator_errors": int(v.sum()),
            "yule_q": float(q)}


def main():
    d = np.load(OUT_DIR / "nslkdd_data.npz", allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    y = {"VAL": d["y_val"], "TEST": d["y_test"]}
    seeds = sorted({int(f.name.split("_s")[1].split("_")[0]) for f in PRED_DIR.glob("nslkdd_s*_risk.npz")})
    rows = []
    for s in seeds:
        R = np.load(PRED_DIR / f"nslkdd_s{s}_risk.npz")
        r = {"VAL": R["r_val"], "TEST": R["r_test"]}
        for b, bname in BASES.items():
            P = np.load(PRED_DIR / f"nslkdd_s{s}_{b}.npz")
            p = {"VAL": P["p_val"], "TEST": P["p_test"]}
            for split in ("VAL", "TEST"):
                yt = y[split] != 0
                e = (p[split].argmax(1) != 0) != yt
                v = (r[split] >= 0.5) != yt
                rows.append({"seed": s, "ensemble": bname, "split": split, **dependence(e, v)})
    df = pd.DataFrame(rows)
    agg = df.drop(columns="seed").groupby(["ensemble", "split"]).agg(["mean", "std"])
    tdir = OUT_DIR / "tables"; tdir.mkdir(exist_ok=True)
    df.to_csv(tdir / "validator_independence.csv", index=False)
    feats = {"ensemble_features (all 41)": names,
             "validator_features (odd-indexed, 20)": [names[i] for i in RISK_FEATURES],
             "features_only_in_ensemble": [n for i, n in enumerate(names) if i not in RISK_FEATURES],
             "features_only_in_validator": [],
             "relation": "validator features are a strict subset of the ensemble features (not disjoint)",
             "validator_model": "LightGBM, 400 trees, depth 6, lr 0.05, class_weight=balanced, "
                                "seed = training seed + 777, trained on real FIT rows only"}
    rc.dump_json({"features": feats, "summary_mean": {" | ".join(k): v for k, v in
                                     agg.xs("mean", axis=1, level=1).round(4).to_dict("index").items()}},
                 tdir / "validator_independence.json")
    pd.set_option("display.width", 250)
    print("Validator features (20):", ", ".join(feats["validator_features (odd-indexed, 20)"]))
    print("\nError dependence, mean (sd) over seeds")
    m = agg.xs("mean", axis=1, level=1); sd = agg.xs("std", axis=1, level=1)
    cols = ["phi", "P(V wrong)", "P(V wrong | E wrong)", "lift", "joint_errors",
            "expected_if_independent", "yule_q"]
    out = m[cols].round(3).astype(str) + " (" + sd[cols].round(3).astype(str) + ")"
    print(out.to_string())


if __name__ == "__main__":
    main()
