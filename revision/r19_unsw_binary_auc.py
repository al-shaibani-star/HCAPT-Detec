"""
Post-hoc (NOT pre-registered) descriptive analysis for UNSW-NB15: threshold-free ROC-AUC of the
attack score 1 - p(Normal), binary recall and FPR per configuration, from the frozen r17
predictions.  Kept separate from r17_unsw_confirmatory.py so that the script hash recorded in
out/unsw/PREREGISTRATION.json still matches the script that produced the confirmatory results.
Output: revision/out/tables/unsw_binary_auc.csv
Usage:  python revision\\r19_unsw_binary_auc.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import r17_unsw_confirmatory as u


def main():
    D = u.load(False); yb = D["y_test"] != 0
    rows = []
    for c in u.CONFIGS:
        for s in u.SEEDS:
            p = np.load(u.PRED / f"s{s}_{c}.npz")["p_test"]; pr = p.argmax(1) != 0
            rows.append({"config": c, "seed": s, "auc_attack_score": roc_auc_score(yb, 1 - p[:, 0]),
                         "binary_recall": (pr & yb).sum() / yb.sum(), "fpr": (pr & ~yb).sum() / (~yb).sum()})
    d = pd.DataFrame(rows).groupby("config")[["auc_attack_score", "binary_recall", "fpr"]].agg(["mean", "std"]).round(4)
    d.to_csv(u.rc.OUT_DIR / "tables" / "unsw_binary_auc.csv")
    print(d.sort_values(("auc_attack_score", "mean"), ascending=False).to_string())


if __name__ == "__main__":
    main()
