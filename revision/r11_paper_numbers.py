"""
Collect every number quoted in the revised manuscript into one file
(revision/out/tables/paper_numbers.json) so the text is generated from results,
not typed by hand.  Also runs the McNemar family the reviewers asked for:
proposed (+ZT) against the simple alternatives, per seed, Holm-adjusted.
Usage: python revision\\r11_paper_numbers.py
"""
import json

import numpy as np
import pandas as pd

import rev_common as rc
from rev_common import OUT_DIR, PRED_DIR
from r02_tables_nslkdd import mcnemar, holm, zt_policy

T = OUT_DIR / "tables"


def main():
    out = {}
    s = json.loads((T / "nslkdd_summary.json").read_text())
    ps = json.loads((T / "nslkdd_per_seed.json").read_text())
    out["ablation"] = s["ablation"]; out["zt"] = s["zt"]
    seeds = s["seeds"]
    # ZT details averaged over seeds
    z = [ps[str(k)]["zt_default"] for k in seeds]
    out["zt_detail"] = {
        "challenges": [x["challenges"] for x in z], "correct": [x["correct_challenges"] for x in z],
        "false": [x["false_challenges"] for x in z], "eligible": [x["eligible_high_conf_benign_errors"] for x in z],
        "all_high_conf_errors": [x["all_high_conf_errors"] for x in z],
        "auto_only_recall": [x["binary_automatic_only"]["recall"] for x in z],
        "with_alert_recall": [x["binary_with_challenge_as_alert"]["recall"] for x in z],
        "actions_seed0": ps[str(seeds[0])]["zt_default"]["actions"],
        "tuning": [ps[str(k)]["zt_tuned"]["tuning"]["source"] for k in seeds]}
    out["bootstrap_seed0"] = ps[str(seeds[0])]["bootstrap_ci_zt_tuned"]

    # McNemar: proposed+ZT vs simple alternatives
    d = np.load(OUT_DIR / "nslkdd_data.npz"); y = d["y_test"]
    comps = {"Ensemble + class weighting only": ("none", None), "Ensemble + SMOTE": ("smote", None),
             "DNN alone (class-weighted)": ("none", "pt_nn"), "Logistic regression (class-weighted)": ("lr", None),
             "Ensemble + WGAN-GP/SMOTE (no ZT)": ("hybrid", None)}
    rows = []
    for k in seeds:
        P = np.load(PRED_DIR / f"nslkdd_s{k}_hybrid.npz"); R = np.load(PRED_DIR / f"nslkdd_s{k}_risk.npz")
        a = zt_policy(P["p_test"], R["r_test"], 0.85, 0.70)[1]
        res = []
        for name, (cfg, key) in comps.items():
            Q = np.load(PRED_DIR / f"nslkdd_s{k}_{cfg}.npz")
            b = Q[key].argmax(1) if key else Q["p_test"].argmax(1)
            res.append({"seed": k, "comparator": name, **mcnemar(y, a, b)})
        adj = holm([r["p_exact"] for r in res])
        for r, q in zip(res, adj):
            r["p_holm"] = q; rows.append(r)
    mc = pd.DataFrame(rows)
    mc.to_csv(T / "mcnemar_proposed_vs_alternatives.csv", index=False)
    out["mcnemar"] = mc.groupby("comparator").agg(
        proposed_right=("a_right_b_wrong", "mean"), comparator_right=("a_wrong_b_right", "mean"),
        p_holm_max=("p_holm", "max"), p_holm_min=("p_holm", "min")).round(6).to_dict("index")

    out["calibration"] = pd.read_csv(T / "calibration_summary.csv").to_dict("records")
    cal = json.loads((T / "calibration_per_seed.json").read_text())
    out["calibration_diag"] = pd.DataFrame([{"model": k.split("_", 1)[1], **{a: b for a, b in v.items() if a != "auc_test"},
                                             **{f"auc_{a}": b for a, b in v["auc_test"].items()}}
                                            for k, v in cal.items()]).groupby("model").mean(numeric_only=True).round(4).to_dict("index")
    out["validator"] = json.loads((T / "validator_independence.json").read_text())
    out["learning_curve_changeover"] = json.loads((T / "learning_curve_changeover.json").read_text())
    lc = pd.read_csv(T / "learning_curve_summary.csv")
    out["learning_curve"] = lc.round(4).to_dict("records")
    out["dapt_overall"] = pd.read_csv(T / "dapt_overall.csv").round(4).to_dict("records")
    out["dapt_stage"] = pd.read_csv(T / "dapt_per_stage.csv").round(4).to_dict("records")
    out["dapt_dayout"] = pd.read_csv(T / "dapt_day_out.csv").round(4).to_dict("records")
    out["adversarial"] = pd.read_csv(T / "adversarial_nslkdd.csv").round(4).to_dict("records")
    prot = json.loads((OUT_DIR / "nslkdd_protocol.json").read_text())
    out["protocol"] = {k: prot[k] for k in ("fit_counts", "val_counts", "test_counts", "risk_features",
                                            "hardware", "provenance")}
    lat = []
    for f in sorted(PRED_DIR.glob("nslkdd_s*_meta.json")):
        m = json.loads(f.read_text())
        if "ensemble_latency" in m["configs"].get("hybrid", {}):
            lat.append({"ens": m["configs"]["hybrid"]["ensemble_latency"], "risk": m["risk_latency"]})
    out["latency"] = {k: {kk: float(np.mean([x[k][kk] for x in lat])) for kk in
                          ("single_sample_median_ms", "single_sample_p95_ms", "batch_ms_per_sample")}
                      for k in ("ens", "risk")}
    # IoT corpora (r14, r15); augmentation is included only when all 15 runs exist
    iot = {"telemetry_audit": pd.read_csv(T / "toniot_telemetry_audit.csv").to_dict("records"),
           "protocols": {}, "unseen": {}, "per_type": {}, "augment": {}}
    for ds in ("ton_net", "edge"):
        iot["protocols"][ds] = pd.read_csv(T / f"iot_{ds}_protocols.csv").round(4).to_dict("records")
        iot["unseen"][ds] = pd.read_csv(T / f"iot_{ds}_unseen.csv").round(4).to_dict("records")
        iot["per_type"][ds] = pd.read_csv(T / f"iot_{ds}_per_type.csv").round(4).to_dict("records")
        f = T / f"iot_{ds}_augment.csv"
        if f.exists():
            a = pd.read_csv(f)
            if len(a) == 15:
                g = a.groupby("config")
                iot["augment"][ds] = {c: {k: {"mean": float(d[k].mean()), "std": float(d[k].std(ddof=1))}
                                          for k in ("macro_f1_type", "binary_f1", "binary_fpr", "type_accuracy")}
                                      for c, d in g}
    out["iot"] = iot
    # equal-budget tuning (r16)
    if (T / "tuning_final.csv").exists():
        tf = pd.read_csv(T / "tuning_final.csv"); ts = pd.read_csv(T / "tuning_summary.csv")
        cols = ["val_macro_f1", "test_binary_accuracy", "test_binary_f1", "test_binary_fpr", "test_macro_f1",
                "recall_Probe", "recall_R2L", "recall_U2R"]
        out["tuning"] = {r.treatment: {"trials": int(r.trials), "best_val_macro_f1": float(r.best_val_macro_f1),
                                       "minutes": float(r.tuning_minutes), "params": json.loads(r.best_params),
                                       **{c: {"mean": float(tf[tf.treatment == r.treatment][c].mean()),
                                              "std": float(tf[tf.treatment == r.treatment][c].std(ddof=1))}
                                          for c in cols}}
                         for r in ts.itertuples()}
    # UNSW-NB15 pre-registered confirmatory test (r17)
    if (T / "unsw_summary.json").exists():
        us = json.loads((T / "unsw_summary.json").read_text())
        ur = pd.read_csv(T / "unsw_results.csv")
        reg = json.loads((OUT_DIR / "unsw" / "PREREGISTRATION.json").read_text())
        agg = {}
        for (cfg, sub), g in ur.groupby(["config", "subset"]):
            agg.setdefault(sub, {})[cfg] = {c: {"mean": float(g[c].mean()), "std": float(g[c].std(ddof=1))}
                                            for c in g.columns if c not in ("seed", "config", "subset")}
        auc = pd.read_csv(T / "unsw_binary_auc.csv", header=[0, 1], index_col=0)
        out["unsw"] = {"summary": us, "results": agg, "prereg_written_utc": reg["written_utc"],
                       "prereg_script_sha256": reg["script_sha256"],
                       "auc": {c: {"auc": float(auc.loc[c, ("auc_attack_score", "mean")]),
                                   "binary_recall": float(auc.loc[c, ("binary_recall", "mean")])} for c in auc.index}}
    rc.dump_json(out, T / "paper_numbers.json")
    print(mc.groupby("comparator")[["a_right_b_wrong", "a_wrong_b_right", "p_holm"]].agg(["mean", "max"]).round(4))
    print("\n[r11] ->", T / "paper_numbers.json")


if __name__ == "__main__":
    main()
