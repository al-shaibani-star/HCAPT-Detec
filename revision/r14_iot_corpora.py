"""
Replace the submitted ToN-IoT/Edge-IIoTset results with leakage-aware protocols
(Reviewer 2, comment 4; Reviewer 1, comment 8).

Datasets
  ton_net  ToN-IoT network flows (train_test_network.csv, 211,043 flows, 51 source IPs).
           Identifiers and free text dropped: src/dst IP and port, dns_query,
           ssl_subject, ssl_issuer, http_uri, http_user_agent, weird_addl.
           Group = source IP (attacker hosts never shared between train and test).
  edge     Edge-IIoTset ML subset (157,800 rows); the leakage-audited column list of
           the submitted version is dropped.  The source addresses are largely spoofed
           and the timestamps unusable, so group = identical feature vector
           (duplicates never cross a split).
  (ToN-IoT telemetry is audited separately by r15 and is not used for modelling.)

Parts
  protocols  multiclass (attack type) models lr / xgb / ens under
               R  stratified 5-fold over rows (the submitted protocol)
               G  stratified group 5-fold (groups above)
             plus U: leave-one-attack-type-out binary detection (train on normal +
             all other types; test on the unseen type + held-out normal groups)
  augment    ens under protocol G with no resampling / SMOTE / WGAN-GP, every class
             topped up to the majority count of the training fold (multiclass);
             RF trees on bootstrap samples of <= 50,000 rows (RF_AUGMENT), same for all three

Preprocessing (categorical vocabulary, scaling) is fitted per training fold.
Outputs: revision/out/tables/iot_<dataset>_<part>*.csv
Usage (PowerShell, from 05-Implementation):
  python -u revision\\r14_iot_corpora.py --dataset ton_net --part protocols --smoke
  python -u revision\\r14_iot_corpora.py --dataset ton_net --part protocols
  python -u revision\\r14_iot_corpora.py --dataset edge    --part protocols
  python -u revision\\r14_iot_corpora.py --dataset ton_net --part augment
  python -u revision\\r14_iot_corpora.py --dataset edge    --part augment
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit

import rev_common as rc
from rev_common import OUT_DIR
from r08_dapt_stages import fit_predict, wilson, release_gpu
import wgan_gp_augment as wg

DATA = rc.SRC_DIR.parent / "data"
TON_DROP = ["src_ip", "dst_ip", "src_port", "dst_port", "dns_query", "ssl_subject", "ssl_issuer",
            "http_uri", "http_user_agent", "weird_addl"]
EDGE_DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size",
             "arp.src.proto_ipv4", "http.file_data", "http.referer", "http.request.full_uri",
             "http.request.uri.query", "http.request.method", "http.request.version", "dns.qry.name",
             "mqtt.topic", "mqtt.msg", "mqtt.msg_decoded_as", "mqtt.protoname", "tcp.payload", "tcp.options",
             "dns.qry.name.len", "mqtt.conack.flags", "tcp.dstport", "tcp.srcport", "tcp.seq", "tcp.ack"]
T = OUT_DIR / "tables"
# Augmented training sets reach 290k-400k rows over 10-15 classes; full-depth RF trees on
# all of them exhausted 31 GB RAM.  In the augment part only, every tree is grown on a
# bootstrap sample of at most 50,000 rows and at most 4 trees are built in parallel.
RF_AUGMENT = {"max_samples": 50000, "n_jobs": 4}


def load(name, smoke):
    if name == "ton_net":
        df = pd.read_csv(DATA / "ton-iot" / "network" / "train_test_network.csv", low_memory=False)
        types = df["type"].astype(str).str.strip().str.lower()
        groups = df["src_ip"].astype(str).values
        feats = df.drop(columns=TON_DROP + ["label", "type"])
    else:
        df = pd.read_csv(DATA / "edge-iiotset" / "ml-dl-selected" / "ML-EdgeIIoT-dataset.csv", low_memory=False)
        types = df["Attack_type"].astype(str).str.strip()
        feats = df.drop(columns=[c for c in EDGE_DROP if c in df.columns] + ["Attack_label", "Attack_type"])
        groups = pd.util.hash_pandas_object(feats.astype(str), index=False).astype(str).values
    normal = "normal" if name == "ton_net" else "Normal"
    classes = [normal] + sorted(t for t in types.unique() if t != normal)
    y = types.map({c: i for i, c in enumerate(classes)}).values.astype(int)
    if smoke:
        idx = np.random.default_rng(0).choice(len(y), 20000, replace=False)
        feats, y, groups = feats.iloc[idx].reset_index(drop=True), y[idx], groups[idx]
    cat = [c for c in feats.columns if feats[c].dtype == object]
    num = [c for c in feats.columns if c not in cat]
    for c in num:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    for c in cat:
        feats[c] = feats[c].astype(str).str.strip()
    print(f"[{name}] rows {len(y):,} | numeric {len(num)} | categorical {len(cat)} | classes {len(classes)} "
          f"| groups {len(set(groups)):,} | duplicate feature rows {feats.duplicated().mean():.1%}")
    return feats, y, groups, classes, num, cat


def encode(F, tr, te, num, cat):
    """Categorical vocabulary from the training rows only (unseen -> extra code)."""
    A = F.iloc[tr]; B = F.iloc[te]
    outA = [np.nan_to_num(A[num].values.astype(np.float64))]; outB = [np.nan_to_num(B[num].values.astype(np.float64))]
    for c in cat:
        vocab = {v: i for i, v in enumerate(pd.unique(A[c]))}
        outA.append(A[c].map(vocab).values[:, None].astype(np.float64))
        outB.append(B[c].map(vocab).fillna(len(vocab)).values[:, None].astype(np.float64))
    Xa, Xb = np.hstack(outA), np.hstack(outB)
    return np.clip(Xa, -1e12, 1e12).astype(np.float32), np.clip(Xb, -1e12, 1e12).astype(np.float32)


def scores(y, pred, classes):
    b = rc.binary_metrics(y, pred)
    f1s = []
    for c in range(len(classes)):
        tp = ((pred == c) & (y == c)).sum(); den = (pred == c).sum() + (y == c).sum()
        f1s.append(2 * tp / den if den else np.nan)
    return {"binary_f1": b["f1"], "binary_recall": b["recall"], "binary_fpr": b["fpr"],
            "type_accuracy": float((pred == y).mean()), "macro_f1_type": float(np.nanmean(f1s))}


def part_protocols(name, F, y, groups, classes, num, cat, models):
    k = len(classes); rows, per_type = [], []
    for proto in ("R random 5-fold", "G grouped 5-fold"):
        cv = (StratifiedKFold(5, shuffle=True, random_state=0).split(F, y) if proto.startswith("R")
              else StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, groups))
        splits = list(cv)
        for m in models:
            t0 = time.time(); oof = np.zeros((len(y), k), np.float32)
            for f, (tr, te) in enumerate(splits):
                Xa, Xb = encode(F, tr, te, num, cat)
                oof[te] = fit_predict(m, Xa, y[tr], Xb, k, f)
            pred = oof.argmax(1)
            rows.append({"dataset": name, "protocol": proto, "model": m, **scores(y, pred, classes),
                         "seconds": round(time.time() - t0, 1)})
            for c in range(1, k):
                msk = y == c; r = wilson(int((pred[msk] == c).sum()), int(msk.sum()))
                d = wilson(int((pred[msk] != 0).sum()), int(msk.sum()))
                per_type.append({"dataset": name, "protocol": proto, "model": m, "type": classes[c],
                                 "n": int(msk.sum()), "type_recall": r[0], "type_ci95": f"[{r[1]:.3f}, {r[2]:.3f}]",
                                 "detected_as_attack": d[0]})
            print(f"[{name}] {proto} | {m}: {rows[-1]}")
    # U: unseen attack type (binary)
    nidx = np.flatnonzero(y == 0)
    gs = GroupShuffleSplit(1, test_size=0.2, random_state=0)
    ntr, nte = next(gs.split(nidx, groups=groups[nidx])); ntr, nte = nidx[ntr], nidx[nte]
    unseen = []
    for c in range(1, k):
        att = np.flatnonzero(y == c); others = np.flatnonzero((y != 0) & (y != c))
        tr = np.r_[ntr, others]; te = np.r_[nte, att]
        yb_tr = (y[tr] != 0).astype(int); yb_te = (y[te] != 0).astype(int)
        Xa, Xb = encode(F, tr, te, num, cat)
        for m in models:
            pr = fit_predict(m, Xa, yb_tr, Xb, 2, 0).argmax(1)
            rec = wilson(int((pr[yb_te == 1] == 1).sum()), int((yb_te == 1).sum()))
            fpr = wilson(int((pr[yb_te == 0] == 1).sum()), int((yb_te == 0).sum()))
            unseen.append({"dataset": name, "unseen_type": classes[c], "model": m, "attack_rows": len(att),
                           "recall": rec[0], "recall_ci95": f"[{rec[1]:.3f}, {rec[2]:.3f}]",
                           "fpr": fpr[0], "fpr_ci95": f"[{fpr[1]:.4f}, {fpr[2]:.4f}]"})
        print(f"[{name}] U unseen {classes[c]} done")
    return rows, per_type, unseen


def topup(kind, X, y, k, seed, gan_epochs):
    counts = np.bincount(y, minlength=k); target = int(counts.max())
    need = {c: target - int(counts[c]) for c in range(k) if 0 < counts[c] < target}
    if kind == "none" or not need:
        return X, y
    if kind == "smote":
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        kk = min(5, min(counts[c] for c in need) - 1)
        s = (SMOTE(sampling_strategy={c: target for c in need}, k_neighbors=kk, random_state=seed) if kk >= 1
             else RandomOverSampler(sampling_strategy={c: target for c in need}, random_state=seed))
        return s.fit_resample(X, y)
    # WGAN-GP on standardised features (the generator has no output bound)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    wg.NUM_CLASSES = k; wg.save_checkpoint = lambda *a, **kw: None
    wg.MODELS_DIR = OUT_DIR / "models_tmp"; wg.MODELS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    G, C, _ = wg.train_wgan_gp(Z, y, epochs=gan_epochs, batch_size=512, resume=False)
    Xs, ys = wg.generate_synthetic(G, need)
    del G, C; release_gpu()
    return np.vstack([X, Xs * sd + mu]).astype(np.float32), np.r_[y, ys].astype(int)


def part_augment(name, F, y, groups, classes, num, cat, gan_epochs, out_csv, only=None):
    k = len(classes)
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv); done = set(zip(prev.fold, prev.config))
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, groups))
    for f, (tr, te) in enumerate(splits):
        if only is not None and f != only[0]:
            continue
        Xa, Xb = encode(F, tr, te, num, cat)
        for cfg in ("none", "smote", "wgan"):
            if (f, cfg) in done or (only is not None and cfg != only[1]):
                continue
            t0 = time.time()
            Xr, yr = topup(cfg, Xa, y[tr], k, f, gan_epochs)
            pred = fit_predict("ens", Xr, yr, Xb, k, f, rf_kw=RF_AUGMENT).argmax(1)
            yt = y[te]
            row = {"dataset": name, "fold": f, "config": cfg, **scores(yt, pred, classes),
                   "seconds": round(time.time() - t0, 1)}
            for c in range(1, k):
                msk = yt == c
                row[f"recall_{classes[c]}"] = float((pred[msk] == c).mean()) if msk.any() else np.nan
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            del Xr, yr; release_gpu()
            print(f"[{name}] augment fold {f} {cfg}: macroF1 {row['macro_f1_type']:.4f} "
                  f"binF1 {row['binary_f1']:.4f} ({row['seconds']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ton_net", "edge"], required=True)
    ap.add_argument("--part", choices=["protocols", "augment"], required=True)
    ap.add_argument("--models", nargs="+", default=["lr", "xgb", "ens"])
    ap.add_argument("--gan-epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", nargs=2, metavar=("FOLD", "CONFIG"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.smoke:
        a.gan_epochs = 1
    sfx = "_smoke" if a.smoke else ""
    T.mkdir(exist_ok=True)
    if a.part == "protocols" or a.worker:      # the augment orchestrator does not load data
        F, y, groups, classes, num, cat = load(a.dataset, a.smoke)
    if a.part == "protocols":
        rows, per_type, unseen = part_protocols(a.dataset, F, y, groups, classes, num, cat, a.models)
        pd.DataFrame(rows).to_csv(T / f"iot_{a.dataset}_protocols{sfx}.csv", index=False)
        pd.DataFrame(per_type).to_csv(T / f"iot_{a.dataset}_per_type{sfx}.csv", index=False)
        pd.DataFrame(unseen).to_csv(T / f"iot_{a.dataset}_unseen{sfx}.csv", index=False)
        pd.set_option("display.width", 250)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print(pd.DataFrame(unseen).round(3).to_string(index=False))
    else:
        out = T / f"iot_{a.dataset}_augment{sfx}.csv"
        if a.worker:
            part_augment(a.dataset, F, y, groups, classes, num, cat, a.gan_epochs, out,
                         only=(int(a.worker[0]), a.worker[1]))
            return
        if a.smoke and out.exists():
            out.unlink()
        # One fresh process per (fold, config): memory held by RF workers, GAN tensors and
        # previous training sets accumulated inside a single process and exhausted RAM.
        import subprocess, sys
        for f in range(5):
            for cfg in ("none", "smote", "wgan"):
                if out.exists() and ((pd.read_csv(out).fold == f) & (pd.read_csv(out).config == cfg)).any():
                    continue
                cmd = [sys.executable, "-u", __file__, "--dataset", a.dataset, "--part", "augment",
                       "--gan-epochs", str(a.gan_epochs), "--worker", str(f), cfg] + (["--smoke"] if a.smoke else [])
                for attempt in (1, 2):
                    if subprocess.run(cmd).returncode == 0:
                        break
                    print(f"[{a.dataset}] fold {f} {cfg}: attempt {attempt} failed" + (", retrying" if attempt == 1 else ""))
                else:
                    raise SystemExit(f"fold {f} {cfg} failed twice; re-run the same command to resume")
        d = pd.read_csv(out)
        print(d.groupby("config")[["macro_f1_type", "binary_f1", "binary_fpr"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
