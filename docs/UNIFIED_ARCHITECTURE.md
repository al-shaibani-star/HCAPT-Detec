# HCAPT-Defense — Unified Architecture Specification

**Purpose:** Formal design document for the end-to-end integration of
the three research innovations with Layer 3 (Federated Learning +
Blockchain Threat Intelligence). This document must be approved
before implementation begins.

**Last updated:** 2026-04-22
**Status:** DRAFT — awaiting approval

---

## 1. Scope

This document specifies:
1. The data-flow contract between every layer and every innovation.
2. The precise module boundaries and function signatures.
3. The temporal phases (offline training, online inference).
4. The trust and privacy guarantees of each edge in the graph.
5. The evaluation protocol that exercises every edge.

It does **not** re-explain what each innovation does — for that see
`RESULTS_SUMMARY.md` and `INTERPRETATION.md`.

---

## 2. Integration Graph (formal)

### 2.1 Vertex set

| ID     | Component                          | File                          |
|--------|------------------------------------|-------------------------------|
| L1     | Preprocessing + cache              | `data_loader.py`              |
| I1     | Innovation #1 — GAN/WGAN-GP        | `gan_augment.py`, `wgan_gp_augment.py` |
| L3a    | Layer 3a — Federated Learning      | `federated_learning.py`       |
| L3b    | Layer 3b — Blockchain + Threat Intel | `blockchain.py`, `threat_intelligence.py` |
| I2     | Innovation #2 — Ensemble + Self-Learning | `ensemble_selflearn.py`  |
| I3     | Innovation #3 — Zero-Trust         | `zero_trust.py`               |
| L5     | Response decision                  | (policy output in I3)         |

### 2.2 Directed edge set (contracts)

```
L1  --[X_tr,  y_tr,   X_te,  y_te]-->  I1
L1  --[X_tr,  y_tr,   X_te,  y_te]-->  I2  (baseline path)
L1  --[X_te,  y_te]----------------->  I3  (inference input)

I1  --[X_bal, y_bal  (336,715 × 41)]->  L3a  (federated clients)
I1  --[X_bal, y_bal]-------------------> I2   (RF + XGBoost training)

L3a --[global_NN_state_dict]----------> I2   (Deep-NN base learner)
L3a --[round_hashes, metadata]--------> L3b  (audit trail)

I2  --[ensemble_preds, confidences]--> L3b  (IoC harvest after training)
L3b --[enriched_preds, enriched_confs]-> I2  (online enrichment)

I2  --[(enriched_pred, enriched_conf)]-> I3   (Zero-Trust input)

I3  --[action ∈ {ALLOW, MONITOR, CHALLENGE, QUARANTINE, BLOCK}]-> L5
```

Every edge MUST be covered by at least one integration test.

---

## 3. Temporal phases

### Phase A — One-time offline setup (re-runnable via cache)

```
t=0  :  Run data_loader.get_preprocessed() → cached .npz
t=1  :  Run wgan_gp_augment (I1)
        - Train 100 epochs with checkpointing
        - Generate X_bal (336K samples) → cached
t=2  :  Run federated_learning (L3a)
        - 5 clients, non-IID partition of X_bal
        - 20 rounds × 3 local epochs
        - Final global NN → saved to disk
        - Every round's weight hash → Blockchain
t=3  :  Train RF + XGBoost centrally on X_bal (still part of I2)
t=4  :  Assemble I2 ensemble:
          ens.rf   = trained RF
          ens.xgb  = trained XGBoost
          ens.nn   = FL global model (from L3a)
          ens.weights = learned from val accs
t=5  :  Each client harvests IoCs from its local correct-attack
        predictions with conf >= 0.85 → submit to Blockchain
t=6  :  Train Zero-Trust Risk Scorer on X_bal (I3)
t=7  :  Verify Blockchain integrity
```

### Phase B — Online inference (per-sample or per-batch)

```
for each sample x in stream:
    1. x_scaled = scaler.transform(x)          # from L1
    2. pred, conf = ensemble.predict(x_scaled)   # from I2
    3. enriched_pred, enriched_conf = TI.enrich(x_scaled, pred, conf)
                                                 # from L3b
    4. action = ZT.validate(x_scaled,
                            enriched_pred,
                            enriched_conf)        # from I3
    5. execute(action)                          # L5
```

Latency budget: 1 second per sample (target from HCAPT spec).
Current measurement: ~0.024 ms/sample — 41,000× headroom.

---

## 4. Integration contracts (precise)

### 4.1 I1 → L3a (GAN → FL)

```python
# I1 produces:
X_bal: np.ndarray of shape (336715, 41), dtype=float32
y_bal: np.ndarray of shape (336715,),   dtype=int64
# class counts: {0: 67343, 1: 67343, 2: 67343, 3: 67343, 4: 67343}

# L3a consumes:
parts = partition_non_iid(X_bal, y_bal, n_clients=5, concentration=0.5)
# Each part: (X_i, y_i) goes to FLClient(client_id, X_i, y_i, ...)
```

**Privacy note:** `X_bal` contains GAN-synthesized samples. Real samples
remain in the aggregator only before partitioning; clients only see
their partition.

### 4.2 L3a → I2 (FL → Ensemble)

```python
# L3a exposes:
server.global_model: torch.nn.Module  # DeepDetector
server.global_state() -> Dict[str, torch.Tensor]

# I2 integration:
ens = EnsembleDetector(feature_dim=41)
ens.rf.fit(X_bal, y_bal)
ens.xgb.fit(X_bal, y_bal)
ens.nn.load_state_dict(server.global_state())  # ← key integration
ens.weights = softmax([rf_val_acc, xgb_val_acc, nn_val_acc] * 10)
```

**Design decision:** RF and XGBoost are trained centrally on balanced
data. Only the Deep NN is federated. The paper must explicitly state
this scoping.

### 4.3 L3a → L3b (FL audit trail)

```python
# Every FL round end:
wh = server.weight_hash()  # SHA-256 of (sorted) state_dict
audit_ioc = IoC(
    feature_vector=encode_hash_as_vector(wh, dim=41),
    attack_class=0,              # "Normal" = metadata carrier
    severity=0,
    confidence=0.0,
    source_node="FL_AGGREGATOR",
    timestamp=time.time(),
    attack_name=f"fl_round_{r}_hash_{wh[:12]}",
)
blockchain.submit_ioc(audit_ioc)
```

**Security guarantee:** If any client tampers with the aggregated
weights afterwards, the reconstructed hash will differ from the
blockchain record — detectable via `verify_weights_against_chain()`.

### 4.4 I2 → L3b (IoC harvest)

```python
# After training, EACH CLIENT (not aggregator):
for client in clients:
    preds, confs = local_predict(client.model, client.X)
    mask = (preds == client.y) & (preds != 0) & (confs >= 0.85)
    for i in np.where(mask)[0][:100]:                # cap per client
        ioc = IoC(
            feature_vector=client.X[i].tolist(),     # 41-dim
            attack_class=int(preds[i]),
            severity=SEV_MAP[int(preds[i])],
            confidence=float(confs[i]),
            source_node=client.client_id,
            timestamp=time.time(),
        )
        blockchain.submit_ioc(ioc)
blockchain.mine_pending(force=True)
```

**Privacy note:** Feature vectors are shared (already scaled / encoded).
Not raw packet data, but reversible in principle. This is the
weakest privacy link in the design. Options for strengthening:
- Hash-only IoCs (lose similarity search)
- Differential-private noise (configurable ε)
- k-anonymity filter before submit

### 4.5 L3b → I2 (enrichment)

```python
# Online inference path:
ens_preds, ens_confs = ensemble.predict_with_confidence(X_batch)

ti = ThreatIntelligenceLayer(blockchain,
                              similarity_threshold=0.90,
                              class_vote_required=2,
                              min_confidence_boost=0.05)
enriched_preds, enriched_confs, _ = ti.enrich(
    X_batch, ens_preds, ens_confs)
```

**Decision rules implemented in `ti.enrich()`:**
1. If `ens_pred == Normal` AND blockchain has ≥2 matching IoCs with
   `class != Normal` AND `similarity >= 0.90` → **OVERRIDE** to attack.
2. If `ens_pred == attack X` AND blockchain has matching IoCs with
   `class == X` → **BOOST** confidence by `min_confidence_boost`.
3. Otherwise — pass through unchanged.

### 4.6 I2 → I3 (Ensemble → Zero-Trust)

```python
# Zero-Trust receives enriched predictions:
zt = ZeroTrustLayer(risk_scorer, policy=PolicyEngine())
decisions = zt.validate_batch(X_batch,
                              enriched_preds,
                              enriched_confs,
                              return_decisions=True)
```

**Double-check nature:** The Zero-Trust RiskScorer is independent of
both the ensemble AND the blockchain — providing a third
independent signal. If all three (ensemble, blockchain, risk scorer)
agree, confidence is maximal. Disagreements trigger CHALLENGE.

---

## 5. Module changes required

### 5.1 `ensemble_selflearn.py`
- Add `EnsembleDetector.load_fl_nn(state_dict)` method to plug FL
  weights into the ensemble's DeepDetector.
- Add `EnsembleDetector.enrich_with_blockchain(bc)` optional wrapper.

### 5.2 `fl_blockchain_demo.py` (new master pipeline)
Already mostly built. Remaining changes:
- Use GAN-balanced data as input to `partition_non_iid`.
- After FL completes, construct full ensemble with FL NN.
- Run Blockchain IoC harvest from clients.
- Run full pipeline evaluation (I2 + L3b + I3 combined).

### 5.3 `master_pipeline.py` (to be created)
Single orchestrator:
```python
def main():
    # Phase A: offline setup
    run_I1_gan()
    run_L3a_fl_training()
    assemble_I2_ensemble()
    harvest_iocs_to_L3b()
    train_I3_risk_scorer()
    verify_blockchain()

    # Phase B: evaluation
    evaluate_all_configurations()
    save_figures()
```

### 5.4 Evaluation matrix

| Configuration | Trained NN | Ensemble | BC enrich | ZT |
|---|---|---|---|---|
| Baseline XGB | — | — | — | — |
| Ens (central NN) | central | yes | no | no |
| Ens + GAN | central | yes | no | no |
| Ens + GAN + ZT | central | yes | no | yes |
| **Ens + GAN + FL** | **federated** | yes | no | no |
| **Ens + GAN + FL + BC** | **federated** | yes | **yes** | no |
| **Ens + GAN + FL + BC + ZT** | **federated** | yes | **yes** | **yes** |

We measure the **incremental contribution of FL and BC** as separate
rows, closing the ablation.

---

## 6. Privacy & security claims

| Claim | Mechanism | Verifiable by |
|---|---|---|
| Raw training data never leaves a client | FL local training | inspection of `federated_learning.py` |
| Shared IoC features cannot be trivially reversed to raw packets | NSL-KDD dataset already provides post-extraction features | dataset provenance |
| Tampering with aggregated weights is detectable | SHA-256 hash per round on Blockchain | `verify_chain()` |
| Tampering with shared IoCs is detectable | IoCs in hash-linked blocks | `verify_chain()` |
| Malicious clients' contributions are bounded | (not implemented) — future work via Krum or median aggregation | — |
| Differential privacy on IoCs | (not implemented in baseline) — future work | — |

---

## 7. Evaluation metrics (unified)

For every row of the matrix in §5.4 we report:

**Multi-class:**
- Accuracy, Precision/Recall/F1 (weighted), FPR
- Per-class P/R/F1 for Normal, DoS, Probe, R2L, U2R

**Binary (attack detection):**
- Same metrics, computed on `y_bin = y != 0`

**Operational:**
- Inference latency (ms/sample)
- Memory footprint
- Blockchain size (blocks, MB)
- FL communication cost (MB/round)

**Privacy / security:**
- Chain verification time
- Number of IoCs on chain
- Hash collisions (should be 0)

---

## 8. Open design questions (requiring user input)

Before implementation, resolve:

1. **IoC content policy** (Section 4.4):
   - (a) Full feature vectors *[currently]*
   - (b) Hash-only
   - (c) DP-noised
2. **Consensus mechanism** (blockchain.py):
   - (a) PoW *[currently, difficulty=3]*
   - (b) PoA (move to authority-based)
   - (c) Simple round-robin (trivial, demo only)
3. **Number of clients** (L3a):
   - 5 *[current]*, or 3, 10
4. **Data partition strategy**:
   - IID / non-IID (concentration=0.5) *[current]*
5. **Similarity threshold** for TI layer:
   - 0.85 *[current]* or tune to maximize challenge precision

---

## 9. Implementation plan (after approval)

1. **Fix FedAvg BatchNorm bug** (already patched in current code; verify).
2. **Wire GAN-balanced data into FL partition** (currently uses raw data).
3. **Create `master_pipeline.py`** executing Phase A end-to-end.
4. **Add `ensemble.load_fl_nn()` integration method.**
5. **Create `evaluate_all_configurations.py`** producing the ablation
   matrix of §5.4.
6. **Regenerate figures** with the new rows.
7. **Update `INTERPRETATION.md`** with FL + BC findings.
8. **Update LaTeX snippets** (new figures + tables).

Estimated time: ~25 min of compute (GAN already cached, FL fresh run).

---

## 10. Sign-off checklist

- [ ] User approves the integration graph (§2.2)
- [ ] User decides on IoC content policy (§8.1)
- [ ] User decides on consensus mechanism (§8.2)
- [ ] User confirms client count & partition (§8.3, §8.4)
- [ ] User confirms similarity threshold (§8.5)

Once all boxes checked, proceed with §9.
