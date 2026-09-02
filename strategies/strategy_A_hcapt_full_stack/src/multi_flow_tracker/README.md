# Multi-Flow APT Graph Tracker (Component 8)

A new component of HCAPT-Defense that operates on *sequences of flows*
rather than individual flows. Closes the gap between single-flow IDS
(NSL-KDD, ToN-IoT, Edge-IIoTset) and authentication-log lateral
movement detectors (HetGLM, T-GAT) — the first to apply graph-based
APT tracking to standard network-flow IDS datasets.

## Architecture

```
                Sliding window of N flows
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │  GraphBuilder                        │
        │  Nodes  = (src_key, dst_key) pairs   │
        │  Edges  = aggregated flow stats      │
        │  Time   = monotonic window index     │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │  GraphSAGE-style message-passing GNN │
        │  2 layers, 64 hidden units           │
        │  Per-node embedding                  │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │  PatternHead                         │
        │  Binary head: per-node risk score    │
        │  R ∈ [0, 1] = P(node is APT-active)  │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │  Zero-Trust integration              │
        │  graph_risk → PolicyEngine boost     │
        │  → final action ∈ {ALLOW, ...,       │
        │                    CHALLENGE, BLOCK} │
        └──────────────────────────────────────┘
```

## Key design choices

* **Per-dataset node-key extractors** — each dataset has its own
  `nodes_from_flow()` function in `graph_builder.py` that derives
  source/destination identifiers from whatever features are
  available (e.g. NSL-KDD has no IP addresses, so we use
  `(protocol_type, service)` as the source key and
  `dst_host_count` bin as the destination key).
* **No PyTorch Geometric dependency** — we implement
  SAGE-style mean aggregation by hand using PyTorch tensors. The
  whole GNN is ~80 lines of code.
* **Window-based, not graph-of-graphs** — each `N`-flow window
  produces one graph, classified independently. Avoids the
  intractable global-graph view.

## Lateral-movement patterns it can flag

| Pattern        | Graph signature                          |
|----------------|------------------------------------------|
| Fan-out        | 1 src node connected to ≥ K dst nodes within window |
| Beaconing      | Same (src,dst) edge with regular inter-arrival times |
| Credential reuse | Same src key spanning ≥ K distinct services |
| Slow-drip exfil | Long-lived edge with low byte rate but high duration |
| Collection     | Many src nodes → 1 dst node, then dst → external |

## Limitations and honest caveats

* **NSL-KDD**: no IP addresses available, so the graph is a
  "service interaction graph" rather than a true host-level
  network graph. Useful for fan-out/credential-reuse patterns but
  not for true lateral movement between hosts.
* **ToN-IoT IoT telemetry**: per-device sensor data, no flow-level
  source/destination → graph degenerates to a per-device temporal
  series. We treat each device as a node and edges as
  attack-similarity. Different scope than network-flow graphs.
* **Edge-IIoTset**: after our leak-aware cleanup
  (`edge_iiotset.py:DROP_COLS`), the `ip.src_host` / `ip.dst_host`
  fields are dropped — same limitation as NSL-KDD. The graph
  becomes a "protocol/service-tuple interaction graph".

For a *true* host-level graph view, the ToN-IoT Network CSV
(`data/ton-iot/network/train_test_network.csv`) is required.
Re-download from the official SharePoint and place at that path.
The graph builder automatically detects and uses real
`src_ip` / `dst_ip` fields when present.

## Files

* `graph_builder.py` — sliding-window graph construction
* `gnn_detector.py` — SAGE-style GNN (~80 LOC, no PyG)
* `tracker.py` — top-level `MultiFlowTracker` integrating builder + GNN
* `zt_integration.py` — bridge to `zero_trust.PolicyEngine`
