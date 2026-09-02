"""
Component 8 — Graph Builder

Converts a sliding window of flow records into a directed graph
suitable for GNN-based pattern detection. Per-dataset node-key
extractors handle the wide schema variation across NSL-KDD,
ToN-IoT IoT telemetry, and Edge-IIoTset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# Graph data structure (no PyG dependency)
# =============================================================================
@dataclass
class FlowGraph:
    """A directed graph extracted from one sliding window."""
    node_features: np.ndarray              # (n_nodes, d_node)
    edge_index:    np.ndarray              # (2, n_edges) [src_idx, dst_idx]
    edge_features: np.ndarray              # (n_edges, d_edge)
    node_labels:   Optional[np.ndarray] = None   # (n_nodes,) — APT-active flag
    node_keys:     List = field(default_factory=list)  # human-readable IDs
    # Per-row mapping: which node each row's src/dst maps to (length = window_size)
    row_src_node:  Optional[np.ndarray] = None
    row_dst_node:  Optional[np.ndarray] = None

    @property
    def n_nodes(self) -> int: return int(self.node_features.shape[0])
    @property
    def n_edges(self) -> int: return int(self.edge_index.shape[1])
    @property
    def d_node(self)  -> int: return int(self.node_features.shape[1])
    @property
    def d_edge(self)  -> int: return int(self.edge_features.shape[1])


# =============================================================================
# Per-dataset node-key extractors
# =============================================================================
def nsl_kdd_keys(row: np.ndarray, feature_names: List[str]) -> Tuple[tuple, tuple]:
    """NSL-KDD has no IP addresses. Use (protocol, service) -> (flag, dst_host_count_bin)."""
    proto = float(row[feature_names.index("protocol_type")])
    svc   = float(row[feature_names.index("service")])
    flag  = float(row[feature_names.index("flag")])
    dhc   = float(row[feature_names.index("dst_host_count")])
    src_key = (round(proto, 2), round(svc, 2))
    dst_key = (round(flag, 2), int(np.clip(dhc, -10, 10)))   # binned
    return src_key, dst_key


def ton_iot_telemetry_keys(row: np.ndarray, feature_names: List[str]) -> Tuple[tuple, tuple]:
    """ToN-IoT IoT telemetry: device id is the only natural identifier.
    Use device as both src and dst (degenerate but tracks per-device pattern)."""
    if "__device" not in feature_names:
        return ("?", 0), ("?", 0)
    dev = round(float(row[feature_names.index("__device")]), 2)
    return (dev, "src"), (dev, "dst")


def edge_iiotset_keys(row: np.ndarray, feature_names: List[str]) -> Tuple[tuple, tuple]:
    """Edge-IIoTset (after leak-cleanup): no IPs. Use protocol-family signature."""
    def get(name, default=0.0):
        return float(row[feature_names.index(name)]) if name in feature_names else default
    tcp_dst = round(get("tcp.dstport"), 0)
    udp_dst = round(get("udp.port"), 0)
    proto   = "tcp" if tcp_dst > 0 else ("udp" if udp_dst > 0 else "other")
    port    = tcp_dst if tcp_dst > 0 else udp_dst
    src_key = (proto, int(np.clip(get("tcp.srcport"), 0, 1e6)))
    dst_key = (proto, int(np.clip(port, 0, 1e6)))
    return src_key, dst_key


DATASET_KEY_EXTRACTORS: Dict[str, Callable] = {
    "nsl-kdd":           nsl_kdd_keys,
    "ton-iot-telemetry": ton_iot_telemetry_keys,
    "edge-iiotset":      edge_iiotset_keys,
}


# =============================================================================
# Graph builder
# =============================================================================
class FlowGraphBuilder:
    """Build one FlowGraph per sliding window."""

    def __init__(self, dataset_name: str, feature_names: List[str],
                  window_size: int = 256, max_nodes_per_window: int = 256):
        self.dataset_name = dataset_name
        self.feature_names = feature_names
        self.window_size = window_size
        self.max_nodes = max_nodes_per_window
        self.key_fn = DATASET_KEY_EXTRACTORS.get(dataset_name, nsl_kdd_keys)

    # -------------------------------------------------------- single window
    def build_window(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> FlowGraph:
        """X: (window_size, d_feat); y: (window_size,) optional labels."""
        # 1) Extract node keys per row
        keys_src, keys_dst = [], []
        for i in range(len(X)):
            s, d = self.key_fn(X[i], self.feature_names)
            keys_src.append(s); keys_dst.append(d)
        # 2) De-duplicate node keys into integer ids
        all_keys = list(dict.fromkeys(keys_src + keys_dst))   # preserves order
        if len(all_keys) > self.max_nodes:
            all_keys = all_keys[:self.max_nodes]               # truncate worst-case
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        # 3) Per-node feature aggregation
        node_feats = np.zeros((len(all_keys), 4 + X.shape[1]), dtype=np.float32)
        node_label = np.zeros(len(all_keys), dtype=np.int64) if y is not None else None
        node_count = np.zeros(len(all_keys), dtype=np.int64)
        for i, (s, d) in enumerate(zip(keys_src, keys_dst)):
            if s in key_to_idx:
                idx = key_to_idx[s]
                node_feats[idx, 0] += 1                    # out-degree count
                node_feats[idx, 2] += X[i].mean()         # aggregate signal
                node_feats[idx, 4:] += X[i]               # raw aggregate
                node_count[idx] += 1
                if y is not None:
                    node_label[idx] |= int(y[i] != 0)
            if d in key_to_idx and d != s:
                idx = key_to_idx[d]
                node_feats[idx, 1] += 1                    # in-degree count
                node_feats[idx, 3] += X[i].mean()
                node_feats[idx, 4:] += X[i]
                node_count[idx] += 1
                if y is not None:
                    node_label[idx] |= int(y[i] != 0)
        # Normalise aggregated raw features by their count
        for i in range(len(all_keys)):
            if node_count[i] > 0:
                node_feats[i, 4:] /= node_count[i]
        # 4) Edge list
        edge_src, edge_dst, edge_attr = [], [], []
        for i, (s, d) in enumerate(zip(keys_src, keys_dst)):
            if s in key_to_idx and d in key_to_idx:
                edge_src.append(key_to_idx[s])
                edge_dst.append(key_to_idx[d])
                # Edge features: position-in-window, raw row mean, raw row std
                edge_attr.append([
                    i / max(len(X) - 1, 1),
                    float(X[i].mean()),
                    float(X[i].std()),
                ])
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64) \
                     if edge_src else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_attr, dtype=np.float32) \
                        if edge_attr else np.zeros((0, 3), dtype=np.float32)
        # Per-row mapping for back-projection
        row_src_node = np.array([key_to_idx.get(s, -1) for s in keys_src], dtype=np.int64)
        row_dst_node = np.array([key_to_idx.get(d, -1) for d in keys_dst], dtype=np.int64)
        return FlowGraph(
            node_features=node_feats,
            edge_index=edge_index,
            edge_features=edge_features,
            node_labels=node_label,
            node_keys=all_keys,
            row_src_node=row_src_node,
            row_dst_node=row_dst_node,
        )

    # --------------------------------------------------------- many windows
    def build_dataset(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                      stride: Optional[int] = None) -> List[FlowGraph]:
        """Slide a window of size `window_size` across (X, y)."""
        stride = stride or self.window_size
        graphs = []
        for start in range(0, len(X) - self.window_size + 1, stride):
            end = start + self.window_size
            yi = y[start:end] if y is not None else None
            graphs.append(self.build_window(X[start:end], yi))
        return graphs


# =============================================================================
# Quick CLI for debugging
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.standard_normal((512, 18)).astype(np.float32)
    y = rng.integers(0, 2, size=512)
    feature_names = [f"f{i}" for i in range(18)]
    builder = FlowGraphBuilder("nsl-kdd", feature_names, window_size=64)
    graphs = builder.build_dataset(X, y)
    print(f"Built {len(graphs)} graphs.")
    g = graphs[0]
    print(f"  n_nodes={g.n_nodes} n_edges={g.n_edges} d_node={g.d_node} d_edge={g.d_edge}")
    print(f"  node_labels sum: {g.node_labels.sum() if g.node_labels is not None else 'n/a'}")
