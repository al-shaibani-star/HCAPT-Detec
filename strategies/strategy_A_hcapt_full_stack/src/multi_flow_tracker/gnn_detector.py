"""
Component 8 — GraphSAGE-style GNN Detector

A SAGE-style 2-layer message-passing network implemented in pure
PyTorch (no torch_geometric dependency, so it installs cleanly on
Windows). Operates on `FlowGraph` instances produced by
`graph_builder.py`.

Architecture:
  Node embedding  = Linear(d_node -> hidden)
  SAGE layer 1    = aggregate(mean of neighbours) + Linear
  SAGE layer 2    = aggregate(mean of neighbours) + Linear
  Pattern head    = Linear(hidden -> 1) [sigmoid for binary risk]
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import FlowGraph


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# SAGE layer (mean-pool aggregation)
# =============================================================================
class SAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h:          (n_nodes, in_dim)
        edge_index: (2, n_edges) — [src, dst]
        Returns:    (n_nodes, out_dim)
        """
        n = h.size(0)
        if edge_index.numel() == 0:
            return F.relu(self.lin_self(h))
        # Mean of source neighbours' features, indexed by dst
        src, dst = edge_index[0], edge_index[1]
        # Compute per-dst sum then divide by per-dst count
        agg = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, h[src])
        # Count
        ones = torch.ones(src.size(0), device=h.device, dtype=h.dtype)
        cnt = torch.zeros(n, device=h.device, dtype=h.dtype)
        cnt.index_add_(0, dst, ones)
        cnt = cnt.clamp(min=1.0).unsqueeze(-1)
        agg = agg / cnt
        return F.relu(self.lin_self(h) + self.lin_neigh(agg))


# =============================================================================
# GAT layer (Graph Attention — alternative to SAGE)
# =============================================================================
class GATLayer(nn.Module):
    """Single-head Graph Attention layer.

    For each destination node v, aggregate source neighbours u with
    learned attention weights:

        α_{u→v} = softmax_u( LeakyReLU( a^T [W h_u || W h_v] ) )
        h'_v    = ReLU( sum_u α_{u→v} W h_u )

    Implemented in pure PyTorch (no torch_geometric dependency).
    """
    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Two-part attention vector: source half + destination half
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        if edge_index.numel() == 0:
            return F.relu(self.W(h))
        Wh = self.W(h)                                     # (n, out_dim)
        src, dst = edge_index[0], edge_index[1]
        # Attention logits per edge:  a_src·h_u + a_dst·h_v
        e = (Wh[src] * self.a_src).sum(-1) + (Wh[dst] * self.a_dst).sum(-1)
        e = self.leaky(e)                                  # (n_edges,)
        # Softmax-per-destination (stable: subtract max per dst)
        # 1. Subtract per-dst max for numerical stability
        max_per_dst = torch.full((n,), -1e9, device=h.device, dtype=h.dtype)
        max_per_dst = max_per_dst.scatter_reduce(0, dst, e, reduce="amax",
                                                   include_self=False)
        e = e - max_per_dst[dst]
        ex = torch.exp(e)
        # 2. Per-dst denom (sum of exp)
        denom = torch.zeros(n, device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, ex)
        alpha = ex / denom[dst].clamp(min=1e-12)           # (n_edges,)
        # 3. Weighted aggregation
        out = torch.zeros_like(Wh)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * Wh[src])
        return F.relu(out)


# =============================================================================
# Multi-Flow Pattern Detector
# =============================================================================
class MultiFlowGNN(nn.Module):
    """Pattern detector. `layer_type` selects SAGE (mean aggregation) or GAT
    (attention-weighted aggregation). Both are pure-PyTorch implementations."""

    def __init__(self, d_node: int, hidden: int = 64, n_layers: int = 2,
                  layer_type: str = "sage"):
        super().__init__()
        self.layer_type = layer_type.lower()
        self.proj_in = nn.Linear(d_node, hidden)
        if self.layer_type == "sage":
            LayerCls = SAGELayer
        elif self.layer_type == "gat":
            LayerCls = GATLayer
        else:
            raise ValueError(f"layer_type must be 'sage' or 'gat', got {layer_type}")
        self.layers = nn.ModuleList([LayerCls(hidden, hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, node_features: torch.Tensor,
                 edge_index: torch.Tensor) -> torch.Tensor:
        """Returns per-node logit (n_nodes, 1)."""
        h = F.relu(self.proj_in(node_features))
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.dropout(h)
        return self.head(h)


# =============================================================================
# Training utilities
# =============================================================================
def train_gnn(model: MultiFlowGNN, graphs: List[FlowGraph],
              epochs: int = 30, lr: float = 1e-3,
              batch_size: int = 16, verbose: bool = True) -> List[float]:
    """Train a MultiFlowGNN on a list of FlowGraphs with node-level labels.

    Each graph contributes its nodes (with labels) to a binary classification
    objective: P(node is APT-active).
    """
    model = model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        ep_loss, n_batches = 0.0, 0
        # Shuffle graphs
        order = np.random.permutation(len(graphs))
        for i in range(0, len(graphs), batch_size):
            batch = [graphs[j] for j in order[i:i+batch_size]]
            opt.zero_grad()
            losses_batch = []
            for g in batch:
                if g.node_labels is None or g.n_nodes == 0:
                    continue
                nf = torch.from_numpy(g.node_features).to(DEVICE)
                ei = torch.from_numpy(g.edge_index).to(DEVICE)
                yl = torch.from_numpy(g.node_labels.astype(np.float32)).to(DEVICE)
                logit = model(nf, ei).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, yl)
                losses_batch.append(loss)
            if not losses_batch:
                continue
            batch_loss = torch.stack(losses_batch).mean()
            batch_loss.backward()
            opt.step()
            ep_loss += float(batch_loss.item()); n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        losses.append(avg)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"   [GNN] epoch {ep+1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def predict_node_risk(model: MultiFlowGNN, graphs: List[FlowGraph]) -> List[np.ndarray]:
    """Per-graph array of node risk scores ∈ [0,1]."""
    model = model.to(DEVICE).eval()
    out = []
    for g in graphs:
        if g.n_nodes == 0:
            out.append(np.zeros(0, dtype=np.float32))
            continue
        nf = torch.from_numpy(g.node_features).to(DEVICE)
        ei = torch.from_numpy(g.edge_index).to(DEVICE)
        logit = model(nf, ei).squeeze(-1)
        prob = torch.sigmoid(logit).cpu().numpy()
        out.append(prob)
    return out


# =============================================================================
# Per-flow risk (back-projection: node risk -> row risk in original window)
# =============================================================================
def project_risk_to_rows(graphs: List[FlowGraph], risks: List[np.ndarray],
                          window_size: int) -> np.ndarray:
    """For each original row, find which node it mapped to and assign its risk.

    Returns a (total_rows,) array of per-row risk scores in [0,1].
    """
    all_row_risks = []
    for g, r in zip(graphs, risks):
        # Reconstruct src→idx map (we don't actually have per-row -> node lookup
        # saved, so we approximate using the first node's risk for every row in
        # the window; in production, builder should record per-row node ids).
        if r.size == 0:
            all_row_risks.append(np.zeros(window_size, dtype=np.float32))
            continue
        # Use mean of all nodes' risk as a conservative per-window proxy
        all_row_risks.append(np.full(window_size, float(r.mean()), dtype=np.float32))
    return np.concatenate(all_row_risks)


# =============================================================================
# CLI demo
# =============================================================================
if __name__ == "__main__":
    from .graph_builder import FlowGraphBuilder
    rng = np.random.default_rng(42)
    n_flows = 1024
    # Use realistic NSL-KDD feature names so the key extractor works
    feature_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
    ]
    d_feat = len(feature_names)
    X = rng.standard_normal((n_flows, d_feat)).astype(np.float32)
    y = rng.integers(0, 2, size=n_flows)
    builder = FlowGraphBuilder("nsl-kdd", feature_names, window_size=64)
    graphs = builder.build_dataset(X, y)
    print(f"Built {len(graphs)} graphs.")
    model = MultiFlowGNN(d_node=graphs[0].d_node, hidden=32)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    train_gnn(model, graphs, epochs=5, batch_size=4, verbose=True)
    risks = predict_node_risk(model, graphs)
    print(f"Per-graph mean risk: {[round(float(r.mean()), 3) for r in risks[:5]]}")
