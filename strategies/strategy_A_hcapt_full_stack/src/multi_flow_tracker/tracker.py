"""
Component 8 — MultiFlowTracker

Top-level API that wraps the graph builder + GNN detector into a single
class with `.fit(X_train, y_train)` and `.predict_flow_risk(X)`. Built
to slot into Phase 2 pipeline.

Reads dataset name from a config and dispatches to the correct
node-key extractor in graph_builder.DATASET_KEY_EXTRACTORS.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import torch

from .graph_builder import FlowGraphBuilder, FlowGraph
from .gnn_detector  import MultiFlowGNN, train_gnn, predict_node_risk, DEVICE


@dataclass
class MultiFlowConfig:
    window_size:    int = 256
    stride:         int = 256          # non-overlapping by default
    hidden:         int = 64
    n_layers:       int = 2
    epochs:         int = 20
    lr:             float = 1e-3
    batch_size:     int = 16
    risk_threshold: float = 0.5        # node risk >= this -> "APT-active"
    layer_type:     str = "sage"       # "sage" | "gat"


class MultiFlowTracker:

    def __init__(self, dataset_name: str, feature_names: List[str],
                 config: Optional[MultiFlowConfig] = None):
        self.dataset_name = dataset_name
        self.feature_names = list(feature_names)
        self.config = config or MultiFlowConfig()
        self.builder = FlowGraphBuilder(
            dataset_name, self.feature_names,
            window_size=self.config.window_size,
        )
        self.model: Optional[MultiFlowGNN] = None
        self.training_loss: List[float] = []

    # ------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        graphs = self.builder.build_dataset(X, y, stride=self.config.stride)
        if not graphs:
            raise ValueError(
                f"Not enough data to build any window of size "
                f"{self.config.window_size} from X of length {len(X)}"
            )
        d_node = graphs[0].d_node
        self.model = MultiFlowGNN(d_node, hidden=self.config.hidden,
                                   n_layers=self.config.n_layers,
                                   layer_type=self.config.layer_type)
        if verbose:
            print(f"[mft] Built {len(graphs)} graphs ({d_node}-dim nodes); "
                  f"GNN params: {sum(p.numel() for p in self.model.parameters())}")
        self.training_loss = train_gnn(
            self.model, graphs,
            epochs=self.config.epochs,
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            verbose=verbose,
        )
        return self

    # --------------------------------------------------------- predict
    def predict_flow_risk(self, X: np.ndarray) -> np.ndarray:
        """Return a per-row risk score in [0,1].

        For each row, the risk = max(src_node_risk, dst_node_risk) where
        src_node and dst_node are the graph nodes that row mapped to in
        its enclosing window.
        """
        if self.model is None:
            raise RuntimeError("MultiFlowTracker.fit() must be called first")
        graphs = self.builder.build_dataset(X, None, stride=self.config.stride)
        node_risks = predict_node_risk(self.model, graphs)
        per_row = np.zeros(len(X), dtype=np.float32)
        for i, (g, r) in enumerate(zip(graphs, node_risks)):
            start = i * self.config.stride
            end   = min(start + self.config.window_size, len(X))
            if r.size == 0 or g.row_src_node is None:
                continue
            # For each row in this window, the risk is the max of its src/dst node risk
            for j in range(end - start):
                src_idx = int(g.row_src_node[j])
                dst_idx = int(g.row_dst_node[j])
                risk = 0.0
                if 0 <= src_idx < r.size: risk = max(risk, float(r[src_idx]))
                if 0 <= dst_idx < r.size: risk = max(risk, float(r[dst_idx]))
                per_row[start + j] = risk
        return per_row

    # ------------------------------------------------------------- I/O
    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            return
        torch.save(self.model.state_dict(), out_dir / "gnn.pt")
        with open(out_dir / "tracker_meta.json", "w") as f:
            json.dump({
                "dataset": self.dataset_name,
                "feature_names": self.feature_names,
                "config": self.config.__dict__,
                "training_loss": [float(x) for x in self.training_loss],
                "d_node_input": int(self.model.proj_in.in_features),
            }, f, indent=2)
        print(f"[mft] saved -> {out_dir}")

    @classmethod
    def load(cls, dir: Path) -> "MultiFlowTracker":
        meta = json.load(open(dir / "tracker_meta.json"))
        cfg = MultiFlowConfig(**meta["config"])
        t = cls(meta["dataset"], meta["feature_names"], cfg)
        d_node = meta["d_node_input"]
        t.model = MultiFlowGNN(d_node, hidden=cfg.hidden, n_layers=cfg.n_layers)
        t.model.load_state_dict(torch.load(dir / "gnn.pt", map_location=DEVICE))
        t.model.to(DEVICE).eval()
        return t


# =============================================================================
# Quick sanity demo on a real loader
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.loaders.nsl_kdd import NslKddLoader
    data_root = Path(__file__).resolve().parents[5] / "data"
    ds = NslKddLoader(data_root).load("binary")
    cfg = MultiFlowConfig(window_size=512, epochs=5)
    tracker = MultiFlowTracker(ds.name, ds.feature_names, cfg).fit(
        ds.X_train, ds.y_train, verbose=True,
    )
    risks = tracker.predict_flow_risk(ds.X_test)
    print(f"\nFlow risks on test: mean={risks.mean():.3f}, "
          f"std={risks.std():.3f}, max={risks.max():.3f}")
