"""
HCAPT-Defense - Layer 3: Blockchain Threat Intelligence

Implements a minimal but complete blockchain to store Indicators of
Compromise (IoCs) contributed by multiple HCAPT nodes. The chain is:
  - append-only
  - tamper-evident (each block hashes the previous block)
  - verifiable (verify_chain() checks every hash + consensus proof)
  - queryable by cosine similarity over stored IoC feature vectors

Consensus options:
  - 'poa'  Proof of Authority  (DEFAULT) - authority-signed blocks,
           appropriate for enterprise/consortium deployments (fast).
  - 'pow'  Proof of Work (difficulty=N leading zeros) - demo-only.

Classes:
  IoC         - dataclass, one threat indicator
  Block       - one block in the chain
  Blockchain  - full chain with mining and similarity queries
"""
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

from config import RESULTS_DIR, NUM_CLASSES, LABEL_MAP

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================
#  IoC - Indicator of Compromise
# ============================================================
@dataclass
class IoC:
    feature_vector: List[float]    # 41-dim, post-scaler
    attack_class: int              # 0..NUM_CLASSES-1 (should be != 0)
    severity: int                  # 1-5
    confidence: float              # [0, 1] - source ensemble confidence
    source_node: str               # node identifier
    timestamp: float               # epoch seconds
    attack_name: str = ""          # optional fine-grained label
    ttl_blocks: int = 1000         # time-to-live in blocks

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "IoC":
        return cls(**d)

    def feature_hash(self) -> str:
        """Content-addressable ID for deduplication."""
        s = json.dumps([round(x, 4) for x in self.feature_vector])
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ============================================================
#  Block
# ============================================================
@dataclass
class Block:
    index: int
    timestamp: float
    iocs: List[IoC] = field(default_factory=list)
    prev_hash: str = ""
    nonce: int = 0
    hash: str = ""
    authority: str = ""        # PoA: id of the authority that sealed
    authority_sig: str = ""    # PoA: signature (HMAC-SHA256 of payload)

    def _payload(self) -> str:
        """Deterministic serialization for hashing."""
        return json.dumps({
            "index": self.index,
            "timestamp": round(self.timestamp, 2),
            "iocs": [ioc.to_dict() for ioc in self.iocs],
            "prev_hash": self.prev_hash,
            "nonce": self.nonce,
            "authority": self.authority,
        }, sort_keys=True)

    def compute_hash(self) -> str:
        return hashlib.sha256(self._payload().encode()).hexdigest()

    def mine_pow(self, difficulty: int) -> int:
        """Proof-of-Work: find nonce such that hash starts with N zeros.
        Returns the number of iterations performed."""
        prefix = "0" * difficulty
        iters = 0
        while True:
            h = self.compute_hash()
            if h.startswith(prefix):
                self.hash = h
                return iters
            self.nonce += 1
            iters += 1

    def seal_poa(self, authority: str, secret: bytes) -> int:
        """Proof-of-Authority: the authority signs the payload with
        HMAC-SHA256, and publishes the block. Very fast (single hash).
        Returns 1 (constant work, for API symmetry with mine_pow).
        """
        import hmac
        self.authority = authority
        self.authority_sig = hmac.new(
            secret, self._payload().encode(), hashlib.sha256
        ).hexdigest()
        self.hash = self.compute_hash()
        return 1

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "iocs": [i.to_dict() for i in self.iocs],
            "prev_hash": self.prev_hash,
            "nonce": self.nonce,
            "hash": self.hash,
            "authority": self.authority,
            "authority_sig": self.authority_sig,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Block":
        return cls(
            index=d["index"], timestamp=d["timestamp"],
            iocs=[IoC.from_dict(i) for i in d["iocs"]],
            prev_hash=d["prev_hash"], nonce=d["nonce"], hash=d["hash"],
            authority=d.get("authority", ""),
            authority_sig=d.get("authority_sig", ""),
        )


# ============================================================
#  Blockchain
# ============================================================
class Blockchain:
    def __init__(self, consensus: str = "poa",
                 difficulty: int = 3, block_size: int = 20,
                 authorities: Optional[Dict[str, bytes]] = None):
        """
        consensus   : 'poa' (default) or 'pow'
        difficulty  : PoW leading-zero count (ignored for PoA)
        block_size  : max IoCs per block
        authorities : dict of {authority_id: secret_bytes} for PoA.
                      Defaults to a single 'HCAPT_CA' authority.
        """
        self.chain: List[Block] = []
        self.pending_iocs: List[IoC] = []
        self.consensus = consensus
        self.difficulty = difficulty
        self.block_size = block_size
        self.authorities = authorities or {
            "HCAPT_CA": b"hcapt-defense-ca-secret-v1"
        }
        # Round-robin sealer selection
        self._sealer_cursor = 0
        self.nodes: Dict[str, List[int]] = {}
        self._seal_genesis()

    def _choose_sealer(self) -> Tuple[str, bytes]:
        names = sorted(self.authorities.keys())
        name = names[self._sealer_cursor % len(names)]
        self._sealer_cursor += 1
        return name, self.authorities[name]

    def _seal_genesis(self):
        genesis = Block(index=0, timestamp=time.time(),
                         iocs=[], prev_hash="0" * 64)
        if self.consensus == "pow":
            genesis.mine_pow(self.difficulty)
        else:
            auth, secret = self._choose_sealer()
            genesis.seal_poa(auth, secret)
        self.chain.append(genesis)

    # ----- Adding IoCs -----
    def submit_ioc(self, ioc: IoC):
        """Queue an IoC; mined automatically when block_size reached."""
        self.pending_iocs.append(ioc)
        # Track node contributions
        if ioc.source_node not in self.nodes:
            self.nodes[ioc.source_node] = []
        if len(self.pending_iocs) >= self.block_size:
            self.mine_pending()

    def mine_pending(self, force: bool = False) -> Optional[Block]:
        if not self.pending_iocs and not force:
            return None
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            iocs=list(self.pending_iocs),
            prev_hash=self.chain[-1].hash,
        )
        if self.consensus == "pow":
            new_block.mine_pow(self.difficulty)
        else:
            auth, secret = self._choose_sealer()
            new_block.seal_poa(auth, secret)
        self.chain.append(new_block)
        # Track node -> block mapping
        for ioc in new_block.iocs:
            self.nodes.setdefault(ioc.source_node, []).append(new_block.index)
        self.pending_iocs.clear()
        return new_block

    # ----- Verification -----
    def verify(self) -> Tuple[bool, Optional[str]]:
        """Verify every block's hash, prev_hash link, and consensus proof.
        Returns (valid, reason_if_invalid)."""
        import hmac
        prefix = "0" * self.difficulty
        for i, blk in enumerate(self.chain):
            # Hash recomputation
            if blk.compute_hash() != blk.hash:
                return False, f"Block {i}: hash does not match payload"
            # Chain link
            if i > 0 and blk.prev_hash != self.chain[i - 1].hash:
                return False, f"Block {i}: prev_hash mismatch"
            # Consensus proof
            if self.consensus == "pow":
                if not blk.hash.startswith(prefix):
                    return False, (f"Block {i}: PoW failed "
                                    f"(hash={blk.hash[:12]}...)")
            else:  # poa
                if blk.authority not in self.authorities:
                    return False, (f"Block {i}: unknown authority "
                                    f"'{blk.authority}'")
                secret = self.authorities[blk.authority]
                expected = hmac.new(secret, blk._payload().encode(),
                                     hashlib.sha256).hexdigest()
                if expected != blk.authority_sig:
                    return False, (f"Block {i}: authority "
                                    f"signature mismatch")
        return True, None

    # ----- Querying -----
    def all_iocs(self) -> List[IoC]:
        return [ioc for b in self.chain for ioc in b.iocs]

    def query_similar(self, feature_vector: np.ndarray,
                      top_k: int = 5,
                      min_similarity: float = 0.85) -> List[Tuple[float, IoC]]:
        """Cosine-similarity search. Returns sorted list of
        (similarity, IoC) pairs with similarity >= min_similarity."""
        iocs = self.all_iocs()
        if not iocs:
            return []
        V = np.array([ioc.feature_vector for ioc in iocs], dtype=np.float32)
        q = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        # Normalize
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = (Vn @ qn.T).ravel()
        mask = sims >= min_similarity
        if not mask.any():
            return []
        idx = np.where(mask)[0]
        ranked = sorted(idx, key=lambda i: -sims[i])[:top_k]
        return [(float(sims[i]), iocs[i]) for i in ranked]

    def query_batch(self, X: np.ndarray, min_similarity: float = 0.85
                    ) -> List[List[Tuple[float, IoC]]]:
        """Vectorised similarity search on a batch of samples."""
        iocs = self.all_iocs()
        if not iocs:
            return [[] for _ in range(len(X))]
        V = np.array([ioc.feature_vector for ioc in iocs], dtype=np.float32)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S = Xn @ Vn.T  # (n_samples, n_iocs)
        out = []
        for row in S:
            mask = row >= min_similarity
            if not mask.any():
                out.append([])
                continue
            idx = np.where(mask)[0]
            ranked = sorted(idx, key=lambda i: -row[i])[:5]
            out.append([(float(row[i]), iocs[i]) for i in ranked])
        return out

    # ----- Persistence -----
    def save(self, path: Path):
        data = {
            "consensus": self.consensus,
            "difficulty": self.difficulty,
            "block_size": self.block_size,
            "chain": [b.to_dict() for b in self.chain],
            "nodes": self.nodes,
            # NOTE: authority secrets are NOT persisted (security)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path,
             authorities: Optional[Dict[str, bytes]] = None) -> "Blockchain":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bc = cls.__new__(cls)
        bc.consensus = data.get("consensus", "pow")
        bc.difficulty = data["difficulty"]
        bc.block_size = data["block_size"]
        bc.chain = [Block.from_dict(b) for b in data["chain"]]
        bc.pending_iocs = []
        bc.nodes = data.get("nodes", {})
        bc.authorities = authorities or {"HCAPT_CA": b"hcapt-defense-ca-secret-v1"}
        bc._sealer_cursor = len(bc.chain)
        return bc

    # ----- Stats -----
    def stats(self) -> Dict:
        iocs = self.all_iocs()
        per_class = {INV_LABEL_MAP[c]: 0 for c in range(NUM_CLASSES)}
        per_node = {n: 0 for n in self.nodes}
        for ioc in iocs:
            per_class[INV_LABEL_MAP[ioc.attack_class]] += 1
            per_node[ioc.source_node] = per_node.get(ioc.source_node, 0) + 1
        return {
            "consensus": self.consensus,
            "blocks": len(self.chain),
            "iocs": len(iocs),
            "pending_iocs": len(self.pending_iocs),
            "difficulty": self.difficulty if self.consensus == "pow" else None,
            "authorities": list(self.authorities.keys()),
            "nodes": list(self.nodes.keys()),
            "iocs_per_class": per_class,
            "iocs_per_node": per_node,
        }


# ============================================================
#  Quick sanity check
# ============================================================
if __name__ == "__main__":
    print("Blockchain smoke test (PoA)")
    bc = Blockchain(consensus="poa", block_size=5)

    # Submit a few IoCs
    rng = np.random.default_rng(42)
    for i in range(12):
        ioc = IoC(
            feature_vector=rng.normal(size=41).astype(float).tolist(),
            attack_class=int(rng.integers(1, 5)),
            severity=int(rng.integers(1, 6)),
            confidence=float(rng.uniform(0.75, 0.99)),
            source_node=f"node_{i % 3}",
            timestamp=time.time(),
        )
        bc.submit_ioc(ioc)
    bc.mine_pending(force=True)

    ok, reason = bc.verify()
    print(f"  verified: {ok}  ({reason or 'all good'})")

    stats = bc.stats()
    print(f"  blocks: {stats['blocks']} | iocs: {stats['iocs']}")
    print(f"  per-class: {stats['iocs_per_class']}")
    print(f"  per-node:  {stats['iocs_per_node']}")

    # Similarity query
    query = rng.normal(size=41)
    hits = bc.query_similar(query, top_k=3, min_similarity=0.0)
    print(f"  top-3 similarity hits: "
          f"{[(round(s, 3), INV_LABEL_MAP[i.attack_class]) for s, i in hits]}")
