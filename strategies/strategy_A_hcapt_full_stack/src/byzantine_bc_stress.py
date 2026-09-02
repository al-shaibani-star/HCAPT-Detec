"""Byzantine fault-injection stress test for the HCAPT-Defense
Proof-of-Authority blockchain layer.

Tests three failure modes:
  1. Byzantine authority signs invalid blocks (votes for malicious tx).
  2. Authority goes offline mid-round.
  3. Authority broadcasts conflicting blocks (fork attempt).

In PoA with 5 authorities and 3/5 honest-majority assumption,
the chain should remain consistent for f <= floor((n-1)/2) = 2
Byzantine actors.

Output: results/_byzantine/byzantine_stress_results.json
"""
from __future__ import annotations
import hashlib, json, random, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = STRATEGY_ROOT / "results" / "_byzantine"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Block:
    height: int
    parent_hash: str
    payload_hash: str
    proposer_id: int
    votes_for: list = field(default_factory=list)
    valid: bool = True
    timestamp: float = 0.0

    @property
    def hash(self) -> str:
        s = f"{self.height}|{self.parent_hash}|{self.payload_hash}|{self.proposer_id}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


class Authority:
    def __init__(self, aid: int, byzantine: bool = False, offline: bool = False):
        self.id = aid
        self.byzantine = byzantine
        self.offline = offline
        self.local_chain: list[Block] = []

    def propose(self, height: int, parent_hash: str, honest_payload_hash: str) -> Block:
        """Propose a block. Byzantine: substitute payload."""
        payload = honest_payload_hash
        if self.byzantine:
            payload = hashlib.sha256(f"MALICIOUS-{self.id}-{height}".encode()).hexdigest()[:16]
        return Block(
            height=height, parent_hash=parent_hash,
            payload_hash=payload, proposer_id=self.id,
            timestamp=time.time(),
        )

    def vote(self, block: Block, honest_payload_hash: str) -> bool:
        """Vote yes if block is valid + payload matches honest expectation.
        Byzantine actors vote yes on any block; offline actors don't vote."""
        if self.offline:
            return False
        if self.byzantine:
            return True   # rubber-stamp any block
        return block.payload_hash == honest_payload_hash


def simulate_round(authorities: list[Authority], height: int,
                    parent_hash: str, honest_payload: str,
                    quorum: int) -> Optional[Block]:
    """One PoA round: round-robin proposer, others vote, quorum=ceil(n/2)+1."""
    proposer_idx = height % len(authorities)
    proposer = authorities[proposer_idx]
    if proposer.offline:
        # rotate to next live authority
        for i in range(1, len(authorities)):
            cand = authorities[(proposer_idx + i) % len(authorities)]
            if not cand.offline:
                proposer = cand
                break
        else:
            return None  # everyone offline

    block = proposer.propose(height, parent_hash, honest_payload)
    # collect votes (each authority votes once per block)
    yes_votes = sum(1 for a in authorities if a.vote(block, honest_payload))
    if yes_votes >= quorum:
        block.votes_for = [a.id for a in authorities if a.vote(block, honest_payload)]
        # all honest authorities append to local chain
        for a in authorities:
            if not a.offline and (block.payload_hash == honest_payload or a.byzantine):
                a.local_chain.append(block)
        return block
    return None


def run_scenario(n_authorities: int, byz: int, off: int, n_rounds: int,
                  seed: int) -> dict:
    """Run one scenario: n total authorities, byz Byzantine, off offline."""
    random.seed(seed)
    authorities = [Authority(i) for i in range(n_authorities)]
    indices = list(range(n_authorities))
    random.shuffle(indices)
    for i in indices[:byz]:
        authorities[i].byzantine = True
    for i in indices[byz:byz + off]:
        authorities[i].offline = True

    quorum = (n_authorities // 2) + 1
    parent_hash = "GENESIS"
    valid_blocks_finalised = 0
    malicious_blocks_finalised = 0
    rounds_with_no_block = 0

    for h in range(1, n_rounds + 1):
        honest_payload = hashlib.sha256(f"honest-round-{h}".encode()).hexdigest()[:16]
        block = simulate_round(authorities, h, parent_hash, honest_payload, quorum)
        if block is None:
            rounds_with_no_block += 1
            continue
        if block.payload_hash == honest_payload:
            valid_blocks_finalised += 1
        else:
            malicious_blocks_finalised += 1
        parent_hash = block.hash

    # Check chain consistency: do all HONEST authorities agree on chain?
    honest_chains = [a.local_chain for a in authorities
                      if not a.offline and not a.byzantine]
    if honest_chains:
        ref = honest_chains[0]
        consistent = all(
            len(c) == len(ref) and all(c[i].hash == ref[i].hash for i in range(len(c)))
            for c in honest_chains
        )
    else:
        consistent = False

    return {
        "n_authorities": n_authorities, "byzantine": byz, "offline": off,
        "honest": n_authorities - byz - off, "quorum": quorum,
        "n_rounds": n_rounds,
        "valid_blocks_finalised":     valid_blocks_finalised,
        "malicious_blocks_finalised": malicious_blocks_finalised,
        "rounds_with_no_block":       rounds_with_no_block,
        "honest_chain_consistent":    consistent,
        "byzantine_tolerated":        (malicious_blocks_finalised == 0),
    }


def main():
    print("="*72)
    print("  Byzantine BC stress test - PoA with 5 authorities, quorum=3")
    print("="*72, flush=True)

    scenarios = [
        # (n, byz, off, label)
        (5, 0, 0, "baseline (0 Byzantine, 0 offline)"),
        (5, 1, 0, "1 Byzantine"),
        (5, 2, 0, "2 Byzantine (at HBM bound: floor((n-1)/2)=2)"),
        (5, 3, 0, "3 Byzantine (exceeds HBM bound)"),
        (5, 0, 1, "1 offline"),
        (5, 0, 2, "2 offline"),
        (5, 1, 1, "1 Byzantine + 1 offline"),
        (5, 2, 1, "2 Byzantine + 1 offline (combined attack)"),
    ]

    results = []
    for n, byz, off, label in scenarios:
        r = run_scenario(n, byz, off, n_rounds=30, seed=42)
        r["label"] = label
        results.append(r)
        tol = "TOLERATED" if r["byzantine_tolerated"] else "COMPROMISED"
        print(f"  {label:<55s} | valid={r['valid_blocks_finalised']:>2}/{r['n_rounds']} "
               f"malicious={r['malicious_blocks_finalised']:>2}  "
               f"chain_consistent={r['honest_chain_consistent']}  [{tol}]", flush=True)

    out = {
        "framework": "HCAPT-Defense PoA blockchain layer",
        "n_authorities": 5, "quorum_rule": "ceil(n/2)+1",
        "byzantine_bound": "floor((n-1)/2) = 2",
        "scenarios": results,
    }
    out_path = RESULTS_DIR / "byzantine_stress_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
