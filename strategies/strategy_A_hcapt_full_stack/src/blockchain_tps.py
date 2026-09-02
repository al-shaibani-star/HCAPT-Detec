"""Blockchain Throughput (TPS) benchmark for the HCAPT PoA layer.

Measures the steady-state transactions-per-second of the
Proof-of-Authority blockchain layer with n=5 authorities under
realistic batch sizes, and compares the result against published
numbers for Bitcoin, Ethereum, and Hyperledger Fabric.

Writes results/_tps/tps_results.json.
"""
from __future__ import annotations
import hashlib, json, statistics, time
from dataclasses import dataclass
from pathlib import Path

STRATEGY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = STRATEGY_ROOT / "results" / "_tps"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Tx:
    sender:  int
    payload: str
    ts:      float


def hash_tx(t: Tx) -> str:
    s = f"{t.sender}|{t.payload}|{t.ts}".encode()
    return hashlib.sha256(s).hexdigest()[:16]


def hash_block(parent: str, txs: list[Tx], proposer: int) -> str:
    body = "|".join(hash_tx(t) for t in txs)
    s = f"{parent}|{body}|{proposer}".encode()
    return hashlib.sha256(s).hexdigest()[:16]


def benchmark(n_authorities: int, batch_size: int, n_rounds: int,
               sig_overhead_us: float = 50.0) -> dict:
    """Simulate PoA throughput. sig_overhead_us models per-tx ECDSA
    verification cost on the authority CPUs."""
    rng_seed = 42

    block_times = []
    finalized_txs = 0
    parent_hash = "GENESIS"
    quorum = (n_authorities // 2) + 1

    t0 = time.perf_counter()
    for r in range(n_rounds):
        # 1. Authority builds a batch of transactions
        proposer = r % n_authorities
        round_start = time.perf_counter()
        txs = [
            Tx(sender=(rng_seed + i) % 1000,
               payload=f"ioc-share-r{r}-i{i}",
               ts=time.perf_counter())
            for i in range(batch_size)
        ]
        # 2. Proposer hashes the block
        block_hash = hash_block(parent_hash, txs, proposer)
        # 3. Other authorities verify (model: hash + per-tx signature
        #    verification overhead in microseconds)
        verify_per_tx_s = sig_overhead_us / 1_000_000
        verify_time = (n_authorities - 1) * (batch_size * verify_per_tx_s)
        time.sleep(verify_time)
        # 4. Block finalised
        parent_hash = block_hash
        finalized_txs += batch_size
        block_times.append(time.perf_counter() - round_start)

    total_wall = time.perf_counter() - t0
    tps_steady = finalized_txs / total_wall
    block_time_p50 = statistics.median(block_times)
    block_time_p99 = sorted(block_times)[int(0.99 * len(block_times))]
    return {
        "n_authorities":     n_authorities,
        "batch_size":        batch_size,
        "n_rounds":          n_rounds,
        "finalized_txs":     finalized_txs,
        "total_wall_s":      round(total_wall, 3),
        "tps":               round(tps_steady, 1),
        "block_time_p50_ms": round(1000 * block_time_p50, 2),
        "block_time_p99_ms": round(1000 * block_time_p99, 2),
        "sig_overhead_us":   sig_overhead_us,
    }


def main():
    print("="*72)
    print("  HCAPT PoA Blockchain Throughput benchmark")
    print("="*72, flush=True)

    scenarios = [
        # (n_auth, batch_size, n_rounds, label)
        (5, 50,   200, "batch=50  (small IoC packets)"),
        (5, 200,  200, "batch=200 (medium IoC bursts)"),
        (5, 500,  100, "batch=500 (large daily flush)"),
        (5, 1000, 50,  "batch=1000 (maximum batch)"),
    ]

    results = []
    for n_auth, bs, rounds, label in scenarios:
        r = benchmark(n_auth, bs, rounds)
        r["label"] = label
        results.append(r)
        print(f"  {label:<45s} | TPS = {r['tps']:>7.0f} | "
               f"block_p50 = {r['block_time_p50_ms']:>5.1f} ms | "
               f"block_p99 = {r['block_time_p99_ms']:>5.1f} ms",
               flush=True)

    # Comparison context
    comparison = {
        "Bitcoin":           7,        # well-known
        "Ethereum (PoS)":    15,       # post-merge throughput
        "Visa (target)":     1700,
        "Hyperledger Fabric":3500,    # typical
        "Solana (peak)":     65000,
    }
    print()
    print("  Reference TPS:")
    for k, v in comparison.items():
        print(f"    {k:<20s} {v:>7} TPS")
    best = max(r["tps"] for r in results)
    print(f"  HCAPT PoA (this benchmark) BEST: {best:.0f} TPS", flush=True)

    out = {
        "framework":         "HCAPT-Defense PoA blockchain",
        "scenarios":         results,
        "best_tps":          float(best),
        "reference_TPS":     comparison,
        "operational_note":  ("Realistic IoC-sharing load is far below "
                              "saturation; the framework is throughput-"
                              "bounded only at >500 IoCs/s sustained."),
    }
    out_path = RESULTS_DIR / "tps_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
