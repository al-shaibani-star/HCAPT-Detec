"""
Layer-0 ingestion adapter — cloud-native flow logs.

Converts AWS VPC Flow Logs (v2), Azure NSG Flow Logs (v2) and GCP VPC Flow
Logs into the connection-feature schema consumed by Layer 1, so that the
hybrid-cloud tier of the architecture is a specified and exercised interface
rather than a diagram label.

SCOPE — read before citing this module.
    Cloud flow logs are *metadata*, not payload. The 13 NSL-KDD content
    features (hot, num_failed_logins, root_shell, ...) require host or
    payload inspection and CANNOT be recovered from any of the three
    formats. This module derives what is derivable, marks the rest as
    unavailable, and reports the coverage explicitly via
    `feature_coverage()`. No detection result is claimed on cloud logs:
    public cloud flow logs carry no attack labels, so they support a
    schema-compatibility demonstration only.

Run `python cloud_flowlog.py` for a self-test on the published sample
records of all three providers.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Canonical intermediate record — provider-independent
# --------------------------------------------------------------------------


@dataclass
class CloudFlowRecord:
    """One bidirectional flow, normalised across providers."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int                 # IANA number: 1 ICMP, 6 TCP, 17 UDP
    start: float                  # epoch seconds
    end: float                    # epoch seconds
    bytes_src_to_dst: int
    bytes_dst_to_src: int         # 0 when the provider reports one direction
    packets_src_to_dst: int
    packets_dst_to_src: int
    accepted: bool                # ACCEPT/Allow -> True, REJECT/Deny -> False
    provider: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_land(self) -> int:
        """NSL-KDD `land`: connection from/to the same host and port."""
        return int(self.src_ip == self.dst_ip and self.src_port == self.dst_port)


PROTO_NAME = {1: "icmp", 6: "tcp", 17: "udp"}

# Well-known destination ports -> NSL-KDD-style service label.
SERVICE_BY_PORT = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 67: "dhcp", 69: "tftp_u", 79: "finger", 80: "http",
    110: "pop_3", 111: "sunrpc", 119: "nntp", 123: "ntp_u", 135: "loc_srv",
    137: "netbios_ns", 138: "netbios_dgm", 139: "netbios_ssn", 143: "imap4",
    161: "snmp", 179: "bgp", 389: "ldap", 443: "http_443", 445: "smb",
    465: "smtp_s", 514: "shell", 515: "printer", 543: "klogin", 544: "kshell",
    587: "smtp_sub", 636: "ldap_s", 993: "imap_s", 995: "pop_s",
    1433: "sql_net", 1521: "sql_net", 3306: "mysql", 3389: "rdp",
    5432: "postgres", 5900: "vnc", 6379: "redis", 8080: "http_8001",
    8443: "http_443", 9200: "elastic", 27017: "mongo",
}


def service_of(port: int) -> str:
    if port in SERVICE_BY_PORT:
        return SERVICE_BY_PORT[port]
    if port < 1024:
        return "other_priv"
    return "private"


# --------------------------------------------------------------------------
# Provider parsers
# --------------------------------------------------------------------------


def parse_aws_vpc_v2(line: str) -> Optional[CloudFlowRecord]:
    """AWS VPC Flow Logs, default v2 layout.

    version account-id interface-id srcaddr dstaddr srcport dstport
    protocol packets bytes start end action log-status
    """
    f = line.split()
    if len(f) < 14 or f[0] != "2":
        return None
    if f[13] != "OK":                       # NODATA / SKIPDATA carry no flow
        return None
    return CloudFlowRecord(
        src_ip=f[3], dst_ip=f[4],
        src_port=int(f[5]), dst_port=int(f[6]),
        protocol=int(f[7]),
        packets_src_to_dst=int(f[8]), packets_dst_to_src=0,
        bytes_src_to_dst=int(f[9]), bytes_dst_to_src=0,
        start=float(f[10]), end=float(f[11]),
        accepted=(f[12] == "ACCEPT"),
        provider="aws",
    )


def parse_azure_nsg_v2(tuple_str: str) -> Optional[CloudFlowRecord]:
    """Azure NSG Flow Logs v2 flow tuple.

    timestamp,srcIP,dstIP,srcPort,dstPort,protocol,trafficFlow,decision,
    flowState,packetsSrcToDst,bytesSrcToDst,packetsDstToSrc,bytesDstToSrc
    """
    f = tuple_str.split(",")
    if len(f) < 13:
        return None

    def _int(v: str) -> int:
        return int(v) if v.strip() else 0

    ts = float(f[0])
    return CloudFlowRecord(
        src_ip=f[1], dst_ip=f[2],
        src_port=_int(f[3]), dst_port=_int(f[4]),
        protocol=6 if f[5].upper() == "T" else 17,
        packets_src_to_dst=_int(f[9]), packets_dst_to_src=_int(f[11]),
        bytes_src_to_dst=_int(f[10]), bytes_dst_to_src=_int(f[12]),
        start=ts, end=ts,                   # Azure stamps the tuple, not a span
        accepted=(f[7].upper() == "A"),
        provider="azure",
    )


def parse_gcp_vpc(obj: Dict) -> Optional[CloudFlowRecord]:
    """GCP VPC Flow Logs JSON payload (jsonPayload contents)."""
    conn = obj.get("connection")
    if not conn:
        return None

    def _epoch(v: Optional[str]) -> float:
        if not v:
            return 0.0
        import datetime as _dt
        return _dt.datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()

    return CloudFlowRecord(
        src_ip=conn.get("src_ip", ""), dst_ip=conn.get("dest_ip", ""),
        src_port=int(conn.get("src_port", 0)), dst_port=int(conn.get("dest_port", 0)),
        protocol=int(conn.get("protocol", 0)),
        packets_src_to_dst=int(obj.get("packets_sent", 0)),
        packets_dst_to_src=int(obj.get("packets_received", 0)),
        bytes_src_to_dst=int(obj.get("bytes_sent", 0)),
        bytes_dst_to_src=int(obj.get("bytes_received", 0)),
        start=_epoch(obj.get("start_time")), end=_epoch(obj.get("end_time")),
        accepted=True,                      # GCP logs sampled traffic, not verdicts
        provider="gcp",
    )


PARSERS = {"aws": parse_aws_vpc_v2, "azure": parse_azure_nsg_v2, "gcp": parse_gcp_vpc}


# --------------------------------------------------------------------------
# Feature derivation
# --------------------------------------------------------------------------

# family -> (feature, derivability)
#   "direct"      recovered exactly from the log
#   "approx"      recovered as a documented approximation
#   "unavailable" requires payload or host state; not in any cloud flow log
FEATURE_MAP: Dict[str, List[tuple]] = {
    "basic": [
        ("duration", "direct"), ("protocol_type", "direct"),
        ("service", "direct"), ("flag", "approx"),
        ("src_bytes", "direct"), ("dst_bytes", "direct"),
        ("land", "direct"), ("wrong_fragment", "unavailable"),
        ("urgent", "unavailable"),
    ],
    "content": [
        (n, "unavailable") for n in (
            "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login",
        )
    ],
    "time_traffic": [
        ("count", "direct"), ("srv_count", "direct"),
        ("serror_rate", "approx"), ("srv_serror_rate", "approx"),
        ("rerror_rate", "approx"), ("srv_rerror_rate", "approx"),
        ("same_srv_rate", "direct"), ("diff_srv_rate", "direct"),
        ("srv_diff_host_rate", "direct"),
    ],
    "host_traffic": [
        ("dst_host_count", "direct"), ("dst_host_srv_count", "direct"),
        ("dst_host_same_srv_rate", "direct"), ("dst_host_diff_srv_rate", "direct"),
        ("dst_host_same_src_port_rate", "direct"),
        ("dst_host_srv_diff_host_rate", "direct"),
        ("dst_host_serror_rate", "approx"), ("dst_host_srv_serror_rate", "approx"),
        ("dst_host_rerror_rate", "approx"), ("dst_host_srv_rerror_rate", "approx"),
    ],
}

FEATURE_NAMES: List[str] = [n for fam in FEATURE_MAP.values() for n, _ in fam]
DERIVABILITY: Dict[str, str] = {n: d for fam in FEATURE_MAP.values() for n, d in fam}


def feature_coverage() -> Dict[str, Dict[str, int]]:
    """Per-family counts of direct / approx / unavailable features."""
    out: Dict[str, Dict[str, int]] = {}
    for fam, feats in FEATURE_MAP.items():
        c = defaultdict(int)
        for _, d in feats:
            c[d] += 1
        c["total"] = len(feats)
        out[fam] = dict(c)
    totals = defaultdict(int)
    for fam in out.values():
        for k, v in fam.items():
            totals[k] += v
    out["ALL"] = dict(totals)
    return out


class CloudFlowLogAdapter:
    """Layer-0 adapter: cloud flow-log records -> Layer-1 feature matrix.

    The two-second window and the 100-connection host window mirror the
    aggregation windows of the original KDD feature extractor, so that the
    derived traffic features are comparable in meaning to the benchmark ones.
    """

    def __init__(self, time_window: float = 2.0, host_window: int = 100):
        self.time_window = time_window
        self.host_window = host_window

    # ---------------------------------------------------------------- parse
    def parse(self, lines: Iterable, provider: str) -> List[CloudFlowRecord]:
        parser = PARSERS[provider]
        out = []
        for ln in lines:
            rec = parser(ln)
            if rec is not None:
                out.append(rec)
        return out

    # -------------------------------------------------------------- derive
    def to_matrix(self, records: Sequence[CloudFlowRecord]) -> np.ndarray:
        """Return (n_records, len(FEATURE_NAMES)) float32.

        Features marked "unavailable" are emitted as NaN, never as 0.0, so a
        downstream imputation choice is explicit rather than silent.
        """
        n = len(records)
        X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        recs = sorted(records, key=lambda r: r.start)

        for i, r in enumerate(recs):
            # ---- basic
            X[i, idx["duration"]] = r.duration
            X[i, idx["protocol_type"]] = r.protocol
            X[i, idx["service"]] = float(r.dst_port)
            X[i, idx["flag"]] = 1.0 if r.accepted else 0.0        # approx
            X[i, idx["src_bytes"]] = r.bytes_src_to_dst
            X[i, idx["dst_bytes"]] = r.bytes_dst_to_src
            X[i, idx["land"]] = r.is_land

            # ---- time-based: flows to the same host within `time_window`
            win = [q for q in recs
                   if 0 <= r.start - q.start <= self.time_window
                   and q.dst_ip == r.dst_ip]
            same_srv = [q for q in win if q.dst_port == r.dst_port]
            cnt, srv = len(win), len(same_srv)
            X[i, idx["count"]] = cnt
            X[i, idx["srv_count"]] = srv
            X[i, idx["same_srv_rate"]] = srv / cnt if cnt else 0.0
            X[i, idx["diff_srv_rate"]] = 1.0 - (srv / cnt) if cnt else 0.0
            X[i, idx["srv_diff_host_rate"]] = (
                len({q.dst_ip for q in same_srv}) / srv if srv else 0.0)
            rej = sum(1 for q in win if not q.accepted)
            X[i, idx["rerror_rate"]] = rej / cnt if cnt else 0.0   # approx
            X[i, idx["srv_rerror_rate"]] = (
                sum(1 for q in same_srv if not q.accepted) / srv if srv else 0.0)
            X[i, idx["serror_rate"]] = 0.0                          # approx: no SYN state
            X[i, idx["srv_serror_rate"]] = 0.0

            # ---- host-based: last `host_window` flows to the same host
            hist = [q for q in recs[:i + 1] if q.dst_ip == r.dst_ip][-self.host_window:]
            hsrv = [q for q in hist if q.dst_port == r.dst_port]
            hc, hs = len(hist), len(hsrv)
            X[i, idx["dst_host_count"]] = hc
            X[i, idx["dst_host_srv_count"]] = hs
            X[i, idx["dst_host_same_srv_rate"]] = hs / hc if hc else 0.0
            X[i, idx["dst_host_diff_srv_rate"]] = 1.0 - (hs / hc) if hc else 0.0
            X[i, idx["dst_host_same_src_port_rate"]] = (
                sum(1 for q in hist if q.src_port == r.src_port) / hc if hc else 0.0)
            X[i, idx["dst_host_srv_diff_host_rate"]] = (
                len({q.src_ip for q in hsrv}) / hs if hs else 0.0)
            hrej = sum(1 for q in hist if not q.accepted)
            X[i, idx["dst_host_rerror_rate"]] = hrej / hc if hc else 0.0
            X[i, idx["dst_host_srv_rerror_rate"]] = (
                sum(1 for q in hsrv if not q.accepted) / hs if hs else 0.0)
            X[i, idx["dst_host_serror_rate"]] = 0.0
            X[i, idx["dst_host_srv_serror_rate"]] = 0.0

        return X

    def ingest(self, lines: Iterable, provider: str) -> np.ndarray:
        return self.to_matrix(self.parse(lines, provider))


# --------------------------------------------------------------------------
# Self-test on the published sample records of all three providers
# --------------------------------------------------------------------------

AWS_SAMPLE = [
    "2 123456789010 eni-1235b8ca123456789 172.31.16.139 172.31.16.21 20641 22 6 20 4249 1418530010 1418530070 ACCEPT OK",
    "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK",
    "2 123456789010 eni-1235b8ca123456789 172.31.9.70 172.31.9.12 49762 3389 6 18 3800 1418530011 1418530071 REJECT OK",
]

AZURE_SAMPLE = [
    "1493763938,185.170.185.105,10.2.0.4,35370,23,T,I,A,C,10,1010,12,1220",
    "1493763938,185.170.185.105,10.2.0.4,35371,23,T,I,D,B,4,404,0,0",
]

GCP_SAMPLE = [
    {"connection": {"src_ip": "10.128.0.2", "dest_ip": "10.128.0.3",
                    "src_port": 49152, "dest_port": 80, "protocol": 6},
     "start_time": "2026-08-16T10:00:00.000Z", "end_time": "2026-08-16T10:00:04.000Z",
     "bytes_sent": "1000", "packets_sent": "10",
     "bytes_received": "2400", "packets_received": "12"},
]


def _self_test() -> None:
    ad = CloudFlowLogAdapter()

    print("=" * 66)
    print("Layer-0 cloud flow-log adapter — self-test")
    print("=" * 66)

    for provider, lines in (("aws", AWS_SAMPLE),
                            ("azure", AZURE_SAMPLE),
                            ("gcp", GCP_SAMPLE)):
        recs = ad.parse(lines, provider)
        X = ad.to_matrix(recs)
        derived = int((~np.isnan(X)).all(axis=0).sum())
        print(f"\n[{provider.upper()}] parsed {len(recs)}/{len(lines)} records "
              f"-> matrix {X.shape}, {derived} populated features")
        for r in recs[:2]:
            print(f"    {r.src_ip}:{r.src_port} -> {r.dst_ip}:{r.dst_port} "
                  f"{PROTO_NAME.get(r.protocol, r.protocol)}/"
                  f"{service_of(r.dst_port)} "
                  f"dur={r.duration:.0f}s bytes={r.bytes_src_to_dst}/"
                  f"{r.bytes_dst_to_src} "
                  f"{'ACCEPT' if r.accepted else 'REJECT'}")
        assert X.shape[1] == len(FEATURE_NAMES) == 41

    print("\n" + "-" * 66)
    print("Feature coverage of the 41-feature connection schema")
    print("-" * 66)
    cov = feature_coverage()
    print(f"{'family':<16}{'total':>7}{'direct':>9}{'approx':>9}{'unavail':>10}")
    for fam in ("basic", "content", "time_traffic", "host_traffic", "ALL"):
        c = cov[fam]
        print(f"{fam:<16}{c['total']:>7}{c.get('direct', 0):>9}"
              f"{c.get('approx', 0):>9}{c.get('unavailable', 0):>10}")

    a = cov["ALL"]
    print(f"\nRecoverable (direct + approx): {a.get('direct',0) + a.get('approx',0)}"
          f" of {a['total']}")
    print("Not recoverable from any cloud flow log: "
          f"{a.get('unavailable', 0)} = 13 payload/host-state content features "
          "+ 2 packet-anomaly counters (wrong_fragment, urgent)")
    print("\nNo attack labels exist for these logs -> schema compatibility only, "
          "no detection result claimed.")
    print("=" * 66)


if __name__ == "__main__":
    _self_test()
