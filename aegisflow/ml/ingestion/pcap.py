"""PCAP / PCAPNG ingestion: raw packets -> bidirectional flows in the canonical schema.

Readers:
  scapy    (default) ``scapy.utils.PcapReader``, streams pcap and pcapng, pure Python.
  pyshark  ``pyshark.FileCapture`` (needs Wireshark's ``tshark`` on PATH).

Both produce the same ``Packet`` records, so flow assembly and every feature are reader-independent.

Flow assembly follows CICFlowMeter's conventions so the output is comparable to the CIC-IDS2017 CSVs
the model was trained on:
  * a flow is the bidirectional 5-tuple; the first packet seen sets the forward direction, so
    ``source_ip`` is the initiator;
  * a flow ends on an idle gap > ``idle_timeout`` (default 120 s), after ``active_timeout``
    (default 3600 s), or on a TCP FIN/RST (the FIN/RST packet belongs to the flow it closes);
  * byte counts and packet-length statistics use transport PAYLOAD bytes (as CICFlowMeter does);
  * ``flow_duration`` is in seconds and ``inter_arrival_mean/std`` in microseconds, matching
    what the CIC-IDS2017 adapter produces;
  * ``tcp_window_size`` is the TCP window of the first forward packet (Init_Win_bytes_forward).

Features CICFlowMeter does not export but packets do carry, filled here from real packets:
  ``ttl_mean`` / ``ttl_std`` (IPv4 TTL / IPv6 hop limit over all packets of the flow) and
  ``retransmission_count`` (TCP segments with payload whose direction, sequence number and length
  were already seen in the flow).

Known differences from CICFlowMeter (documented, not hidden): no flow "bulk"/"subflow"/active-idle
statistics, and timestamps are UTC from the capture (CIC-IDS2017 CSVs use local capture time).
"""
from __future__ import annotations

import gzip
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ...errors import AegisFlowError, DatasetNotFoundError, MissingDependencyError
from ...schema import coerce_canonical_frame, validate_canonical_frame

UNLABELED = "UNLABELED"
PROTOCOL_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}
FLAGS = ("F", "S", "R", "P", "A", "U")  # FIN SYN RST PSH ACK URG


@dataclass(frozen=True)
class Packet:
    ts: float                 # seconds since epoch
    src: str
    dst: str
    sport: int | None
    dport: int | None
    proto: int
    payload_len: int
    ttl: int | None
    flags: str = ""           # subset of FLAGS
    window: int | None = None
    seq: int | None = None


# ------------------------------------------------------------------------------------------- readers
@contextmanager
def open_pcap_reader(path: str | Path):
    """``scapy.utils.PcapReader`` over a file handle we own. Given a path, scapy opens the file itself
    and leaks the handle when the content is neither pcap nor pcapng; on Windows that open handle then
    blocks deleting the file (e.g. the upload endpoint's temp directory)."""
    from scapy.utils import PcapReader
    with open(path, "rb") as raw:
        gzipped = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        fh = gzip.GzipFile(fileobj=raw) if gzipped else raw
        with PcapReader(fh) as reader:
            yield reader


def read_packets_scapy(path: str | Path) -> Iterator[Packet]:
    try:
        from scapy.layers.inet import ICMP, IP, TCP, UDP
        from scapy.layers.inet6 import IPv6
        import scapy.utils  # noqa: F401
    except ImportError as exc:  # pragma: no cover - scapy is in requirements.txt
        raise MissingDependencyError("PCAP ingestion needs 'scapy' (pip install scapy)") from exc
    with open_pcap_reader(path) as reader:
        for pkt in reader:
            if IP in pkt:
                ip = pkt[IP]; src, dst, proto, ttl = ip.src, ip.dst, int(ip.proto), int(ip.ttl)
                ip_payload = int(ip.len) - int(ip.ihl) * 4
            elif IPv6 in pkt:
                ip = pkt[IPv6]; src, dst, proto, ttl = ip.src, ip.dst, int(ip.nh), int(ip.hlim)
                ip_payload = int(ip.plen)
            else:
                continue
            ts = float(pkt.time)
            if TCP in pkt:
                t = pkt[TCP]
                yield Packet(ts, src, dst, int(t.sport), int(t.dport), 6, max(0, ip_payload - int(t.dataofs) * 4), ttl,
                             "".join(f for f in FLAGS if f in str(t.flags)), int(t.window), int(t.seq))
            elif UDP in pkt:
                u = pkt[UDP]
                yield Packet(ts, src, dst, int(u.sport), int(u.dport), 17, max(0, int(u.len) - 8), ttl)
            else:
                yield Packet(ts, src, dst, None, None, proto, max(0, ip_payload), ttl)


def read_packets_pyshark(path: str | Path) -> Iterator[Packet]:
    try:
        import pyshark
    except ImportError as exc:
        raise MissingDependencyError("reader='pyshark' needs 'pyshark' and Wireshark's tshark") from exc
    cap = pyshark.FileCapture(str(path), keep_packets=False)
    try:
        for pkt in cap:
            if hasattr(pkt, "ip"):
                ip = pkt.ip; src, dst, proto, ttl = ip.src, ip.dst, int(ip.proto), int(ip.ttl)
                ip_payload = int(ip.len) - int(ip.hdr_len)
            elif hasattr(pkt, "ipv6"):
                ip = pkt.ipv6; src, dst, proto, ttl = ip.src, ip.dst, int(ip.nxt), int(ip.hlim)
                ip_payload = int(ip.plen)
            else:
                continue
            ts = float(pkt.sniff_timestamp)
            if hasattr(pkt, "tcp"):
                t = pkt.tcp
                bits = int(str(t.flags), 16)
                flags = "".join(f for f, b in zip(FLAGS, (0x01, 0x02, 0x04, 0x08, 0x10, 0x20)) if bits & b)
                yield Packet(ts, src, dst, int(t.srcport), int(t.dstport), 6, max(0, ip_payload - int(t.hdr_len)), ttl,
                             flags, int(t.window_size_value), int(t.seq_raw))
            elif hasattr(pkt, "udp"):
                u = pkt.udp
                yield Packet(ts, src, dst, int(u.srcport), int(u.dstport), 17, max(0, int(u.length) - 8), ttl)
            else:
                yield Packet(ts, src, dst, None, None, proto, max(0, ip_payload), ttl)
    finally:
        cap.close()


READERS = {"scapy": read_packets_scapy, "pyshark": read_packets_pyshark}


# ------------------------------------------------------------------------------------------- flows
@dataclass
class _Flow:
    src: str
    dst: str
    sport: int | None
    dport: int | None
    proto: int
    first: float
    last: float
    times: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    ttls: list[int] = field(default_factory=list)
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    flag_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FLAGS, 0))
    init_window: int | None = None
    seen_segments: set = field(default_factory=set)
    retransmissions: int = 0
    fin_dirs: set = field(default_factory=set)

    def add(self, p: Packet, forward: bool) -> None:
        self.last = p.ts
        self.times.append(p.ts); self.sizes.append(p.payload_len)
        if p.ttl is not None:
            self.ttls.append(p.ttl)
        if forward:
            self.fwd_pkts += 1; self.fwd_bytes += p.payload_len
            if self.init_window is None and p.window is not None:
                self.init_window = p.window
        else:
            self.bwd_pkts += 1; self.bwd_bytes += p.payload_len
        for f in p.flags:
            self.flag_counts[f] += 1
        if p.proto == 6 and p.payload_len > 0 and p.seq is not None:
            key = (forward, p.seq, p.payload_len)
            if key in self.seen_segments:
                self.retransmissions += 1
            else:
                self.seen_segments.add(key)
        if "F" in p.flags:
            self.fin_dirs.add(forward)

    def finished_by_tcp(self, p: Packet) -> bool:
        return p.proto == 6 and ("R" in p.flags or len(self.fin_dirs) == 2)

    def record(self) -> dict:
        times = np.asarray(self.times); sizes = np.asarray(self.sizes, dtype=float)
        iat = np.diff(times) * 1e6
        ttls = np.asarray(self.ttls, dtype=float)
        return {
            "timestamp": pd.Timestamp(datetime.fromtimestamp(self.first, tz=timezone.utc)).tz_localize(None),
            "source_ip": self.src, "destination_ip": self.dst,
            "source_port": self.sport, "destination_port": self.dport,
            "protocol": PROTOCOL_NAMES.get(self.proto, str(self.proto)),
            "flow_duration": float(self.last - self.first),
            "packet_count": float(self.fwd_pkts + self.bwd_pkts), "byte_count": float(self.fwd_bytes + self.bwd_bytes),
            "fwd_packet_count": float(self.fwd_pkts), "bwd_packet_count": float(self.bwd_pkts),
            "fwd_byte_count": float(self.fwd_bytes), "bwd_byte_count": float(self.bwd_bytes),
            "packet_size_mean": float(sizes.mean()), "packet_size_std": float(sizes.std(ddof=1)) if len(sizes) > 1 else 0.0,
            "packet_size_min": float(sizes.min()), "packet_size_max": float(sizes.max()),
            "inter_arrival_mean": float(iat.mean()) if len(iat) else 0.0,
            "inter_arrival_std": float(iat.std(ddof=1)) if len(iat) > 1 else 0.0,
            "syn_count": float(self.flag_counts["S"]), "ack_count": float(self.flag_counts["A"]),
            "rst_count": float(self.flag_counts["R"]), "fin_count": float(self.flag_counts["F"]),
            "psh_count": float(self.flag_counts["P"]), "urg_count": float(self.flag_counts["U"]),
            "tcp_window_size": float(self.init_window) if self.init_window is not None else math.nan,
            "ttl_mean": float(ttls.mean()) if len(ttls) else math.nan,
            "ttl_std": float(ttls.std(ddof=1)) if len(ttls) > 1 else (0.0 if len(ttls) else math.nan),
            "retransmission_count": float(self.retransmissions) if self.proto == 6 else math.nan,
            "dataset_label": UNLABELED,
        }


def flows_from_packets(packets: Iterable[Packet], *, idle_timeout: float = 120.0,
                       active_timeout: float = 3600.0) -> pd.DataFrame:
    """Assemble packets (any order is accepted; they are processed by timestamp) into canonical flows."""
    if idle_timeout <= 0 or active_timeout <= 0:
        raise ValueError("timeouts must be positive")
    active: dict[tuple, _Flow] = {}
    done: list[dict] = []
    for p in sorted(packets, key=lambda q: q.ts):
        a, b = (p.src, -1 if p.sport is None else p.sport), (p.dst, -1 if p.dport is None else p.dport)
        key = (p.proto,) + (a + b if a <= b else b + a)
        flow = active.get(key)
        if flow is not None and (p.ts - flow.last > idle_timeout or p.ts - flow.first > active_timeout):
            done.append(flow.record()); flow = None
        if flow is None:
            flow = _Flow(p.src, p.dst, p.sport, p.dport, p.proto, p.ts, p.ts)
            active[key] = flow
        flow.add(p, forward=(p.src, p.sport) == (flow.src, flow.sport))
        if flow.finished_by_tcp(p):
            done.append(flow.record()); del active[key]
    done.extend(f.record() for f in active.values())
    if not done:
        raise AegisFlowError("no IPv4/IPv6 packets found in capture")
    df = pd.DataFrame(done).sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = coerce_canonical_frame(df)
    df["source_dataset"] = "pcap"
    validate_canonical_frame(df, context="pcap")
    return df


def pcap_to_flows(path: str | Path, *, reader: str = "scapy", idle_timeout: float = 120.0,
                  active_timeout: float = 3600.0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(f"capture not found: {path}")
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}; choose from {sorted(READERS)}")
    return flows_from_packets(READERS[reader](path), idle_timeout=idle_timeout, active_timeout=active_timeout)


def apply_label_file(flows: pd.DataFrame, labels_csv: str | Path, mapping: dict) -> pd.DataFrame:
    """Label flows from analyst-supplied ground truth.

    ``labels_csv`` columns: ``source_ip, start, end, label`` (start/end parseable timestamps, UTC).
    A flow gets the label of the first row whose source_ip matches and whose [start, end] contains the
    flow's first-packet time; other flows are ``BENIGN``. Every label must be a key of ``mapping``
    (a ``configs/stages.yaml`` section), so stage/class mapping stays explicit.
    """
    rules = pd.read_csv(labels_csv)
    need = {"source_ip", "start", "end", "label"}
    if need - set(rules.columns):
        raise ValueError(f"label file needs columns {sorted(need)}")
    unknown = sorted(set(rules["label"]) - set(mapping))
    if unknown:
        raise ValueError(f"labels not in the stage mapping: {unknown}")
    if "BENIGN" not in mapping:
        raise ValueError("stage mapping has no BENIGN entry")
    out = flows.copy()
    labels = pd.Series("BENIGN", index=out.index, dtype="string")
    ts = pd.to_datetime(out["timestamp"])
    for r in rules.itertuples(index=False):
        hit = (out["source_ip"] == str(r.source_ip)) & (ts >= pd.Timestamp(r.start)) & (ts <= pd.Timestamp(r.end))
        hit &= labels == "BENIGN"
        labels[hit] = str(r.label)
    out["dataset_label"] = labels
    out["normalized_attack_class"] = labels.map(lambda l: mapping[l]["normalized_attack_class"]).astype("string")
    out["attack_stage"] = labels.map(lambda l: mapping[l]["attack_stage"]).astype("string")
    out["label_confidence"] = labels.map(lambda l: mapping[l]["confidence"]).astype("string")
    return out
