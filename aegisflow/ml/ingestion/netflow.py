"""NetFlow v5 / v9 / IPFIX ingestion -> canonical flow schema.

Inputs:
  * a packet capture (.pcap/.pcapng) of the UDP export traffic a router/probe sends to a collector
    (``read_export_capture``): every UDP payload that parses as NetFlow v5, v9 or IPFIX (v10) is decoded,
    with v9/IPFIX templates tracked per exporter + observation domain;
  * ``nfdump -o csv`` output from nfcapd files (``read_nfdump_csv``), which covers whatever versions
    the collector received.

What NetFlow can and cannot supply (nothing is invented):
  * NetFlow records are UNIDIRECTIONAL: request and response are separate flows, each with its own
    ``source_ip``. CIC-IDS2017 flows are bidirectional, so per-host windows differ in shape.
  * byte counts are layer-3 bytes INCLUDING headers (CICFlowMeter counts payload bytes).
  * TCP flags arrive as the OR of all packets' flags, so ``syn_count`` etc. are set to 1 when the
    flag was seen and 0 otherwise: a lower bound on the true count, documented as such.
  * packet-length std/min/max, inter-arrival statistics, TCP window, TTL and retransmissions are not
    in NetFlow and stay NA; ``packet_size_mean`` is bytes / packets.
"""
from __future__ import annotations

import ipaddress
import socket
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ...errors import AegisFlowError, DatasetNotFoundError
from ...schema import coerce_canonical_frame, validate_canonical_frame
from .pcap import PROTOCOL_NAMES, UNLABELED, open_pcap_reader

# NetFlow v9 / IPFIX information-element ids used here (identical numbering in both).
IN_BYTES, IN_PKTS, PROTOCOL, TCP_FLAGS, L4_SRC_PORT, IPV4_SRC, L4_DST_PORT, IPV4_DST = 1, 2, 4, 6, 7, 8, 11, 12
LAST_SWITCHED, FIRST_SWITCHED, IPV6_SRC, IPV6_DST = 21, 22, 27, 28
FLOW_START_SEC, FLOW_END_SEC, FLOW_START_MS, FLOW_END_MS = 150, 151, 152, 153
SYSTEM_INIT_MS = 160
FLAG_BITS = {"fin": 0x01, "syn": 0x02, "rst": 0x04, "psh": 0x08, "ack": 0x10, "urg": 0x20}


@dataclass
class FlowRecord:
    start: float      # epoch seconds
    end: float
    src: str
    dst: str
    sport: int | None
    dport: int | None
    proto: int
    packets: int
    bytes: int
    tcp_flags: int | None
    version: int


@dataclass
class NetflowDecoder:
    """Stateful decoder: keeps v9/IPFIX templates and IPFIX system-init times per exporter/domain."""
    templates: dict = field(default_factory=dict)        # (exporter, domain, template_id) -> [(id, len)]
    options: dict = field(default_factory=dict)          # (exporter, domain, template_id) -> [(id, len)] (scope+fields)
    init_ms: dict = field(default_factory=dict)          # (exporter, domain) -> systemInitTimeMilliseconds
    skipped_no_template: int = 0

    def decode(self, data: bytes, exporter: str = "") -> list[FlowRecord]:
        if len(data) < 4:
            return []
        version = struct.unpack("!H", data[:2])[0]
        if version == 5:
            return _decode_v5(data)
        if version == 9:
            return self._decode_templated(data, exporter, 9)
        if version == 10:
            return self._decode_templated(data, exporter, 10)
        return []

    # -------------------------------------------------------------------------------------- v9 / IPFIX
    def _decode_templated(self, data: bytes, exporter: str, version: int) -> list[FlowRecord]:
        if version == 9:
            if len(data) < 20:
                return []
            _, _, uptime_ms, unix_secs, _, domain = struct.unpack("!HHIIII", data[:20])
            export_time, off, end = float(unix_secs), 20, len(data)
            tmpl_set, opt_set = 0, 1
        else:
            if len(data) < 16:
                return []
            _, length, export_secs, _, domain = struct.unpack("!HHIII", data[:16])
            export_time, uptime_ms, off, end = float(export_secs), None, 16, min(len(data), length)
            tmpl_set, opt_set = 2, 3
        out: list[FlowRecord] = []
        while off + 4 <= end:
            set_id, set_len = struct.unpack("!HH", data[off:off + 4])
            if set_len < 4:
                break
            body = data[off + 4:off + set_len]
            if set_id == tmpl_set:
                self._read_templates(body, exporter, domain)
            elif set_id == opt_set:
                self._read_option_templates(body, exporter, domain, version)
            elif set_id >= 256:
                key = (exporter, domain, set_id)
                if key in self.templates:
                    for values in _records(body, self.templates[key]):
                        rec = self._to_record(values, version, export_time, uptime_ms, exporter, domain)
                        if rec is not None:
                            out.append(rec)
                elif key in self.options:
                    for values in _records(body, self.options[key]):
                        if SYSTEM_INIT_MS in values:
                            self.init_ms[(exporter, domain)] = _uint(values[SYSTEM_INIT_MS])
                else:
                    self.skipped_no_template += 1
            off += set_len
        return out

    def _read_templates(self, body: bytes, exporter: str, domain: int) -> None:
        p = 0
        while p + 4 <= len(body):
            tid, count = struct.unpack("!HH", body[p:p + 4]); p += 4
            if tid < 256:
                break  # padding
            fields = []
            for _ in range(count):
                fid, flen = struct.unpack("!HH", body[p:p + 4]); p += 4
                if fid & 0x8000:      # IPFIX enterprise-specific element: skip the enterprise number
                    p += 4; fid = -1
                fields.append((fid, flen))
            self.templates[(exporter, domain, tid)] = fields

    def _read_option_templates(self, body: bytes, exporter: str, domain: int, version: int) -> None:
        p = 0
        while p + 6 <= len(body):
            if version == 9:
                tid, scope_len, opt_len = struct.unpack("!HHH", body[p:p + 6]); p += 6
                n = (scope_len + opt_len) // 4
            else:
                tid, total, _scope_count = struct.unpack("!HHH", body[p:p + 6]); p += 6
                n = total
            if tid < 256:
                break
            fields = []
            for _ in range(n):
                fid, flen = struct.unpack("!HH", body[p:p + 4]); p += 4
                if version == 10 and fid & 0x8000:
                    p += 4; fid = -1
                fields.append((fid, flen))
            self.options[(exporter, domain, tid)] = fields

    def _to_record(self, v: dict, version: int, export_time: float, uptime_ms: int | None,
                   exporter: str, domain: int) -> FlowRecord | None:
        if IPV4_SRC in v:
            src, dst = str(ipaddress.IPv4Address(v[IPV4_SRC])), str(ipaddress.IPv4Address(v.get(IPV4_DST, b"\0" * 4)))
        elif IPV6_SRC in v:
            src, dst = str(ipaddress.IPv6Address(v[IPV6_SRC])), str(ipaddress.IPv6Address(v.get(IPV6_DST, b"\0" * 16)))
        else:
            return None
        if FLOW_START_MS in v:
            start, end = _uint(v[FLOW_START_MS]) / 1000, _uint(v.get(FLOW_END_MS, v[FLOW_START_MS])) / 1000
        elif FLOW_START_SEC in v:
            start, end = float(_uint(v[FLOW_START_SEC])), float(_uint(v.get(FLOW_END_SEC, v[FLOW_START_SEC])))
        elif FIRST_SWITCHED in v:
            first, last = _uint(v[FIRST_SWITCHED]), _uint(v.get(LAST_SWITCHED, v[FIRST_SWITCHED]))
            if uptime_ms is not None:          # v9: relative to the header's sysUptime
                start = export_time - (uptime_ms - first) / 1000
                end = export_time - (uptime_ms - last) / 1000
            elif (exporter, domain) in self.init_ms:  # IPFIX: relative to systemInitTimeMilliseconds
                base = self.init_ms[(exporter, domain)] / 1000
                start, end = base + first / 1000, base + last / 1000
            else:
                start = end = export_time      # no reference time: fall back to the export time
        else:
            start = end = export_time
        proto = _uint(v.get(PROTOCOL, b"\0"))
        ports = proto in (6, 17)
        return FlowRecord(start, end, src, dst, _uint(v[L4_SRC_PORT]) if ports and L4_SRC_PORT in v else None,
                          _uint(v[L4_DST_PORT]) if ports and L4_DST_PORT in v else None, proto,
                          _uint(v.get(IN_PKTS, b"\0")), _uint(v.get(IN_BYTES, b"\0")),
                          _uint(v[TCP_FLAGS]) if proto == 6 and TCP_FLAGS in v else None, version)


def _uint(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _records(body: bytes, fields: list[tuple[int, int]]) -> Iterator[dict]:
    size = sum(l for _, l in fields)
    if size == 0:
        return
    p = 0
    while p + size <= len(body):
        rec, q = {}, p
        for fid, flen in fields:
            rec[fid] = body[q:q + flen]; q += flen
        yield rec
        p += size


def _decode_v5(data: bytes) -> list[FlowRecord]:
    if len(data) < 24:
        return []
    _, count, uptime_ms, secs, nsecs = struct.unpack("!HHIII", data[:16])
    export = secs + nsecs / 1e9
    out = []
    for i in range(count):
        r = data[24 + 48 * i:24 + 48 * (i + 1)]
        if len(r) < 48:
            break
        (src, dst, _nh, _in, _out, pkts, octets, first, last, sport, dport, _pad, flags, proto,
         _tos, _sas, _das, _sm, _dm, _pad2) = struct.unpack("!4s4s4sHHIIIIHHBBBBHHBBH", r)
        ports = proto in (6, 17)
        out.append(FlowRecord(export - (uptime_ms - first) / 1000, export - (uptime_ms - last) / 1000,
                              socket.inet_ntoa(src), socket.inet_ntoa(dst), sport if ports else None,
                              dport if ports else None, proto, pkts, octets, flags if proto == 6 else None, 5))
    return out


# ------------------------------------------------------------------------------------------- readers
def read_export_capture(path: str | Path, decoder: NetflowDecoder | None = None) -> Iterator[FlowRecord]:
    """Decode NetFlow/IPFIX export datagrams from a capture of collector-bound UDP traffic."""
    from scapy.layers.inet import IP, UDP
    from scapy.layers.inet6 import IPv6
    dec = decoder or NetflowDecoder()
    with open_pcap_reader(path) as reader:
        for pkt in reader:
            if UDP not in pkt:
                continue
            exporter = pkt[IP].src if IP in pkt else pkt[IPv6].src if IPv6 in pkt else ""
            yield from dec.decode(bytes(pkt[UDP].payload), f"{exporter}:{pkt[UDP].sport}")


def read_nfdump_csv(path: str | Path) -> Iterator[FlowRecord]:
    """Records from ``nfdump -r <file> -o csv`` (times as printed by nfdump, i.e. the collector's local time)."""
    df = pd.read_csv(path, skipinitialspace=True)
    need = {"ts", "te", "sa", "da", "sp", "dp", "pr", "flg", "ipkt", "ibyt"}
    if need - set(df.columns):
        raise ValueError(f"not nfdump CSV output: missing columns {sorted(need - set(df.columns))}")
    df = df.dropna(subset=["sa", "da"])                    # nfdump appends a summary block after a blank line
    df = df[df["ts"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
    names = {v: k for k, v in PROTOCOL_NAMES.items()}
    for r in df.itertuples(index=False):
        proto = names.get(str(r.pr).strip().upper(), None)
        if proto is None:
            proto = int(r.pr) if str(r.pr).strip().isdigit() else 0
        start = pd.Timestamp(r.ts).timestamp()
        # ts/te are printed at 1 s resolution; td carries the duration with milliseconds
        end = start + float(r.td) if "td" in df.columns and pd.notna(r.td) else pd.Timestamp(r.te).timestamp()
        flg = str(r.flg).strip()
        flags = None
        if proto == 6:   # nfdump prints flags as 8 chars "CEUAPRSF", '.' when unset
            flags = sum(b for ch, b in zip(flg[-6:], (0x20, 0x10, 0x08, 0x04, 0x02, 0x01)) if ch != ".")
        ports = proto in (6, 17)
        yield FlowRecord(start, end, str(r.sa), str(r.da), int(r.sp) if ports else None, int(r.dp) if ports else None,
                         proto, int(r.ipkt), int(r.ibyt), flags, 0)
        if "opkt" in df.columns and int(getattr(r, "opkt", 0) or 0) > 0:   # bidirectional (NSEL) records
            yield FlowRecord(start, end, str(r.da), str(r.sa), int(r.dp) if ports else None,
                             int(r.sp) if ports else None, proto, int(r.opkt), int(r.obyt), flags, 0)


# ------------------------------------------------------------------------------------------- canonical
def records_to_flows(records: Iterable[FlowRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        pk = max(r.packets, 0)
        row = {"timestamp": pd.Timestamp(datetime.fromtimestamp(r.start, tz=timezone.utc)).tz_localize(None),
               "source_ip": r.src, "destination_ip": r.dst, "source_port": r.sport, "destination_port": r.dport,
               "protocol": PROTOCOL_NAMES.get(r.proto, str(r.proto)), "flow_duration": max(0.0, r.end - r.start),
               "packet_count": float(pk), "byte_count": float(r.bytes),
               "fwd_packet_count": float(pk), "bwd_packet_count": 0.0,
               "fwd_byte_count": float(r.bytes), "bwd_byte_count": 0.0,
               "packet_size_mean": float(r.bytes) / pk if pk else np.nan,
               "dataset_label": UNLABELED}
        for name, bit in FLAG_BITS.items():  # presence (0/1): a lower bound on the per-flow count
            row[f"{name}_count"] = float(bool(r.tcp_flags & bit)) if r.tcp_flags is not None else np.nan
        rows.append(row)
    if not rows:
        raise AegisFlowError("no NetFlow/IPFIX flow records decoded")
    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = coerce_canonical_frame(df)
    df["source_dataset"] = "netflow"
    validate_canonical_frame(df, context="netflow")
    return df


def netflow_to_flows(path: str | Path, fmt: str = "auto") -> pd.DataFrame:
    """``fmt``: 'capture' (pcap of exports), 'nfdump-csv', or 'auto' (by file extension)."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(f"NetFlow input not found: {path}")
    if fmt == "auto":
        fmt = "nfdump-csv" if path.suffix.lower() == ".csv" else "capture"
    if fmt == "capture":
        return records_to_flows(read_export_capture(path))
    if fmt == "nfdump-csv":
        return records_to_flows(read_nfdump_csv(path))
    raise ValueError(f"unknown NetFlow input format {fmt!r}")
