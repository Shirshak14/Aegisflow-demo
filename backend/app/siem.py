"""Export ledger alerts to SOC tooling: CEF (ArcSight/most SIEMs), RFC 5424 syslog, or JSON lines.

Read-only over the hash-chained ledger: exporting never changes alerts. Each exported event carries
the alert's ledger id and chain hash, so the SIEM copy can be checked against ``/audit/verify``.
``since_id`` lets a collector poll incrementally ("give me everything after the last id I saw").
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from typing import Any, Iterable

VENDOR, PRODUCT = "AegisFlow", "AegisFlow Forecaster"
FORMATS = ("cef", "syslog", "jsonl")


def _cef_header(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _cef_ext(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace("=", "\\=").replace("\n", "\\n").replace("\r", "\\r")


def severity(risk_score: float | None) -> int:
    """CEF severity 0-10 from the 0-100 risk score."""
    return 0 if risk_score is None else max(0, min(10, int(round(float(risk_score) / 10))))


def _epoch_ms(ts: str | None) -> int | None:
    if not ts:
        return None
    t = datetime.fromisoformat(str(ts))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return int(t.timestamp() * 1000)


def to_cef(alert: dict[str, Any], version: str = "0.1.0") -> str:
    stage = alert.get("predicted_stage") or "UNCERTAIN"
    ext = {"rt": _epoch_ms(alert.get("predicted_at")), "src": alert.get("host_id"),
           "externalId": alert.get("id"), "cfp1": alert.get("lstm_probability"), "cfp1Label": "attackProbability",
           "cn1": alert.get("risk_score"), "cn1Label": "riskScore", "cs1": stage, "cs1Label": "predictedStage",
           "cs2": alert.get("hash"), "cs2Label": "ledgerHash", "cs3": alert.get("sequence_id"),
           "cs3Label": "sequenceId", "cs4": alert.get("model_version"), "cs4Label": "modelVersion",
           "start": _epoch_ms(alert.get("target_window_start")), "end": _epoch_ms(alert.get("target_window_end"))}
    body = " ".join(f"{k}={_cef_ext(v)}" for k, v in ext.items() if v is not None)
    head = ["CEF:0", VENDOR, PRODUCT, version, "aegisflow-forecast", "Predicted attack in next window",
            str(severity(alert.get("risk_score")))]
    return "|".join(_cef_header(h) for h in head) + "|" + body


def to_syslog(alert: dict[str, Any], hostname: str = "aegisflow", version: str = "0.1.0") -> str:
    """RFC 5424 line (facility local4, severity from risk) with the CEF event as the message."""
    sev = severity(alert.get("risk_score"))
    syslog_sev = 2 if sev >= 8 else 3 if sev >= 6 else 4 if sev >= 4 else 5   # crit/err/warning/notice
    pri = 20 * 8 + syslog_sev
    ts = alert.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"<{pri}>1 {ts} {hostname} aegisflow - alert - {to_cef(alert, version)}"


def to_jsonl(alert: dict[str, Any]) -> str:
    """ECS-style field names for Elastic / Splunk / OpenSearch ingestion."""
    rc = alert.get("risk_components")
    doc = {"@timestamp": alert.get("predicted_at"), "event": {"kind": "alert", "module": "aegisflow",
           "id": alert.get("id"), "risk_score": alert.get("risk_score"), "severity": severity(alert.get("risk_score")),
           "start": alert.get("target_window_start"), "end": alert.get("target_window_end"), "hash": alert.get("hash")},
           "source": {"ip": alert.get("host_id")},
           "aegisflow": {"attack_probability": alert.get("lstm_probability"), "threshold": alert.get("lstm_threshold"),
                         "predicted_stage": alert.get("predicted_stage"), "sequence_id": alert.get("sequence_id"),
                         "risk_components": json.loads(rc) if isinstance(rc, str) else rc,
                         "model_version": alert.get("model_version"), "prev_hash": alert.get("prev_hash")}}
    return json.dumps(doc, sort_keys=True)


def render(alerts: Iterable[dict[str, Any]], fmt: str) -> list[str]:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    f = {"cef": to_cef, "syslog": to_syslog, "jsonl": to_jsonl}[fmt]
    return [f(a) for a in alerts]


def alerts_since(ledger, since_id: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    """Alerts with id > since_id, oldest first."""
    with ledger._connect() as con:
        rows = con.execute("SELECT * FROM alerts WHERE id > ? ORDER BY id ASC LIMIT ?", (since_id, limit)).fetchall()
    return [dict(r) for r in rows]


def send_syslog_udp(lines: Iterable[str], host: str, port: int = 514) -> int:
    """Send each line as one UDP syslog datagram; returns how many were sent."""
    n = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for line in lines:
            s.sendto(line.encode("utf-8"), (host, port)); n += 1
    return n
