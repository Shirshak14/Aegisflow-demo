"""Investigation bundle: everything an analyst needs to hand an alert (or a set of alerts) to someone else.

One JSON document with the ledger records, the temporal context of each alert, the hosts' risk timelines
around them, the model version and a configuration fingerprint, every analyst action, and the result of
verifying both hash chains at export time.

Ground truth (``truth_attack``, ``truth_class``, ``truth_stage``) is left out unless the caller asks for it
for evaluation. Those fields are part of each record's ledger hash, so a bundle without them cannot be
re-hashed on its own; the record hashes and the chain head are included so they can be checked against the
server's ledger instead.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import replay as _replay
from . import temporal_context
from .info import ROOT

GROUND_TRUTH_FIELDS = ("truth_attack", "truth_class", "truth_stage")
CONFIG_FILES = ("configs/config.yaml", "configs/stages.yaml", "configs/mitre_mapping.yaml", "configs/datasets.yaml")
MODEL_FILES = ("best.pt", "logistic.joblib", "preprocessor.joblib", "metadata.json", "metrics.json", "calibration.json")
TIMELINE_MARGIN = pd.Timedelta(minutes=30)
MAX_ALERTS = 1000


def _sha256(path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def fingerprint() -> dict[str, Any]:
    """SHA-256 of every config file and model artifact the replay reads, and one hash over all of them."""
    files = {name: _sha256(ROOT / name) for name in CONFIG_FILES}
    files |= {f"model/{name}": _sha256(_replay.MODEL_DIR / name) for name in MODEL_FILES}
    combined = hashlib.sha256("\n".join(f"{k}={v}" for k, v in sorted(files.items())).encode("utf-8")).hexdigest()
    return {"sha256": combined, "files": files, "note": "null = file not present on this server"}


def _timeline(engine, host_id: str, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    h = engine.host(host_id)
    if h is None:
        return {"host_id": host_id, "points": [], "note": "host not in the current replay session"}
    times = [pd.Timestamp(a["target_window_start"]) for a in alerts if a.get("target_window_start")]
    lo, hi = (min(times) - TIMELINE_MARGIN, max(times) + TIMELINE_MARGIN) if times else (None, None)
    pts = [p for p in h["timeline"] if lo is None or lo <= pd.Timestamp(p["t"]) <= hi]
    return {"host_id": host_id, "from": None if lo is None else lo.isoformat(), "to": None if hi is None else hi.isoformat(),
            "points": pts, "note": f"risk of every replayed sequence of this host within {TIMELINE_MARGIN} of its exported alerts"}


def bundle(app_state, alert_ids: list[int], include_ground_truth: bool = False) -> dict[str, Any]:
    """Build the bundle for ``alert_ids`` (unknown ids are listed, not fatal)."""
    ledger, engine, analyst = app_state.ledger, app_state.engine, app_state.analyst
    ids = list(dict.fromkeys(int(i) for i in alert_ids))[:MAX_ALERTS]
    records, missing = [], []
    for i in ids:
        rec = ledger.get(i)
        if rec is None:
            missing.append(i)
        else:
            records.append(engine.with_anomaly([rec])[0])
    alerts = []
    for rec in records:
        out = {k: v for k, v in rec.items() if include_ground_truth or k not in GROUND_TRUTH_FIELDS}
        try:
            ctx = temporal_context.context(str(rec["sequence_id"]))
        except (OSError, ValueError):
            ctx = None
        alerts.append({"record": out, "temporal_context": ctx, "analyst": analyst.history(rec["id"])})
    by_host: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_host.setdefault(str(rec["host_id"]), []).append(rec)
    versions = sorted({str(r.get("model_version")) for r in records})
    return {
        "bundle": "aegisflow-investigation", "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": {"alert_ids": [r["id"] for r in records], "missing_alert_ids": missing,
                  "hosts": sorted(by_host), "replay_session": engine.session_id},
        "ground_truth_included": bool(include_ground_truth),
        "ground_truth_note": ("Dataset labels included for evaluation; they were never shown to the model."
                              if include_ground_truth else
                              f"Dataset labels ({', '.join(GROUND_TRUTH_FIELDS)}) are left out of this operational export. "
                              "They are covered by each record's ledger hash, so verify records against the server "
                              "ledger (hash, prev_hash, chain head below) rather than re-hashing this file."),
        "model": {"versions": versions, "lstm_threshold": getattr(engine, "lstm_threshold", None),
                  "lr_threshold": getattr(engine, "lr_threshold", None)},
        "config_fingerprint": fingerprint(),
        "alerts": alerts,
        "timelines": [_timeline(engine, h, recs) for h, recs in sorted(by_host.items())],
        "verification": {"alerts_chain": ledger.verify(), "analyst_actions_chain": analyst.verify(),
                         "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
