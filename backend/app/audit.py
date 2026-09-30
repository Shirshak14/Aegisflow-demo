"""Hash-chained, append-only alert ledger in SQLite (stdlib sqlite3).

hash_n = SHA256(canonical_json(alert fields of record n) + hash_{n-1}), with
hash_0's predecessor = 64 zeros. Verification recomputes every hash from the
stored COLUMNS (the values the API and dashboard display), so editing any field,
deleting or reordering a record breaks the chain at that record.

Known limit: deleting the most recent record(s) leaves a shorter valid chain;
detect that by comparing the reported head hash with one recorded externally.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64

# Every displayed alert field is covered by the hash (id/prev_hash/hash excluded).
HASH_FIELDS: tuple[str, ...] = (
    "session_id", "sequence_id", "host_id", "predicted_at", "target_window_start", "target_window_end",
    "lstm_probability", "lstm_threshold", "lr_probability", "lr_threshold", "lr_flag", "reference_rule_flag",
    "risk_score", "risk_components", "confidence", "confidence_label", "predicted_stage",
    "truth_attack", "truth_class", "truth_stage", "model_version", "created_at",
)
_REAL = {"lstm_probability", "lstm_threshold", "lr_probability", "lr_threshold", "risk_score", "confidence"}
_INT = {"lr_flag", "reference_rule_flag", "truth_attack"}

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {", ".join(f"{f} {'REAL' if f in _REAL else 'INTEGER' if f in _INT else 'TEXT'}" for f in HASH_FIELDS)},
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_host ON alerts(host_id);
"""


def canonical(fields: dict[str, Any]) -> str:
    return json.dumps({k: fields.get(k) for k in HASH_FIELDS}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def compute_hash(fields: dict[str, Any], prev_hash: str) -> str:
    return hashlib.sha256((canonical(fields) + prev_hash).encode("utf-8")).hexdigest()


def _normalize(fields: dict[str, Any]) -> dict[str, Any]:
    """Coerce to the exact types SQLite returns, so hashes recompute identically."""
    out: dict[str, Any] = {}
    for k in HASH_FIELDS:
        v = fields.get(k)
        if v is None:
            out[k] = None
        elif k in _REAL:
            out[k] = round(float(v), 6)
        elif k in _INT:
            out[k] = int(v)
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, sort_keys=True, separators=(",", ":"))
        else:
            out[k] = str(v)
    return out


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def append(self, fields: dict[str, Any]) -> dict[str, Any]:
        rec = _normalize({**fields, "created_at": fields.get("created_at")
                          or datetime.now(timezone.utc).isoformat(timespec="seconds")})
        with self._lock, self._connect() as con:
            row = con.execute("SELECT hash FROM alerts ORDER BY id DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            h = compute_hash(rec, prev)
            cols = ", ".join(HASH_FIELDS) + ", prev_hash, hash"
            cur = con.execute(f"INSERT INTO alerts ({cols}) VALUES ({', '.join('?' * (len(HASH_FIELDS) + 2))})",
                              [rec[k] for k in HASH_FIELDS] + [prev, h])
            return {"id": cur.lastrowid, **rec, "prev_hash": prev, "hash": h}

    def list(self, limit: int = 100, offset: int = 0, host_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM alerts", []
        if host_id:
            q, args = q + " WHERE host_id = ?", [host_id]
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._connect() as con:
            return [dict(r) for r in con.execute(q, args + [limit, offset]).fetchall()]

    def get(self, alert_id: int) -> dict[str, Any] | None:
        with self._connect() as con:
            r = con.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return dict(r) if r else None

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])

    def reset(self) -> None:
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM alerts")
            con.execute("DELETE FROM sqlite_sequence WHERE name = 'alerts'")

    def verify(self) -> dict[str, Any]:
        expected_prev, checked = GENESIS, 0
        with self._connect() as con:
            for r in con.execute("SELECT * FROM alerts ORDER BY id ASC"):
                row = dict(r)
                if row["prev_hash"] != expected_prev:
                    return {"status": "CORRUPTED", "record_id": row["id"], "records_checked": checked,
                            "reason": "chain link broken: prev_hash does not match the previous record's hash "
                                      "(a record before it was deleted, reordered, or its hash rewritten)"}
                if compute_hash(row, row["prev_hash"]) != row["hash"]:
                    return {"status": "CORRUPTED", "record_id": row["id"], "records_checked": checked,
                            "reason": "record content does not match its stored hash (a field was modified)"}
                expected_prev = row["hash"]
                checked += 1
        return {"status": "VERIFIED", "record_id": None, "records_checked": checked,
                "head_hash": expected_prev, "reason": None}
