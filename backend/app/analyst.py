"""Analyst review of alerts: acknowledge, approve a response, dismiss, override the stage, reopen.

AegisFlow never acts on traffic by itself. A response (e.g. "block 172.16.0.1 at the perimeter") is
only *recorded* as approved by a named analyst; carrying it out is left to the SOC's own tooling.

Every action is appended to its own SHA-256 hash chain (table ``alert_actions`` in the same SQLite
file as the alert ledger). Each action's hash also covers the hash of the alert it refers to, so an
approval is bound to the exact alert record the analyst saw: editing the alert afterwards, or editing,
deleting or reordering an action, is detected by ``verify()``.

Status of an alert = result of its latest status-changing action (``open`` if none):
  acknowledge       open | reopened              -> acknowledged
  approve_response  open | acknowledged | reopened -> response_approved   (needs ``response``)
  dismiss           open | acknowledged | reopened -> dismissed           (false positive / benign)
  reopen            response_approved | dismissed -> reopened
  override_stage    any status, status unchanged    (needs ``stage`` from stages.yaml stages_order)
  comment           any status, status unchanged    (needs ``note``)

Known limit: there is no authentication, so ``analyst`` is self-declared; put the API behind the
SOC's SSO before relying on names.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .audit import GENESIS, Ledger

ACTION_FIELDS = ("alert_id", "alert_hash", "action", "analyst", "note", "response", "stage", "status_after",
                 "created_at")
TRANSITIONS: dict[str, tuple[set[str] | None, str | None]] = {
    "acknowledge": ({"open", "reopened"}, "acknowledged"),
    "approve_response": ({"open", "acknowledged", "reopened"}, "response_approved"),
    "dismiss": ({"open", "acknowledged", "reopened"}, "dismissed"),
    "reopen": ({"response_approved", "dismissed"}, "reopened"),
    "override_stage": (None, None),
    "comment": (None, None),
}

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS alert_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {", ".join(f"{f} {'INTEGER' if f == 'alert_id' else 'TEXT'}" for f in ACTION_FIELDS)},
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_alert ON alert_actions(alert_id);
"""


class ActionError(ValueError):
    """Invalid action, missing field, or a transition the current status does not allow."""


def _hash(rec: dict[str, Any], prev: str) -> str:
    canon = json.dumps({k: rec.get(k) for k in ACTION_FIELDS}, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False)
    return hashlib.sha256((canon + prev).encode("utf-8")).hexdigest()


class AnalystLog:
    def __init__(self, ledger: Ledger, stages: list[str]):
        self.ledger = ledger
        self.stages = list(stages)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return self.ledger._connect()

    # ------------------------------------------------------------------ writes
    def record(self, alert_id: int, action: str, analyst: str, *, note: str | None = None,
               response: str | None = None, stage: str | None = None) -> dict[str, Any]:
        if action not in TRANSITIONS:
            raise ActionError(f"unknown action '{action}'; choose from {sorted(TRANSITIONS)}")
        if not analyst or not analyst.strip():
            raise ActionError("analyst is required")
        if action == "approve_response" and not (response and response.strip()):
            raise ActionError("approve_response needs the response being approved")
        if action == "override_stage" and stage not in self.stages:
            raise ActionError(f"override_stage needs stage from {self.stages}")
        if action == "comment" and not (note and note.strip()):
            raise ActionError("comment needs a note")
        with self.ledger._lock, self._connect() as con:
            alert = con.execute("SELECT id, hash FROM alerts WHERE id = ?", (alert_id,)).fetchone()
            if alert is None:
                raise LookupError(f"alert {alert_id} not found")
            current = self._status(con, alert_id)
            allowed, after = TRANSITIONS[action]
            if allowed is not None and current not in allowed:
                raise ActionError(f"cannot {action} an alert that is {current}")
            rec = {"alert_id": int(alert_id), "alert_hash": alert["hash"], "action": action,
                   "analyst": analyst.strip(), "note": note, "response": response,
                   "stage": stage if action == "override_stage" else None,
                   "status_after": after or current,
                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            row = con.execute("SELECT hash FROM alert_actions ORDER BY id DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            h = _hash(rec, prev)
            cols = ", ".join(ACTION_FIELDS) + ", prev_hash, hash"
            cur = con.execute(f"INSERT INTO alert_actions ({cols}) VALUES ({', '.join('?' * (len(ACTION_FIELDS) + 2))})",
                              [rec[k] for k in ACTION_FIELDS] + [prev, h])
            return {"id": cur.lastrowid, **rec, "prev_hash": prev, "hash": h}

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _status(con: sqlite3.Connection, alert_id: int) -> str:
        r = con.execute("SELECT status_after FROM alert_actions WHERE alert_id = ? ORDER BY id DESC LIMIT 1",
                        (alert_id,)).fetchone()
        return r["status_after"] if r else "open"

    def history(self, alert_id: int) -> dict[str, Any] | None:
        with self._connect() as con:
            alert = con.execute("SELECT id, predicted_stage FROM alerts WHERE id = ?", (alert_id,)).fetchone()
            if alert is None:
                return None
            actions = [dict(r) for r in con.execute("SELECT * FROM alert_actions WHERE alert_id = ? ORDER BY id",
                                                     (alert_id,))]
            status = self._status(con, alert_id)
        overrides = [a["stage"] for a in actions if a["action"] == "override_stage"]
        approved = [a for a in actions if a["action"] == "approve_response"]
        return {"alert_id": alert_id, "status": status, "model_stage": alert["predicted_stage"],
                "effective_stage": overrides[-1] if overrides else alert["predicted_stage"],
                "approved_response": approved[-1]["response"] if approved and status == "response_approved" else None,
                "actions": actions}

    def statuses(self) -> dict[int, str]:
        """Current status of every alert that has at least one status-changing action (others are 'open')."""
        with self._connect() as con:
            rows = con.execute("""SELECT a.alert_id, a.status_after FROM alert_actions a
                                  JOIN (SELECT alert_id, MAX(id) AS last FROM alert_actions GROUP BY alert_id) l
                                  ON a.id = l.last""").fetchall()
        return {int(r["alert_id"]): r["status_after"] for r in rows}

    def summary(self) -> dict[str, int]:
        """Number of alerts in each status (alerts with no action are 'open')."""
        with self._connect() as con:
            total = int(con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
            rows = con.execute("""SELECT a.status_after AS s, COUNT(*) AS n FROM alert_actions a
                                  JOIN (SELECT alert_id, MAX(id) AS last FROM alert_actions GROUP BY alert_id) l
                                  ON a.id = l.last GROUP BY a.status_after""").fetchall()
        out = {r["s"]: int(r["n"]) for r in rows}
        out["open"] = out.get("open", 0) + total - sum(out.values())
        return out

    def verify(self) -> dict[str, Any]:
        expected, checked = GENESIS, 0
        with self._connect() as con:
            alert_hashes = {r["id"]: r["hash"] for r in con.execute("SELECT id, hash FROM alerts")}
            for r in con.execute("SELECT * FROM alert_actions ORDER BY id ASC"):
                row = dict(r)
                bad = None
                if row["prev_hash"] != expected:
                    bad = "chain link broken: an earlier action was deleted, reordered, or its hash rewritten"
                elif _hash(row, row["prev_hash"]) != row["hash"]:
                    bad = "action content does not match its stored hash (a field was modified)"
                elif alert_hashes.get(row["alert_id"]) != row["alert_hash"]:
                    bad = "the alert this action refers to was changed or removed after the action was recorded"
                if bad:
                    return {"status": "CORRUPTED", "action_id": row["id"], "actions_checked": checked, "reason": bad}
                expected = row["hash"]
                checked += 1
        return {"status": "VERIFIED", "action_id": None, "actions_checked": checked, "head_hash": expected,
                "reason": None}
