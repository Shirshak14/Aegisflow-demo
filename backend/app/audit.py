"""Hash-chained, append-only alert ledger in SQLite (stdlib sqlite3).

hash_n = SHA256(canonical_json(alert fields of record n) + hash_{n-1}), with
hash_0's predecessor = 64 zeros. Verification recomputes every hash from the
stored COLUMNS (the values the API and dashboard display), so editing any field,
deleting or reordering a record breaks the chain at that record.

Sessions: one replay run is one session (``session_id``). The chain is NOT reset between sessions, so
earlier sessions stay in the ledger and keep verifying; list/count can be scoped to one session. Only the
explicit ``Ledger.reset()`` (not called by the replay engine or the API) erases records.

Head anchor: after every append the id and hash of the newest record are written to a second file
(``<db>.head``). ``verify()`` checks the anchor, which catches deleting the most recent record(s) and
recomputing the whole chain, neither of which the chain alone can see. If ``AEGISFLOW_LEDGER_KEY`` is
set the anchor is HMAC-SHA256 signed, so someone who can edit the DB and the anchor file but does not
hold the key cannot forge it; without a key the anchor is only a second copy of the head.

External checkpoint: ``Ledger.export_checkpoint(path)`` writes a small signed record (record count, newest id, head hash,
time) that is meant to be stored OFF the host (another machine, a bucket, a printout). ``verify_checkpoint(path)``
later proves the current ledger still contains the exact history the checkpoint saw, so truncation, rewriting or
wholesale replacement of the DB is caught even when the attacker also rewrote the local anchor. Setting
``AEGISFLOW_LEDGER_CHECKPOINT`` mirrors the checkpoint to that path after every append (point it at a synced or
network location). ``recover_prefix`` copies the intact prefix into a NEW database and never edits the original. The
CLI is ``python -m backend.app.audit {checkpoint,verify,recover}``. This is tamper-EVIDENT, not tamper-proof:
nothing here stops someone with the signing key and the host from forging history; it makes undetected edits hard.

Known limits: the anchor sits next to the DB by default, so an attacker with full filesystem access and
no signing key can rewrite both; keep the key off the host and copy ``head_hash`` elsewhere for stronger
guarantees. Records appended outside the API after the last anchor are reported as ``anchor: behind``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
KEY_ENV = "AEGISFLOW_LEDGER_KEY"
CHECKPOINT_ENV = "AEGISFLOW_LEDGER_CHECKPOINT"
CHECKPOINT_VERSION = 1

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
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
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
    def __init__(self, db_path: str | Path, anchor_key: str | bytes | None = None):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        key = anchor_key if anchor_key is not None else os.environ.get(KEY_ENV)
        self._key = key.encode("utf-8") if isinstance(key, str) else (key or None)
        self.anchor_path = Path(self.db_path + ".head")
        self.checkpoint_path = Path(os.environ[CHECKPOINT_ENV]) if os.environ.get(CHECKPOINT_ENV) else None
        self._lock = threading.Lock()
        self._conns_lock = threading.Lock()
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        con = self._connect()
        con.execute("PRAGMA journal_mode=WAL")  # persistent in the DB file, so set once
        with con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """One connection per thread, reused. ``with con:`` commits but does not close it."""
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
            with self._conns_lock:
                self._conns.append(con)
        return con

    def close(self) -> None:
        with self._conns_lock:
            for con in self._conns:
                con.close()
            self._conns.clear()
        self._local = threading.local()

    # ------------------------------------------------------------------ head anchor
    def _sign(self, last_id: int, head_hash: str) -> str | None:
        if not self._key:
            return None
        return hmac.new(self._key, f"{last_id}:{head_hash}".encode("utf-8"), hashlib.sha256).hexdigest()

    def _write_anchor(self, last_id: int, head_hash: str) -> None:
        payload = {"last_id": last_id, "head_hash": head_hash, "hmac": self._sign(last_id, head_hash)}
        tmp = self.anchor_path.with_name(self.anchor_path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        # Windows refuses to replace a file another handle (a concurrent verify read, antivirus) has open for a
        # moment; that is transient, so retry briefly instead of failing the whole replay.
        for attempt in range(20):
            try:
                os.replace(tmp, self.anchor_path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))

    def _read_anchor(self) -> dict[str, Any] | None:
        try:
            for attempt in range(20):  # same transient Windows lock as in _write_anchor, seen from the reader
                try:
                    raw = self.anchor_path.read_text(encoding="utf-8")
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.01 * (attempt + 1))
            a = json.loads(raw)
            return {"last_id": int(a["last_id"]), "head_hash": str(a["head_hash"]), "hmac": a.get("hmac")}
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError):
            return {"last_id": -1, "head_hash": "", "hmac": None, "unreadable": True}

    def _anchor_problem(self, anchor: dict[str, Any], hash_by_id: dict[int, str]) -> str | None:
        """Why the head anchor does not match the chain, or None if it does."""
        if anchor.get("unreadable"):
            return "head anchor file is unreadable (it was edited or truncated)"
        if self._key:
            expected = self._sign(anchor["last_id"], anchor["head_hash"])
            if not anchor["hmac"] or not hmac.compare_digest(str(anchor["hmac"]), expected):
                return "head anchor signature is invalid (the anchor was edited or signed with a different key)"
        if anchor["last_id"] == 0:
            return None
        if anchor["last_id"] not in hash_by_id:
            return (f"newest record(s) deleted: the head anchor points at record {anchor['last_id']}, "
                    "which is no longer in the ledger")
        if hash_by_id[anchor["last_id"]] != anchor["head_hash"]:
            return "chain was recomputed: the hash of the anchored record differs from the recorded head hash"
        return None

    # ------------------------------------------------------------------ records
    def append(self, fields: dict[str, Any]) -> dict[str, Any]:
        rec = _normalize({**fields, "created_at": fields.get("created_at")
                          or datetime.now(timezone.utc).isoformat(timespec="seconds")})
        with self._lock:
            con = self._connect()
            with con:
                row = con.execute("SELECT hash FROM alerts ORDER BY id DESC LIMIT 1").fetchone()
                prev = row["hash"] if row else GENESIS
                h = compute_hash(rec, prev)
                cols = ", ".join(HASH_FIELDS) + ", prev_hash, hash"
                cur = con.execute(f"INSERT INTO alerts ({cols}) VALUES ({', '.join('?' * (len(HASH_FIELDS) + 2))})",
                                  [rec[k] for k in HASH_FIELDS] + [prev, h])
            self._write_anchor(cur.lastrowid, h)  # after the commit, so the anchor never points at a missing row
            if self.checkpoint_path is not None:
                self._write_checkpoint(self.checkpoint_path, cur.lastrowid, h)
            return {"id": cur.lastrowid, **rec, "prev_hash": prev, "hash": h}

    def list(self, limit: int = 100, offset: int = 0, host_id: str | None = None,
             session_id: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        if host_id:
            where.append("host_id = ?"); args.append(host_id)
        if session_id:
            where.append("session_id = ?"); args.append(session_id)
        q = "SELECT * FROM alerts" + (" WHERE " + " AND ".join(where) if where else "")
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._connect() as con:
            return [dict(r) for r in con.execute(q, args + [limit, offset]).fetchall()]

    def get(self, alert_id: int) -> dict[str, Any] | None:
        with self._connect() as con:
            r = con.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return dict(r) if r else None

    def count(self, session_id: str | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM alerts", []
        if session_id:
            q, args = q + " WHERE session_id = ?", [session_id]
        with self._connect() as con:
            return int(con.execute(q, args).fetchone()[0])

    def sessions(self) -> list[dict[str, Any]]:
        """Sessions in the ledger, oldest first, with their alert count and first/last record id."""
        with self._connect() as con:
            rows = con.execute("SELECT session_id, COUNT(*) AS alerts, MIN(id) AS first_id, MAX(id) AS last_id "
                               "FROM alerts GROUP BY session_id ORDER BY first_id").fetchall()
        return [dict(r) for r in rows]

    def reset(self) -> None:
        """Erase every alert and analyst action. Explicit and manual: nothing in the app calls this."""
        with self._lock:
            with self._connect() as con:
                con.execute("DELETE FROM alerts")
                con.execute("DELETE FROM sqlite_sequence WHERE name = 'alerts'")
                # analyst actions (backend/app/analyst.py) refer to alerts by id; clear them with the alerts
                if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='alert_actions'").fetchone():
                    con.execute("DELETE FROM alert_actions")
                    con.execute("DELETE FROM sqlite_sequence WHERE name = 'alert_actions'")
            self._write_anchor(0, GENESIS)  # the external checkpoint, if any, is deliberately left alone

    # ------------------------------------------------------------------ external checkpoint
    def _sign_checkpoint(self, body: dict[str, Any]) -> str | None:
        if not self._key:
            return None
        msg = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hmac.new(self._key, b"checkpoint:" + msg.encode("utf-8"), hashlib.sha256).hexdigest()

    def _write_checkpoint(self, path: Path, last_id: int, head_hash: str, records: int | None = None) -> dict[str, Any]:
        if records is None:
            with self._connect() as con:
                records = int(con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
        body = {"version": CHECKPOINT_VERSION, "last_id": last_id, "head_hash": head_hash, "records": records,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "signed": bool(self._key)}
        doc = {**body, "hmac": self._sign_checkpoint(body)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return doc

    def export_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Write a signed checkpoint of the current head to ``path`` (store it away from this host).

        Refuses to checkpoint a ledger that does not verify, so a checkpoint never certifies a corrupted chain."""
        v = self.verify()
        if v["status"] != "VERIFIED":
            raise ValueError(f"ledger does not verify ({v['reason']}); refusing to checkpoint it")
        with self._connect() as con:
            row = con.execute("SELECT id, hash FROM alerts ORDER BY id DESC LIMIT 1").fetchone()
        last_id, head = (row["id"], row["hash"]) if row else (0, GENESIS)
        return self._write_checkpoint(Path(path), last_id, head, v["records_checked"])

    def verify_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Check the ledger against an externally stored checkpoint.

        Returns ``{"status": "MATCHES" | "MISMATCH" | "CHECKPOINT_INVALID", ...}``. MATCHES means the checkpointed
        record is still in the chain with the same hash (new records after it are expected and reported as
        ``records_since_checkpoint``). The chain itself is verified first."""
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            body = {k: doc[k] for k in ("version", "last_id", "head_hash", "records", "created_at", "signed")}
            last_id, head = int(body["last_id"]), str(body["head_hash"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"status": "CHECKPOINT_INVALID", "reason": f"checkpoint unreadable or malformed ({exc!r})"}
        if self._key:
            expected = self._sign_checkpoint(body)
            if not body["signed"] or not doc.get("hmac") or not hmac.compare_digest(str(doc["hmac"]), expected):
                return {"status": "CHECKPOINT_INVALID",
                        "reason": "checkpoint signature is missing or invalid (edited, or signed with another key)"}
        elif body["signed"]:
            return {"status": "CHECKPOINT_INVALID",
                    "reason": f"checkpoint is signed but {KEY_ENV} is not set, so it cannot be authenticated"}
        chain = self.verify()
        if chain["status"] != "VERIFIED":
            return {"status": "MISMATCH", "reason": f"the ledger chain itself is corrupted: {chain['reason']}",
                    "checkpoint_last_id": last_id}
        with self._connect() as con:
            row = con.execute("SELECT hash FROM alerts WHERE id = ?", (last_id,)).fetchone() if last_id else None
            total = int(con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
        base = {"checkpoint_last_id": last_id, "checkpoint_created_at": body["created_at"],
                "signed": bool(self._key and body["signed"]), "records_now": total}
        if last_id == 0:
            return {"status": "MATCHES", **base, "records_since_checkpoint": total, "reason": None}
        if row is None:
            return {"status": "MISMATCH", **base, "reason": f"record {last_id}, the checkpointed head, is no longer "
                                                            "in the ledger (records were deleted or the DB was replaced)"}
        if row["hash"] != head:
            return {"status": "MISMATCH", **base, "reason": f"record {last_id} exists but its hash differs from the "
                                                            "checkpoint (history was rewritten)"}
        return {"status": "MATCHES", **base, "records_since_checkpoint": total - int(body["records"]), "reason": None}

    def recover_prefix(self, out_db: str | Path) -> dict[str, Any]:
        """Copy the longest intact prefix of the chain into a NEW database (the original is never modified).

        The copy gets its own head anchor (signed if a key is set). Records after the first bad one are not copied;
        they stay in the original for investigation. Compare the result with an external checkpoint afterwards."""
        out = Path(out_db)
        if out.exists():
            raise FileExistsError(f"{out} already exists; recovery never overwrites")
        good, prev = [], GENESIS
        for r in self._connect().execute("SELECT * FROM alerts ORDER BY id ASC"):
            row = dict(r)
            if row["prev_hash"] != prev or compute_hash(row, row["prev_hash"]) != row["hash"]:
                break
            good.append(row)
            prev = row["hash"]
        total = self.count()
        new = Ledger(out, anchor_key=self._key)
        try:
            with new._connect() as con:
                cols = ["id", *HASH_FIELDS, "prev_hash", "hash"]
                con.executemany(f"INSERT INTO alerts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                [[g[c] for c in cols] for g in good])
            new._write_anchor(good[-1]["id"] if good else 0, prev)
            verdict = new.verify()
        finally:
            new.close()
        return {"recovered_records": len(good), "dropped_records": total - len(good), "output": str(out),
                "output_verify": verdict["status"], "head_hash": prev}

    def verify(self) -> dict[str, Any]:
        expected_prev, checked, hash_by_id = GENESIS, 0, {}
        con = self._connect()
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
            hash_by_id[row["id"]] = row["hash"]
            checked += 1
        anchor = self._read_anchor()
        if anchor is None:
            state = "missing"  # legacy ledger, or the anchor file was removed
        else:
            problem = self._anchor_problem(anchor, hash_by_id)
            if problem:
                return {"status": "CORRUPTED", "record_id": anchor["last_id"] if anchor["last_id"] > 0 else None,
                        "records_checked": checked, "reason": problem}
            if checked and anchor["last_id"] != max(hash_by_id):
                state = "behind"
            else:
                state = "signed" if self._key else "unsigned"
        return {"status": "VERIFIED", "record_id": None, "records_checked": checked,
                "head_hash": expected_prev, "anchor": state, "reason": None}


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m backend.app.audit", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("checkpoint", "write a signed checkpoint of the ledger head (store it off this host)"),
                      ("verify", "verify the chain, and a checkpoint if --checkpoint is given"),
                      ("recover", "copy the intact prefix of the chain into a new database")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--db", required=True, help="ledger SQLite file")
        if name == "checkpoint":
            p.add_argument("--out", required=True)
        if name == "verify":
            p.add_argument("--checkpoint")
        if name == "recover":
            p.add_argument("--out", required=True, help="new database to create (must not exist)")
    a = ap.parse_args(argv)
    if not Path(a.db).exists():
        print(f"no such ledger: {a.db}")
        return 2
    led = Ledger(a.db)
    try:
        if a.cmd == "checkpoint":
            try:
                print(json.dumps(led.export_checkpoint(a.out), indent=2))
            except ValueError as exc:
                print(exc)
                return 1
            if not led._key:
                print(f"warning: {KEY_ENV} is not set, so this checkpoint is NOT signed and only detects accidents")
            return 0
        if a.cmd == "verify":
            chain = led.verify()
            print(json.dumps(chain, indent=2))
            ok = chain["status"] == "VERIFIED"
            if a.checkpoint:
                cp = led.verify_checkpoint(a.checkpoint)
                print(json.dumps(cp, indent=2))
                ok = ok and cp["status"] == "MATCHES"
            return 0 if ok else 1
        print(json.dumps(led.recover_prefix(a.out), indent=2))
        return 0
    finally:
        led.close()


if __name__ == "__main__":
    raise SystemExit(_main())
