import json

import pytest
import sqlite3

from fastapi.testclient import TestClient

from tests.api.conftest import AUTH
from backend.app.audit import GENESIS, Ledger
from backend.app.main import create_app


def _alert(i: int) -> dict:
    return {"sequence_id": f"h_seq_{i}", "host_id": "10.0.0.1", "lstm_probability": 0.1 * i,
            "risk_score": 10.0 * i, "risk_components": {"attack_probability": 0.1 * i}, "truth_attack": i % 2}


def test_chain_links_and_verifies(tmp_path):
    led = Ledger(tmp_path / "a.db")
    recs = [led.append(_alert(i)) for i in range(1, 6)]
    assert recs[0]["prev_hash"] == GENESIS
    assert all(b["prev_hash"] == a["hash"] for a, b in zip(recs, recs[1:]))
    assert led.verify() == {"status": "VERIFIED", "record_id": None, "records_checked": 5,
                            "head_hash": recs[-1]["hash"], "anchor": "unsigned", "reason": None}


def test_api_detects_direct_db_tampering_and_names_record(tmp_path):
    db = tmp_path / "a.db"
    client = TestClient(create_app(db, warm_up=False))
    for i in range(1, 6):
        assert client.post("/alerts", headers=AUTH, json=_alert(i)).status_code == 201
    assert client.get("/audit/verify").json()["status"] == "VERIFIED"
    with sqlite3.connect(db) as con:  # tamper outside the API, directly in the DB
        con.execute("UPDATE alerts SET risk_score = 0.0 WHERE id = 3")
    v = client.get("/audit/verify").json()
    assert v["status"] == "CORRUPTED" and v["record_id"] == 3 and "modified" in v["reason"]


def test_rewriting_a_hash_breaks_the_next_link(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db)
    for i in range(1, 5):
        led.append(_alert(i))
    # attacker edits record 2 AND recomputes its hash: record 3's prev_hash no longer matches
    row = led.get(2); row["risk_score"] = 99.0
    from backend.app.audit import compute_hash
    with sqlite3.connect(db) as con:
        con.execute("UPDATE alerts SET risk_score = 99.0, hash = ? WHERE id = 2", (compute_hash(row, row["prev_hash"]),))
    v = led.verify()
    assert v["status"] == "CORRUPTED" and v["record_id"] == 3 and "chain link broken" in v["reason"]


def test_deleting_a_middle_record_is_detected(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db)
    for i in range(1, 5):
        led.append(_alert(i))
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM alerts WHERE id = 2")
    v = led.verify()
    assert v["status"] == "CORRUPTED" and v["record_id"] == 3


def _recompute_from(db, start_id: int) -> str:
    """Attacker edits record `start_id` and recomputes every hash after it, so the chain alone looks valid."""
    from backend.app.audit import compute_hash
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        prev = con.execute("SELECT prev_hash FROM alerts WHERE id = ?", (start_id,)).fetchone()[0]
        for r in con.execute("SELECT * FROM alerts WHERE id >= ? ORDER BY id", (start_id,)).fetchall():
            row = dict(r)
            if row["id"] == start_id:
                row["risk_score"] = 1.0
            h = compute_hash(row, prev)
            con.execute("UPDATE alerts SET risk_score = ?, prev_hash = ?, hash = ? WHERE id = ?",
                        (row["risk_score"], prev, h, row["id"]))
            prev = h
    return prev


def test_tail_deletion_is_caught_by_the_anchor(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db)
    for i in range(1, 6):
        led.append(_alert(i))
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM alerts WHERE id >= 4")  # chain alone is still valid
    v = led.verify()
    assert v["status"] == "CORRUPTED" and v["record_id"] == 5 and "deleted" in v["reason"]


def test_full_recomputation_is_caught_by_the_anchor(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db)
    for i in range(1, 5):
        led.append(_alert(i))
    _recompute_from(db, 2)
    v = led.verify()
    assert v["status"] == "CORRUPTED" and "recomputed" in v["reason"]


def test_signed_anchor_cannot_be_forged_without_the_key(tmp_path):
    import json
    db = tmp_path / "a.db"
    led = Ledger(db, anchor_key="secret")
    for i in range(1, 5):
        led.append(_alert(i))
    assert led.verify()["anchor"] == "signed"
    new_head = _recompute_from(db, 2)
    # attacker also rewrites the anchor file, but cannot sign it
    led.anchor_path.write_text(json.dumps({"last_id": 4, "head_hash": new_head, "hmac": None}))
    v = led.verify()
    assert v["status"] == "CORRUPTED" and "signature" in v["reason"]
    forged = Ledger(db, anchor_key="wrong-key")
    assert forged.verify()["status"] == "CORRUPTED"


def test_missing_anchor_is_reported_not_fatal_and_reset_reanchors(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db)
    led.append(_alert(1))
    led.anchor_path.unlink()
    assert led.verify()["anchor"] == "missing" and led.verify()["status"] == "VERIFIED"
    led.append(_alert(2))
    assert led.verify()["anchor"] == "unsigned"
    led.reset()
    v = led.verify()
    assert v["status"] == "VERIFIED" and v["records_checked"] == 0
    led.append(_alert(3))
    assert led.verify()["records_checked"] == 1


def test_connections_are_reused_per_thread(tmp_path):
    import threading
    led = Ledger(tmp_path / "a.db")
    assert led._connect() is led._connect()
    other = []
    t = threading.Thread(target=lambda: other.append(led._connect()))
    t.start(); t.join()
    assert other[0] is not led._connect()
    led.close()
    assert led.count() == 0  # reconnects lazily after close


def test_post_alert_requires_api_key(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path / "a.db", warm_up=False))
    assert client.post("/alerts", json=_alert(1)).status_code == 401
    assert client.post("/alerts", headers={"X-API-Key": "nope"}, json=_alert(1)).status_code == 401
    assert client.post("/alerts", headers=AUTH, json=_alert(1)).status_code == 201
    monkeypatch.delenv("AEGISFLOW_API_KEY")
    r = client.post("/alerts", headers=AUTH, json=_alert(2))
    assert r.status_code == 503 and "AEGISFLOW_API_KEY" in r.json()["detail"]
    assert client.get("/health").json()["alerts_in_ledger"] == 1


def test_post_alert_rejects_out_of_range_values(tmp_path):
    client = TestClient(create_app(tmp_path / "a.db", warm_up=False))
    bad = [{"lstm_probability": 1.5}, {"lstm_probability": -0.1}, {"confidence": 2}, {"lr_threshold": 7},
           {"risk_score": 101}, {"risk_score": -1}, {"lr_flag": 2}, {"truth_attack": -1}, {"reference_rule_flag": 5},
           {"host_id": "x" * 129}, {"model_version": "y" * 500}, {"risk_components": {f"k{i}": 1 for i in range(33)}},
           {"risk_components": {"a": {"nested": 1}}}]
    for patch in bad:
        r = client.post("/alerts", headers=AUTH, json={**_alert(1), **patch})
        assert r.status_code == 422, patch
    assert client.get("/health").json()["alerts_in_ledger"] == 0
    ok = {**_alert(1), "lstm_probability": 1.0, "confidence": 0.0, "risk_score": 100, "lr_flag": 1}
    assert client.post("/alerts", headers=AUTH, json=ok).status_code == 201


# ---- transient Windows "Access is denied" on the head-anchor file (seen killing a 1000x replay) ----
def _flaky(real, failures: int):
    """Wrap ``real`` so its first ``failures`` calls raise PermissionError, as Windows does for a briefly locked file."""
    calls = {"n": 0}

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise PermissionError(13, "Access is denied")
        return real(*args, **kwargs)

    return wrapper, calls


def test_anchor_write_retries_a_transient_permission_error(tmp_path, monkeypatch):
    from backend.app import audit

    led = Ledger(tmp_path / "a.db")
    monkeypatch.setattr(audit.time, "sleep", lambda s: None)
    flaky, calls = _flaky(audit.os.replace, failures=5)
    monkeypatch.setattr(audit.os, "replace", flaky)
    led.append(_alert(1))  # would raise PermissionError without the retry
    assert calls["n"] == 6
    monkeypatch.undo()
    assert led.verify()["status"] == "VERIFIED"


def test_anchor_write_still_raises_a_persistent_permission_error(tmp_path, monkeypatch):
    import pytest

    from backend.app import audit

    led = Ledger(tmp_path / "a.db")
    monkeypatch.setattr(audit.time, "sleep", lambda s: None)
    flaky, calls = _flaky(audit.os.replace, failures=10**6)
    monkeypatch.setattr(audit.os, "replace", flaky)
    with pytest.raises(PermissionError):
        led.append(_alert(1))
    assert calls["n"] == 20  # bounded: gives up after 20 attempts


def test_anchor_read_retries_a_transient_permission_error(tmp_path, monkeypatch):
    from pathlib import Path

    from backend.app import audit

    led = Ledger(tmp_path / "a.db")
    rec = led.append(_alert(1))
    monkeypatch.setattr(audit.time, "sleep", lambda s: None)
    flaky, calls = _flaky(Path.read_text, failures=3)
    monkeypatch.setattr(Path, "read_text", flaky)
    anchor = led._read_anchor()
    assert anchor["head_hash"] == rec["hash"] and calls["n"] == 4


# ---- external signed checkpoint, verify and recovery
def _filled(tmp_path, n=5, key="k" * 32):
    led = Ledger(tmp_path / "c.db", anchor_key=key)
    for i in range(1, n + 1):
        led.append(_alert(i))
    return led


def test_checkpoint_matches_and_allows_new_records(tmp_path):
    led = _filled(tmp_path)
    cp = tmp_path / "off_host" / "cp.json"
    led.export_checkpoint(cp)
    led.append(_alert(6))
    r = led.verify_checkpoint(cp)
    assert r["status"] == "MATCHES" and r["records_since_checkpoint"] == 1 and r["signed"]


def test_checkpoint_catches_truncation_even_if_anchor_is_rewritten(tmp_path):
    led = _filled(tmp_path)
    cp = tmp_path / "cp.json"
    led.export_checkpoint(cp)
    with sqlite3.connect(led.db_path) as con:
        con.execute("DELETE FROM alerts WHERE id >= 4")
    led._write_anchor(3, led.get(3)["hash"])  # attacker with the key on the host repairs the local anchor
    assert led.verify()["status"] == "VERIFIED"  # the local checks alone are fooled
    assert led.verify_checkpoint(cp)["status"] == "MISMATCH"


def test_checkpoint_rejects_edit_and_wrong_key(tmp_path):
    led = _filled(tmp_path)
    cp = tmp_path / "cp.json"
    led.export_checkpoint(cp)
    doc = json.loads(cp.read_text())
    doc["head_hash"] = "0" * 64
    cp.write_text(json.dumps(doc))
    assert led.verify_checkpoint(cp)["status"] == "CHECKPOINT_INVALID"
    led.export_checkpoint(cp)
    other = Ledger(led.db_path, anchor_key="z" * 32)
    assert other.verify_checkpoint(cp)["status"] == "CHECKPOINT_INVALID"


def test_checkpoint_refuses_corrupted_ledger_and_mirror_env(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror.json"
    monkeypatch.setenv("AEGISFLOW_LEDGER_CHECKPOINT", str(mirror))
    led = _filled(tmp_path, 3)
    assert json.loads(mirror.read_text())["last_id"] == 3
    with sqlite3.connect(led.db_path) as con:
        con.execute("UPDATE alerts SET risk_score = 1.0 WHERE id = 2")
    with pytest.raises(ValueError):
        led.export_checkpoint(tmp_path / "x.json")
    assert led.verify_checkpoint(mirror)["status"] == "MISMATCH"


def test_recover_prefix_copies_intact_records_into_new_db(tmp_path):
    led = _filled(tmp_path, 6)
    with sqlite3.connect(led.db_path) as con:
        con.execute("UPDATE alerts SET risk_score = 1.0 WHERE id = 4")
    r = led.recover_prefix(tmp_path / "recovered.db")
    assert r["recovered_records"] == 3 and r["dropped_records"] == 3 and r["output_verify"] == "VERIFIED"
    assert led.verify()["status"] == "CORRUPTED"  # original untouched
    with pytest.raises(FileExistsError):
        led.recover_prefix(tmp_path / "recovered.db")


def test_audit_cli_roundtrip(tmp_path, monkeypatch, capsys):
    from backend.app.audit import _main
    monkeypatch.setenv("AEGISFLOW_LEDGER_KEY", "k" * 32)
    led = _filled(tmp_path, 3)
    led.close()
    cp = str(tmp_path / "cp.json")
    assert _main(["checkpoint", "--db", str(led.db_path), "--out", cp]) == 0
    assert _main(["verify", "--db", str(led.db_path), "--checkpoint", cp]) == 0
    with sqlite3.connect(led.db_path) as con:
        con.execute("DELETE FROM alerts WHERE id = 3")
    assert _main(["verify", "--db", str(led.db_path), "--checkpoint", cp]) == 1
