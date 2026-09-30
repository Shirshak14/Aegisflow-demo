import sqlite3

from fastapi.testclient import TestClient

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
                            "head_hash": recs[-1]["hash"], "reason": None}


def test_api_detects_direct_db_tampering_and_names_record(tmp_path):
    db = tmp_path / "a.db"
    client = TestClient(create_app(db, warm_up=False))
    for i in range(1, 6):
        assert client.post("/alerts", json=_alert(i)).status_code == 201
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
