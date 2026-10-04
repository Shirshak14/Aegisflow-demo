"""Analyst review workflow: status transitions, stage override, and its own tamper-evident chain."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.app.analyst import ActionError, AnalystLog
from backend.app.audit import Ledger
from backend.app.main import create_app

STAGES = ["Benign", "Reconnaissance", "Impact"]


def _alert(i):
    return {"sequence_id": f"h_seq_{i}", "host_id": "172.16.0.1", "lstm_probability": 0.9,
            "predicted_stage": "UNCERTAIN", "risk_score": 50.0}


@pytest.fixture()
def log(tmp_path):
    led = Ledger(tmp_path / "a.db")
    for i in range(3):
        led.append(_alert(i))
    return AnalystLog(led, STAGES)


def test_lifecycle_and_history(log):
    log.record(1, "acknowledge", "asha")
    log.record(1, "override_stage", "asha", stage="Impact", note="flood pattern")
    log.record(1, "approve_response", "ravi", response="rate-limit 172.16.0.1 at the edge")
    h = log.history(1)
    assert h["status"] == "response_approved" and h["model_stage"] == "UNCERTAIN" and h["effective_stage"] == "Impact"
    assert h["approved_response"] == "rate-limit 172.16.0.1 at the edge"
    assert [a["action"] for a in h["actions"]] == ["acknowledge", "override_stage", "approve_response"]
    log.record(1, "reopen", "ravi", note="re-check")
    assert log.history(1)["status"] == "reopened" and log.history(1)["approved_response"] is None
    log.record(2, "dismiss", "asha", note="scheduled backup")
    assert log.summary() == {"open": 1, "reopened": 1, "dismissed": 1}
    assert log.history(99) is None


@pytest.mark.parametrize("args,msg", [
    (("bogus", "a"), "unknown action"), (("acknowledge", " "), "analyst"),
    (("approve_response", "a"), "response"), (("override_stage", "a"), "stage"),
    (("comment", "a"), "note")])
def test_invalid_actions(log, args, msg):
    with pytest.raises(ActionError, match=msg):
        log.record(1, *args)


def test_disallowed_transitions_and_missing_alert(log):
    log.record(1, "dismiss", "a")
    with pytest.raises(ActionError, match="cannot acknowledge an alert that is dismissed"):
        log.record(1, "acknowledge", "a")
    with pytest.raises(ActionError, match="cannot reopen"):
        log.record(2, "reopen", "a")
    with pytest.raises(LookupError):
        log.record(42, "acknowledge", "a")


def test_chain_detects_edits_deletions_and_alert_changes(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db); led.append(_alert(0)); led.append(_alert(1))
    log = AnalystLog(led, STAGES)
    for who in ("a", "b"):
        log.record(1, "comment", who, note="x")
    log.record(2, "acknowledge", "c")
    assert log.verify()["status"] == "VERIFIED" and log.verify()["actions_checked"] == 3
    with sqlite3.connect(db) as con:
        con.execute("UPDATE alert_actions SET analyst = 'mallory' WHERE id = 2")
    v = log.verify(); assert (v["status"], v["action_id"]) == ("CORRUPTED", 2) and "modified" in v["reason"]
    with sqlite3.connect(db) as con:
        con.execute("UPDATE alert_actions SET analyst = 'b' WHERE id = 2")
        con.execute("DELETE FROM alert_actions WHERE id = 1")
    v = log.verify(); assert (v["status"], v["action_id"]) == ("CORRUPTED", 2) and "link" in v["reason"]


def test_action_is_bound_to_the_alert_record(tmp_path):
    db = tmp_path / "a.db"
    led = Ledger(db); led.append(_alert(0))
    log = AnalystLog(led, STAGES)
    log.record(1, "approve_response", "a", response="block")
    with sqlite3.connect(db) as con:  # someone rewrites the alert AND recomputes its hash
        con.execute("UPDATE alerts SET hash = ? WHERE id = 1", ("de" * 32,))
    v = log.verify()
    assert v["status"] == "CORRUPTED" and "alert this action refers to" in v["reason"]


def test_reset_clears_actions(log):
    log.record(1, "acknowledge", "a")
    log.ledger.reset()
    assert log.summary() == {"open": 0} and log.verify()["actions_checked"] == 0


def test_api(tmp_path):
    client = TestClient(create_app(tmp_path / "a.db", warm_up=False))
    for i in range(2):
        assert client.post("/alerts", json=_alert(i)).status_code == 201
    r = client.post("/alerts/1/actions", json={"action": "override_stage", "analyst": "asha", "stage": "Impact"})
    assert r.status_code == 201 and r.json()["status_after"] == "open"
    assert client.post("/alerts/1/actions", json={"action": "approve_response", "analyst": "ravi",
                                                  "response": "block at firewall"}).status_code == 201
    h = client.get("/alerts/1/actions").json()
    assert h["status"] == "response_approved" and h["effective_stage"] == "Impact"
    assert client.post("/alerts/1/actions", json={"action": "dismiss", "analyst": "x"}).status_code == 422
    assert client.post("/alerts/9/actions", json={"action": "acknowledge", "analyst": "x"}).status_code == 404
    assert client.get("/alerts/9/actions").status_code == 404
    assert client.get("/triage/summary").json() == {"response_approved": 1, "open": 1}
    assert client.get("/audit/verify-actions").json()["status"] == "VERIFIED"
    assert client.get("/audit/verify").json()["status"] == "VERIFIED"  # alert chain untouched
