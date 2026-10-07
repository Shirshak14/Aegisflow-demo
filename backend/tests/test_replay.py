"""Replay runs a saved model over the test split.

The synthetic test always runs (a tiny model is trained on generated sequences, see conftest.py). The
real-artifact test additionally needs data/processed and artifacts/models and is skipped without them.
"""
import json

import pytest

from backend.app.audit import Ledger
from backend.app.info import DATA_DIR, MODEL_DIR
from backend.app.replay import ReplayEngine

HAVE_REAL_ARTIFACTS = (DATA_DIR / "sequences.parquet").exists() and (MODEL_DIR / "best.pt").exists()


def _replay_and_check(tmp_path, model_dir, *, stage_label="UNCERTAIN"):
    engine = ReplayEngine(Ledger(tmp_path / "r.db"))
    engine.prepare()
    engine.session_id = "test"
    ev = engine.events
    assert ev.target_window_start.is_monotonic_increasing
    for i in range(len(ev)):  # synchronous: same _process path the thread uses
        engine._process(ev.iloc[i])
    alerts = engine.ledger.list(limit=100000)
    tp = sum(a["truth_attack"] for a in alerts)
    # Expected counts come from the artifacts under test: the offline LSTM test confusion matrix
    # [[TN, FP], [FN, TP]] saved by `train` at the validation-selected threshold. Alerts = FP + TP.
    # (For the committed demo artifacts this is [[10622, 270], [408, 79]] -> 349 alerts, 79 true positives.)
    (_, fp), (_, offline_tp) = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["models"]["lstm"]["confusion_matrix"]
    assert (len(alerts), tp) == (fp + offline_tp, offline_tp)
    assert {a["predicted_stage"] for a in alerts} == {stage_label}
    assert all(a["risk_components"] and 0 <= a["risk_score"] <= 75 for a in alerts)
    assert engine.ledger.verify()["status"] == "VERIFIED"
    return engine, alerts


def test_synthetic_replay_matches_offline_confusion_and_chain_verifies(tmp_path, synthetic_replay):
    _, model_dir = synthetic_replay
    engine, alerts = _replay_and_check(tmp_path, model_dir)
    assert len(engine.events) == 120 and len(alerts) > 0  # the signal is learnable, so something alerts
    assert {a["model_version"] for a in alerts} == {"synthetic-test-model"}
    assert {a["session_id"] for a in alerts} == {"test"}


def test_synthetic_replay_detects_tampering(tmp_path, synthetic_replay):
    import sqlite3
    _, model_dir = synthetic_replay
    engine, alerts = _replay_and_check(tmp_path, model_dir)
    with sqlite3.connect(tmp_path / "r.db") as con:
        con.execute("UPDATE alerts SET risk_score = 0 WHERE id = 1")
    assert engine.ledger.verify()["status"] == "CORRUPTED"


@pytest.mark.skipif(not HAVE_REAL_ARTIFACTS, reason="processed data / Phase 3 artifacts not present")
def test_replay_reproduces_offline_test_confusion_and_chain_verifies(tmp_path):
    _replay_and_check(tmp_path, MODEL_DIR)


def _process_session(engine, session_id, n):
    engine.session_id = session_id
    for i in range(n):
        engine._process(engine.events.iloc[i])


def test_sessions_accumulate_in_one_verifiable_chain_and_reads_are_scoped(tmp_path, synthetic_replay):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    app = create_app(tmp_path / "s.db", warm_up=False)
    engine, ledger = app.state.engine, app.state.ledger
    engine.prepare()
    _process_session(engine, "s1", len(engine.events))
    n1 = ledger.count()
    assert n1 > 0
    _process_session(engine, "s2", len(engine.events))  # a second run on the same ledger: nothing is wiped
    assert ledger.count() == 2 * n1 and ledger.count("s1") == ledger.count("s2") == n1
    assert [s["session_id"] for s in ledger.sessions()] == ["s1", "s2"]
    v = ledger.verify()
    assert v["status"] == "VERIFIED" and v["records_checked"] == 2 * n1

    c = TestClient(app)
    assert {a["session_id"] for a in c.get("/alerts", params={"limit": 1000}).json()} == {"s2"}  # default = current
    assert len(c.get("/alerts", params={"limit": 1000, "session_id": "all"}).json()) == 2 * n1
    assert len(c.get("/alerts", params={"limit": 1000, "session_id": "s1"}).json()) == n1
    h = c.get("/health").json()
    assert (h["alerts_in_ledger"], h["alerts_in_session"], h["session_id"]) == (2 * n1, n1, "s2")
    assert sum(c.get("/triage/summary").json().values()) == n1
    assert sum(c.get("/triage/summary", params={"session_id": "all"}).json().values()) == 2 * n1
    assert len(c.get("/audit/sessions").json()) == 2


def test_starting_a_replay_keeps_earlier_sessions(tmp_path, synthetic_replay):
    ledger = Ledger(tmp_path / "k.db")
    engine = ReplayEngine(ledger)
    ledger.append({"session_id": "old", "sequence_id": "x", "host_id": "h", "lstm_probability": 0.9})
    started = engine.start(speed=max(engine.allowed_speeds), reset=True)
    engine.stop()
    assert started["session_id"] != "old"
    assert ledger.count("old") == 1
    assert ledger.verify()["status"] == "VERIFIED"
