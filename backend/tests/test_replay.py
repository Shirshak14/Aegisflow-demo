"""Replay runs the real saved model over the real test split (needs data/ and artifacts/)."""
import pytest

from backend.app.audit import Ledger
from backend.app.info import DATA_DIR, MODEL_DIR
from backend.app.replay import ReplayEngine

pytestmark = pytest.mark.skipif(not (DATA_DIR / "sequences.parquet").exists() or not (MODEL_DIR / "best.pt").exists(),
                                reason="processed data / Phase 3 artifacts not present")


def test_replay_reproduces_offline_test_confusion_and_chain_verifies(tmp_path):
    engine = ReplayEngine(Ledger(tmp_path / "r.db"))
    engine.prepare()
    engine.session_id = "test"
    ev = engine.events
    assert ev.target_window_start.is_monotonic_increasing
    for i in range(len(ev)):  # synchronous: same _process path the thread uses
        engine._process(ev.iloc[i])
    alerts = engine.ledger.list(limit=100000)
    tp = sum(a["truth_attack"] for a in alerts)
    # Phase 3 LSTM test confusion matrix at the validation-selected threshold: [[10622, 270], [408, 79]]
    assert (len(alerts), tp) == (349, 79)
    assert {a["predicted_stage"] for a in alerts} == {"UNCERTAIN"}
    assert all(a["risk_components"] and 0 <= a["risk_score"] <= 75 for a in alerts)
    assert engine.ledger.verify()["status"] == "VERIFIED"
