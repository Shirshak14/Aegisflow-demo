"""Replay runs the real saved model over the real test split (needs data/ and artifacts/)."""
import json

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
    # Expected counts come from the artifacts under test: the offline LSTM test confusion matrix
    # [[TN, FP], [FN, TP]] saved by `train` at the validation-selected threshold. Alerts = FP + TP.
    # (For the committed demo artifacts this is [[10622, 270], [408, 79]] -> 349 alerts, 79 true positives.)
    (_, fp), (_, offline_tp) = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))["models"]["lstm"]["confusion_matrix"]
    assert (len(alerts), tp) == (fp + offline_tp, offline_tp)
    assert {a["predicted_stage"] for a in alerts} == {"UNCERTAIN"}
    assert all(a["risk_components"] and 0 <= a["risk_score"] <= 75 for a in alerts)
    assert engine.ledger.verify()["status"] == "VERIFIED"
