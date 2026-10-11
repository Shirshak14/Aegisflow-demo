"""Risk settings come from config.yaml: validated at start-up and served to the dashboard."""
import pytest
from fastapi.testclient import TestClient

from aegisflow.config import load_config
from backend.app.audit import Ledger
from backend.app.main import create_app
from backend.app.replay import RISK_COMPONENTS, ReplayEngine, validate_risk_weights

GOOD = {"attack_probability": 0.4, "stage_severity": 0.25, "prediction_confidence": 0.15,
        "abnormality_score": 0.1, "recent_attack_history": 0.1}


def test_shipped_weights_are_valid():
    assert validate_risk_weights(load_config().config.risk_scoring.weights) == GOOD


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_risk_weights({**GOOD, "attack_probability": 0.5})


def test_weights_must_have_exactly_the_known_components():
    with pytest.raises(ValueError, match="exactly"):
        validate_risk_weights({k: v for k, v in GOOD.items() if k != "abnormality_score"})
    with pytest.raises(ValueError, match="exactly"):
        validate_risk_weights({**GOOD, "made_up": 0.0})


def test_weights_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        validate_risk_weights({**GOOD, "attack_probability": 0.6, "stage_severity": -0.05})


def test_components_constant_matches_config_keys():
    assert set(RISK_COMPONENTS) == set(load_config().config.risk_scoring.weights)


def _row(p=0.9):
    class R:
        lstm_p, abnormality = p, 0.0
    return R()


def test_stage_term_is_zero_for_the_uncertain_label_and_risk_is_capped_at_75(tmp_path):
    eng = ReplayEngine(Ledger(tmp_path / "w.db"))
    eng.lstm_threshold = 0.3
    risk, comp = eng._risk(_row(1.0), recent_alerts=99)
    assert comp["stage_severity"] == 0.0
    assert risk <= 75.0 and eng.risk_config()["max_reachable_risk"] == 75.0


def test_stage_severity_table_is_read_for_the_predicted_stage(tmp_path):
    eng = ReplayEngine(Ledger(tmp_path / "w.db"))
    eng.lstm_threshold = 0.3
    eng.stage_severity[eng.predicted_stage] = 50.0  # pretend a stage model predicted a 50-severity stage
    _, comp = eng._risk(_row(0.5), recent_alerts=0)
    assert comp["stage_severity"] == 0.5
    assert eng.risk_config()["max_reachable_risk"] == pytest.approx(87.5)


def test_risk_config_endpoint_matches_config_yaml(tmp_path):
    cfg = load_config().config
    body = TestClient(create_app(tmp_path / "a.db", warm_up=False)).get("/risk/config").json()
    assert body["weights"] == GOOD
    assert body["thresholds"] == {"low": 33.0, "medium": 66.0}
    assert body["max_reachable_risk"] == 75.0
    assert body["confidence_threshold"] == cfg.confidence.stage_prediction_threshold
    assert body["uncertain_label"] == cfg.confidence.uncertain_label
    assert body["default_speed"] == cfg.replay.default_speed in body["allowed_speeds"]
    assert body["allowed_speeds"] == list(cfg.replay.allowed_speeds)
