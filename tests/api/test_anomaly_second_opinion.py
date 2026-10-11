"""The opt-in anomaly model (replay.anomaly_model_dir) is a second opinion: it labels alerts, never changes them."""
import json

import numpy as np
import pytest

from backend.app.audit import Ledger
from backend.app.replay import ReplayEngine

from .conftest import build_synthetic_sequences


@pytest.fixture(scope="module")
def anomaly_dir(tmp_path_factory):
    pytest.importorskip("sklearn")
    from aegisflow.ml.anomaly import train_anomaly
    from aegisflow.ml.modeling import ForecastDataset

    out = tmp_path_factory.mktemp("anomaly")
    train_anomaly(ForecastDataset.from_frame(build_synthetic_sequences()), out, seed=0, n_estimators=50)
    return out


def test_fit_set_is_benign_train_only_and_threshold_is_the_fit_quantile(anomaly_dir):
    from aegisflow.ml.anomaly import AnomalyScorer, fit_indices
    from aegisflow.ml.modeling import ForecastDataset

    data = ForecastDataset.from_frame(build_synthetic_sequences())
    ix = fit_indices(data)
    assert len(ix) > 0 and (data.attack[ix] == 0).all() and set(data.frame.split.iloc[ix]) == {"train"}
    scorer = AnomalyScorer(anomaly_dir)
    meta = json.loads((anomaly_dir / "anomaly.json").read_text(encoding="utf-8"))
    assert meta["labels_used_to_fit"] is False and meta["fit_sequences"] == len(ix)
    assert scorer.flag(scorer.score(data.X[ix])).mean() == pytest.approx(0.01, abs=0.01)
    # the synthetic attacks carry 40x traffic, so a benign-only model ranks them as more unusual
    assert scorer.score(data.X[data.attack == 1]).mean() > scorer.score(data.X[ix]).mean()


def _run(engine):
    engine.prepare()
    engine.session_id = "t"
    for i in range(len(engine.events)):
        engine._process(engine.events.iloc[i])
    return engine.ledger.list(limit=100000)


def test_replay_alerts_are_unchanged_and_get_the_anomaly_verdict(tmp_path, synthetic_replay, anomaly_dir):
    plain = _run(ReplayEngine(Ledger(tmp_path / "a.db")))
    engine = ReplayEngine(Ledger(tmp_path / "b.db"))
    engine.anomaly_model_dir = anomaly_dir
    with_anom = _run(engine)
    key = lambda a: (a["sequence_id"], a["lstm_probability"], a["risk_score"])
    assert [key(a) for a in with_anom] == [key(a) for a in plain] and len(plain) > 0
    assert all("anomaly_flag" not in a for a in with_anom)  # the ledger record itself is untouched
    assert engine.ledger.verify()["status"] == "VERIFIED"

    enriched = engine.with_anomaly(with_anom)
    assert all(isinstance(a["anomaly_flag"], bool) and a["anomaly_threshold"] == engine.anomaly_threshold for a in enriched)
    both = [a for a in enriched if a["anomaly_flag"]]
    st = engine.status()["anomaly"]
    assert st["alerts_both_models"] == len(both) > 0
    assert st["alerts_both_models_matching_attack_label"] == sum(a["truth_attack"] for a in both)
    assert engine.with_anomaly([{"sequence_id": "not-replayed"}]) == [{"sequence_id": "not-replayed"}]


def test_api_attaches_the_verdict_only_when_configured(tmp_path, synthetic_replay, anomaly_dir):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    app = create_app(tmp_path / "c.db", warm_up=False)
    client = TestClient(app)
    assert client.get("/features").json()["anomaly_model"]["configured"] is False
    _run(app.state.engine)
    assert "anomaly_flag" not in client.get("/alerts").json()[0]
    assert client.get("/replay/status").json()["anomaly"] is None

    app = create_app(tmp_path / "d.db", warm_up=False)
    app.state.engine.anomaly_model_dir = anomaly_dir
    client = TestClient(app)
    _run(app.state.engine)
    assert client.get("/features").json()["anomaly_model"]["available"] is True
    alerts = client.get("/alerts?limit=1000").json()
    assert all("anomaly_flag" in a for a in alerts)
    one = client.get(f"/alerts/{alerts[0]['id']}").json()
    assert one["anomaly_flag"] == alerts[0]["anomaly_flag"]
    assert client.get("/replay/status").json()["anomaly"]["alerts_both_models"] == sum(a["anomaly_flag"] for a in alerts)
    assert client.get("/audit/verify").json()["status"] == "VERIFIED"
    assert np.isfinite([a["anomaly_score"] for a in alerts]).all()
