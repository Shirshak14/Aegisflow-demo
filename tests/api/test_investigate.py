"""Investigation routes: temporal context (/context), model health and drift (/health/model), evidence export (/evidence)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import health, replay, temporal_context
from backend.app.main import create_app


def _sequences(window_starts, host="10.0.0.1", length=10):
    """Sequence rows the way sequences.py builds them: <host>_seq_<k> starts at the host's k-th window."""
    w = [pd.Timestamp("2017-07-07 10:00:00") + pd.Timedelta(seconds=s) for s in window_starts]
    rows = []
    for k in range(len(w) - length):
        end = w[k + length - 1] + pd.Timedelta(seconds=60)
        target = next((x for x in w[k + length:] if x > end), None)
        if target is None:
            break
        rows.append({"sequence_id": f"{host}_seq_{k}", "host_id": host, "seq_start_time": w[k], "seq_end_time": end,
                     "target_window_start": target, "target_window_end": target + pd.Timedelta(seconds=60), "split": "test"})
    return pd.DataFrame(rows)


def test_temporal_context_reports_spans_silent_windows_gaps_and_lead(tmp_path, monkeypatch):
    # 14 contiguous windows (30 s stride), then about an hour of silence, then 12 more windows
    starts = [30 * i for i in range(14)] + [4000 + 30 * i for i in range(12)]
    frame = _sequences(starts)
    frame.to_parquet(tmp_path / "sequences.parquet", index=False)
    monkeypatch.setattr(replay, "DATA_DIR", tmp_path)

    first = temporal_context.context("10.0.0.1_seq_0")
    assert first["input_span_seconds"] == 9 * 30 + 60 and first["silent_windows"] == 0 and first["contiguous"]
    assert first["previous_gap_seconds"] == 30 and first["largest_gap_seconds"] == 30
    assert first["lead_seconds"] == 30 == first["min_lead_seconds"]  # target = first window starting after the input ends

    # seq_4's input ends just before the silence, so its target is about an hour away
    lead = temporal_context.context("10.0.0.1_seq_4")
    assert lead["lead_seconds"] == 4000 - 450 and lead["silent_windows"] == 0
    assert lead["lead_rank_in_test"] > 0.5

    # seq_5's last input window is the first one after the silence
    gap = temporal_context.context("10.0.0.1_seq_5")
    assert gap["previous_gap_seconds"] == 4000 - 390 == gap["largest_gap_seconds"]
    assert gap["expected_windows"] == 129 and gap["silent_windows"] == 119 and not gap["contiguous"]
    assert temporal_context.context("unknown") is None

    s = TestClient(create_app(tmp_path / "d.db", warm_up=False)).get("/context/summary").json()
    assert s["sequences"] == len(frame) and s["contiguous_span_seconds"] == 330
    assert 0 < s["with_silent_windows_share"] < 1 and s["lead_seconds"]["max"] == 4000 - 390


def _replayed_app(tmp_path):
    app = create_app(tmp_path / "r.db", warm_up=False)
    engine = app.state.engine
    engine.prepare()
    engine.session_id = "inv"
    for i in range(len(engine.events)):
        engine._process(engine.events.iloc[i])
    engine.cursor = len(engine.events)
    return app


def test_model_health_needs_enough_sequences_then_reports_drift(tmp_path, synthetic_replay, monkeypatch):
    app = create_app(tmp_path / "h.db", warm_up=False)
    client = TestClient(app)
    app.state.engine.prepare()
    assert client.get("/health/model").json()["state"] == "too_few_sequences"

    monkeypatch.setattr(health, "MIN_SEQUENCES", 50)
    app = _replayed_app(tmp_path)
    d = TestClient(app).get("/health/model").json()
    assert d["state"] == "ok" and d["replayed"] == len(app.state.engine.events)
    assert len(d["features"]) == 28 and all(f["psi"] >= 0 for f in d["features"])
    assert [f["psi"] for f in d["features"]] == sorted((f["psi"] for f in d["features"]), reverse=True)
    assert sum(d["scores"]["replay"]) == d["replayed"] and d["non_finite_inputs"] == 0
    assert sum(r["sequences"] for r in d["volume"]["replay"]) == d["replayed"]
    assert "not measured" in d["processing_lag"]


def test_evidence_bundle_leaves_out_ground_truth_unless_asked_and_changes_nothing(tmp_path, synthetic_replay):
    app = _replayed_app(tmp_path)
    client, ledger = TestClient(app), app.state.ledger
    ids = [a["id"] for a in ledger.list(limit=3)]
    assert ids, "the synthetic model should alert"
    app.state.analyst.record(ids[0], "acknowledge", "ana")
    before = (ledger.count(), ledger.verify()["head_hash"])

    r = client.post("/evidence", json={"alert_ids": ids + [999999]})
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    b = r.json()
    assert b["ground_truth_included"] is False and b["scope"]["missing_alert_ids"] == [999999]
    assert sorted(b["scope"]["alert_ids"]) == sorted(ids)
    for a in b["alerts"]:
        assert not {"truth_attack", "truth_class", "truth_stage"} & set(a["record"])
        assert a["record"]["hash"] and a["analyst"]["alert_id"] == a["record"]["id"]
    assert b["alerts"][[a["record"]["id"] for a in b["alerts"]].index(ids[0])]["analyst"]["status"] == "acknowledged"
    assert b["verification"]["alerts_chain"]["status"] == "VERIFIED"
    assert b["verification"]["analyst_actions_chain"]["status"] == "VERIFIED"
    assert len(b["config_fingerprint"]["sha256"]) == 64 and b["config_fingerprint"]["files"]["configs/config.yaml"]
    assert b["model"]["versions"] == ["synthetic-test-model"]
    assert b["timelines"] and all(p["t"] for t in b["timelines"] for p in t["points"])

    labelled = client.post("/evidence", json={"alert_ids": ids[:1], "include_ground_truth": True}).json()
    assert labelled["ground_truth_included"] and "truth_attack" in labelled["alerts"][0]["record"]

    assert client.post("/evidence", json={"alert_ids": [999999]}).status_code == 404
    assert client.post("/evidence", json={"alert_ids": []}).status_code == 422
    assert (ledger.count(), ledger.verify()["head_hash"]) == before


@pytest.mark.skipif(not (replay.DATA_DIR / "sequences.parquet").exists(), reason="processed data not present")
def test_real_test_split_has_variable_horizons():
    s = temporal_context.split_summary()
    assert s["sequences"] > 0 and s["lead_seconds"]["max"] >= s["lead_seconds"]["median"] >= 30
