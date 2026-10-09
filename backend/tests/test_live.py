"""B2: the dashboard's server-sent event stream (GET /live)."""
import asyncio
import json

from fastapi.testclient import TestClient

from backend.app.live import snapshot_events
from backend.app.main import create_app
from backend.tests.conftest import AUTH


def _collect(snapshots, *, polls, heartbeat=1e9):
    """Run the generator for `polls` loop iterations over a scripted list of snapshots."""
    it = iter(snapshots)
    calls = {"n": 0}

    async def disconnected():
        calls["n"] += 1
        return calls["n"] > polls

    def snap():
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v

    async def run():
        return [f async for f in snapshot_events(snap, disconnected, interval=0, heartbeat=heartbeat)]
    return asyncio.run(run())


def test_sends_only_changes_then_heartbeats():
    frames = _collect([{"a": 1}, {"a": 1}, {"a": 2}, {"a": 2}], polls=4)
    assert frames[0].startswith("retry:")
    events = [f for f in frames if f.startswith("event:")]
    assert [json.loads(e.split("data: ", 1)[1]) for e in events] == [{"a": 1}, {"a": 2}]
    hb = _collect([{"a": 1}] * 3, polls=3, heartbeat=0)
    assert hb.count(": keep-alive\n\n") == 2


def test_snapshot_errors_become_error_events_and_stream_continues():
    frames = _collect([RuntimeError("model warm-up"), {"a": 1}], polls=2)
    assert frames[1].startswith("event: error") and "model warm-up" in frames[1]
    assert frames[2].startswith("event: snapshot")


def test_live_endpoint_matches_rest_bodies(tmp_path):
    client = TestClient(create_app(tmp_path / "d.db", warm_up=False))
    client.post("/alerts", json={"sequence_id": "s1", "host_id": "10.0.0.1", "lstm_probability": 0.9,
                                 "predicted_stage": "UNCERTAIN", "risk_score": 50.0}, headers=AUTH)
    r = client.get("/live?max_events=1")
    assert r.headers["content-type"].startswith("text/event-stream")
    frame = next(f for f in r.text.split("\n\n") if f.startswith("event: snapshot"))
    snap = json.loads(frame.split("data: ", 1)[1])
    assert set(snap) == {"st", "hosts", "alerts", "statuses", "triage"}
    assert snap["st"]["state"] == client.get("/replay/status").json()["state"]
    assert snap["alerts"] == client.get("/alerts?limit=50").json()
    assert snap["triage"] == client.get("/triage/summary").json()
