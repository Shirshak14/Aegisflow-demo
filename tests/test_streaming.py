"""Streaming scorer: same results as the batch path, bounded memory, late data, API."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scapy")

from aegisflow.config import load_config
from aegisflow.ml.ingestion.pcap import pcap_to_flows
from aegisflow.ml.scoring import score_flows
from aegisflow.ml.streaming import StreamScorer
from test_pcap import long_capture, tiny_model  # noqa: F401  (fixtures)


def batch_vs(stream_rows):
    return {(r["host_id"], pd.Timestamp(r["seq_end_time"])): r["attack_probability"] for r in stream_rows}


@pytest.mark.parametrize("chunk", [1, 7, 45, 1000])
def test_stream_matches_batch(long_capture, tiny_model, chunk):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture)
    batch = score_flows(flows, tiny_model, cfg)
    st = StreamScorer(tiny_model, cfg)
    rows = []
    for i in range(0, len(flows), chunk):
        rows += st.push(flows.iloc[i:i + chunk])
    rows += st.flush()
    got = batch_vs(rows)
    want = {(h, pd.Timestamp(e)): p for h, e, p in zip(batch.host_id, batch.seq_end_time, batch.attack_probability)}
    assert len(want) > 0 and got.keys() == want.keys()
    assert all(abs(got[k] - want[k]) < 1e-6 for k in want)
    s = st.status()
    assert s["sequences_scored"] == len(batch) and s["late_flows"] == 0 and s["alerts"] == int(batch.predicted_attack.sum())


def test_scores_arrive_before_the_stream_ends_and_memory_is_bounded(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture)
    st = StreamScorer(tiny_model, cfg)
    early, max_buffer = 0, 0
    for i in range(len(flows)):
        early += len(st.push(flows.iloc[i:i + 1]))
        max_buffer = max(max_buffer, st.status()["buffered_flows"])
    assert early > 0  # windows were scored while data was still arriving
    assert max_buffer < len(flows) / 2
    assert all(len(h) <= st.L for h in st.history.values())


def test_late_flows_are_counted_and_lateness_allows_them(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture)
    late = flows.iloc[[5]].copy()
    st = StreamScorer(tiny_model, cfg)
    st.push(flows.iloc[:30]); st.push(late)
    assert st.status()["late_flows"] == 1
    st2 = StreamScorer(tiny_model, cfg, allowed_lateness=10_000)
    st2.push(flows.iloc[:30]); st2.push(flows.iloc[[29]].assign(source_port=1))
    assert st2.status()["late_flows"] == 0 and st2.status()["windows_closed"] == 0
    st2.reset()
    assert st2.status()["flows_seen"] == 0


def test_stream_api(long_capture, tiny_model, tmp_path, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient
    import backend.app.main as main
    monkeypatch.setattr(main, "ROOT", tmp_path)
    cfg = load_config(overrides=[f"stream.model_dir={tiny_model}"])
    monkeypatch.setattr(main, "load_config", lambda *a, **k: cfg)
    client = TestClient(main.create_app(tmp_path / "l.db", warm_up=False))
    flows = pcap_to_flows(long_capture)
    recs = flows.drop(columns=["dataset_label"]).astype(object).where(flows.drop(columns=["dataset_label"]).notna(), None)
    recs["timestamp"] = flows.timestamp.astype(str)
    scored = 0
    for i in range(0, len(recs), 10):
        r = client.post("/stream/flows", json={"flows": recs.iloc[i:i + 10].to_dict("records")})
        assert r.status_code == 200, r.text
        scored += len(r.json()["scored"])
    scored += len(client.post("/stream/flush").json()["scored"])
    assert scored == len(score_flows(flows, tiny_model, cfg)) and scored > 0
    assert client.get("/stream/status").json()["sequences_scored"] == scored
    assert len(client.get("/stream/results", params={"limit": 3}).json()) == 3
    assert client.post("/stream/flows", json={"flows": [{"timestamp": "2017-07-03"}]}).status_code == 422
    assert client.post("/stream/reset").json()["flows_seen"] == 0
    assert client.get("/health").json()["alerts_in_ledger"] == 0  # streaming never writes to the ledger


def test_cli_stream(long_capture, tiny_model, capsys):  # noqa: F811
    from aegisflow.cli import main
    assert main(["stream", "--input", str(long_capture), "--model-dir", str(tiny_model), "--chunk-seconds", "15"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert '"status"' in out[-1] and len(out) > 1
