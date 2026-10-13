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


# ---------------------------------------------------------------- integrity: duplicates, late, gaps
def _quiet(flows, drop_between):
    """Remove the flows whose timestamp offset (s) from the start lies in [lo, hi): a feed outage."""
    lo, hi = drop_between
    off = (flows.timestamp - flows.timestamp.min()).dt.total_seconds()
    return flows[(off < lo) | (off >= hi)].reset_index(drop=True)


def _run(st, flows, chunk):
    rows = []
    for i in range(0, len(flows), chunk):
        rows += st.push(flows.iloc[i:i + chunk])
    return rows + st.flush()


def test_duplicates_across_chunks_are_dropped_and_results_match_batch(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture)
    want = batch_vs(_run(StreamScorer(tiny_model, cfg), flows, 1000))
    st = StreamScorer(tiny_model, cfg)
    rows = []
    for i in range(0, len(flows), 10):  # every chunk is pushed twice, and the previous chunk is replayed too
        part = flows.iloc[i:i + 10]
        rows += st.push(part) + st.push(part) + st.push(flows.iloc[max(i - 10, 0):i + 5])
    rows += st.flush()
    got = batch_vs(rows)
    assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-6 for k in want)
    s = st.status()
    assert s["duplicate_flows"] > 0 and s["sequences_scored"] == len(want)


def test_duplicate_after_its_windows_closed_counts_as_late_not_double_counted(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture)
    st = StreamScorer(tiny_model, cfg)
    st.push(flows)
    before = st.status()
    st.push(flows.iloc[:5])  # repeats of flows in windows that closed already
    after = st.status()
    assert after["late_flows"] - before["late_flows"] + after["duplicate_flows"] - before["duplicate_flows"] == 5
    assert after["buffered_flows"] == before["buffered_flows"]


def test_flow_in_an_overlap_of_a_closed_window_is_late_not_half_counted(long_capture, tiny_model):  # noqa: F811
    """With 60 s windows on a 30 s stride a flow sits in two windows. If one of them closed, the flow is
    late: it must not be aggregated into the other one only."""
    cfg = load_config()
    flows = pcap_to_flows(long_capture).sort_values("timestamp").reset_index(drop=True)
    st = StreamScorer(tiny_model, cfg)
    st.push(flows.iloc[:30])
    closed_end = st._closed_until()
    assert st.next_k > 0
    straggler = flows.iloc[[0]].assign(timestamp=closed_end - pd.Timedelta(seconds=1), source_port=7)
    # still inside the next, open window's range, but also inside the last closed window
    assert straggler.timestamp.iloc[0] >= st._start(st.next_k)
    n = st.status()["buffered_flows"]
    st.push(straggler)
    assert st.status()["late_flows"] == 1 and st.status()["buffered_flows"] == n


def test_out_of_order_inside_allowed_lateness_matches_in_order(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture)
    want = batch_vs(_run(StreamScorer(tiny_model, cfg), flows.sort_values("timestamp"), 1000))
    shuffled = flows.sort_values("timestamp").reset_index(drop=True)
    order = np.argsort(np.arange(len(shuffled)) // 4 * 4 + np.random.default_rng(1).random(len(shuffled)) * 4)
    st = StreamScorer(tiny_model, cfg, allowed_lateness=120)  # disorder is only within a few flows (< 120 s)
    got = batch_vs(_run(st, shuffled.iloc[order].reset_index(drop=True), 5))
    assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-6 for k in want)
    assert st.status()["late_flows"] == 0


def test_grid_origin_follows_the_earliest_flow_until_a_window_closes(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = pcap_to_flows(long_capture).sort_values("timestamp").reset_index(drop=True)
    want = batch_vs(_run(StreamScorer(tiny_model, cfg), flows, 1000))
    st = StreamScorer(tiny_model, cfg)
    rows = st.push(flows.iloc[[3]]) + st.push(flows.iloc[[0, 1, 2]])  # earliest flows arrive second
    rows += _run(st, flows.iloc[4:], 7)
    got = batch_vs(rows)
    assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-6 for k in want)
    assert st.status()["late_flows"] == 0


def test_feed_outage_is_reported_and_sequences_are_flagged(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = _quiet(pcap_to_flows(long_capture), (150, 270))  # 2 minutes with no traffic from anyone
    st = StreamScorer(tiny_model, cfg)
    rows = _run(st, flows, 20)
    s = st.status()
    assert s["empty_windows"] > 0 and s["feed_gaps"] and s["feed_gaps"][0]["windows"] >= 2
    gapped = [r for r in rows if not r["contiguous"]]
    assert gapped and all(r["input_missing_windows"] > 0 and r["max_gap_windows"] > 0 for r in gapped)
    assert all(r["input_span_seconds"] >= 330 for r in rows)   # (10 - 1) * 30 + 60: never shorter
    assert any(r["input_span_seconds"] > 330 for r in gapped)  # a gap stretches the real time the input covers
    ok = [r for r in rows if r["contiguous"]]
    assert all(r["input_span_seconds"] == 330 for r in ok)
    assert all(pd.Timestamp(r["target_not_before"]) == pd.Timestamp(r["seq_end_time"]) + pd.Timedelta(seconds=30) for r in rows)


def test_max_missing_windows_skips_gapped_sequences_in_stream_and_batch_alike(long_capture, tiny_model):  # noqa: F811
    cfg = load_config()
    flows = _quiet(pcap_to_flows(long_capture), (25, 100))  # one empty slot early on; later sequences are clean
    default = _run(StreamScorer(tiny_model, cfg), flows, 20)
    st = StreamScorer(tiny_model, cfg, max_missing_windows=0)
    strict = _run(st, flows, 20)
    batch = score_flows(flows, tiny_model, cfg, max_missing_windows=0)
    assert 0 < len(strict) < len(default) and all(r["contiguous"] for r in strict)
    assert st.status()["sequences_skipped_gaps"] == len(default) - len(strict)
    assert batch_vs(strict).keys() == {(h, e) for h, e in zip(batch.host_id, batch.seq_end_time)}
    # the unrestricted batch path reports the same continuity as the stream
    full = score_flows(flows, tiny_model, cfg)
    by_end = {(h, e): m for h, e, m in zip(full.host_id, full.seq_end_time, full.input_missing_windows)}
    assert all(by_end[(r["host_id"], pd.Timestamp(r["seq_end_time"]))] == r["input_missing_windows"] for r in default)


def test_slot_gaps_and_continuity_helpers():
    from aegisflow.ml.temporal.sequences import sequence_continuity, slot_gaps
    t = pd.date_range("2017-07-03", periods=4, freq="30s")
    assert slot_gaps(t, 30).tolist() == [0, 0, 0]
    assert slot_gaps(t[[0, 1, 3]], 30).tolist() == [0, 1]
    assert slot_gaps(t[[0, 4 - 1]], 30).tolist() == [2] and slot_gaps(t[:1], 30).size == 0
    c = sequence_continuity(t[[0, 1, 3]], t[[0, 1, 3]] + pd.Timedelta("60s"), 30, 60)
    assert c["input_missing_windows"] == 1 and c["max_gap_windows"] == 1 and not c["contiguous"]
    assert c["input_span_seconds"] == 150 and c["target_not_before"] == t[3] + pd.Timedelta("90s")
