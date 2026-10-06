"""PCAP ingestion (scapy + pyshark readers), labelling, and scoring captured traffic."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

scapy_all = pytest.importorskip("scapy.all")
from scapy.all import ICMP, IP, TCP, UDP, Ether, IPv6, Raw, wrpcap  # noqa: E402

from aegisflow.config import load_config
from aegisflow.ml.ingestion.pcap import (UNLABELED, Packet, apply_label_file, flows_from_packets,
                                         pcap_to_flows)

T0 = 1499077800.0  # 2017-07-03 10:30:00 UTC
A, B, C = "192.168.10.5", "192.168.10.50", "8.8.8.8"


def at(pkt, t):
    pkt.time = t
    return pkt


def tcp_session(t, sport=40000, ttl_fwd=(64, 64, 62, 64, 64), retransmit=True):
    """Handshake, 100 B request (retransmitted once), 200 B response, FIN/ACK both ways."""
    f = lambda flags, seq, ttl, payload=b"": Ether() / IP(src=A, dst=B, ttl=ttl) / TCP(
        sport=sport, dport=80, flags=flags, seq=seq, window=8192) / (Raw(payload) if payload else b"")
    r = lambda flags, seq, payload=b"": Ether() / IP(src=B, dst=A, ttl=128) / TCP(
        sport=80, dport=sport, flags=flags, seq=seq, window=29200) / (Raw(payload) if payload else b"")
    pk = [at(f("S", 1000, ttl_fwd[0]), t), at(r("SA", 5000), t + 0.001), at(f("A", 1001, ttl_fwd[1]), t + 0.002),
          at(f("PA", 1001, ttl_fwd[2], b"x" * 100), t + 0.003)]
    if retransmit:
        pk.append(at(f("PA", 1001, ttl_fwd[3], b"x" * 100), t + 0.203))
    pk += [at(r("PA", 5001, b"y" * 200), t + 0.25), at(f("FA", 1101, ttl_fwd[4]), t + 0.3),
           at(r("FA", 5201), t + 0.301)]
    return pk


@pytest.fixture()
def capture(tmp_path):
    pk = tcp_session(T0)
    pk += tcp_session(T0 + 10, retransmit=False)                       # same 5-tuple, new connection after FIN
    pk += [at(Ether() / IP(src=A, dst=C, ttl=64) / UDP(sport=5353, dport=53) / Raw(b"q" * 30), T0 + s)
           for s in (1, 2, 400)]                                       # idle gap 398 s > 120 s -> 2 flows
    pk += [at(Ether() / IP(src=C, dst=A, ttl=50) / ICMP(), T0 + 3)]
    pk += [at(Ether() / IPv6(src="fe80::1", dst="fe80::2", hlim=255) / TCP(sport=1, dport=22, flags="S"), T0 + 4)]
    path = tmp_path / "cap.pcap"
    wrpcap(str(path), pk)
    return path


def test_tcp_flow_features_match_cicflowmeter_conventions(capture):
    f = pcap_to_flows(capture)
    tcp = f[(f.protocol == "TCP") & (f.destination_port == 80)].reset_index(drop=True)
    assert len(tcp) == 2  # FIN in both directions closed the first connection
    s = tcp.iloc[0]
    assert (s.source_ip, s.destination_ip, s.source_port, s.destination_port) == (A, B, 40000, 80)
    assert (s.fwd_packet_count, s.bwd_packet_count) == (5, 3)
    assert (s.fwd_byte_count, s.bwd_byte_count, s.byte_count) == (200, 200, 400)  # payload bytes
    assert (s.syn_count, s.fin_count, s.psh_count, s.rst_count) == (2, 2, 3, 0)
    assert s.tcp_window_size == 8192 and s.retransmission_count == 1
    assert s.flow_duration == pytest.approx(0.301, abs=1e-6)
    assert s.inter_arrival_mean == pytest.approx(0.301 / 7 * 1e6, rel=1e-6)  # microseconds
    ttls = [64, 128, 64, 62, 64, 128, 64, 128]
    assert s.ttl_mean == pytest.approx(np.mean(ttls)) and s.ttl_std == pytest.approx(np.std(ttls, ddof=1))
    assert s.timestamp == pd.Timestamp("2017-07-03 10:30:00")
    assert tcp.iloc[1].retransmission_count == 0
    assert set(f.dataset_label) == {UNLABELED}


def test_udp_idle_timeout_icmp_and_ipv6(capture):
    f = pcap_to_flows(capture)
    udp = f[f.protocol == "UDP"]
    assert sorted(udp.packet_count.tolist()) == [1, 2]
    assert pcap_to_flows(capture, idle_timeout=1000)[lambda d: d.protocol == "UDP"].packet_count.tolist() == [3]
    icmp = f[f.protocol == "ICMP"].iloc[0]
    assert pd.isna(icmp.source_port) and pd.isna(icmp.retransmission_count) and icmp.ttl_mean == 50
    v6 = f[f.source_ip == "fe80::1"].iloc[0]
    assert v6.ttl_mean == 255 and v6.syn_count == 1


def test_pcapng_and_pyshark_reader_agree(capture, tmp_path):
    from scapy.utils import rdpcap, wrpcapng
    ng = tmp_path / "cap.pcapng"
    wrpcapng(str(ng), rdpcap(str(capture)))
    base = pcap_to_flows(capture)
    pd.testing.assert_frame_equal(base, pcap_to_flows(ng))
    if shutil.which("tshark") is None:
        pytest.skip("tshark not installed")
    pytest.importorskip("pyshark")
    pd.testing.assert_frame_equal(base, pcap_to_flows(capture, reader="pyshark"))


def test_rst_ends_flow_and_bad_inputs(tmp_path):
    p = lambda t, flags: Packet(T0 + t, A, B, 1, 2, 6, 0, 64, flags, 100, 0)
    f = flows_from_packets([p(0, "S"), p(1, "R"), p(2, "S")])
    assert f.packet_count.tolist() == [2, 1]
    with pytest.raises(ValueError):
        flows_from_packets([p(0, "S")], idle_timeout=0)
    with pytest.raises(Exception, match="no IPv4"):
        flows_from_packets([])
    with pytest.raises(Exception, match="not found"):
        pcap_to_flows(tmp_path / "missing.pcap")


def test_label_file(capture, tmp_path):
    cfg = load_config()
    labels = tmp_path / "labels.csv"
    labels.write_text(f"source_ip,start,end,label\n{A},2017-07-03 10:30:05,2017-07-03 10:31:00,PortScan\n")
    f = apply_label_file(pcap_to_flows(capture), labels, dict(cfg.stages.cic_ids2017))
    hit = f[f.dataset_label == "PortScan"]
    assert len(hit) == 1 and hit.iloc[0].attack_stage == "Reconnaissance"
    assert (f.dataset_label == "BENIGN").sum() == len(f) - 1
    bad = tmp_path / "bad.csv"
    bad.write_text(f"source_ip,start,end,label\n{A},2017-07-03,2017-07-04,Teleport\n")
    with pytest.raises(ValueError, match="not in the stage mapping"):
        apply_label_file(pcap_to_flows(capture), bad, dict(cfg.stages.cic_ids2017))


# --------------------------------------------------------------------------------------------- scoring
@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from aegisflow.ml.modeling import ForecastDataset, train_experiment
    from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES
    rng = np.random.default_rng(0)
    n, L = 120, 10
    X = rng.gamma(2.0, 5.0, size=(n, L, len(STANDARD_NUMERIC_WINDOW_FEATURES)))
    y = rng.random(n) < 0.3
    frame = pd.DataFrame({"sequence_id": [f"s{i}" for i in range(n)], "sequence_features": [x.tolist() for x in X],
                          "target_features": X[:, -1].tolist(), "target_attack_present": y.astype(int),
                          "target_dominant_class": "Benign", "target_dominant_stage": "Benign",
                          "seq_end_time": pd.Timestamp("2017-07-03"), "target_window_start": pd.Timestamp("2017-07-03 00:01"),
                          "split": ["train"] * 80 + ["val"] * 20 + ["test"] * 20})
    out = tmp_path_factory.mktemp("model")
    train_experiment(ForecastDataset.from_frame(frame), out, epochs=1, batch_size=32, hidden_size=4)
    return out


@pytest.fixture()
def long_capture(tmp_path):
    """Two hosts sending a short TCP session every 20 s for 8 minutes -> enough 60 s/30 s windows."""
    pk = []
    for i in range(24):
        pk += tcp_session(T0 + 20 * i, sport=41000 + i, retransmit=i % 3 == 0)
        pk += [at(Ether() / IP(src=C, dst=A, ttl=50) / UDP(sport=53, dport=6000 + i) / Raw(b"r" * 60), T0 + 20 * i + 5)]
    path = tmp_path / "long.pcap"
    wrpcap(str(path), pk)
    return path


def test_score_flows_end_to_end(long_capture, tiny_model):
    from aegisflow.ml.scoring import score_flows
    cfg = load_config()
    scored = score_flows(pcap_to_flows(long_capture), tiny_model, cfg)
    assert set(scored.host_id) <= {A, C, B} and A in set(scored.host_id)
    assert scored.attack_probability.between(0, 1).all()
    assert (scored.seq_end_time > scored.seq_start_time).all()
    a = scored[scored.host_id == A]
    assert a.ttl_mean.notna().all() and (a.retransmissions >= 0).all() and a.retransmissions.sum() > 0
    meta = json.loads((tiny_model / "metadata.json").read_text())
    assert (scored.threshold == meta["threshold"]).all()


def test_build_inference_sequences_counts():
    from aegisflow.ml.modeling import MODEL_FEATURES
    from aegisflow.ml.scoring import build_inference_sequences
    t = pd.date_range("2017-07-03", periods=12, freq="30s")
    w = pd.DataFrame({"host_id": ["h"] * 12, "window_start": t, "window_end": t + pd.Timedelta("60s"),
                      **{f: np.arange(12.0) for f in MODEL_FEATURES}})
    X, meta = build_inference_sequences(w, 10)
    assert X.shape == (3, 10, len(MODEL_FEATURES)) and X[-1, -1, 0] == 11
    assert meta.seq_end_time.iloc[-1] == t[-1] + pd.Timedelta("60s")
    assert build_inference_sequences(w.iloc[:5], 10)[0].shape[0] == 0


def test_cli_ingest_and_score(long_capture, tiny_model, tmp_path, capsys):
    from aegisflow.cli import main
    out = tmp_path / "flows.parquet"
    assert main(["ingest-pcap", "--input", str(long_capture), "--output", str(out)]) == 0
    f = pd.read_parquet(out)
    assert len(f) == 48 and f.retransmission_count.sum() == 8
    assert main(["score-pcap", "--input", str(long_capture), "--model-dir", str(tiny_model),
                 "--output", str(tmp_path / "scored.csv")]) == 0
    assert "scored host sequences" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "scored.csv")) > 0
