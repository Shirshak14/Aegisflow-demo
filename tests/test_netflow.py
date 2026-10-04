"""NetFlow v5 / v9 / IPFIX and nfdump CSV decoding, checked against real softflowd/nfdump output."""
import struct
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("scapy")

from aegisflow.ml.ingestion.netflow import (FlowRecord, NetflowDecoder, netflow_to_flows, read_nfdump_csv,
                                            records_to_flows)

FX = Path(__file__).parent / "fixtures" / "netflow"
SOURCES = ["softflowd_v5.pcap", "softflowd_v9.pcap", "softflowd_v10.pcap", "nfdump_v9.csv"]


def key(df):
    return Counter(zip(df.source_ip, df.destination_ip, df.source_port.astype("Int64").astype(str),
                       df.destination_port.astype("Int64").astype(str), df.protocol, df.packet_count, df.byte_count))


def ground_truth():
    """Packets and L3 bytes per unidirectional 5-tuple, counted directly from the source capture."""
    from scapy.all import IP, TCP, UDP, rdpcap
    c = Counter()
    for p in rdpcap(str(FX / "source_traffic.pcap")):
        l4 = p[TCP] if TCP in p else p[UDP]
        c[(p[IP].src, p[IP].dst, str(l4.sport), str(l4.dport), "TCP" if TCP in p else "UDP")] += 1
    return c


@pytest.mark.parametrize("name", SOURCES)
def test_every_format_matches_the_source_traffic(name):
    f = netflow_to_flows(FX / name)
    truth = ground_truth()
    assert len(f) == len(truth) == 72
    got = {(s, d, str(sp), str(dp), pr): n for (s, d, sp, dp, pr, n, _b) in key(f)}
    assert got == {k: float(v) for k, v in truth.items()}
    # bytes are L3 (IP header included): first session request direction = 40+40+140+140+40
    first = f[(f.source_port == 41000)].iloc[0]
    assert first.byte_count == 400 and first.packet_count == 5
    assert (f.bwd_packet_count == 0).all()  # NetFlow is unidirectional
    tcp = f[f.protocol == "TCP"]
    assert (tcp.syn_count == 1).all() and (tcp.fin_count == 1).all() and (tcp.rst_count == 0).all()
    assert f[f.protocol == "UDP"].syn_count.isna().all()
    assert f.ttl_mean.isna().all() and f.retransmission_count.isna().all()  # not in NetFlow: never invented


def test_all_formats_agree_and_times_are_consistent():
    # (Absolute times differ by about a second between fixtures: each came from a separate softflowd run.)
    frames = [netflow_to_flows(FX / n) for n in SOURCES]
    assert all(key(fr) == key(frames[0]) for fr in frames)
    for fr in frames:
        tcp = fr[fr.protocol == "TCP"]
        assert tcp.flow_duration.max() == pytest.approx(0.301, abs=2e-3)
        # 24 sessions 20 s apart -> about 460 s between the first and last flow start
        assert (fr.timestamp.max() - fr.timestamp.min()).total_seconds() == pytest.approx(465, abs=2)


def test_v9_data_before_template_is_counted_not_guessed():
    dec = NetflowDecoder()
    hdr = struct.pack("!HHIIII", 9, 1, 1000, 1700000000, 1, 0)
    data_set = struct.pack("!HH", 300, 8) + b"\0" * 4
    assert dec.decode(hdr + data_set, "x") == [] and dec.skipped_no_template == 1
    tmpl = struct.pack("!HHHH", 0, 16, 300, 2) + struct.pack("!HHHH", 8, 4, 2, 4)
    recs = dec.decode(hdr + tmpl + struct.pack("!HH", 300, 12) + bytes([10, 0, 0, 9]) + struct.pack("!I", 7), "x")
    assert len(recs) == 1 and recs[0].src == "10.0.0.9" and recs[0].packets == 7
    assert dec.decode(b"\x00\x07garbage", "x") == []


def test_bad_inputs(tmp_path):
    with pytest.raises(Exception, match="not found"):
        netflow_to_flows(tmp_path / "nope.pcap")
    bad = tmp_path / "x.csv"; bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not nfdump"):
        list(read_nfdump_csv(bad))
    with pytest.raises(Exception, match="no NetFlow"):
        records_to_flows([])
    one = records_to_flows([FlowRecord(1.0, 2.0, "1.1.1.1", "2.2.2.2", None, None, 1, 3, 300, None, 5)])
    assert one.protocol.iloc[0] == "ICMP" and one.packet_size_mean.iloc[0] == 100


def test_cli_ingest_and_score_netflow(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    import numpy as np
    from aegisflow.cli import main
    from aegisflow.ml.modeling import ForecastDataset, train_experiment
    from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES
    out = tmp_path / "flows.csv"
    assert main(["ingest-netflow", "--input", str(FX / "softflowd_v9.pcap"), "--output", str(out)]) == 0
    assert len(pd.read_csv(out)) == 72
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 5.0, size=(60, 10, len(STANDARD_NUMERIC_WINDOW_FEATURES)))
    frame = pd.DataFrame({"sequence_features": [x.tolist() for x in X], "target_features": X[:, -1].tolist(),
                          "target_attack_present": (rng.random(60) < .3).astype(int), "target_dominant_class": "B",
                          "target_dominant_stage": "Benign", "seq_end_time": pd.Timestamp("2017-07-03"),
                          "target_window_start": pd.Timestamp("2017-07-03 00:01"),
                          "split": ["train"] * 40 + ["val"] * 10 + ["test"] * 10})
    train_experiment(ForecastDataset.from_frame(frame), tmp_path / "m", epochs=1, batch_size=16, hidden_size=4)
    assert main(["score-netflow", "--input", str(FX / "nfdump_v9.csv"), "--model-dir", str(tmp_path / "m")]) == 0
    assert "scored host sequences" in capsys.readouterr().out
