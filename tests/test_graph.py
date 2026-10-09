"""Temporal GNN: graph construction from flows, and that the graph adds information per-host models lack."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aegisflow.ml.features.flow_features import compute_flow_features
from aegisflow.ml.graph import build_graph_sequences, predict_tgnn, train_tgnn, window_neighbours
from aegisflow.ml.modeling import MODEL_FEATURES, ForecastDataset, train_experiment
from aegisflow.ml.temporal.sequences import build_host_sequences
from aegisflow.ml.temporal.windowing import aggregate_host_windows


def flows_frame(rows):
    df = pd.DataFrame(rows)
    for c in ("ack_count", "rst_count", "fin_count", "psh_count", "urg_count", "packet_size_mean", "packet_size_std",
              "inter_arrival_mean", "tcp_window_size"):
        if c not in df:
            df[c] = 0.0
    df["normalized_attack_class"] = np.where(df.dataset_label == "BENIGN", "Benign", "Botnet")
    df["attack_stage"] = np.where(df.dataset_label == "BENIGN", "Benign", "Command and Control")
    return compute_flow_features(df)


def synthetic(seed=0, clients=40, minutes=90):
    """Clients talk to 4 servers at random. Server S0 periodically turns 'bad' (its own outbound traffic
    explodes); a client's flows to S0 during a bad period are labelled Bot. A client's OWN features do not
    change when it talks to S0, so only the graph (who it talked to, and what that peer was doing) predicts
    its next-window label."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2017-07-03 09:00:00")
    servers = [f"10.9.0.{i}" for i in range(4)]
    bad = np.zeros(minutes * 2, bool)
    for s in rng.choice(minutes * 2 - 20, 6, replace=False):
        bad[s:s + 16] = True
    rows = []
    for half in range(minutes * 2):                   # 30 s steps
        base = t0 + pd.Timedelta(seconds=30 * half)
        for c in range(clients):
            for _ in range(2):
                srv = rng.integers(0, 4)
                ts = base + pd.Timedelta(seconds=float(rng.uniform(0, 30)))
                rows.append(dict(timestamp=ts, source_ip=f"10.0.0.{c}", destination_ip=servers[srv],
                                 source_port=int(rng.integers(1024, 65000)), destination_port=443, protocol="TCP",
                                 flow_duration=float(rng.uniform(0.1, 2)), packet_count=float(rng.integers(5, 15)),
                                 byte_count=float(rng.integers(500, 5000)), syn_count=1.0,
                                 dataset_label="Bot" if srv == 0 and bad[half] else "BENIGN"))
        for i, srv in enumerate(servers):            # server outbound traffic
            heavy = i == 0 and bad[half]
            for _ in range(3):
                ts = base + pd.Timedelta(seconds=float(rng.uniform(0, 30)))
                rows.append(dict(timestamp=ts, source_ip=srv, destination_ip=f"10.0.0.{rng.integers(0, clients)}",
                                 source_port=443, destination_port=int(rng.integers(1024, 65000)), protocol="TCP",
                                 flow_duration=float(rng.uniform(0.1, 2)),
                                 packet_count=float(rng.integers(5, 15)) * (50 if heavy else 1),
                                 byte_count=float(rng.integers(500, 5000)) * (200 if heavy else 1), syn_count=1.0,
                                 dataset_label="BENIGN"))
    flows = flows_frame(rows).sort_values("timestamp").reset_index(drop=True)
    windows = aggregate_host_windows(flows, window_size_seconds=60, stride_seconds=30)
    seq = build_host_sequences(windows, sequence_length=5)
    seq = seq[seq.host_id.str.startswith("10.0.0.")].reset_index(drop=True)   # forecast clients
    cut1, cut2 = seq.seq_end_time.quantile([0.6, 0.8])
    seq["split"] = np.where(seq.seq_end_time <= cut1, "train", np.where(seq.seq_end_time <= cut2, "val", "test"))
    return flows, windows, seq


@pytest.fixture(scope="module")
def data():
    return synthetic()


def test_neighbours_come_from_flows_in_the_window():
    t = pd.Timestamp("2017-07-03 09:00:00")
    flows = pd.DataFrame({"timestamp": [t, t + pd.Timedelta("10s"), t + pd.Timedelta("70s")],
                          "source_ip": ["a", "c", "a"], "destination_ip": ["b", "a", "d"]})
    windows = pd.DataFrame({"host_id": ["a", "a"], "window_start": [t, t + pd.Timedelta("60s")]})
    nb = window_neighbours(flows, windows, 60)
    assert nb[("a", t)] == {"b", "c"} and nb[("b", t)] == {"a"} and nb[("a", t + pd.Timedelta("60s"))] == {"d"}


def test_graph_sequences_align_with_host_windows(data):
    flows, windows, seq = data
    g = build_graph_sequences(seq, windows, flows, 60)
    assert g.neighbours.shape == g.base.X.shape and g.degree.shape == g.base.X.shape[:2] + (1,)
    assert (g.degree > 0).all()                    # every client talks to servers every window
    # Neighbour features are exactly the mean of the server rows the host talked to.
    i, t = 3, 2
    host = seq.host_id.iloc[i]
    ws = windows[(windows.host_id == host) & (windows.window_start >= seq.seq_start_time.iloc[i])].window_start.iloc[t]
    peers = window_neighbours(flows, windows, 60)[(host, ws)]
    rows = windows[(windows.window_start == ws) & windows.host_id.isin(peers)][MODEL_FEATURES].fillna(0).to_numpy()
    assert np.allclose(g.neighbours[i, t], rows.mean(0), rtol=1e-5)
    with pytest.raises(ValueError, match="same preprocess run"):
        build_graph_sequences(seq, windows.iloc[::2], flows, 60)


def test_graph_carries_signal_the_lstm_cannot_see(data, tmp_path):
    flows, windows, seq = data
    g = build_graph_sequences(seq, windows, flows, 60)
    res = train_tgnn(g, tmp_path / "gnn", epochs=30, batch_size=64, hidden_size=16, learning_rate=0.005,
                     dropout=0.0, patience=8)
    lstm = train_experiment(ForecastDataset.from_frame(seq), tmp_path / "lstm", epochs=30, batch_size=64,
                            hidden_size=16, learning_rate=0.005, dropout=0.0, patience=8)
    gnn_auc, lstm_auc = res["models"]["temporal_gnn"]["roc_auc"], lstm["models"]["lstm"]["roc_auc"]
    assert gnn_auc > 0.8 and gnn_auc > lstm_auc + 0.15, (gnn_auc, lstm_auc)
    p = predict_tgnn(g, tmp_path / "gnn")
    assert p.shape == (len(seq),) and ((p >= 0) & (p <= 1)).all()
    with pytest.raises(ValueError, match="not a temporal GNN"):
        predict_tgnn(g, tmp_path / "lstm")
