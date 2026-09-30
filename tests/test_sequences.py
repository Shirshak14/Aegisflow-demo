"""
Tests for temporal sequence generation, future target alignment, and leakage prevention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegisflow.ml.temporal.sequences import build_host_sequences


def _create_mock_windows(n_windows_h1=15, n_windows_h2=12):
    base_t = pd.Timestamp("2017-07-03 10:00:00")
    records = []

    # Host 1
    for i in range(n_windows_h1):
        ws = base_t + pd.Timedelta(seconds=i * 30)
        we = ws + pd.Timedelta(seconds=60)
        records.append({
            "host_id": "host_1",
            "window_idx": i,
            "window_start": ws,
            "window_end": we,
            "flow_count": float(i + 1),
            "packet_count_sum": float((i + 1) * 10),
            "packet_count_mean": 10.0,
            "byte_count_sum": float((i + 1) * 100),
            "byte_count_mean": 100.0,
            "duration_mean": 1.0,
            "packets_per_second_mean": 5.0,
            "bytes_per_second_mean": 50.0,
            "bytes_per_packet_mean": 10.0,
            "packet_size_mean": 100.0,
            "packet_size_std_mean": 10.0,
            "inter_arrival_mean": 0.1,
            "syn_count_sum": 1.0,
            "ack_count_sum": 9.0,
            "rst_count_sum": 0.0,
            "fin_count_sum": 0.0,
            "psh_count_sum": 0.0,
            "syn_ratio": 0.1,
            "ack_ratio": 0.9,
            "rst_ratio": 0.0,
            "fin_ratio": 0.0,
            "psh_ratio": 0.0,
            "tcp_window_size_mean": 8192.0,
            "unique_destination_ips": 1.0,
            "unique_destination_ports": 1.0,
            "unique_source_ports": 1.0,
            "destination_port_entropy": 0.0,
            "connection_burst_max": 2.0,
            "attack_flow_ratio": 1.0 if i >= 10 else 0.0,
            "benign_flow_ratio": 0.0 if i >= 10 else 1.0,
            "attack_present": 1 if i >= 10 else 0,
            "dominant_class": "Denial of Service" if i >= 10 else "Benign",
            "dominant_stage": "Impact" if i >= 10 else "Benign",
            "stage_distribution": '{"Impact": 1}' if i >= 10 else '{"Benign": 1}',
            "label_confidence": "inferred" if i >= 10 else "ground_truth",
        })

    # Host 2
    for j in range(n_windows_h2):
        ws = base_t + pd.Timedelta(seconds=j * 30)
        we = ws + pd.Timedelta(seconds=60)
        records.append({
            "host_id": "host_2",
            "window_idx": j,
            "window_start": ws,
            "window_end": we,
            "flow_count": 5.0,
            "packet_count_sum": 50.0,
            "packet_count_mean": 10.0,
            "byte_count_sum": 500.0,
            "byte_count_mean": 100.0,
            "duration_mean": 2.0,
            "packets_per_second_mean": 5.0,
            "bytes_per_second_mean": 50.0,
            "bytes_per_packet_mean": 10.0,
            "packet_size_mean": 100.0,
            "packet_size_std_mean": 10.0,
            "inter_arrival_mean": 0.1,
            "syn_count_sum": 1.0,
            "ack_count_sum": 9.0,
            "rst_count_sum": 0.0,
            "fin_count_sum": 0.0,
            "psh_count_sum": 0.0,
            "syn_ratio": 0.1,
            "ack_ratio": 0.9,
            "rst_ratio": 0.0,
            "fin_ratio": 0.0,
            "psh_ratio": 0.0,
            "tcp_window_size_mean": 8192.0,
            "unique_destination_ips": 2.0,
            "unique_destination_ports": 2.0,
            "unique_source_ports": 2.0,
            "destination_port_entropy": 1.0,
            "connection_burst_max": 1.0,
            "attack_flow_ratio": 0.0,
            "benign_flow_ratio": 1.0,
            "attack_present": 0,
            "dominant_class": "Benign",
            "dominant_stage": "Benign",
            "stage_distribution": '{"Benign": 5}',
            "label_confidence": "ground_truth",
        })

    return pd.DataFrame(records)


def test_sequence_creation_and_future_target():
    windows = _create_mock_windows(n_windows_h1=15, n_windows_h2=12)
    seq_df = build_host_sequences(
        windows,
        sequence_length=5,
        forecast_horizon=2,
        stride=1,
    )

    # Horizon counts windows whose starts are strictly after the final input end.
    # This leaves 7 valid host_1 starts and 4 host_2 starts.
    assert len(seq_df) == 11

    # Test Host 1 sequence 0:
    # Inputs are windows 0..4 (end=180s); eligible future windows start at 210s.
    # Horizon 2 selects eligible window 8 (start=240s), not row 6 which overlaps.
    h1_seq0 = seq_df[seq_df["sequence_id"] == "host_1_seq_0"].iloc[0]
    assert h1_seq0["sequence_length"] == 5
    assert h1_seq0["forecast_horizon"] == 2
    assert len(h1_seq0["sequence_features"]) == 5
    assert h1_seq0["target_window_start"] == pd.Timestamp("2017-07-03 10:04:00")
    assert h1_seq0["target_window_start"] > h1_seq0["seq_end_time"]
    assert h1_seq0["target_attack_present"] == 0  # window 6 is Benign

    # Host 1 start 4 targets window 12, which is attack-positive.
    h1_seq4 = seq_df[seq_df["sequence_id"] == "host_1_seq_4"].iloc[0]
    assert h1_seq4["target_attack_present"] == 1
    assert h1_seq4["target_dominant_class"] == "Denial of Service"
    assert h1_seq4["target_dominant_stage"] == "Impact"


def test_regression_old_minus_30_second_overlap_is_rejected():
    windows = _create_mock_windows(n_windows_h1=15, n_windows_h2=12)
    seq_df = build_host_sequences(windows, sequence_length=10, forecast_horizon=1)
    gaps = pd.to_datetime(seq_df["target_window_start"]) - pd.to_datetime(seq_df["seq_end_time"])
    assert len(seq_df) > 0
    assert (gaps > pd.Timedelta(0)).all()
    # Under the old row-offset rule, inputs 0..9 ended at +330s and target row 10
    # started at +300s: the exact regression was a -30 second gap.
    old_gap = (windows.query("host_id == 'host_1'").iloc[10]["window_start"] -
               windows.query("host_id == 'host_1'").iloc[9]["window_end"])
    assert old_gap == pd.Timedelta(seconds=-30)


def test_forecast_horizon_counts_strictly_future_windows():
    windows = _create_mock_windows(n_windows_h1=18, n_windows_h2=18)
    one = build_host_sequences(windows, sequence_length=5, forecast_horizon=1)
    two = build_host_sequences(windows, sequence_length=5, forecast_horizon=2)
    a = one[one.sequence_id == "host_1_seq_0"].iloc[0]
    b = two[two.sequence_id == "host_1_seq_0"].iloc[0]
    assert a.target_window_start > a.seq_end_time
    assert b.target_window_start > a.target_window_start
    assert (b.target_window_start - a.target_window_start) == pd.Timedelta(seconds=30)


def test_every_target_is_strict_future_nonoverlapping_and_same_host():
    windows = _create_mock_windows(n_windows_h1=20, n_windows_h2=18)
    seq_df = build_host_sequences(windows, sequence_length=5, forecast_horizon=1)
    assert (pd.to_datetime(seq_df.target_window_start) > pd.to_datetime(seq_df.seq_end_time)).all()
    assert (pd.to_datetime(seq_df.target_window_end) > pd.to_datetime(seq_df.target_window_start)).all()
    assert (seq_df.host_id == seq_df.target_host_id).all()
    assert all(len(row.sequence_features) == row.sequence_length for row in seq_df.itertuples())
    for row in seq_df.itertuples():
        # The test fixture's flow_count increases by input row, so ordering is visible.
        if row.host_id == "host_1":
            assert np.diff(np.asarray(row.sequence_features)[:, 0]).min() > 0


def test_no_cross_host_sequence_leakage():
    windows = _create_mock_windows(n_windows_h1=15, n_windows_h2=12)
    seq_df = build_host_sequences(
        windows,
        sequence_length=5,
        forecast_horizon=1,
    )

    # Ensure every sequence belongs to only one host
    for _, row in seq_df.iterrows():
        host = row["host_id"]
        assert host in {"host_1", "host_2"}
        assert row["sequence_id"].startswith(host)


def test_no_future_to_past_leakage():
    windows = _create_mock_windows(n_windows_h1=15, n_windows_h2=12)
    seq_df = build_host_sequences(
        windows,
        sequence_length=5,
        forecast_horizon=1,
    )

    for _, row in seq_df.iterrows():
        # Target window start MUST be strictly in the future relative to the last sequence window's start
        assert row["target_window_start"] > row["seq_start_time"]
        assert row["target_window_end"] > row["seq_end_time"]


def test_target_is_onset_flags_only_attacks_absent_from_all_inputs():
    # host_1 windows >= 10 are attacks. With L=5, H=1 (60s windows, 30s stride), start i's
    # inputs are i..i+4 and its target is window i+7 (first start > input end).
    # Target is attack for i >= 3; inputs contain attack for i >= 6 -> onsets are i in {3, 4, 5}.
    windows = _create_mock_windows(n_windows_h1=20, n_windows_h2=12)
    seq_df = build_host_sequences(windows, sequence_length=5, forecast_horizon=1)
    h1 = seq_df[seq_df.host_id == "host_1"].set_index("sequence_id")
    onsets = sorted(int(s.rsplit("_", 1)[1]) for s in h1.index[h1.target_is_onset])
    assert onsets == [3, 4, 5]
    # the existing any-positive target is unchanged; onset is a strict subset of it
    assert (seq_df.loc[seq_df.target_is_onset, "target_attack_present"] == 1).all()
    assert int(h1.loc["host_1_seq_6", "target_attack_present"]) == 1 and not h1.loc["host_1_seq_6", "target_is_onset"]
    assert not seq_df[seq_df.host_id == "host_2"].target_is_onset.any()
