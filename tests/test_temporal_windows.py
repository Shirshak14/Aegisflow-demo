"""
Tests for host-level temporal window aggregation and label preservation.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from aegisflow.ml.temporal.windowing import aggregate_host_windows


def _create_sample_flows():
    # 2 hosts: Host A (attacking) and Host B (benign)
    # Timestamps spaced across 2 minutes
    base_t = pd.Timestamp("2017-07-03 10:00:00")
    data = []

    # Host A flows: at t=0s, 10s, 20s, 40s, 70s
    data.append({
        "timestamp": base_t + pd.Timedelta(seconds=0),
        "source_ip": "10.0.0.1",
        "destination_ip": "192.168.1.1",
        "source_port": 1000,
        "destination_port": 80,
        "protocol": "TCP",
        "flow_duration": 1.0,
        "packet_count": 5.0,
        "byte_count": 500.0,
        "syn_count": 1.0,
        "ack_count": 4.0,
        "normalized_attack_class": "Benign",
        "attack_stage": "Benign",
        "dataset_label": "BENIGN",
    })
    data.append({
        "timestamp": base_t + pd.Timedelta(seconds=10),
        "source_ip": "10.0.0.1",
        "destination_ip": "192.168.1.2",
        "source_port": 1001,
        "destination_port": 81,
        "protocol": "TCP",
        "flow_duration": 1.5,
        "packet_count": 10.0,
        "byte_count": 1000.0,
        "syn_count": 1.0,
        "ack_count": 9.0,
        "normalized_attack_class": "Reconnaissance",
        "attack_stage": "Reconnaissance",
        "dataset_label": "PortScan",
    })
    data.append({
        "timestamp": base_t + pd.Timedelta(seconds=40),  # inside [0, 60) AND [30, 90)
        "source_ip": "10.0.0.1",
        "destination_ip": "192.168.1.3",
        "source_port": 1002,
        "destination_port": 82,
        "protocol": "TCP",
        "flow_duration": 2.0,
        "packet_count": 5.0,
        "byte_count": 500.0,
        "syn_count": 1.0,
        "ack_count": 4.0,
        "normalized_attack_class": "Benign",
        "attack_stage": "Benign",
        "dataset_label": "BENIGN",
    })
    data.append({
        "timestamp": base_t + pd.Timedelta(seconds=70),  # inside [30, 90) AND [60, 120)
        "source_ip": "10.0.0.1",
        "destination_ip": "192.168.1.4",
        "source_port": 1003,
        "destination_port": 83,
        "protocol": "TCP",
        "flow_duration": 1.0,
        "packet_count": 20.0,
        "byte_count": 2000.0,
        "syn_count": 2.0,
        "ack_count": 18.0,
        "normalized_attack_class": "Benign",
        "attack_stage": "Benign",
        "dataset_label": "BENIGN",
    })

    # Host B flows: all benign
    data.append({
        "timestamp": base_t + pd.Timedelta(seconds=5),
        "source_ip": "10.0.0.2",
        "destination_ip": "192.168.1.1",
        "source_port": 2000,
        "destination_port": 443,
        "protocol": "TCP",
        "flow_duration": 5.0,
        "packet_count": 50.0,
        "byte_count": 5000.0,
        "syn_count": 1.0,
        "ack_count": 49.0,
        "normalized_attack_class": "Benign",
        "attack_stage": "Benign",
        "dataset_label": "BENIGN",
    })

    return pd.DataFrame(data)


def test_window_creation_and_overlapping():
    df = _create_sample_flows()
    # Shuffle dataframe to test chronological sorting requirement
    df_shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    windows = aggregate_host_windows(
        df_shuffled,
        window_size_seconds=60.0,
        stride_seconds=30.0,
        min_flows_per_window=1,
    )

    assert not windows.empty
    # Host A should have windows covering its activity
    host_a_wins = windows[windows["host_id"] == "10.0.0.1"].sort_values("window_start").reset_index(drop=True)
    assert len(host_a_wins) >= 2

    # Flow at t=40s should be counted in both window [0, 60) and window [30, 90)
    w0 = host_a_wins.iloc[0]
    w1 = host_a_wins.iloc[1]
    assert w0["window_start"] < w1["window_start"]
    assert w0["flow_count"] == 3  # flows at 0s, 10s, 40s
    assert w1["flow_count"] == 2  # flows at 40s, 70s


def test_attack_present_and_stage_aggregation():
    df = _create_sample_flows()
    windows = aggregate_host_windows(
        df,
        window_size_seconds=60.0,
        stride_seconds=30.0,
        min_flows_per_window=1,
    )

    host_a_w0 = windows[(windows["host_id"] == "10.0.0.1") & (windows["window_idx"] == 0)].iloc[0]
    # In window 0 of Host A: 2 Benign flows, 1 PortScan flow
    # The attack must NOT be silently called Benign!
    assert host_a_w0["attack_present"] == 1
    assert host_a_w0["dominant_class"] == "Reconnaissance"
    assert host_a_w0["dominant_stage"] == "Reconnaissance"
    assert host_a_w0["label_confidence"] == "inferred"

    # Stage distribution JSON check
    dist = json.loads(host_a_w0["stage_distribution"])
    assert dist["Benign"] == 2
    assert dist["Reconnaissance"] == 1

    # Host B should be completely benign with ground_truth confidence
    host_b_w0 = windows[windows["host_id"] == "10.0.0.2"].iloc[0]
    assert host_b_w0["attack_present"] == 0
    assert host_b_w0["dominant_class"] == "Benign"
    assert host_b_w0["dominant_stage"] == "Benign"
    assert host_b_w0["label_confidence"] == "ground_truth"


def test_entropy_and_fanout():
    df = _create_sample_flows()
    windows = aggregate_host_windows(
        df,
        window_size_seconds=60.0,
        stride_seconds=30.0,
        min_flows_per_window=1,
    )

    host_a_w0 = windows[(windows["host_id"] == "10.0.0.1") & (windows["window_idx"] == 0)].iloc[0]
    # 3 distinct ports (80, 81, 82), entropy should be > 1.0 (log2(3) ~ 1.58)
    assert host_a_w0["destination_port_entropy"] > 1.0
    assert host_a_w0["unique_destination_ports"] == 3
    assert host_a_w0["unique_destination_ips"] == 3


def test_configurable_host_grouping():
    df = _create_sample_flows()
    # Group by destination_ip instead of source_ip
    windows = aggregate_host_windows(
        df,
        window_size_seconds=60.0,
        stride_seconds=30.0,
        min_flows_per_window=1,
        group_by="destination_ip",
    )
    assert set(windows["host_id"].unique()) == {"192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.4"}
