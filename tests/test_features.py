"""
Tests for canonical flow feature engineering and feature registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegisflow.ml.features.flow_features import compute_flow_features
from aegisflow.ml.features.registry import (
    FEATURE_REGISTRY,
    get_features_by_level,
    get_unavailable_features,
    registry_as_markdown,
)
from aegisflow.schema import empty_canonical_frame


def _make_dummy_flows() -> pd.DataFrame:
    df = empty_canonical_frame()
    rows = [
        {
            "timestamp": pd.Timestamp("2017-07-03 10:00:00"),
            "source_ip": "192.168.1.10",
            "destination_ip": "10.0.0.1",
            "source_port": 50000,
            "destination_port": 80,
            "protocol": "TCP",
            "flow_duration": 2.0,
            "packet_count": 10.0,
            "byte_count": 1000.0,
            "syn_count": 1.0,
            "ack_count": 9.0,
            "rst_count": 0.0,
            "fin_count": 1.0,
            "psh_count": 2.0,
            "dataset_label": "BENIGN",
        },
        {
            "timestamp": pd.Timestamp("2017-07-03 10:00:05"),
            "source_ip": "192.168.1.20",
            "destination_ip": "10.0.0.2",
            "source_port": 50001,
            "destination_port": 443,
            "protocol": "TCP",
            "flow_duration": 0.0,  # Zero duration edge case
            "packet_count": 0.0,  # Zero packet count edge case
            "byte_count": 0.0,
            "syn_count": np.nan,  # Missing flag count edge case
            "dataset_label": "BENIGN",
        },
    ]
    for i, r in enumerate(rows):
        for k, v in r.items():
            df.loc[i, k] = v
    return df


def test_canonical_feature_calculations():
    df = _make_dummy_flows()
    featured = compute_flow_features(df)

    row0 = featured.iloc[0]
    assert row0["packets_per_second"] == pytest.approx(10.0 / 2.0)
    assert row0["bytes_per_second"] == pytest.approx(1000.0 / 2.0)
    assert row0["bytes_per_packet"] == pytest.approx(1000.0 / 10.0)
    assert row0["syn_ratio"] == pytest.approx(1.0 / 10.0)
    assert row0["ack_ratio"] == pytest.approx(9.0 / 10.0)
    assert row0["rst_ratio"] == pytest.approx(0.0)
    assert row0["fin_ratio"] == pytest.approx(1.0 / 10.0)
    assert row0["psh_ratio"] == pytest.approx(2.0 / 10.0)


def test_zero_duration_and_zero_packet_handling():
    df = _make_dummy_flows()
    featured = compute_flow_features(df)

    row1 = featured.iloc[1]
    # Check that zero duration is safely clamped to 1e-6 and packet count to 1.0 without inf/nan crash
    assert np.isfinite(row1["packets_per_second"])
    assert np.isfinite(row1["bytes_per_second"])
    assert np.isfinite(row1["bytes_per_packet"])
    assert row1["packets_per_second"] == pytest.approx(0.0)
    assert row1["bytes_per_second"] == pytest.approx(0.0)
    assert row1["bytes_per_packet"] == pytest.approx(0.0)
    assert row1["syn_ratio"] == pytest.approx(0.0)


def test_feature_registry_metadata():
    assert len(FEATURE_REGISTRY) > 20
    flow_features = get_features_by_level("flow")
    window_features = get_features_by_level("host-window")
    seq_features = get_features_by_level("sequence")

    assert len(flow_features) > 0
    assert len(window_features) > 0
    assert len(seq_features) > 0

    unavail = get_unavailable_features()
    unavail_names = {f.feature_name for f in unavail}
    assert "ttl_mean" in unavail_names
    assert "ttl_std" in unavail_names
    assert "retransmission_count" in unavail_names

    md = registry_as_markdown()
    assert "| `timestamp` |" in md
    assert "| `ttl_mean` |" in md
