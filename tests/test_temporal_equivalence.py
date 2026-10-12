"""The vectorised window and sequence builders (B6, B7) must match the old row-by-row code exactly.

The old implementations live in tests/_temporal_reference.py. Synthetic flows cover the edge cases
named in the backlog: ties between attack classes, windows without attacks, single-flow hosts, flows
exactly on window boundaries, interleaved hosts, gaps between a host's windows and missing labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _temporal_reference import old_aggregate_host_windows, old_build_host_sequences
from aegisflow.ml.modeling import ForecastDataset
from aegisflow.ml.temporal.sequences import (
    STANDARD_NUMERIC_WINDOW_FEATURES,
    build_host_sequences,
    feature_array,
    write_sequences,
)
from aegisflow.ml.temporal.windowing import aggregate_host_windows

CLASSES = [("Benign", "Benign", "BENIGN"), ("DoS", "Impact", "DoS Hulk"), ("PortScan", "Reconnaissance", "PortScan"),
           ("Botnet", "Command and Control", "Bot"), (None, None, None)]


def _flows(seed: int, n: int = 600, hosts: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2017-07-03 09:00:00")
    # Half the timestamps land exactly on a 30 s grid point (a window start and another window's end).
    on_grid = rng.random(n) < 0.5
    secs = np.where(on_grid, rng.integers(0, 120, n) * 30.0, rng.uniform(0, 3600, n).round(3))
    # Host gaps: host 0 is silent between 20 and 40 minutes.
    host = rng.integers(0, hosts, n)
    secs = np.where((host == 0) & (secs > 1200) & (secs < 2400), secs + 1200, secs)
    label = rng.choice(len(CLASSES), n, p=[0.55, 0.15, 0.15, 0.1, 0.05])
    rows = {
        "timestamp": base + pd.to_timedelta(secs, unit="s"),
        "source_ip": [f"10.0.0.{h}" for h in host],
        "destination_ip": [f"192.168.1.{d}" for d in rng.integers(1, 8, n)],
        "source_port": rng.integers(1000, 1010, n),
        "destination_port": rng.choice([22, 80, 443, 8080], n),
        "protocol": "TCP",
        "flow_duration": rng.exponential(2.0, n),
        "packet_count": rng.integers(1, 50, n).astype(float),
        "byte_count": rng.integers(40, 5000, n).astype(float),
        "syn_count": rng.integers(0, 3, n).astype(float),
        "ack_count": rng.integers(0, 40, n).astype(float),
        "normalized_attack_class": [CLASSES[i][0] for i in label],
        "attack_stage": [CLASSES[i][1] for i in label],
        "dataset_label": [CLASSES[i][2] for i in label],
    }
    df = pd.DataFrame(rows)
    # A host with exactly one flow
    lone = df.iloc[[0]].assign(source_ip="10.0.0.99", timestamp=base + pd.Timedelta(seconds=90))
    return pd.concat([df, lone], ignore_index=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("window,stride", [(60.0, 30.0), (60.0, 60.0), (90.0, 30.0), (20.0, 30.0)])
def test_windows_match_old_implementation(seed, window, stride):
    flows = _flows(seed)
    old = old_aggregate_host_windows(flows, window_size_seconds=window, stride_seconds=stride)
    new = aggregate_host_windows(flows, window_size_seconds=window, stride_seconds=stride)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    assert old.attack_present.sum() > 0 and (old.attack_present == 0).sum() > 0


def test_dominant_class_ties_pick_the_first_attack_column_like_idxmax():
    base = pd.Timestamp("2017-07-03 09:00:00")
    flows = _flows(0, n=4).iloc[:4].copy()
    flows["source_ip"] = "10.0.0.1"
    flows["timestamp"] = base + pd.to_timedelta([1, 2, 3, 4], unit="s")
    # One DoS and one PortScan flow (a tie), plus benign flows.
    flows["normalized_attack_class"] = ["PortScan", "DoS", "Benign", "Benign"]
    flows["attack_stage"] = ["Reconnaissance", "Impact", "Benign", "Benign"]
    new = aggregate_host_windows(flows)
    old = old_aggregate_host_windows(flows)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    assert set(new.dominant_class) == {"DoS"} and set(new.dominant_stage) == {"Impact"}


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("length,horizon,stride", [(1, 1, 1), (3, 1, 1), (5, 2, 1), (4, 3, 2), (10, 1, 1)])
def test_sequences_match_old_implementation(seed, length, horizon, stride):
    windows = aggregate_host_windows(_flows(seed, n=1500, hosts=4))
    old = old_build_host_sequences(windows, sequence_length=length, forecast_horizon=horizon, stride=stride)
    new = build_host_sequences(windows, sequence_length=length, forecast_horizon=horizon, stride=stride)
    assert len(old) > 0
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    assert np.array_equal(feature_array(new.sequence_features, n_features=len(STANDARD_NUMERIC_WINDOW_FEATURES)),
                          feature_array(old.sequence_features, n_features=len(STANDARD_NUMERIC_WINDOW_FEATURES)))


def test_sequences_empty_when_no_host_has_enough_windows():
    windows = aggregate_host_windows(_flows(0, n=20))
    old = old_build_host_sequences(windows, sequence_length=200)
    new = build_host_sequences(windows, sequence_length=200)
    assert old.empty and new.empty


def _with_split(seq: pd.DataFrame) -> pd.DataFrame:
    return seq.assign(split=np.where(np.arange(len(seq)) % 3 == 0, "test", "train"))


@pytest.mark.parametrize("sequence_format", ["nested", "arrays"])
def test_saved_sequences_have_no_future_inputs_and_load_identically(tmp_path, sequence_format):
    windows = aggregate_host_windows(_flows(4, n=1500, hosts=4))
    windows = windows.assign(**{c: 0.0 for c in STANDARD_NUMERIC_WINDOW_FEATURES if c not in windows.columns})
    seq = _with_split(build_host_sequences(windows, sequence_length=5))
    path = tmp_path / "sequences.parquet"
    write_sequences(seq, path, sequence_format=sequence_format)
    back = pd.read_parquet(path)
    assert list(back.columns) == list(seq.columns)
    # Leakage: every input window starts no later than seq_end_time, and the target starts after it.
    assert (back.seq_start_time <= back.seq_end_time).all()
    assert (back.target_window_start > back.seq_end_time).all()
    assert (back.target_host_id == back.host_id).all()
    built, loaded = ForecastDataset.from_frame(seq), ForecastDataset.from_frame(back)
    assert np.array_equal(loaded.X, built.X) and np.array_equal(loaded.future_state, built.future_state)
    assert np.array_equal(loaded.attack, built.attack)
    pd.testing.assert_frame_equal(back.drop(columns=["sequence_features", "target_features"]),
                                  seq.drop(columns=["sequence_features", "target_features"]), check_exact=True)
