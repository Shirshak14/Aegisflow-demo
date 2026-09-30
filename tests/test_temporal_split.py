"""
Tests for time-based chronological train/val/test splitting and leakage prevention.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from aegisflow.ml.temporal.sequences import build_host_sequences
from aegisflow.ml.temporal.split import compute_temporal_splits
from tests.test_sequences import _create_mock_windows


def test_time_based_split_chronology_and_quarantine(tmp_path: Path):
    windows = _create_mock_windows(n_windows_h1=30, n_windows_h2=25)
    seq_df = build_host_sequences(
        windows,
        sequence_length=5,
        forecast_horizon=1,
    )

    split_df, meta = compute_temporal_splits(
        seq_df,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    assert "split" in split_df.columns
    assert set(split_df["split"].unique()).issubset({"train", "val", "test", "boundary_excluded"})

    val_cutoff = pd.Timestamp(meta.val_cutoff_time)
    test_cutoff = pd.Timestamp(meta.test_cutoff_time)

    # 1. Train sequences must strictly complete on or before val_cutoff
    train_seqs = split_df[split_df["split"] == "train"]
    if not train_seqs.empty:
        assert (train_seqs["target_window_end"] <= val_cutoff).all()

    # 2. Val sequences must strictly start on or after val_cutoff and complete on or before test_cutoff
    val_seqs = split_df[split_df["split"] == "val"]
    if not val_seqs.empty:
        assert (val_seqs["seq_start_time"] >= val_cutoff).all()
        assert (val_seqs["target_window_end"] <= test_cutoff).all()

    # 3. Test sequences must strictly start on or after test_cutoff
    test_seqs = split_df[split_df["split"] == "test"]
    if not test_seqs.empty:
        assert (test_seqs["seq_start_time"] >= test_cutoff).all()

    # 4. Check metadata serialization
    meta_path = tmp_path / "split_metadata.json"
    meta.save(meta_path)
    assert meta_path.exists()
    with meta_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["dataset_key"] == "cic_ids2017"
    assert data["total_sequences"] == len(split_df)


def _split_fixture():
    # 2-hour timeline so each 20% partition is longer than one 5-window sequence span.
    windows = _create_mock_windows(n_windows_h1=240, n_windows_h2=200)
    return build_host_sequences(windows, sequence_length=5, forecast_horizon=1)


def test_split_assignment_is_label_independent():
    # Cutoffs must come from time alone: no stratification, no label-driven boundaries.
    seq_df = _split_fixture()
    base, meta = compute_temporal_splits(seq_df, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    flipped = seq_df.copy()
    flipped["target_attack_present"] = 1 - flipped["target_attack_present"]
    flipped["target_dominant_class"] = "MUTATED"
    other, other_meta = compute_temporal_splits(flipped, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    assert (base["split"].to_numpy() == other["split"].to_numpy()).all()
    assert (meta.val_cutoff_time, meta.test_cutoff_time) == (other_meta.val_cutoff_time, other_meta.test_cutoff_time)


def test_split_partitions_are_disjoint_and_ordered_in_time():
    # Every train interval ends before any val interval begins; same for val -> test.
    split_df, _ = compute_temporal_splits(_split_fixture(), train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    parts = {s: split_df[split_df["split"] == s] for s in ("train", "val", "test")}
    assert all(len(p) > 0 for p in parts.values())
    assert parts["train"]["target_window_end"].max() <= parts["val"]["seq_start_time"].min()
    assert parts["val"]["target_window_end"].max() <= parts["test"]["seq_start_time"].min()
    straddling = split_df[split_df["split"] == "boundary_excluded"]
    assert len(split_df) == sum(len(p) for p in parts.values()) + len(straddling)


def test_split_is_deterministic():
    seq_df = _split_fixture()
    a, _ = compute_temporal_splits(seq_df, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    b, _ = compute_temporal_splits(seq_df.sample(frac=1.0, random_state=0),
                                   train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    assert a.set_index("sequence_id")["split"].sort_index().equals(b.set_index("sequence_id")["split"].sort_index())
