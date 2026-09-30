"""
Tests for Phase 2 data quality reporting and output dataset schema.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from aegisflow.ml.temporal.reporting import generate_data_quality_report
from aegisflow.ml.temporal.sequences import build_host_sequences
from aegisflow.ml.temporal.split import compute_temporal_splits
from tests.test_sequences import _create_mock_windows
from tests.test_temporal_windows import _create_sample_flows


def test_data_quality_report_generation(tmp_path: Path):
    flows = _create_sample_flows()
    windows = _create_mock_windows(n_windows_h1=20, n_windows_h2=15)
    seqs = build_host_sequences(windows, sequence_length=5, forecast_horizon=1)
    seqs, split_meta = compute_temporal_splits(seqs, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)

    md_path, json_path, metrics = generate_data_quality_report(
        dataset_key="cic_ids2017",
        raw_count=len(flows) + 10,
        cleaned_df=flows,
        cleaning_dropped_by_reason={"missing_timestamp": 5, "exact_duplicate_flow": 5},
        windows_df=windows,
        sequences_df=seqs,
        split_metadata=split_meta,
        reports_dir=tmp_path,
    )

    assert md_path.exists()
    assert json_path.exists()

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["dataset_key"] == "cic_ids2017"
    assert data["cleaned_rows_kept"] == len(flows)
    assert data["total_host_windows"] == len(windows)
    assert data["total_sequences"] == len(seqs)

    with md_path.open("r", encoding="utf-8") as fh:
        md_text = fh.read()
    assert "# AegisFlow Phase 2: Data Quality & Temporal Distribution Report" in md_text
    assert "Cleaned Rows Kept" in md_text
    assert "ttl_mean" in md_text
    assert "PROXY INFERENCES" in md_text
