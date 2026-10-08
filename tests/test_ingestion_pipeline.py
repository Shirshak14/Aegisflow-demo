"""Integration test: raw fixture -> ingest -> interim parquet, end to end."""
from __future__ import annotations

import pytest

from aegisflow.ml.ingestion.pipeline import run_ingestion


def test_run_ingestion_end_to_end(tmp_path, cfg, cic_ids2017_entry, monkeypatch):
    cfg.datasets["cic_ids2017"] = cic_ids2017_entry
    monkeypatch.setitem(cfg.config.paths, "data_interim", str(tmp_path / "interim"))

    pytest.importorskip("pyarrow")
    result = run_ingestion(cfg, "cic_ids2017", sample_size=None)

    assert result.rows_raw == 9
    assert result.cleaning.rows_out == 9
    assert result.output_path.exists()
    assert sum(result.stage_distribution.values()) == 9


def test_ingestion_records_real_counts_for_preprocess(tmp_path, cfg, cic_ids2017_entry, monkeypatch):
    from aegisflow.ml.ingestion.pipeline import summary_path
    from aegisflow.ml.temporal.pipeline import recorded_ingest_counts

    cfg.datasets["cic_ids2017"] = cic_ids2017_entry
    monkeypatch.setitem(cfg.config.paths, "data_interim", str(tmp_path / "interim"))
    pytest.importorskip("pyarrow")
    result = run_ingestion(cfg, "cic_ids2017", sample_size=None)

    raw, drops, recorded = recorded_ingest_counts(result.output_path, result.cleaning.rows_out)
    assert recorded and raw == result.rows_raw and drops == result.cleaning.dropped_by_reason

    # a summary written for a different interim file is not trusted
    assert recorded_ingest_counts(result.output_path, result.cleaning.rows_out + 1) == (result.cleaning.rows_out + 1, {}, False)
    summary_path(result.output_path).unlink()
    assert recorded_ingest_counts(result.output_path, 9) == (9, {}, False)
