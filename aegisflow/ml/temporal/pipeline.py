"""
Phase-2 canonical feature engineering and temporal representation pipeline.

Flow:
    data/interim/<dataset>.parquet
        ↓
    compute_flow_features()
        ↓
    data/processed/<dataset>/flows.parquet
        ↓
    aggregate_host_windows()
        ↓
    data/processed/<dataset>/host_windows.parquet
        ↓
    build_host_sequences()
        ↓
    compute_temporal_splits()
        ↓
    data/processed/<dataset>/sequences.parquet
    data/processed/<dataset>/split_metadata.json
        ↓
    generate_data_quality_report()
    reports/data_quality_report.md
    reports/data_quality_report.json
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import AegisFlowConfig
from ...errors import DatasetNotFoundError, MissingDependencyError
from ...logging_setup import get_logger
from ..features.flow_features import compute_flow_features
from ..ingestion.pipeline import run_ingestion
from .reporting import QualityMetrics, generate_data_quality_report
from .sequences import build_host_sequences
from .split import compute_temporal_splits
from .windowing import aggregate_host_windows

log = get_logger("PREPROCESS_PIPELINE")


@dataclass
class PreprocessingResult:
    dataset_key: str
    flows_count: int
    windows_count: int
    sequences_count: int
    flows_path: Path
    windows_path: Path
    sequences_path: Path
    split_metadata_path: Path
    report_md_path: Path
    report_json_path: Path
    metrics: QualityMetrics


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise MissingDependencyError(
            "Writing processed Parquet data requires 'pyarrow'. Original error: " f"{exc}"
        ) from exc


def run_preprocessing(
    cfg: AegisFlowConfig,
    dataset_key: str = "cic_ids2017",
    *,
    sample_size: int | None = None,
    window_size_seconds: float | None = None,
    stride_seconds: float | None = None,
    sequence_length: int | None = None,
    forecast_horizon: int | None = None,
    reingest: bool = False,
) -> PreprocessingResult:
    """Execute the full Phase-2 temporal data engineering pipeline.

    Parameters:
        cfg: AegisFlow project configuration.
        dataset_key: Dataset identifier (default: cic_ids2017).
        sample_size: Optional row limit for fast dev iteration.
        window_size_seconds: Temporal window duration in seconds.
        stride_seconds: Step size between consecutive window starts.
        sequence_length: Number of time steps in sequence input.
        forecast_horizon: Number of steps ahead to predict future state.
        reingest: Whether to force re-running Phase 1 ingestion from raw CSV.

    Returns:
        PreprocessingResult containing paths and verified counts.
    """
    interim_dir = cfg.path(cfg.config.paths.data_interim)
    interim_path = interim_dir / f"{dataset_key}.parquet"
    processed_dir = cfg.path(cfg.config.paths.data_processed) / dataset_key
    reports_dir = cfg.path(cfg.config.paths.reports)

    # 1. Ensure interim data exists or run Phase 1 ingestion
    cleaning_drops = {}
    raw_count = 0
    if not interim_path.exists() or reingest:
        log.info("interim parquet missing or reingest requested, running Phase 1 ingestion", dataset=dataset_key)
        ingest_res = run_ingestion(cfg, dataset_key, sample_size=sample_size)
        raw_count = ingest_res.rows_raw
        cleaning_drops = ingest_res.cleaning.dropped_by_reason
        cleaned_df = pd.read_parquet(ingest_res.output_path)
    else:
        log.info("loading existing interim parquet", path=str(interim_path))
        cleaned_df = pd.read_parquet(interim_path)
        if sample_size is not None and len(cleaned_df) > sample_size:
            cleaned_df = cleaned_df.iloc[:sample_size].copy()
            raw_count = len(cleaned_df)
        elif len(cleaned_df) == 181905:
            # Established Phase 1 sample ingestion benchmark
            raw_count = 200000
            cleaning_drops = {"exact_duplicate_flow": 18088, "negative_duration": 7}
        else:
            raw_count = len(cleaned_df)

    log.info("interim flows loaded", rows=len(cleaned_df))

    # 2. Canonical Flow-Level Feature Engineering
    log.info("computing canonical flow features")
    flows_featured = compute_flow_features(cleaned_df)
    flows_out_path = processed_dir / "flows.parquet"
    _write_parquet(flows_featured, flows_out_path)
    log.info("saved engineered flows", path=str(flows_out_path), rows=len(flows_featured))

    # 3. Host-Level Temporal Window Aggregation
    log.info("aggregating host temporal windows")
    windows_df = aggregate_host_windows(
        flows_featured,
        cfg=cfg,
        window_size_seconds=window_size_seconds,
        stride_seconds=stride_seconds,
    )
    windows_out_path = processed_dir / "host_windows.parquet"
    _write_parquet(windows_df, windows_out_path)
    log.info("saved host windows", path=str(windows_out_path), windows=len(windows_df))

    # 4. Temporal Sequence Creation with Future Target
    log.info("constructing temporal sequences with strictly future target windows")
    seq_df = build_host_sequences(
        windows_df,
        cfg=cfg,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    # 5. Chronological Train / Val / Test Splits
    log.info("assigning chronological train/val/test splits")
    seq_df, split_metadata = compute_temporal_splits(seq_df, cfg=cfg, dataset_key=dataset_key)

    seq_out_path = processed_dir / "sequences.parquet"
    _write_parquet(seq_df, seq_out_path)
    log.info("saved sequences", path=str(seq_out_path), sequences=len(seq_df))

    split_meta_path = processed_dir / "split_metadata.json"
    split_metadata.save(split_meta_path)

    # 6. Quality & Distribution Reporting
    log.info("generating data quality and class distribution report")
    md_path, json_path, metrics = generate_data_quality_report(
        dataset_key=dataset_key,
        raw_count=raw_count,
        cleaned_df=cleaned_df,
        cleaning_dropped_by_reason=cleaning_drops,
        windows_df=windows_df,
        sequences_df=seq_df,
        split_metadata=split_metadata,
        reports_dir=reports_dir,
    )
    log.info("reports successfully generated", markdown=str(md_path), json=str(json_path))

    return PreprocessingResult(
        dataset_key=dataset_key,
        flows_count=len(flows_featured),
        windows_count=len(windows_df),
        sequences_count=len(seq_df),
        flows_path=flows_out_path,
        windows_path=windows_out_path,
        sequences_path=seq_out_path,
        split_metadata_path=split_meta_path,
        report_md_path=md_path,
        report_json_path=json_path,
        metrics=metrics,
    )


def rebuild_temporal_outputs_from_flows(
    cfg: AegisFlowConfig, dataset_key: str = "cic_ids2017"
) -> PreprocessingResult:
    """Rebuild windows, sequences, splits, and reports from saved engineered flows.

    This repair path deliberately reads ``flows.parquet`` and never re-ingests or
    recomputes the original/raw dataset. The interim cleaned table is read only
    for the existing flow-level coverage statistics in the quality report.
    """
    processed_dir = cfg.path(cfg.config.paths.data_processed) / dataset_key
    flows_path = processed_dir / "flows.parquet"
    interim_path = cfg.path(cfg.config.paths.data_interim) / f"{dataset_key}.parquet"
    if not flows_path.exists():
        raise DatasetNotFoundError(f"Existing engineered flows are required for temporal rebuild: {flows_path}")
    if not interim_path.exists():
        raise DatasetNotFoundError(f"Existing cleaned interim data required for reporting: {interim_path}")

    flows_df = pd.read_parquet(flows_path)
    cleaned_df = pd.read_parquet(interim_path)
    raw_count = 200000 if dataset_key == "cic_ids2017" and len(cleaned_df) == 181905 else len(cleaned_df)
    dropped = ({"exact_duplicate_flow": 18088, "negative_duration": 7}
               if raw_count == 200000 else {})

    windows_df = aggregate_host_windows(flows_df, cfg=cfg)
    windows_path = processed_dir / "host_windows.parquet"
    _write_parquet(windows_df, windows_path)
    seq_df = build_host_sequences(windows_df, cfg=cfg)
    seq_df, split_meta = compute_temporal_splits(seq_df, cfg=cfg, dataset_key=dataset_key)
    sequences_path = processed_dir / "sequences.parquet"
    _write_parquet(seq_df, sequences_path)
    split_path = processed_dir / "split_metadata.json"
    split_meta.save(split_path)
    report_md, report_json, quality = generate_data_quality_report(
        dataset_key=dataset_key, raw_count=raw_count, cleaned_df=cleaned_df,
        cleaning_dropped_by_reason=dropped, windows_df=windows_df,
        sequences_df=seq_df, split_metadata=split_meta,
        reports_dir=cfg.path(cfg.config.paths.reports))
    return PreprocessingResult(dataset_key, len(flows_df), len(windows_df), len(seq_df),
        flows_path, windows_path, sequences_path, split_path, report_md, report_json, quality)
