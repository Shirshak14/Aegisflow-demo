"""Phase-1 ingestion pipeline: raw dataset -> canonical -> cleaned -> labeled -> interim Parquet."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import AegisFlowConfig
from ...errors import MissingDependencyError
from ...logging_setup import get_logger
from ..datasets import get_adapter
from ..datasets.base import DatasetAdapter
from ..preprocessing.cleaning import CleaningReport, clean_canonical_frame
from ..preprocessing.labels import apply_stage_mapping

log = get_logger("INGEST")


@dataclass
class IngestResult:
    dataset_key: str
    rows_raw: int
    cleaning: CleaningReport
    output_path: Path
    class_distribution: dict[str, int]
    stage_distribution: dict[str, int]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise MissingDependencyError(
            "Writing processed data requires the 'pyarrow' package "
            "(pip install pyarrow, part of requirements.txt). "
            f"Original error: {exc}"
        ) from exc


def summary_path(interim_path: Path) -> Path:
    """Sidecar next to the interim Parquet that records the measured ingestion counts."""
    return interim_path.with_name(interim_path.stem + ".ingest.json")


def write_ingest_summary(interim_path: Path, *, rows_raw: int, rows_clean: int,
                         dropped_by_reason: dict[str, int], sample_size: int | None) -> Path:
    path = summary_path(interim_path)
    path.write_text(json.dumps({"rows_raw": int(rows_raw), "rows_clean": int(rows_clean),
                                "dropped_by_reason": {k: int(v) for k, v in dropped_by_reason.items()},
                                "sample_size": sample_size}, indent=2), encoding="utf-8")
    return path


def read_ingest_summary(interim_path: Path, rows_clean: int) -> dict | None:
    """The recorded counts, or None when missing, unreadable, or written for a different interim file."""
    path = summary_path(interim_path)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict) or summary.get("rows_clean") != rows_clean:
        log.warning("ingest summary does not match interim parquet, ignoring it", path=str(path),
                    recorded=summary.get("rows_clean") if isinstance(summary, dict) else None, actual=rows_clean)
        return None
    return summary


def run_ingestion(cfg: AegisFlowConfig, dataset_key: str, sample_size: int | None = None) -> IngestResult:
    """Run the full Phase-1 pipeline for one dataset and write data/interim/<dataset_key>.parquet."""
    entry = cfg.datasets[dataset_key]
    adapter: DatasetAdapter = get_adapter(entry.adapter, cfg)

    log.info("dataset discovered, loading raw files", dataset=dataset_key)
    raw = adapter.load_raw(sample_size=sample_size)
    log.info("raw rows loaded", rows=len(raw))

    cleaned, cleaning_report = clean_canonical_frame(raw)

    labeled = apply_stage_mapping(cleaned, cfg, dataset_key)
    labeled = labeled.sort_values("timestamp").reset_index(drop=True)

    out_path = cfg.path(cfg.config.paths.data_interim) / f"{dataset_key}.parquet"
    _write_parquet(labeled, out_path)
    log.info("wrote interim parquet", path=str(out_path), rows=len(labeled))
    write_ingest_summary(out_path, rows_raw=len(raw), rows_clean=len(labeled),
                         dropped_by_reason=cleaning_report.dropped_by_reason, sample_size=sample_size)

    class_dist = labeled["normalized_attack_class"].value_counts(dropna=False).to_dict()
    stage_dist = labeled["attack_stage"].value_counts(dropna=False).to_dict()
    log.info("class distribution", **{str(k): int(v) for k, v in class_dist.items()})

    return IngestResult(
        dataset_key=dataset_key,
        rows_raw=len(raw),
        cleaning=cleaning_report,
        output_path=out_path,
        class_distribution={str(k): int(v) for k, v in class_dist.items()},
        stage_distribution={str(k): int(v) for k, v in stage_dist.items()},
    )
