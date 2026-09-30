"""Standalone dataset validation, used by the CLI and importable for tests."""
from __future__ import annotations

from dataclasses import dataclass

from ...config import AegisFlowConfig
from ...logging_setup import get_logger
from . import get_adapter
from .base import DiscoveryReport

log = get_logger("VALIDATE")


@dataclass
class ValidationResult:
    dataset_key: str
    report: DiscoveryReport
    sample_load_ok: bool
    sample_load_error: str | None


def validate_dataset(cfg: AegisFlowConfig, dataset_key: str, *, try_sample_load: bool = True) -> ValidationResult:
    """Check discovery + (optionally) attempt a small real load to catch schema drift early."""
    entry = cfg.datasets[dataset_key]
    adapter = get_adapter(entry.adapter, cfg)
    report = adapter.discover()

    log.info("discovery", dataset=dataset_key, found=report.found, n_files=len(report.files), n_problems=len(report.problems))
    if not report.found:
        log.error("dataset not found", dataset=dataset_key)
        print(adapter.download_instructions())
        return ValidationResult(dataset_key, report, sample_load_ok=False, sample_load_error="dataset not found")

    if report.problems:
        for p in report.problems:
            log.warning("discovery problem", detail=p)

    sample_ok, sample_err = True, None
    if try_sample_load:
        try:
            df = adapter.load_raw(sample_size=200)
            log.info("sample load ok", rows=len(df), columns=len(df.columns))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as validation failure detail
            sample_ok, sample_err = False, str(exc)
            log.error("sample load failed", error=sample_err)

    return ValidationResult(dataset_key, report, sample_load_ok=sample_ok, sample_load_error=sample_err)
