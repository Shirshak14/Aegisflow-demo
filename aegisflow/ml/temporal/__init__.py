"""
Temporal data engineering and sequence representation package for AegisFlow.
"""
from .windowing import aggregate_host_windows
from .sequences import build_host_sequences
from .split import compute_temporal_splits, SplitMetadata
from .reporting import generate_data_quality_report
from .pipeline import run_preprocessing

__all__ = [
    "aggregate_host_windows",
    "build_host_sequences",
    "compute_temporal_splits",
    "SplitMetadata",
    "generate_data_quality_report",
    "run_preprocessing",
]
