"""Abstract dataset adapter interface every dataset integration must implement."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import AegisFlowConfig, DotDict
from ...logging_setup import get_logger


@dataclass
class DiscoveryReport:
    """Result of checking whether a dataset's raw files are present and look right."""

    dataset_name: str
    raw_dir: Path
    found: bool
    files: list[Path]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return self.found and not self.problems


class DatasetAdapter(abc.ABC):
    """Base class for all dataset adapters.

    A concrete adapter turns whatever files the official dataset ships
    (CSV/PCAP/binetflow/...) into a DataFrame that matches
    ``aegisflow.schema`` exactly. Adapters must NEVER invent rows: if the
    raw files are missing, :meth:`discover` reports that clearly and
    :meth:`load_raw` raises ``DatasetNotFoundError``.
    """

    #: overridden by subclasses, used for log tags and error messages
    name: str = "base"

    def __init__(self, cfg: AegisFlowConfig, entry: DotDict) -> None:
        self.cfg = cfg
        self.entry = entry
        self.raw_dir = cfg.path(entry.raw_dir)
        self.log = get_logger(f"DATASET:{self.name}")

    # ------------------------------------------------------------------ #
    # Interface every adapter must implement
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def discover(self) -> DiscoveryReport:
        """Check whether raw files are present under ``self.raw_dir`` and look correct.

        Must NOT raise; return a DiscoveryReport describing what's wrong so
        the CLI can print actionable instructions.
        """

    @abc.abstractmethod
    def download_instructions(self) -> str:
        """Human-readable, copy-pasteable instructions for obtaining this dataset manually."""

    @abc.abstractmethod
    def load_raw(self, sample_size: int | None = None) -> pd.DataFrame:
        """Load raw files and return a DataFrame matching ``aegisflow.schema`` exactly.

        Raises:
            DatasetNotFoundError: if :meth:`discover` would report ``found=False``.
        """

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def expected_directory_structure(self) -> str:
        return (
            f"{self.raw_dir}/\n"
            f"    (files matching: {self.entry.get('expected_files_glob', '*')})"
        )
