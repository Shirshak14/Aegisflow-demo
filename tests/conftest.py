"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from aegisflow.config import AegisFlowConfig, DotDict, load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def cfg() -> AegisFlowConfig:
    return load_config()


@pytest.fixture()
def cic_ids2017_entry(cfg: AegisFlowConfig) -> DotDict:
    """A datasets.yaml-shaped entry pointed at the small CSV fixture instead of real data."""
    entry = DotDict(dict(cfg.datasets.cic_ids2017))
    entry["raw_dir"] = str(FIXTURES / "cic_ids2017")
    return entry
