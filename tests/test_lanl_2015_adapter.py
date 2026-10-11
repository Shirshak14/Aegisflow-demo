from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aegisflow.config import DotDict
from aegisflow.errors import DatasetNotFoundError
from aegisflow.ml.datasets import get_adapter
from aegisflow.ml.datasets.lanl_2015 import Lanl2015Adapter, load_redteam, victim_onsets
from aegisflow.ml.preprocessing.labels import apply_stage_mapping

FIXTURES = Path(__file__).parent / "fixtures" / "lanl"


@pytest.fixture()
def lanl_entry(cfg) -> DotDict:
    entry = DotDict(dict(cfg.datasets.lanl_2015))
    entry["raw_dir"] = str(FIXTURES)
    return entry


def test_default_dataset_is_still_cic(cfg):
    assert cfg.config.active_dataset == "cic_ids2017"
    assert isinstance(get_adapter("lanl_2015", cfg), Lanl2015Adapter)


def test_missing_dir(cfg, lanl_entry):
    lanl_entry["raw_dir"] = "tests/fixtures/does_not_exist"
    adapter = Lanl2015Adapter(cfg=cfg, entry=lanl_entry)
    assert adapter.discover().found is False
    with pytest.raises(DatasetNotFoundError):
        adapter.load_raw()


def test_victim_onsets():
    on = victim_onsets(load_redteam(FIXTURES / "redteam.txt.gz"))
    assert on["computer"].tolist() == ["C1003", "C728"]
    assert on.set_index("computer").loc["C728", "red_auths"] == 2


def test_load_raw_fields_and_labels(cfg, lanl_entry):
    df = Lanl2015Adapter(cfg=cfg, entry=lanl_entry).load_raw()
    assert len(df) == 6
    first = df.iloc[0]
    assert first["timestamp"] == pd.Timestamp("1970-01-01 00:01:40")
    assert first["destination_port"] == 443 and pd.isna(first["source_port"])  # N10 is de-identified
    assert first["protocol"] == "TCP" and first["packet_size_mean"] == 500
    assert pd.isna(df.iloc[5]["packet_count"])
    # contact 885 s before the first red-team auth is inside the default 3600 s slack
    assert df["dataset_label"].tolist() == ["Benign", "RedTeam", "Benign", "RedTeam", "RedTeamSource", "Benign"]
    mapped = apply_stage_mapping(df, cfg, dataset_key="lanl_2015")
    assert set(mapped["normalized_attack_class"]) == {"APT", "Benign"}


def test_filters(cfg, lanl_entry):
    lanl_entry["computers"] = ["C5"]
    assert len(Lanl2015Adapter(cfg=cfg, entry=lanl_entry).load_raw()) == 1
    lanl_entry["computers"] = None
    lanl_entry["time_range_s"] = [150000, 151000]
    assert len(Lanl2015Adapter(cfg=cfg, entry=lanl_entry).load_raw()) == 4
