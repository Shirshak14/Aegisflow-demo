from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from aegisflow.config import DotDict
from aegisflow.errors import DatasetNotFoundError
from aegisflow.ml.datasets import get_adapter
from aegisflow.ml.datasets.optc import OptcAdapter, host_key, load_ground_truth, parse_local_time
from aegisflow.ml.preprocessing.labels import apply_stage_mapping

FIXTURES = Path(__file__).parent / "fixtures" / "optc"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture()
def optc_entry(cfg) -> DotDict:
    entry = DotDict(dict(cfg.datasets.optc))
    entry["raw_dir"] = str(FIXTURES / "flows")
    entry["labels_dir"] = str(FIXTURES / "labels")
    return entry


def test_default_dataset_is_still_cic(cfg):
    assert cfg.config.active_dataset == "cic_ids2017"
    assert isinstance(get_adapter("optc", cfg), OptcAdapter)


def test_discover_and_missing_dir(cfg, optc_entry):
    assert OptcAdapter(cfg=cfg, entry=optc_entry).discover().ok
    optc_entry["raw_dir"] = "tests/fixtures/does_not_exist"
    adapter = OptcAdapter(cfg=cfg, entry=optc_entry)
    assert adapter.discover().found is False
    with pytest.raises(DatasetNotFoundError):
        adapter.load_raw()


def test_host_filter(cfg, optc_entry):
    optc_entry["hosts"] = [402]
    assert OptcAdapter(cfg=cfg, entry=optc_entry).discover().found is False


def test_host_key_and_time():
    assert host_key("SysClient0201.systemia.com") == "sysclient0201"
    assert host_key("DC1") == "dc1"
    t = parse_local_time(pd.Series(["2019-09-23T11:23:54.717-04:00", "2019-09-23T11:23:54-04:00"]))
    assert t.iloc[0] == pd.Timestamp("2019-09-23 11:23:54.717")
    assert t.iloc[1] == pd.Timestamp("2019-09-23 11:23:54")


def test_ground_truth_parsing():
    gt = load_ground_truth(FIXTURES / "labels")
    assert set(gt["host"]) == {"sysclient0201", "sysclient0402", "sysclient0104", "sysclient0170", "dc1"}
    assert gt.loc[gt["host"] == "dc1", "end"].isna().all()  # "Infinity" -> open-ended
    assert (gt["scenario"] == 1).all()


def test_load_raw_flows_bytes_and_labels(cfg, optc_entry):
    df = OptcAdapter(cfg=cfg, entry=optc_entry).load_raw()
    # f1, f2 (repeated START kept once), f4, f5; f3 is sensor telemetry to Kafka and is dropped
    assert len(df) == 4
    assert (df["source_dataset"] == "optc").all()
    f1 = df[df["destination_port"] == 443].iloc[0]
    assert f1["byte_count"] == 1500 and f1["protocol"] == "TCP" and f1["optc_direction"] == "outbound"
    assert pd.isna(df.loc[df["destination_port"] == 138, "byte_count"]).all()
    assert df["packet_count"].isna().all() and df["syn_count"].isna().all()
    labels = df.set_index("destination_port")["dataset_label"]
    # pid 1284 is red-team from 11:23:54 to 11:27:10: 11:24 is inside, 11:30 is after it ended
    assert labels.loc[8000].tolist() == ["Malicious", "Benign"]
    assert labels.loc[443] == "Benign"
    mapped = apply_stage_mapping(df, cfg, dataset_key="optc")
    assert set(mapped["normalized_attack_class"]) == {"APT", "Benign"}


def test_keep_telemetry(cfg, optc_entry):
    optc_entry["keep_telemetry"] = True
    assert len(OptcAdapter(cfg=cfg, entry=optc_entry).load_raw()) == 5


def test_census_lockstep_and_history(tmp_path):
    spec = importlib.util.spec_from_file_location("optc_onset_census", SCRIPTS / "optc_onset_census.py")
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)
    gt = load_ground_truth(FIXTURES / "labels")
    t = census.onset_table(gt)
    # 0104 and 0170 start 1 s apart: one lockstep event
    assert t["onset_event"].nunique() == 4
    assert t.loc[t["host"] == "sysclient0104", "batch_size"].item() == 2
    t = census.history_from_flows(t, FIXTURES / "flows", gt)
    row = t[t["host"] == "sysclient0201"].iloc[0]
    assert row["own_out_60m"] == 1 and row["own_out_10m"] == 0   # the 11:00 flow; telemetry dropped
    assert row["inbound_60m"] == 1
    assert row["red_flows_extracted"] == 1 and row["first_red_flow_lag_s"] == 6.0
