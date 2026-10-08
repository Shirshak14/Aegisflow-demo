from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aegisflow.config import DotDict
from aegisflow.errors import DatasetNotFoundError
from aegisflow.ml.datasets import get_adapter
from aegisflow.ml.datasets.ctu_13 import Ctu13Adapter, label_family, parse_port, scenario_of
from aegisflow.ml.preprocessing.cleaning import clean_canonical_frame
from aegisflow.ml.preprocessing.labels import apply_stage_mapping

FIXTURES = Path(__file__).parent / "fixtures" / "ctu_13"


@pytest.fixture()
def ctu_entry(cfg) -> DotDict:
    entry = DotDict(dict(cfg.datasets.ctu_13))
    entry["raw_dir"] = str(FIXTURES)
    return entry


def _only_plain(entry: DotDict) -> DotDict:
    entry["scenarios"] = [11]
    return entry


def test_default_dataset_is_still_cic(cfg):
    assert cfg.config.active_dataset == "cic_ids2017"
    assert isinstance(get_adapter("ctu_13", cfg), Ctu13Adapter)


def test_discover_finds_plain_and_gzipped_files(cfg, ctu_entry):
    report = Ctu13Adapter(cfg=cfg, entry=ctu_entry).discover()
    assert report.ok
    assert sorted(scenario_of(f) for f in report.files) == [11, 12]


def test_scenario_filter_and_missing_dir(cfg, ctu_entry):
    assert len(Ctu13Adapter(cfg=cfg, entry=_only_plain(ctu_entry)).discover().files) == 1
    ctu_entry["raw_dir"] = "tests/fixtures/does_not_exist"
    adapter = Ctu13Adapter(cfg=cfg, entry=ctu_entry)
    assert adapter.discover().found is False
    with pytest.raises(DatasetNotFoundError):
        adapter.load_raw()


def test_scenario_names():
    assert scenario_of(Path("capture20110810.binetflow")) == 1
    assert scenario_of(Path("scenario_13_botnet_54.binetflow.gz")) == 13
    assert scenario_of(Path("CTU-Malware-Capture-Botnet-52.binetflow")) == 11
    assert scenario_of(Path("other.binetflow")) is None


def test_load_raw_maps_fields(cfg, ctu_entry):
    df = Ctu13Adapter(cfg=cfg, entry=_only_plain(ctu_entry)).load_raw()
    assert len(df) == 7
    assert (df["source_dataset"] == "ctu_13").all()
    assert (df["ctu_scenario"] == 11).all()
    first = df.iloc[0]
    assert first["timestamp"] == pd.Timestamp("2011-08-18 15:39:35.087798")
    assert first["protocol"] == "TCP"
    assert first["flow_duration"] == pytest.approx(83.062141)
    assert first["packet_count"] == 43065 and first["byte_count"] == 40974671
    assert first["fwd_byte_count"] == 1033777 and first["bwd_byte_count"] == 40974671 - 1033777
    assert first["packet_size_mean"] == pytest.approx(40974671 / 43065)
    # RPA_FPA: SYN never seen, ACK/PSH/FIN/RST seen
    assert (first["syn_count"], first["ack_count"], first["psh_count"], first["fin_count"], first["rst_count"]) == (0, 1, 1, 1, 1)


def test_ports_flags_and_absent_fields(cfg, ctu_entry):
    df = Ctu13Adapter(cfg=cfg, entry=_only_plain(ctu_entry)).load_raw()
    icmp = df[df["protocol"] == "ICMP"].iloc[0]
    assert icmp["source_port"] == 0x0008 and icmp["destination_port"] == 0x9A13
    assert icmp["syn_count"] == 0  # flags only for TCP
    arp = df[df["protocol"] == "ARP"].iloc[0]
    assert pd.isna(arp["source_port"]) and pd.isna(arp["destination_port"])
    zero = df[df["packet_count"] == 2].iloc[-1]
    assert zero["byte_count"] == 0
    for col in ("fwd_packet_count", "bwd_packet_count", "packet_size_std", "inter_arrival_mean",
                "tcp_window_size", "ttl_mean", "retransmission_count"):
        assert df[col].isna().all(), col


def test_label_family_and_stage_mapping(cfg, ctu_entry):
    fam = label_family(pd.Series(["flow=From-Botnet-V42-TCP-CC6", "flow=To-Normal-V42-Grill",
                                  "flow=Background-UDP-Established", "flow=To-Background"]))
    assert fam.tolist() == ["Botnet", "Normal", "Background", "Background"]
    df = Ctu13Adapter(cfg=cfg, entry=ctu_entry).load_raw()
    assert df["ctu_label"].str.startswith("flow=").all()
    cleaned, _ = clean_canonical_frame(df)
    labeled = apply_stage_mapping(cleaned, cfg, "ctu_13")
    bot = labeled[labeled["dataset_label"] == "Botnet"]
    assert (bot["normalized_attack_class"] == "Botnet").all()
    assert (bot["attack_stage"] == "Command and Control").all()
    other = labeled[labeled["dataset_label"] != "Botnet"]
    assert (other["normalized_attack_class"] == "Benign").all()
    bg = labeled[labeled["dataset_label"] == "Background"]
    assert (bg["label_confidence"] == "inferred").all()


def test_gzip_matches_plain_and_sampling_is_deterministic(cfg, ctu_entry):
    adapter = Ctu13Adapter(cfg=cfg, entry=ctu_entry)
    df = adapter.load_raw()
    plain, gz = df[df["ctu_scenario"] == 11], df[df["ctu_scenario"] == 12]
    cols = ["timestamp", "source_ip", "byte_count", "dataset_label"]
    assert plain[cols].reset_index(drop=True).equals(gz[cols].reset_index(drop=True))
    a, b = adapter.load_raw(sample_size=6), adapter.load_raw(sample_size=6)
    assert len(a) == 6 and a["timestamp"].tolist() == b["timestamp"].tolist()


def test_parse_port():
    out = parse_port(pd.Series(["80", "0x0303", "", " 443 "]))
    assert out.tolist()[:2] == [80, 0x0303] and pd.isna(out.iloc[2]) and out.iloc[3] == 443
