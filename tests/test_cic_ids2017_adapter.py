from __future__ import annotations

from pathlib import Path

import pytest

from aegisflow.errors import DatasetNotFoundError
from aegisflow.ml.datasets.cic_ids2017 import CicIds2017Adapter


def test_discover_finds_fixture_file(cfg, cic_ids2017_entry):
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    report = adapter.discover()
    assert report.found is True
    assert len(report.files) == 1
    assert report.problems == []


def test_discover_reports_missing_directory(cfg, cic_ids2017_entry):
    cic_ids2017_entry["raw_dir"] = "tests/fixtures/does_not_exist"
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    report = adapter.discover()
    assert report.found is False
    assert report.problems


def test_load_raw_returns_canonical_columns(cfg, cic_ids2017_entry):
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    df = adapter.load_raw()
    assert len(df) == 9  # 10 data rows in the fixture minus 1 re-embedded header row
    assert set(["timestamp", "source_ip", "destination_ip", "dataset_label"]).issubset(df.columns)


def test_load_raw_drops_reembedded_header_row(cfg, cic_ids2017_entry):
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    df = adapter.load_raw()
    assert (df["dataset_label"] == "Label").sum() == 0


def test_load_raw_converts_duration_to_seconds(cfg, cic_ids2017_entry):
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    df = adapter.load_raw()
    # fixture's first row has Flow Duration = 1500000 microseconds = 1.5 seconds
    assert pytest.approx(df["flow_duration"].iloc[0], abs=1e-6) == 1.5


def test_load_raw_missing_dataset_raises(cfg, cic_ids2017_entry):
    cic_ids2017_entry["raw_dir"] = "tests/fixtures/does_not_exist"
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    with pytest.raises(DatasetNotFoundError):
        adapter.load_raw()


def test_ttl_columns_are_na_not_zero(cfg, cic_ids2017_entry):
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    df = adapter.load_raw()
    assert df["ttl_mean"].isna().all()


def _write_raw_copy(entry, tmp_path, name, timestamps):
    """Copy the fixture CSV under a CIC-IDS2017 file name with the given raw Timestamp strings."""
    import pandas as pd
    src = next(Path(entry["raw_dir"]).rglob("*.csv"))
    raw = pd.read_csv(src, encoding="cp1252")
    raw = raw[raw.iloc[:, -1].astype(str).str.strip() != "Label"].reset_index(drop=True)
    raw = raw.iloc[[i % len(raw) for i in range(len(timestamps))]].reset_index(drop=True)
    raw[next(c for c in raw.columns if c.strip() == "Timestamp")] = timestamps
    out_dir = tmp_path / "TrafficLabelling"
    out_dir.mkdir(exist_ok=True)
    raw.to_csv(out_dir / name, index=False)
    entry["raw_dir"] = str(tmp_path)


def test_afternoon_12_hour_timestamps_are_parsed_as_pm(cfg, cic_ids2017_entry, tmp_path):
    # Raw CIC-IDS2017 uses a 12-hour clock without AM/PM; hours < 8 are afternoon.
    import pandas as pd
    raw_ts = ["7/7/2017 1:30", "7/7/2017 3:30", "7/7/2017 5:30", "7/7/2017 8:59", "7/7/2017 12:15",
              "03/07/2017 08:55:58", "03/07/2017 01:02:03"]
    expected = ["2017-07-07 13:30", "2017-07-07 15:30", "2017-07-07 17:30", "2017-07-07 08:59",
                "2017-07-07 12:15", "2017-07-03 08:55:58", "2017-07-03 13:02:03"]
    _write_raw_copy(cic_ids2017_entry, tmp_path, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", raw_ts)
    df = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry).load_raw()
    assert sorted(pd.to_datetime(df["timestamp"])) == sorted(pd.Timestamp(t) for t in expected)


def test_spread_sample_plan_is_proportional_deterministic_and_spans_each_file():
    from aegisflow.ml.datasets.cic_ids2017 import spread_sample_plan
    counts = [1000, 3000, 0]
    plan = spread_sample_plan(counts, 400, seed=42)
    assert [len(p) for p in plan] == [100, 300, 0]
    for p, n in zip(plan[:2], counts[:2]):
        assert len(set(p.tolist())) == len(p) and (p[:-1] < p[1:]).all()
        assert p.min() < 0.1 * n and p.max() > 0.9 * n  # not a head slice
    again = spread_sample_plan(counts, 400, seed=42)
    assert all((a == b).all() for a, b in zip(plan, again))
    assert not (spread_sample_plan(counts, 400, seed=7)[1] == plan[1]).all()
    assert spread_sample_plan(counts, 10_000, seed=42) == [None, None, None]


def test_load_raw_sample_reads_throughout_each_file(cfg, cic_ids2017_entry, tmp_path):
    import pandas as pd
    base = pd.Timestamp("2017-07-04 09:00:00")
    stamps = [(base + pd.Timedelta(seconds=10 * i)).strftime("4/7/2017 %H:%M:%S") for i in range(200)]
    _write_raw_copy(cic_ids2017_entry, tmp_path, "Tuesday-WorkingHours.pcap_ISCX.csv", stamps)
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    a, b = adapter.load_raw(sample_size=40), adapter.load_raw(sample_size=40)
    assert len(a) == 40
    ts = pd.to_datetime(a["timestamp"])
    span = pd.Timedelta(seconds=10 * 199)
    assert ts.min() < base + span * 0.25 and ts.max() > base + span * 0.75  # not a head slice
    assert (pd.to_datetime(b["timestamp"]).to_numpy() == ts.to_numpy()).all()
    assert len(adapter.load_raw()) == 200


def test_sampling_does_not_exclude_late_file_attack_burst(cfg, cic_ids2017_entry, tmp_path):
    # Regression for the old per-file prefix read (nrows=N), which dropped attacks that start
    # late in a day-file (e.g. Bot in Friday-WorkingHours-Morning). Last 10% of rows are attacks.
    import pandas as pd
    from aegisflow.ml.datasets.cic_ids2017 import spread_sample_plan
    base = pd.Timestamp("2017-07-07 09:00:00")
    stamps = [(base + pd.Timedelta(seconds=10 * i)).strftime("7/7/2017 %H:%M:%S") for i in range(400)]
    _write_raw_copy(cic_ids2017_entry, tmp_path, "Friday-WorkingHours-Morning.pcap_ISCX.csv", stamps)
    csv = tmp_path / "TrafficLabelling" / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    raw = pd.read_csv(csv)
    raw[raw.columns[-1]] = ["BENIGN"] * 360 + ["Bot"] * 40
    raw.to_csv(csv, index=False)
    df = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry).load_raw(sample_size=80)
    assert len(df) == 80
    assert (df["dataset_label"].astype(str).str.strip() == "Bot").sum() > 0
    # And not systematically: across many seeds, the tail keeps roughly its 10% share.
    tail_share = [(p[0] >= 9_000).mean() for p in (spread_sample_plan([10_000], 500, s) for s in range(50))]
    assert 0.07 < sum(tail_share) / len(tail_share) < 0.13
