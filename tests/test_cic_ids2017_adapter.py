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


# --------------------------------------------------------------------------------------------------
# Chunked, column-restricted reading must give exactly what the old whole-file read gave (B5).
def _old_whole_file_load(adapter, path: Path):
    """The pre-B5 reader, kept as the reference: whole file, every column, normalised headers."""
    import pandas as pd

    from aegisflow.ml.datasets.cic_ids2017 import _normalize_header
    from aegisflow.schema import coerce_canonical_frame

    df = pd.read_csv(path, low_memory=False, encoding="cp1252")
    df.columns = [_normalize_header(c) for c in df.columns]
    df = df[df["label"].astype(str).str.strip().str.lower() != "label"]
    df["__source_file__"] = path.name
    canonical = coerce_canonical_frame(adapter._to_canonical(df.reset_index(drop=True)))
    canonical["source_dataset"] = "cic_ids2017"
    return canonical


def test_chunked_usecols_read_equals_old_whole_file_read(tmp_path, cfg, cic_ids2017_entry, monkeypatch):
    import pandas as pd

    from aegisflow.ml.datasets import cic_ids2017 as mod

    header = ["Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port", " Protocol",
              " Timestamp", " Flow Duration", " Total Fwd Packets", "Total Length of Fwd Packets",
              " Fwd Header Length", " Bwd IAT Max", " Label"]  # last-but-two: columns the adapter never uses
    rows = []
    for i in range(7):  # integers in the early chunks ...
        rows.append([f"f{i}", "10.0.0.1", 1000 + i, "10.0.0.2", 80, 6, f"7/7/2017 9:0{i}", 1_000_000 * (i + 1),
                     3, 100 + i, 20, 5, "BENIGN"])
    rows.append(["f7", "10.0.0.1", 1007, "10.0.0.2", 80, 17, "7/7/2017 9:08", 2_500_000, 3, 100.5, 20, 5, "DDoS"])
    rows.append(["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
                 "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
                 "Fwd Header Length", "Bwd IAT Max", "Label"])  # re-embedded header row (mid-file)
    rows.append(["f8", "10.0.0.3", 1008, "10.0.0.2", 80, 6, "7/7/2017 9:09", 3_000_000, 3, 101, 20, 5, "BENIGN"])
    rows += [["", "", "", "", "", "", "", "", "", "", "", "", ""]] * 5  # trailing blank rows: an all-empty chunk
    path = tmp_path / "Day.pcap_ISCX.csv"
    pd.DataFrame(rows, columns=header).to_csv(path, index=False, encoding="cp1252")

    cic_ids2017_entry["raw_dir"] = str(tmp_path)
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    expected = _old_whole_file_load(adapter, path)
    monkeypatch.setattr(mod, "_CSV_CHUNK_ROWS", 2)  # many chunks, including all-empty ones at the end
    got = adapter.load_raw()
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


def test_only_used_columns_are_parsed(cfg, cic_ids2017_entry):
    from aegisflow.ml.datasets.cic_ids2017 import _COLUMN_MAP, _USED_HEADERS, _read_used_columns

    assert _USED_HEADERS == {k for k, v in _COLUMN_MAP.items() if v} and "flow id" not in _USED_HEADERS
    adapter = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry)
    path = adapter.discover().files[0]
    df = _read_used_columns(path, None)
    assert set(df.columns) <= _USED_HEADERS and "label" in df.columns
