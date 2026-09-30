from __future__ import annotations

import numpy as np
import pandas as pd

from aegisflow.ml.preprocessing.cleaning import clean_canonical_frame
from aegisflow.schema import empty_canonical_frame


def _row(**overrides):
    base = dict(
        timestamp=pd.Timestamp("2017-01-01"),
        source_ip="1.1.1.1",
        destination_ip="2.2.2.2",
        source_port=1234,
        destination_port=80,
        protocol="TCP",
        flow_duration=1.0,
        packet_count=10.0,
        byte_count=100.0,
        dataset_label="BENIGN",
    )
    base.update(overrides)
    return base


def test_drops_missing_timestamp():
    df = empty_canonical_frame()
    df.loc[0] = pd.Series(_row())
    df.loc[1] = pd.Series(_row(timestamp=pd.NaT))
    out, report = clean_canonical_frame(df)
    assert len(out) == 1
    assert report.dropped_by_reason["missing_timestamp"] == 1


def test_drops_infinite_values():
    df = empty_canonical_frame()
    df.loc[0] = pd.Series(_row())
    df.loc[1] = pd.Series(_row(source_port=9999, byte_count=np.inf))
    out, report = clean_canonical_frame(df)
    assert len(out) == 1
    assert report.dropped_by_reason["infinite_values"] == 1
    assert report.dropped_by_reason["exact_duplicate_flow"] == 0


def test_drops_exact_duplicates():
    df = empty_canonical_frame()
    df.loc[0] = pd.Series(_row())
    df.loc[1] = pd.Series(_row())
    out, report = clean_canonical_frame(df)
    assert len(out) == 1
    assert report.dropped_by_reason["exact_duplicate_flow"] == 1


def test_keeps_valid_rows_untouched():
    df = empty_canonical_frame()
    df.loc[0] = pd.Series(_row())
    df.loc[1] = pd.Series(_row(source_ip="9.9.9.9"))
    out, report = clean_canonical_frame(df)
    assert len(out) == 2
    assert sum(report.dropped_by_reason.values()) == 0
