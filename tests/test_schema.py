from __future__ import annotations

import pandas as pd
import pytest

from aegisflow.errors import SchemaError
from aegisflow.schema import (
    REQUIRED_COLUMNS,
    coerce_canonical_frame,
    empty_canonical_frame,
    validate_canonical_frame,
)


def test_empty_canonical_frame_has_all_required_columns():
    df = empty_canonical_frame()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_validate_rejects_missing_required_column():
    df = empty_canonical_frame().drop(columns=["source_ip"])
    df.loc[0] = None
    with pytest.raises(SchemaError):
        validate_canonical_frame(df)


def test_validate_rejects_empty_frame():
    df = empty_canonical_frame()
    with pytest.raises(SchemaError):
        validate_canonical_frame(df)


def test_coerce_fills_optional_columns_with_na_not_zero():
    df = empty_canonical_frame()
    df.loc[0] = pd.Series(dict(
        timestamp=pd.Timestamp("2017-01-01"),
        source_ip="1.2.3.4",
        destination_ip="5.6.7.8",
        protocol="TCP",
        flow_duration=1.0,
        packet_count=1.0,
        byte_count=1.0,
        dataset_label="BENIGN",
    ))
    out = coerce_canonical_frame(df.drop(columns=["ttl_mean"]))
    assert "ttl_mean" in out.columns
    assert out["ttl_mean"].isna().all()
