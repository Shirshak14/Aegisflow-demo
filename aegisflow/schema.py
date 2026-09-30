"""
Canonical network-flow schema.

Every dataset adapter must produce a DataFrame containing (at minimum) the
REQUIRED_COLUMNS below, using exactly these names and dtypes. Columns in
OPTIONAL_COLUMNS are filled with pandas.NA by adapters that cannot supply
them -- the feature pipeline is required to handle that gracefully (see
ml/features/registry.py), never to silently invent values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import SchemaError

# column_name -> pandas dtype string
REQUIRED_COLUMNS: dict[str, str] = {
    "timestamp": "datetime64[ns]",
    "source_ip": "string",
    "destination_ip": "string",
    "source_port": "Int64",
    "destination_port": "Int64",
    "protocol": "string",
    "flow_duration": "float64",
    "packet_count": "float64",
    "byte_count": "float64",
    "dataset_label": "string",           # raw label exactly as it appears in the source dataset
}

OPTIONAL_COLUMNS: dict[str, str] = {
    "packet_size_mean": "float64",
    "packet_size_std": "float64",
    "packet_size_min": "float64",
    "packet_size_max": "float64",
    "inter_arrival_mean": "float64",
    "inter_arrival_std": "float64",
    "syn_count": "float64",
    "ack_count": "float64",
    "rst_count": "float64",
    "fin_count": "float64",
    "psh_count": "float64",
    "urg_count": "float64",
    "ttl_mean": "float64",
    "ttl_std": "float64",
    "tcp_window_size": "float64",
    "retransmission_count": "float64",
    "fwd_packet_count": "float64",
    "bwd_packet_count": "float64",
    "fwd_byte_count": "float64",
    "bwd_byte_count": "float64",
}

# columns added later in the pipeline (label mapping, windowing) -- not required from adapters
DERIVED_COLUMNS: dict[str, str] = {
    "normalized_attack_class": "string",
    "attack_stage": "string",
    "label_confidence": "string",   # "ground_truth" | "inferred"
    "source_dataset": "string",
}

ALL_KNOWN_COLUMNS = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS, **DERIVED_COLUMNS}


def validate_canonical_frame(df: pd.DataFrame, *, context: str = "") -> None:
    """Raise SchemaError if ``df`` is missing required canonical columns or has bad dtypes.

    This does not mutate ``df``. Call :func:`coerce_canonical_frame` first if
    you need dtypes fixed up.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{context + ': ' if context else ''}canonical frame is missing required columns: {missing}. "
            f"Every dataset adapter must populate all of REQUIRED_COLUMNS."
        )
    if df.empty:
        raise SchemaError(f"{context + ': ' if context else ''}canonical frame has zero rows.")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise SchemaError(f"{context + ': ' if context else ''}'timestamp' column must be datetime64.")


def coerce_canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing OPTIONAL_COLUMNS as all-NA and coerce dtypes where safe.

    Never invents optional feature *values* -- only ensures the column exists
    so downstream code can check ``.notna()`` uniformly instead of branching
    on column presence.
    """
    df = df.copy()
    for col, dtype in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            # float64 has no pd.NA sentinel -- use np.nan; other dtypes (Int64/string,
            # both pandas nullable extension types) accept pd.NA directly.
            fill = np.nan if dtype == "float64" else pd.NA
            df[col] = pd.array([fill] * len(df), dtype=dtype)
    for col in ("source_ip", "destination_ip", "protocol", "dataset_label"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in ("source_port", "destination_port"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    numeric_cols = [c for c, dt in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items() if dt == "float64"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def empty_canonical_frame() -> pd.DataFrame:
    """Return a zero-row frame with every known canonical column and correct dtype.

    Useful for tests and for adapters to build up via concatenation.
    """
    data = {}
    for col, dtype in ALL_KNOWN_COLUMNS.items():
        if dtype == "datetime64[ns]":
            data[col] = pd.Series([], dtype="datetime64[ns]")
        elif dtype == "Int64":
            data[col] = pd.Series([], dtype="Int64")
        elif dtype == "string":
            data[col] = pd.Series([], dtype="string")
        else:
            data[col] = pd.Series([], dtype="float64")
    return pd.DataFrame(data)
