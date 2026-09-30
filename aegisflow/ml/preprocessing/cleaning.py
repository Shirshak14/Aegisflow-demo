"""
Row-level cleaning of a canonical flow frame.

Every drop is counted and logged (never silent) so ingestion output always
shows: rows loaded -> rows dropped (by reason) -> rows kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...logging_setup import get_logger
from ...schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

log = get_logger("PREPROCESS")


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    dropped_by_reason: dict[str, int] = field(default_factory=dict)

    def log(self) -> None:
        log.info(
            "cleaning complete",
            rows_in=self.rows_in,
            rows_out=self.rows_out,
            rows_dropped=self.rows_in - self.rows_out,
            **{f"dropped_{k}": v for k, v in self.dropped_by_reason.items()},
        )


def clean_canonical_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop rows that cannot be used: missing timestamp/IPs, negative durations, inf values.

    Does not impute or fabricate values -- only removes rows that are
    structurally unusable, and reports exactly how many and why.
    """
    rows_in = len(df)
    dropped: dict[str, int] = {}
    out = df

    mask = out["timestamp"].notna()
    dropped["missing_timestamp"] = int((~mask).sum())
    out = out[mask]

    mask = out["source_ip"].notna() & (out["source_ip"].str.len() > 0)
    dropped["missing_source_ip"] = int((~mask).sum())
    out = out[mask]

    mask = out["destination_ip"].notna() & (out["destination_ip"].str.len() > 0)
    dropped["missing_destination_ip"] = int((~mask).sum())
    out = out[mask]

    mask = out["flow_duration"].fillna(0) >= 0
    dropped["negative_duration"] = int((~mask).sum())
    out = out[mask]

    # Coerce to numeric explicitly rather than trusting live column dtype:
    # a column can end up as 'object' dtype after certain assignment paths
    # even though schema.py declares it float64, which would silently hide
    # infinite values from a dtype-based select_dtypes() check.
    numeric_cols = [c for c, dt in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items() if dt == "float64" and c in out.columns]
    numeric_view = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    inf_mask = np.isinf(numeric_view).any(axis=1)
    dropped["infinite_values"] = int(inf_mask.sum())
    out = out[~inf_mask]

    before_dedup = len(out)
    out = out.drop_duplicates(
        subset=["timestamp", "source_ip", "destination_ip", "source_port", "destination_port", "protocol"]
    )
    dropped["exact_duplicate_flow"] = before_dedup - len(out)

    out = out.reset_index(drop=True)
    report = CleaningReport(rows_in=rows_in, rows_out=len(out), dropped_by_reason=dropped)
    report.log()
    return out, report
