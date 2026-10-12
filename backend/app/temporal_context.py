"""Temporal context of a replayed sequence: how much real time its input covers and how far ahead its target is.

Windows only exist when a host sent traffic (windowing.min_flows_per_window), so "10 input windows" can cover
5.5 minutes or several hours, and "the next window" can start 30 seconds or days after the input ends. This
module reports those spans for one sequence, read from the sequence table (no model, no labels):

  input span       seq_end_time - seq_start_time (first input window start to last input window end)
  silent windows   window slots at the configured stride inside that span with no traffic, so no window
                   (expected slots - sequence_length; 0 means the input is contiguous)
  previous gap     start of the last input window - start of the one before it ("time since the previous
                   observation" when the forecast is made)
  lead             target_window_start - seq_end_time (how far past the input the forecast reaches; the
                   shortest possible is one stride, because the target must start strictly after the input ends)

Per-window start times come from the host's other sequences: sequence ``<host>_seq_<k>`` starts at the host's
k-th window, so window k's start is that sequence's seq_start_time; the last input window starts at
seq_end_time - window size. A window with no sequence of its own (near a host's last windows) is reported as
unknown rather than guessed.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegisflow.config import load_config

from . import replay as _replay

_COLUMNS = ("sequence_id", "host_id", "seq_start_time", "seq_end_time", "target_window_start", "target_window_end",
            "split")
_SEQ_INDEX = re.compile(r"_seq_(\d+)$")
_lock = threading.Lock()


def _windowing() -> tuple[float, float, int]:
    w = load_config().config.windowing
    return float(w.window_size_seconds), float(w.stride_seconds), int(w.sequence_length)


@lru_cache(maxsize=2)
def _table(path: str, mtime_ns: int) -> dict[str, Any]:
    import pyarrow.parquet as pq
    present = set(pq.read_schema(path).names)
    f = pd.read_parquet(path, columns=[c for c in _COLUMNS if c in present])
    for c in ("seq_start_time", "seq_end_time", "target_window_start", "target_window_end"):
        if c in f:
            f[c] = pd.to_datetime(f[c])
    f["sequence_id"] = f["sequence_id"].astype(str)
    f["host_id"] = f["host_id"].astype(str)
    local = f["sequence_id"].str.extract(_SEQ_INDEX, expand=False)
    starts: dict[tuple[str, int], pd.Timestamp] = {}
    if "seq_start_time" in f:
        ok = local.notna()
        starts = dict(zip(zip(f.loc[ok, "host_id"], local[ok].astype(int)), f.loc[ok, "seq_start_time"]))
    test = f[f["split"] == "test"] if "split" in f else f
    return {"rows": f.set_index("sequence_id", drop=False), "starts": starts, "summary": _summary(test)}


def _load() -> dict[str, Any]:
    path = Path(_replay.DATA_DIR) / "sequences.parquet"
    with _lock:
        return _table(str(path), path.stat().st_mtime_ns)


def _seconds(a, b) -> float | None:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return float((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds())


def _dist(values: pd.Series) -> dict[str, float] | None:
    v = values.dropna().to_numpy(dtype=float)
    if not len(v):
        return None
    return {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(v.max())}


def _summary(test: pd.DataFrame) -> dict[str, Any]:
    """How variable the horizon is across the whole test split (what one alert's numbers are compared with)."""
    size, stride, length = _windowing()
    out: dict[str, Any] = {"sequences": int(len(test)), "window_seconds": size, "stride_seconds": stride,
                           "sequence_length": length, "contiguous_span_seconds": size + stride * (length - 1)}
    if {"seq_end_time", "target_window_start"} <= set(test.columns):
        lead = (test["target_window_start"] - test["seq_end_time"]).dt.total_seconds()
        out["lead_seconds"] = _dist(lead)
        out["lead_over_1h_share"] = float((lead > 3600).mean()) if len(lead) else None
        out["_lead"] = np.sort(lead.dropna().to_numpy(dtype=float))
    if {"seq_start_time", "seq_end_time"} <= set(test.columns):
        span = (test["seq_end_time"] - test["seq_start_time"]).dt.total_seconds()
        out["span_seconds"] = _dist(span)
        out["with_silent_windows_share"] = float((span > out["contiguous_span_seconds"] + 1e-6).mean()) if len(span) else None
        out["_span"] = np.sort(span.dropna().to_numpy(dtype=float))
    return out


def _rank(sorted_values: np.ndarray | None, v: float | None) -> float | None:
    """Share of test sequences with a value <= v."""
    if sorted_values is None or v is None or not len(sorted_values):
        return None
    return float(np.searchsorted(sorted_values, v, side="right") / len(sorted_values))


def split_summary() -> dict[str, Any]:
    return {k: v for k, v in _load()["summary"].items() if not k.startswith("_")}


def context(sequence_id: str) -> dict[str, Any] | None:
    """Temporal context of one sequence, or None if it is not in the sequence table."""
    t = _load()
    rows = t["rows"]
    if sequence_id not in rows.index:
        return None
    r = rows.loc[sequence_id]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]
    size, stride, length = _windowing()
    s, e = r.get("seq_start_time"), r.get("seq_end_time")
    ts, te = r.get("target_window_start"), r.get("target_window_end")
    span, lead = _seconds(s, e), _seconds(e, ts)
    expected = None if span is None else int(round((span - size) / stride)) + 1
    m = _SEQ_INDEX.search(sequence_id)
    window_starts: list[str | None] = []
    gaps: list[float | None] = []
    if m and t["starts"]:
        k0 = int(m.group(1))
        ws = [t["starts"].get((str(r["host_id"]), k)) for k in range(k0, k0 + length)]
        if e is not None and not pd.isna(e):  # windows have a fixed length, so the last one starts at seq_end - size
            ws[-1] = pd.Timestamp(e) - pd.Timedelta(seconds=size)
        window_starts = [None if w is None else pd.Timestamp(w).isoformat() for w in ws]
        gaps = [_seconds(a, b) for a, b in zip(ws[:-1], ws[1:])]
    summary = t["summary"]
    return {
        "sequence_id": sequence_id, "host_id": str(r["host_id"]),
        "input_start": None if s is None or pd.isna(s) else pd.Timestamp(s).isoformat(),
        "input_end": None if e is None or pd.isna(e) else pd.Timestamp(e).isoformat(),
        "input_span_seconds": span,
        "observed_windows": length,
        "expected_windows": expected,
        "silent_windows": None if expected is None else max(0, expected - length),
        "contiguous": None if expected is None else expected <= length,
        "window_starts": window_starts,
        "previous_gap_seconds": gaps[-1] if gaps else None,
        "largest_gap_seconds": max(gaps) if gaps and all(g is not None for g in gaps) else None,
        "target_window_start": None if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat(),
        "target_window_end": None if te is None or pd.isna(te) else pd.Timestamp(te).isoformat(),
        "lead_seconds": lead,
        "min_lead_seconds": stride,
        "lead_rank_in_test": _rank(summary.get("_lead"), lead),
        "span_rank_in_test": _rank(summary.get("_span"), span),
        "window_seconds": size, "stride_seconds": stride,
    }
