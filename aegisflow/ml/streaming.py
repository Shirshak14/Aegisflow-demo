"""Streaming telemetry: score host traffic window by window as flows arrive.

Flows (canonical schema, any source: PCAP, NetFlow, CICFlowMeter, a collector) are pushed in time
order, in chunks of any size. A window [t0 + k*stride, t0 + k*stride + window_size) is CLOSED once the
watermark (latest flow time seen minus ``allowed_lateness``) passes its end; it is then aggregated with
exactly the batch code (``aggregate_host_windows``), appended to each host's window history, and every
host that now has ``sequence_length`` windows gets its newest sequence scored immediately.

Fed the same flows, the stream emits the same sequences and probabilities as the batch path
(``scoring.score_flows``) when flows arrive in order (tested). Differences, by design:
  * flows older than the earliest still-open window are counted as ``late_flows`` and dropped;
  * duplicates are removed within each pushed chunk, not across the whole stream;
  * memory is bounded: only flows of open windows and the last ``sequence_length`` windows per host
    are kept.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AegisFlowConfig
from .scoring import LSTMScorer, prepare_flows
from .temporal.windowing import WindowingConfig, aggregate_host_windows


class StreamScorer:
    def __init__(self, model_dir: str | Path, cfg: AegisFlowConfig, *, allowed_lateness: float = 0.0,
                 threshold: float | None = None, keep_results: int = 1000):
        self.scorer = LSTMScorer(model_dir, threshold)
        w = WindowingConfig.from_config(cfg)
        self.window = float(w.window_size_seconds)
        self.stride = float(w.stride_seconds)
        self.host_col = w.group_by
        self.min_flows = int(w.min_flows_per_window)
        self.lateness = float(allowed_lateness)
        self.L = self.scorer.sequence_length
        self._lock = threading.Lock()
        self.results: deque[dict[str, Any]] = deque(maxlen=keep_results)
        self.reset()

    def reset(self) -> None:
        self.t0: pd.Timestamp | None = None
        self.next_k = 0                       # first window index not yet closed
        self.watermark: pd.Timestamp | None = None
        self.buffer = pd.DataFrame()
        self.history: dict[str, deque] = {}   # host -> last L window feature rows
        self.windows_closed = self.flows_seen = self.late_flows = self.sequences_scored = self.alerts = 0
        self.results.clear()

    # ------------------------------------------------------------------ internals
    def _start(self, k: int) -> pd.Timestamp:
        return self.t0 + pd.Timedelta(seconds=k * self.stride)

    def _close_ready(self, final: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.t0 is None:
            return out
        if final:
            if self.buffer.empty:
                return out
            last = self.buffer["timestamp"].max()
            # last window that contains the newest flow
            limit = int(math.floor((last - self.t0).total_seconds() / self.stride))
        else:
            limit = int(math.floor(((self.watermark - self.t0).total_seconds() - self.window) / self.stride))
        while self.next_k <= limit:
            out.extend(self._close_window(self.next_k))
            self.next_k += 1
        keep_from = self._start(self.next_k)
        self.buffer = self.buffer[self.buffer["timestamp"] >= keep_from]
        return out

    def _close_window(self, k: int) -> list[dict[str, Any]]:
        start = self._start(k); end = start + pd.Timedelta(seconds=self.window)
        self.windows_closed += 1
        sl = self.buffer[(self.buffer["timestamp"] >= start) & (self.buffer["timestamp"] < end)]
        if sl.empty:
            return []
        # One window covering the whole slice: the batch aggregation, membership fixed by [start, end).
        win = aggregate_host_windows(sl, window_size_seconds=self.window, stride_seconds=self.window,
                                     min_flows_per_window=self.min_flows, group_by=self.host_col)
        if win.empty:
            return []
        win = win.assign(window_start=start, window_end=end)
        win[self.scorer.feature_names] = win[self.scorer.feature_names].fillna(0.0)
        ready, meta = [], []
        for row in win.itertuples(index=False):
            host = str(row.host_id)
            hist = self.history.setdefault(host, deque(maxlen=self.L))
            hist.append((start, end, np.array([getattr(row, f) for f in self.scorer.feature_names], np.float32)))
            if len(hist) == self.L:
                ready.append(np.stack([h[2] for h in hist]))
                meta.append((host, hist[0][0], hist[-1][1]))
        if not ready:
            return []
        p = self.scorer.probabilities(np.stack(ready))
        out = []
        for (host, s, e), prob in zip(meta, p):
            r = {"host_id": host, "seq_start_time": s.isoformat(), "seq_end_time": e.isoformat(),
                 "attack_probability": float(prob), "predicted_attack": bool(prob >= self.scorer.threshold),
                 "threshold": self.scorer.threshold}
            out.append(r)
            self.sequences_scored += 1
            self.alerts += int(r["predicted_attack"])
            self.results.append(r)
        return out

    # ------------------------------------------------------------------ API
    def push(self, flows: pd.DataFrame) -> list[dict[str, Any]]:
        """Add canonical flows; returns the sequences scored because windows closed."""
        if flows is None or len(flows) == 0:
            return []
        f = prepare_flows(flows)
        with self._lock:
            self.flows_seen += len(f)
            if self.t0 is None:
                self.t0 = f["timestamp"].min()
                self.watermark = self.t0
            late = f["timestamp"] < self._start(self.next_k)
            self.late_flows += int(late.sum())
            f = f[~late]
            if f.empty:
                return []
            self.buffer = pd.concat([self.buffer, f], ignore_index=True) if not self.buffer.empty else f.reset_index(drop=True)
            self.watermark = max(self.watermark, f["timestamp"].max() - pd.Timedelta(seconds=self.lateness))
            return self._close_ready(final=False)

    def flush(self) -> list[dict[str, Any]]:
        """End of stream: close every window that still holds flows."""
        with self._lock:
            return self._close_ready(final=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"t0": None if self.t0 is None else self.t0.isoformat(),
                    "watermark": None if self.watermark is None else self.watermark.isoformat(),
                    "next_window_start": None if self.t0 is None else self._start(self.next_k).isoformat(),
                    "flows_seen": self.flows_seen, "late_flows": self.late_flows,
                    "buffered_flows": int(len(self.buffer)), "windows_closed": self.windows_closed,
                    "hosts_tracked": len(self.history), "sequences_scored": self.sequences_scored,
                    "alerts": self.alerts, "threshold": self.scorer.threshold,
                    "window_size_seconds": self.window, "stride_seconds": self.stride,
                    "sequence_length": self.L, "allowed_lateness_seconds": self.lateness}
