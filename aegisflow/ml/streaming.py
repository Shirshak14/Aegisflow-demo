"""Streaming telemetry: score host traffic window by window as flows arrive.

Flows (canonical schema, any source: PCAP, NetFlow, CICFlowMeter, a collector) are pushed in time
order, in chunks of any size. A window [t0 + k*stride, t0 + k*stride + window_size) is CLOSED once the
watermark (latest flow time seen minus ``allowed_lateness``) passes its end; it is then aggregated with
exactly the batch code (``aggregate_host_windows``), appended to each host's window history, and every
host that now has ``sequence_length`` windows gets its newest sequence scored immediately.

Fed the same flows, the stream emits the same sequences and probabilities as the batch path
(``scoring.score_flows``) when flows arrive in order (tested). Behaviour at the edges, by design:

* Flow identity (``FLOW_IDENTITY``: timestamp, source/destination IP, source/destination port, protocol)
  is the same key the batch cleaner uses. A flow whose identity is already in the open windows, in the
  same chunk or an earlier one, is dropped and counted in ``duplicate_flows``. Identities are only kept
  while their flow is buffered; a repeat of a flow whose windows have all closed is late, so it is
  counted in ``late_flows`` instead (it cannot be told from any other late flow).
* A flow is LATE when it belongs to a window that has already closed, i.e. it is older than the end of
  the last closed window. Late flows are dropped and counted, never added to some windows and not
  others. ``allowed_lateness`` holds windows open that many seconds longer so out-of-order arrivals
  inside that margin are still aggregated.
* The window grid starts at the earliest timestamp of the first flows pushed. Until the first window
  closes, an earlier arrival moves the origin back, so the grid equals the batch grid.
* Grid slots are closed in order, so no slot is skipped silently. A slot in which NO host had traffic
  is an ``empty_windows`` (a feed outage looks like this); runs of them are listed in ``feed_gaps``.
  A host that is silent in some slots has non-consecutive windows; each scored sequence says how many
  silent slots it spans (``input_missing_windows``) and, with ``max_missing_windows``, such
  sequences are not scored (``sequences_skipped_gaps``). The default scores and flags them, as batch does.
* Memory is bounded: only flows of open windows and the last ``sequence_length`` windows per host.
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
from .preprocessing.cleaning import FLOW_IDENTITY
from .scoring import LSTMScorer, prepare_flows
from .temporal.sequences import sequence_continuity
from .temporal.windowing import WindowingConfig, aggregate_host_windows


class StreamScorer:
    def __init__(self, model_dir: str | Path, cfg: AegisFlowConfig, *, allowed_lateness: float = 0.0,
                 threshold: float | None = None, keep_results: int = 1000,
                 max_missing_windows: int | None = None):
        self.scorer = LSTMScorer(model_dir, threshold)
        w = WindowingConfig.from_config(cfg)
        self.window = float(w.window_size_seconds)
        self.stride = float(w.stride_seconds)
        self.host_col = w.group_by
        self.min_flows = int(w.min_flows_per_window)
        self.lateness = float(allowed_lateness)
        self.L = self.scorer.sequence_length
        self.max_missing = None if max_missing_windows is None else int(max_missing_windows)
        self._lock = threading.Lock()
        self.results: deque[dict[str, Any]] = deque(maxlen=keep_results)
        self.reset()

    def reset(self) -> None:
        self.t0: pd.Timestamp | None = None
        self.next_k = 0                       # first window index not yet closed
        self.watermark: pd.Timestamp | None = None
        self.buffer = pd.DataFrame()
        self.history: dict[str, deque] = {}   # host -> last L (slot, start, end, features)
        self.windows_closed = self.flows_seen = self.late_flows = self.sequences_scored = self.alerts = 0
        self.duplicate_flows = self.empty_windows = self.sequences_skipped_gaps = 0
        self.feed_gaps: deque[dict[str, Any]] = deque(maxlen=100)  # runs of grid slots with no traffic at all
        self._empty_run: list[int] = []
        self.results.clear()

    # ------------------------------------------------------------------ internals
    def _start(self, k: int) -> pd.Timestamp:
        return self.t0 + pd.Timedelta(seconds=k * self.stride)

    def _closed_until(self) -> pd.Timestamp:
        """End of the last closed window: a flow before it belongs to a window that is already scored."""
        if self.next_k == 0:
            return self.t0
        return self._start(self.next_k - 1) + pd.Timedelta(seconds=self.window)

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
            self.empty_windows += 1
            self._empty_run.append(k)
            return []
        self._end_empty_run()
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
                cont = sequence_continuity([h[0] for h in hist], [h[1] for h in hist], self.stride, self.window)
                if self.max_missing is not None and cont["input_missing_windows"] > self.max_missing:
                    self.sequences_skipped_gaps += 1
                    continue
                ready.append(np.stack([h[2] for h in hist]))
                meta.append((host, hist[0][0], hist[-1][1], cont))
        if not ready:
            return []
        p = self.scorer.probabilities(np.stack(ready))
        out = []
        for (host, s, e, cont), prob in zip(meta, p):
            cont = {**cont, "target_not_before": cont["target_not_before"].isoformat()}
            r = {"host_id": host, "seq_start_time": s.isoformat(), "seq_end_time": e.isoformat(),
                 "attack_probability": float(prob), "predicted_attack": bool(prob >= self.scorer.threshold),
                 "threshold": self.scorer.threshold, **cont}
            out.append(r)
            self.sequences_scored += 1
            self.alerts += int(r["predicted_attack"])
            self.results.append(r)
        return out

    def _end_empty_run(self) -> None:
        if self._empty_run:
            first, last = self._empty_run[0], self._empty_run[-1]
            self.feed_gaps.append({"from": self._start(first).isoformat(),
                                   "to": (self._start(last) + pd.Timedelta(seconds=self.stride)).isoformat(),
                                   "windows": len(self._empty_run)})
            self._empty_run = []

    # ------------------------------------------------------------------ API
    def push(self, flows: pd.DataFrame) -> list[dict[str, Any]]:
        """Add canonical flows; returns the sequences scored because windows closed."""
        if flows is None or len(flows) == 0:
            return []
        f = prepare_flows(flows)
        with self._lock:
            self.flows_seen += len(f)
            first = f["timestamp"].min()
            if self.t0 is None:
                self.t0 = first
                self.watermark = self.t0
            elif self.next_k == 0 and first < self.t0:
                self.t0 = first  # nothing closed yet: move the grid origin back to match the batch grid
            late = f["timestamp"] < self._closed_until()
            self.late_flows += int(late.sum())
            f = f[~late]
            if not self.buffer.empty and not f.empty:
                seen = pd.MultiIndex.from_frame(self.buffer[list(FLOW_IDENTITY)])
                dup = pd.MultiIndex.from_frame(f[list(FLOW_IDENTITY)]).isin(seen)
                self.duplicate_flows += int(dup.sum())
                f = f[~dup]
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
                    "flows_seen": self.flows_seen, "late_flows": self.late_flows, "duplicate_flows": self.duplicate_flows,
                    "empty_windows": self.empty_windows, "current_gap_windows": len(self._empty_run), "feed_gaps": list(self.feed_gaps),
                    "sequences_skipped_gaps": self.sequences_skipped_gaps,
                    "max_missing_windows": self.max_missing,
                    "buffered_flows": int(len(self.buffer)), "windows_closed": self.windows_closed,
                    "hosts_tracked": len(self.history), "sequences_scored": self.sequences_scored,
                    "alerts": self.alerts, "threshold": self.scorer.threshold,
                    "window_size_seconds": self.window, "stride_seconds": self.stride,
                    "sequence_length": self.L, "allowed_lateness_seconds": self.lateness}
