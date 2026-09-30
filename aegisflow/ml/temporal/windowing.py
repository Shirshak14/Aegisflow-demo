"""
Host-level temporal aggregation and configurable time-window generation.

Takes cleaned and canonically feature-engineered flows and groups them into
sliding host-level temporal windows [window_start, window_end).
Preserves strict chronological ordering and never allows future traffic into earlier windows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...config import AegisFlowConfig
from ...logging_setup import get_logger

log = get_logger("TEMPORAL_WINDOWING")


@dataclass
class WindowingConfig:
    window_size_seconds: float = 60.0
    stride_seconds: float = 30.0
    min_flows_per_window: int = 1
    group_by: str = "source_ip"

    @classmethod
    def from_config(cls, cfg: AegisFlowConfig, **overrides: Any) -> WindowingConfig:
        w_cfg = cfg.config.windowing
        params = {
            "window_size_seconds": float(overrides.get("window_size_seconds") or w_cfg.window_size_seconds),
            "stride_seconds": float(overrides.get("stride_seconds") or w_cfg.stride_seconds),
            "min_flows_per_window": int(overrides.get("min_flows_per_window") or w_cfg.min_flows_per_window),
            "group_by": str(overrides.get("group_by") or w_cfg.group_by),
        }
        return cls(**params)


def aggregate_host_windows(
    flows_df: pd.DataFrame,
    cfg: AegisFlowConfig | None = None,
    *,
    window_size_seconds: float | None = None,
    stride_seconds: float | None = None,
    min_flows_per_window: int | None = None,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Aggregate flow records into chronological host-level time windows.

    Parameters:
        flows_df: Cleaned canonical flows with timestamps and feature columns.
        cfg: Optional project configuration.
        window_size_seconds: Window duration in seconds (defaults to config: 60s).
        stride_seconds: Step size between window starts in seconds (defaults to config: 30s).
        min_flows_per_window: Minimum flows for a window to be retained (default: 1).
        group_by: Host identifier column (default: "source_ip").

    Returns:
        DataFrame where each row is a host-window observation with aggregated features
        and aggregated labels.
    """
    if flows_df.empty:
        log.warning("empty flows DataFrame passed to aggregate_host_windows")
        return pd.DataFrame()

    if cfg is not None:
        w_params = WindowingConfig.from_config(
            cfg,
            window_size_seconds=window_size_seconds,
            stride_seconds=stride_seconds,
            min_flows_per_window=min_flows_per_window,
            group_by=group_by,
        )
    else:
        w_params = WindowingConfig(
            window_size_seconds=float(window_size_seconds or 60.0),
            stride_seconds=float(stride_seconds or 30.0),
            min_flows_per_window=int(min_flows_per_window or 1),
            group_by=str(group_by or "source_ip"),
        )

    host_col = w_params.group_by
    if host_col not in flows_df.columns:
        raise ValueError(f"Configured group_by host column '{host_col}' not found in DataFrame.")

    # 1. Strict chronological sorting
    df = flows_df.sort_values("timestamp").reset_index(drop=True)
    df[host_col] = df[host_col].astype(str)

    w_size = w_params.window_size_seconds
    s_size = w_params.stride_seconds

    # Global time grid origin
    t0 = df["timestamp"].min()
    dt_seconds = (df["timestamp"] - t0).dt.total_seconds().values

    # Calculate active window indices [k_min, k_max] for each flow
    # A flow at offset dt is inside window k (start = t0 + k*s) if k*s <= dt < k*s + w
    # -> (dt - w)/s < k <= dt/s -> k_min = floor((dt - w)/s) + 1, k_max = floor(dt/s)
    k_min = np.maximum(0, np.floor((dt_seconds - w_size) / s_size).astype(int) + 1)
    k_max = np.floor(dt_seconds / s_size).astype(int)

    # Expand flows to active window assignments
    repeats = np.maximum(0, k_max - k_min + 1)
    if repeats.sum() == 0:
        return pd.DataFrame()

    idx_expanded = np.repeat(np.arange(len(df)), repeats)
    # Generate list of k values
    offsets = np.concatenate([np.arange(k1, k2 + 1) for k1, k2 in zip(k_min, k_max)])

    expanded = df.iloc[idx_expanded].copy()
    expanded["__window_idx__"] = offsets

    # Compute actual window start/end timestamps
    # window_start = t0 + k * s
    window_starts = t0 + pd.to_timedelta(offsets * s_size, unit="s")
    expanded["window_start"] = window_starts
    expanded["window_end"] = window_starts + pd.to_timedelta(w_size, unit="s")

    # Grouping key: [host_id, window_idx]
    expanded["host_id"] = expanded[host_col]

    # 2. Main Groupby Aggregations
    # Ensure numeric types
    for col in [
        "packet_count", "byte_count", "flow_duration",
        "packets_per_second", "bytes_per_second", "bytes_per_packet",
        "packet_size_mean", "packet_size_std", "packet_size_min", "packet_size_max",
        "inter_arrival_mean", "inter_arrival_std",
        "syn_count", "ack_count", "rst_count", "fin_count", "psh_count", "urg_count",
        "tcp_window_size"
    ]:
        if col in expanded.columns:
            expanded[col] = pd.to_numeric(expanded[col], errors="coerce")

    # Safe aggregations mapping
    agg_spec: dict[str, Any] = {
        "window_start": ("window_start", "first"),
        "window_end": ("window_end", "first"),
        "flow_count": ("timestamp", "count"),
        "packet_count_sum": ("packet_count", "sum"),
        "packet_count_mean": ("packet_count", "mean"),
        "byte_count_sum": ("byte_count", "sum"),
        "byte_count_mean": ("byte_count", "mean"),
        "duration_mean": ("flow_duration", "mean"),
        "duration_total": ("flow_duration", "sum"),
        "unique_destination_ips": ("destination_ip", "nunique"),
        "unique_destination_ports": ("destination_port", "nunique"),
        "unique_source_ports": ("source_port", "nunique"),
    }

    # Optional column aggregations if present in dataframe
    optional_aggs = [
        ("packets_per_second", "packets_per_second_mean", "mean"),
        ("bytes_per_second", "bytes_per_second_mean", "mean"),
        ("bytes_per_packet", "bytes_per_packet_mean", "mean"),
        ("packet_size_mean", "packet_size_mean", "mean"),
        ("packet_size_std", "packet_size_std_mean", "mean"),
        ("packet_size_min", "packet_size_min", "min"),
        ("packet_size_max", "packet_size_max", "max"),
        ("inter_arrival_mean", "inter_arrival_mean", "mean"),
        ("inter_arrival_std", "inter_arrival_std_mean", "mean"),
        ("syn_count", "syn_count_sum", "sum"),
        ("ack_count", "ack_count_sum", "sum"),
        ("rst_count", "rst_count_sum", "sum"),
        ("fin_count", "fin_count_sum", "sum"),
        ("psh_count", "psh_count_sum", "sum"),
        ("urg_count", "urg_count_sum", "sum"),
        ("tcp_window_size", "tcp_window_size_mean", "mean"),
    ]

    for src_col, target_col, func in optional_aggs:
        if src_col in expanded.columns:
            agg_spec[target_col] = (src_col, func)

    grouped = expanded.groupby(["host_id", "__window_idx__"])
    windows_df = grouped.agg(**agg_spec).reset_index()

    # Filter by min_flows_per_window
    windows_df = windows_df[windows_df["flow_count"] >= w_params.min_flows_per_window].copy()
    if windows_df.empty:
        return pd.DataFrame()

    # 3. Shannon Entropy of destination ports
    # Vectorized: count per [host_id, window_idx, destination_port]
    port_counts = expanded.groupby(["host_id", "__window_idx__", "destination_port"]).size().reset_index(name="p_count")
    port_totals = port_counts.groupby(["host_id", "__window_idx__"])["p_count"].transform("sum")
    p = port_counts["p_count"] / port_totals
    port_counts["p_entropy"] = -p * np.log2(p)
    entropy_df = port_counts.groupby(["host_id", "__window_idx__"])["p_entropy"].sum().reset_index(name="destination_port_entropy")

    # 4. Connection Burst Indicator (max flows in any 1-second bin within the window)
    expanded["__sec_bin__"] = expanded["timestamp"].dt.floor("1s")
    burst_counts = expanded.groupby(["host_id", "__window_idx__", "__sec_bin__"]).size().reset_index(name="b_count")
    burst_df = burst_counts.groupby(["host_id", "__window_idx__"])["b_count"].max().reset_index(name="connection_burst_max")

    # 5. TCP Flag Ratios at Window Level
    pkt_sum_safe = windows_df["packet_count_sum"].clip(lower=1.0)
    for flag_col, ratio_col in [
        ("syn_count_sum", "syn_ratio"),
        ("ack_count_sum", "ack_ratio"),
        ("rst_count_sum", "rst_ratio"),
        ("fin_count_sum", "fin_ratio"),
        ("psh_count_sum", "psh_ratio"),
    ]:
        if flag_col in windows_df.columns:
            windows_df[ratio_col] = (windows_df[flag_col].fillna(0.0) / pkt_sum_safe).clip(lower=0.0, upper=1.0)
        else:
            windows_df[ratio_col] = 0.0

    # Communication fan-out ratio
    windows_df["communication_fan_out"] = windows_df["unique_destination_ips"] / windows_df["flow_count"].clip(lower=1)

    # 6. Attack vs Benign & Stage Distribution (Explicit Strategy)
    # Strategy:
    # - attack_flow_ratio = non-benign flows / total flows
    # - benign_flow_ratio = benign flows / total flows
    # - attack_present = 1 if attack_flow_ratio > 0 else 0
    # - dominant_class = highest count among attack classes if attack_present else 'Benign'
    # - dominant_stage = highest count among attack stages if attack_present else 'Benign'
    # - stage_distribution = JSON dict of exact counts {stage: count}
    # - label_confidence = 'ground_truth' if 100% benign, else 'inferred'

    # Compute counts per stage
    expanded["attack_stage_clean"] = expanded["attack_stage"].fillna("Benign").astype(str)
    expanded["attack_class_clean"] = expanded["normalized_attack_class"].fillna("Benign").astype(str)
    expanded["raw_label_clean"] = expanded["dataset_label"].fillna("BENIGN").astype(str)

    # Dominant raw dataset label
    raw_label_counts = expanded.groupby(["host_id", "__window_idx__", "raw_label_clean"]).size().reset_index(name="raw_cnt")
    dominant_raw = raw_label_counts.sort_values(["host_id", "__window_idx__", "raw_cnt"], ascending=[True, True, False]).drop_duplicates(subset=["host_id", "__window_idx__"])[["host_id", "__window_idx__", "raw_label_clean"]]
    dominant_raw = dominant_raw.rename(columns={"raw_label_clean": "dominant_dataset_label"})

    # Stage matrix
    stage_matrix = expanded.groupby(["host_id", "__window_idx__", "attack_stage_clean"]).size().unstack(fill_value=0)
    class_matrix = expanded.groupby(["host_id", "__window_idx__", "attack_class_clean"]).size().unstack(fill_value=0)

    # Non-benign flow count
    attack_classes = [c for c in class_matrix.columns if c.lower() != "benign"]
    if attack_classes:
        attack_flow_counts = class_matrix[attack_classes].sum(axis=1)
    else:
        attack_flow_counts = pd.Series(0, index=class_matrix.index)

    total_flow_counts = class_matrix.sum(axis=1)
    attack_ratio = attack_flow_counts / total_flow_counts.clip(lower=1)
    benign_ratio = 1.0 - attack_ratio
    attack_present = (attack_flow_counts > 0).astype(int)

    # Dominant class selection
    def _pick_dominant_class(row: pd.Series) -> str:
        if attack_classes:
            atk_sub = row[attack_classes]
            if atk_sub.sum() > 0:
                return str(atk_sub.idxmax())
        return "Benign"

    dominant_classes = class_matrix.apply(_pick_dominant_class, axis=1)

    # Dominant stage selection
    attack_stages = [s for s in stage_matrix.columns if s.lower() != "benign"]
    def _pick_dominant_stage(row: pd.Series) -> str:
        if attack_stages:
            atk_sub = row[attack_stages]
            if atk_sub.sum() > 0:
                return str(atk_sub.idxmax())
        return "Benign"

    dominant_stages = stage_matrix.apply(_pick_dominant_stage, axis=1)

    # Stage distribution JSON
    def _serialize_stage_dist(row: pd.Series) -> str:
        non_zero = {k: int(v) for k, v in row.items() if v > 0}
        return json.dumps(non_zero, sort_keys=True)

    stage_distributions = stage_matrix.apply(_serialize_stage_dist, axis=1)

    # Label confidence: ground_truth if strictly Benign, inferred if any attack stage proxy was mapped
    label_confidences = np.where(attack_present == 1, "inferred", "ground_truth")

    label_summary_df = pd.DataFrame({
        "attack_flow_ratio": attack_ratio.values,
        "benign_flow_ratio": benign_ratio.values,
        "attack_present": attack_present.values,
        "dominant_class": dominant_classes.values,
        "dominant_stage": dominant_stages.values,
        "stage_distribution": stage_distributions.values,
        "label_confidence": label_confidences,
    }, index=class_matrix.index).reset_index()

    # Merge all components together on [host_id, __window_idx__]
    windows_df = windows_df.merge(entropy_df, on=["host_id", "__window_idx__"], how="left")
    windows_df = windows_df.merge(burst_df, on=["host_id", "__window_idx__"], how="left")
    windows_df = windows_df.merge(dominant_raw, on=["host_id", "__window_idx__"], how="left")
    windows_df = windows_df.merge(label_summary_df, on=["host_id", "__window_idx__"], how="left")

    # Fill NA and clean column names
    windows_df["destination_port_entropy"] = windows_df["destination_port_entropy"].fillna(0.0).clip(lower=0.0)
    windows_df["connection_burst_max"] = windows_df["connection_burst_max"].fillna(1.0)
    windows_df = windows_df.rename(columns={"__window_idx__": "window_idx"})

    # Sort chronologically by host and window_start
    windows_df = windows_df.sort_values(["host_id", "window_start"]).reset_index(drop=True)

    log.info(
        "host temporal windows aggregated",
        total_windows=len(windows_df),
        unique_hosts=windows_df["host_id"].nunique(),
        window_size_seconds=w_size,
        stride_seconds=s_size,
        attack_windows=int((windows_df["attack_present"] == 1).sum()),
        benign_windows=int((windows_df["attack_present"] == 0).sum()),
    )
    return windows_df
