"""Pre-B6/B7 temporal builders, kept verbatim as the reference the vectorised code must match.

Copied from aegisflow/ml/temporal/windowing.py and sequences.py before the change (only the
function names and imports differ). Used by tests/test_temporal_equivalence.py; not shipped.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from aegisflow.config import AegisFlowConfig
from aegisflow.logging_setup import get_logger
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES, SequenceConfig
from aegisflow.ml.temporal.windowing import WindowingConfig

log = get_logger("TEMPORAL_REFERENCE")


def old_aggregate_host_windows(
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


def old_build_host_sequences(
    windows_df: pd.DataFrame,
    cfg: AegisFlowConfig | None = None,
    *,
    sequence_length: int | None = None,
    forecast_horizon: int | None = None,
    stride: int = 1,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build chronological temporal sequences and strictly future targets per host.

    Parameters:
        windows_df: Host windows generated by ``aggregate_host_windows``.
        cfg: Optional project configuration.
        sequence_length: Number of consecutive past windows forming model input (default 10).
        forecast_horizon: One-based count of eligible same-host windows with
            ``window_start > final_input_window.window_end`` (default 1).
        stride: Step stride between consecutive sequences for a host (default 1).
        feature_columns: Specific numeric columns to include in the feature vector.

    Returns:
        DataFrame containing sequences, future targets, and temporal bounds.
    """
    if windows_df.empty:
        log.warning("empty windows DataFrame passed to build_host_sequences")
        return pd.DataFrame()

    if cfg is not None:
        seq_cfg = SequenceConfig.from_config(
            cfg,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            stride=stride,
        )
    else:
        seq_cfg = SequenceConfig(
            sequence_length=int(sequence_length or 10),
            forecast_horizon=int(forecast_horizon or 1),
            stride=int(stride or 1),
        )

    L = seq_cfg.sequence_length
    H = seq_cfg.forecast_horizon
    step = seq_cfg.stride

    if L < 1:
        raise ValueError(f"sequence_length must be >= 1, got {L}")
    if H < 1:
        raise ValueError(f"forecast_horizon must be >= 1 (eligible future-window count), got {H}")

    # Determine numerical feature columns to extract
    if feature_columns is None:
        feat_cols = [c for c in STANDARD_NUMERIC_WINDOW_FEATURES if c in windows_df.columns]
    else:
        feat_cols = [c for c in feature_columns if c in windows_df.columns]

    if not feat_cols:
        raise ValueError("No valid numeric feature columns available to construct sequence features.")

    # Fill NA in feature columns with 0.0 for model readiness
    clean_df = windows_df.copy()
    clean_df[feat_cols] = clean_df[feat_cols].fillna(0.0)

    # Sort strictly by host and window_start
    clean_df = clean_df.sort_values(["host_id", "window_start"]).reset_index(drop=True)

    records: list[dict[str, Any]] = []

    # Process each host independently to guarantee zero cross-host leakage
    for host_id, host_group in clean_df.groupby("host_id", sort=False):
        n_windows = len(host_group)
        if n_windows < L + 1:
            # Not enough chronological windows for this host to form a sequence + future target
            continue

        host_group_sorted = host_group.sort_values("window_start").reset_index(drop=True)
        feat_matrix = host_group_sorted[feat_cols].to_numpy(dtype=np.float32)

        window_starts = host_group_sorted["window_start"].values
        window_ends = host_group_sorted["window_end"].values
        attack_presents = host_group_sorted["attack_present"].values
        dom_classes = host_group_sorted["dominant_class"].values
        dom_stages = host_group_sorted["dominant_stage"].values
        stage_dists = host_group_sorted["stage_distribution"].values
        confidences = host_group_sorted["label_confidence"].values

        parsed_starts = pd.to_datetime(window_starts)
        parsed_ends = pd.to_datetime(window_ends)
        if not parsed_starts.is_monotonic_increasing or not parsed_ends.is_monotonic_increasing:
            raise ValueError(f"Host windows are not chronological for host {host_id}")

        for i in range(0, n_windows - L + 1, step):
            in_slice = slice(i, i + L)
            final_input_idx = i + L - 1
            seq_end = pd.Timestamp(window_ends[final_input_idx])
            # Horizon is the one-based rank among windows whose start is strictly
            # after the final input end. With 60s windows/30s stride, horizon=1
            # skips the next overlapping windows and selects the first non-overlap.
            eligible = np.flatnonzero(parsed_starts[final_input_idx + 1:] > seq_end) + final_input_idx + 1
            if len(eligible) < H:
                continue
            target_idx = int(eligible[H - 1])
            input_ends = parsed_ends[in_slice]
            if pd.Timestamp(window_starts[target_idx]) <= input_ends.max():
                raise ValueError(f"Overlapping target window detected for host {host_id}")

            input_features = feat_matrix[in_slice].tolist()  # shape (L, len(feat_cols))
            target_features = feat_matrix[target_idx].tolist()  # shape (len(feat_cols),)

            seq_start = pd.Timestamp(window_starts[i])
            seq_end = pd.Timestamp(window_ends[final_input_idx])
            target_start = pd.Timestamp(window_starts[target_idx])
            target_end = pd.Timestamp(window_ends[target_idx])

            # Explicit strict no-overlap and future-order validation.
            if target_start <= seq_end:
                raise ValueError(
                    f"Temporal leakage detected: target_start ({target_start}) is not after "
                    f"sequence input window end ({seq_end}) for host {host_id}."
                )

            records.append({
                "sequence_id": f"{host_id}_seq_{i}",
                "host_id": str(host_id),
                "target_host_id": str(host_id),
                "seq_start_time": seq_start,
                "seq_end_time": seq_end,
                "target_window_start": target_start,
                "target_window_end": target_end,
                "forecast_horizon": H,
                "sequence_length": L,
                "sequence_features": input_features,
                "target_features": target_features,
                "target_attack_present": int(attack_presents[target_idx]),
                "target_dominant_class": str(dom_classes[target_idx]),
                "target_dominant_stage": str(dom_stages[target_idx]),
                "target_stage_distribution": str(stage_dists[target_idx]),
                "target_label_confidence": str(confidences[target_idx]),
                # Onset: attack target with no attack in ANY input window. Positives that are
                # not onsets are continuations of an attack already visible in the input.
                "target_is_onset": bool(attack_presents[target_idx]) and not bool(np.any(attack_presents[in_slice])),
            })

    seq_df = pd.DataFrame(records)
    if not seq_df.empty:
        seq_df = seq_df.sort_values(["seq_start_time", "host_id"]).reset_index(drop=True)

    log.info(
        "host sequences generated",
        total_sequences=len(seq_df),
        sequence_length=L,
        forecast_horizon=H,
        unique_hosts=seq_df["host_id"].nunique() if not seq_df.empty else 0,
        target_attack_sequences=int((seq_df["target_attack_present"] == 1).sum()) if not seq_df.empty else 0,
        target_benign_sequences=int((seq_df["target_attack_present"] == 0).sum()) if not seq_df.empty else 0,
    )
    return seq_df
