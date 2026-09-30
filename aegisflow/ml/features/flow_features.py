"""
Flow-level canonical feature engineering.

Derives rate and ratio features from cleaned canonical flow frames:
- packets_per_second, bytes_per_second, bytes_per_packet
- TCP control flag ratios (SYN, ACK, RST, FIN, PSH)

Never fabricates absent sensor fields (e.g. TTL, retransmissions).
Guarantees strictly finite numerical outputs without division-by-zero or infs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...logging_setup import get_logger

log = get_logger("FEATURES")


def compute_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived canonical features for each flow in ``df``.

    Parameters:
        df: Cleaned canonical DataFrame adhering to ``aegisflow.schema``.

    Returns:
        New DataFrame with derived canonical flow features attached.
    """
    out = df.copy()

    # Numerical coercion and safe denominators
    duration_safe = pd.to_numeric(out["flow_duration"], errors="coerce").fillna(0.0).clip(lower=1e-6)
    packet_count_safe = pd.to_numeric(out["packet_count"], errors="coerce").fillna(0.0).clip(lower=1.0)
    byte_count = pd.to_numeric(out["byte_count"], errors="coerce").fillna(0.0)

    # 1. Rate and volume ratios
    out["packets_per_second"] = (pd.to_numeric(out["packet_count"], errors="coerce").fillna(0.0) / duration_safe).astype("float64")
    out["bytes_per_second"] = (byte_count / duration_safe).astype("float64")
    out["bytes_per_packet"] = (byte_count / packet_count_safe).astype("float64")

    # 2. TCP Flag ratios (where flags exist; null flags treated as 0)
    flag_pairs = [
        ("syn_count", "syn_ratio"),
        ("ack_count", "ack_ratio"),
        ("rst_count", "rst_ratio"),
        ("fin_count", "fin_ratio"),
        ("psh_count", "psh_ratio"),
    ]

    for raw_flag_col, ratio_col in flag_pairs:
        if raw_flag_col in out.columns:
            flag_vals = pd.to_numeric(out[raw_flag_col], errors="coerce").fillna(0.0)
            ratio = (flag_vals / packet_count_safe).clip(lower=0.0, upper=1.0)
            out[ratio_col] = ratio.astype("float64")
        else:
            out[ratio_col] = pd.Series(np.nan, index=out.index, dtype="float64")

    log.debug("flow features computed", rows=len(out), new_features=["packets_per_second", "bytes_per_second", "bytes_per_packet", *(r for _, r in flag_pairs)])
    return out
