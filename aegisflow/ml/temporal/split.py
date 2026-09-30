"""
Time-based chronological train / validation / test splitting.

Strict Leakage-Prevention Strategy:
- Split boundaries are derived strictly from dataset timestamps.
- Earliest time period -> Train
- Intermediate time period -> Validation
- Latest time period -> Test
- Sequences bridging split boundaries are cleanly tagged as 'boundary_excluded'
  to prevent any future-to-past or test-to-train contamination.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ...config import AegisFlowConfig
from ...logging_setup import get_logger

log = get_logger("SPLIT")


@dataclass
class SplitMetadata:
    dataset_key: str
    global_min_time: str
    global_max_time: str
    val_cutoff_time: str
    test_cutoff_time: str
    train_fraction: float
    val_fraction: float
    test_fraction: float
    total_sequences: int
    train_sequences: int
    val_sequences: int
    test_sequences: int
    boundary_excluded_sequences: int
    class_distribution_by_split: dict[str, dict[str, int]]
    stage_distribution_by_split: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        log.info("saved split metadata", path=str(path))


def compute_temporal_splits(
    sequences_df: pd.DataFrame,
    cfg: AegisFlowConfig | None = None,
    *,
    train_fraction: float | None = None,
    val_fraction: float | None = None,
    test_fraction: float | None = None,
    dataset_key: str = "cic_ids2017",
) -> tuple[pd.DataFrame, SplitMetadata]:
    """Assign sequences to chronological train, val, and test splits with zero leakage.

    Parameters:
        sequences_df: DataFrame output of ``build_host_sequences``.
        cfg: Optional project configuration.
        train_fraction: Fraction for training (default: 0.6 from config).
        val_fraction: Fraction for validation (default: 0.2 from config).
        test_fraction: Fraction for testing (default: 0.2 from config).
        dataset_key: Identifying key of dataset.

    Returns:
        tuple of (sequences_df with 'split' column added, SplitMetadata).
    """
    if sequences_df.empty:
        log.warning("empty sequences DataFrame passed to compute_temporal_splits")
        empty_meta = SplitMetadata(
            dataset_key=dataset_key,
            global_min_time="",
            global_max_time="",
            val_cutoff_time="",
            test_cutoff_time="",
            train_fraction=train_fraction or 0.6,
            val_fraction=val_fraction or 0.2,
            test_fraction=test_fraction or 0.2,
            total_sequences=0,
            train_sequences=0,
            val_sequences=0,
            test_sequences=0,
            boundary_excluded_sequences=0,
            class_distribution_by_split={},
            stage_distribution_by_split={},
        )
        return sequences_df.copy(), empty_meta

    if cfg is not None:
        tr_frac = float(train_fraction if train_fraction is not None else cfg.config.split.train_fraction)
        va_frac = float(val_fraction if val_fraction is not None else cfg.config.split.val_fraction)
        te_frac = float(test_fraction if test_fraction is not None else cfg.config.split.test_fraction)
    else:
        tr_frac = float(train_fraction if train_fraction is not None else 0.6)
        va_frac = float(val_fraction if val_fraction is not None else 0.2)
        te_frac = float(test_fraction if test_fraction is not None else 0.2)

    total_frac = tr_frac + va_frac + te_frac
    if not (0.99 <= total_frac <= 1.01):
        raise ValueError(f"Split fractions must sum to 1.0, got {tr_frac} + {va_frac} + {te_frac} = {total_frac}")

    df = sequences_df.copy()

    # Derived timestamps
    t_min = df["seq_start_time"].min()
    t_max = df["target_window_end"].max()

    # Compute cutoffs based on target_window_end quantile
    # This guarantees proportional distribution across the chronological duration
    val_cutoff = df["target_window_end"].quantile(tr_frac)
    test_cutoff = df["target_window_end"].quantile(tr_frac + va_frac)

    # Assign splits with strict boundary rules
    # 1. train: strictly everything ends before or at val_cutoff
    # 2. val: sequence starts at or after val_cutoff and ends at or before test_cutoff
    # 3. test: sequence starts at or after test_cutoff
    # 4. boundary_excluded: sequence straddles across val_cutoff or test_cutoff
    conditions = [
        (df["target_window_end"] <= val_cutoff),
        (df["seq_start_time"] >= val_cutoff) & (df["target_window_end"] <= test_cutoff),
        (df["seq_start_time"] >= test_cutoff),
    ]
    choices = ["train", "val", "test"]

    df["split"] = "boundary_excluded"
    for cond, choice in zip(conditions, choices):
        df.loc[cond, "split"] = choice

    counts = df["split"].value_counts().to_dict()

    class_dist_by_split: dict[str, dict[str, int]] = {}
    stage_dist_by_split: dict[str, dict[str, int]] = {}

    for s in ["train", "val", "test", "boundary_excluded"]:
        sub = df[df["split"] == s]
        if not sub.empty:
            class_dist_by_split[s] = {str(k): int(v) for k, v in sub["target_dominant_class"].value_counts().items()}
            stage_dist_by_split[s] = {str(k): int(v) for k, v in sub["target_dominant_stage"].value_counts().items()}
        else:
            class_dist_by_split[s] = {}
            stage_dist_by_split[s] = {}

    metadata = SplitMetadata(
        dataset_key=dataset_key,
        global_min_time=str(t_min),
        global_max_time=str(t_max),
        val_cutoff_time=str(val_cutoff),
        test_cutoff_time=str(test_cutoff),
        train_fraction=tr_frac,
        val_fraction=va_frac,
        test_fraction=te_frac,
        total_sequences=len(df),
        train_sequences=int(counts.get("train", 0)),
        val_sequences=int(counts.get("val", 0)),
        test_sequences=int(counts.get("test", 0)),
        boundary_excluded_sequences=int(counts.get("boundary_excluded", 0)),
        class_distribution_by_split=class_dist_by_split,
        stage_distribution_by_split=stage_dist_by_split,
    )

    log.info(
        "chronological split assigned",
        total=len(df),
        train=metadata.train_sequences,
        val=metadata.val_sequences,
        test=metadata.test_sequences,
        excluded=metadata.boundary_excluded_sequences,
        val_cutoff=metadata.val_cutoff_time,
        test_cutoff=metadata.test_cutoff_time,
    )

    return df, metadata
