"""
Data quality, distribution, and temporal health reporting for AegisFlow Phase 2.

Generates comprehensive Markdown and JSON reports containing exact measured
statistics from the processed dataset.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..features.registry import get_unavailable_features
from .split import SplitMetadata


@dataclass
class QualityMetrics:
    dataset_key: str
    raw_rows_loaded: int
    cleaned_rows_kept: int
    dropped_rows_total: int
    dropped_by_reason: dict[str, int]
    unique_hosts_count: int
    temporal_range_start: str
    temporal_range_end: str
    temporal_duration_hours: float
    total_host_windows: int
    attack_host_windows: int
    benign_host_windows: int
    total_sequences: int
    target_attack_sequences: int
    target_benign_sequences: int
    flow_class_distribution: dict[str, int]
    flow_stage_distribution: dict[str, int]
    window_dominant_class_distribution: dict[str, int]
    window_dominant_stage_distribution: dict[str, int]
    sequence_target_class_distribution: dict[str, int]
    sequence_target_stage_distribution: dict[str, int]
    target_gap_seconds: dict[str, Any]
    unavailable_features: list[str]
    split_info: dict[str, Any]
    warnings: list[str]
    onset_by_split: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_data_quality_report(
    *,
    dataset_key: str,
    raw_count: int,
    cleaned_df: pd.DataFrame,
    cleaning_dropped_by_reason: dict[str, int],
    ingest_counts_recorded: bool = True,
    windows_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    split_metadata: SplitMetadata,
    reports_dir: Path,
) -> tuple[Path, Path, QualityMetrics]:
    """Compile and save comprehensive data quality and imbalance reports.

    Returns:
        tuple of (markdown_path, json_path, metrics).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    t_start = str(cleaned_df["timestamp"].min()) if not cleaned_df.empty else ""
    t_end = str(cleaned_df["timestamp"].max()) if not cleaned_df.empty else ""
    duration_hours = 0.0
    if not cleaned_df.empty:
        duration_hours = float((cleaned_df["timestamp"].max() - cleaned_df["timestamp"].min()).total_seconds() / 3600.0)

    unique_hosts = int(cleaned_df["source_ip"].nunique()) if not cleaned_df.empty else 0

    flow_classes = {str(k): int(v) for k, v in cleaned_df["normalized_attack_class"].value_counts().items()} if "normalized_attack_class" in cleaned_df.columns else {}
    flow_stages = {str(k): int(v) for k, v in cleaned_df["attack_stage"].value_counts().items()} if "attack_stage" in cleaned_df.columns else {}

    win_classes = {str(k): int(v) for k, v in windows_df["dominant_class"].value_counts().items()} if not windows_df.empty else {}
    win_stages = {str(k): int(v) for k, v in windows_df["dominant_stage"].value_counts().items()} if not windows_df.empty else {}
    atk_wins = int((windows_df["attack_present"] == 1).sum()) if not windows_df.empty else 0
    ben_wins = int((windows_df["attack_present"] == 0).sum()) if not windows_df.empty else 0

    seq_classes = {str(k): int(v) for k, v in sequences_df["target_dominant_class"].value_counts().items()} if not sequences_df.empty else {}
    seq_stages = {str(k): int(v) for k, v in sequences_df["target_dominant_stage"].value_counts().items()} if not sequences_df.empty else {}
    atk_seqs = int((sequences_df["target_attack_present"] == 1).sum()) if not sequences_df.empty else 0
    ben_seqs = int((sequences_df["target_attack_present"] == 0).sum()) if not sequences_df.empty else 0
    if not sequences_df.empty:
        gap = (pd.to_datetime(sequences_df["target_window_start"]) -
               pd.to_datetime(sequences_df["seq_end_time"])).dt.total_seconds()
        target_gap = {"count": int(len(gap)), "min": float(gap.min()), "median": float(gap.median()),
                      "max": float(gap.max()), "nonpositive_count": int((gap <= 0).sum()),
                      "overlap_count": int((gap < 0).sum())}
    else:
        target_gap = {"count": 0, "min": None, "median": None, "max": None,
                      "nonpositive_count": 0, "overlap_count": 0}

    # Positive targets split into onsets (no attack in any input window) vs continuations.
    onset_by_split: dict[str, dict[str, int]] = {}
    if not sequences_df.empty and "target_is_onset" in sequences_df.columns and "split" in sequences_df.columns:
        for name in ["train", "val", "test", "boundary_excluded"]:
            sub = sequences_df[sequences_df["split"] == name]
            pos = int((sub["target_attack_present"] == 1).sum())
            onset = int(sub["target_is_onset"].astype(bool).sum())
            onset_by_split[name] = {"positive": pos, "onset": onset, "continuation": pos - onset}

    unavail = [f.feature_name for f in get_unavailable_features()]

    # Collect data quality warnings
    warnings: list[str] = []
    if flow_classes.get("Benign", 0) > 0 and len(flow_classes) > 1:
        benign_ratio = flow_classes["Benign"] / max(1, sum(flow_classes.values()))
        if benign_ratio > 0.95:
            warnings.append(
                f"Severe flow-level class imbalance: Benign traffic constitutes {benign_ratio:.1%} of total flows. "
                f"Minority attack classes (e.g. Botnet, Reconnaissance) are heavily under-represented."
            )

    if atk_seqs < 100:
        warnings.append(
            f"Limited positive sequence targets ({atk_seqs} sequences with target_attack_present=1). "
            f"Phase 3 models must employ class-weighted loss functions or focal loss."
        )

    if not ingest_counts_recorded:
        warnings.append(
            "Raw row count and cleaning drops were not recorded for this interim file (no ingest summary); "
            "'Raw Rows Loaded' shows the cleaned count. Re-run with --reingest to measure them."
        )

    if target_gap["nonpositive_count"]:
        warnings.append(f"Temporal target alignment error: {target_gap['nonpositive_count']} sequences have target_window_start <= seq_end_time.")

    warnings.append(
        "Attack stages (Reconnaissance, Initial Access, Lateral Movement, C2, Impact) are PROXY INFERENCES "
        "derived from dataset labels (configs/stages.yaml) and must not be presented as ground truth."
    )

    metrics = QualityMetrics(
        dataset_key=dataset_key,
        raw_rows_loaded=raw_count,
        cleaned_rows_kept=len(cleaned_df),
        dropped_rows_total=raw_count - len(cleaned_df),
        dropped_by_reason=cleaning_dropped_by_reason,
        unique_hosts_count=unique_hosts,
        temporal_range_start=t_start,
        temporal_range_end=t_end,
        temporal_duration_hours=round(duration_hours, 2),
        total_host_windows=len(windows_df),
        attack_host_windows=atk_wins,
        benign_host_windows=ben_wins,
        total_sequences=len(sequences_df),
        target_attack_sequences=atk_seqs,
        target_benign_sequences=ben_seqs,
        flow_class_distribution=flow_classes,
        flow_stage_distribution=flow_stages,
        window_dominant_class_distribution=win_classes,
        window_dominant_stage_distribution=win_stages,
        sequence_target_class_distribution=seq_classes,
        sequence_target_stage_distribution=seq_stages,
        target_gap_seconds=target_gap,
        unavailable_features=unavail,
        split_info=split_metadata.to_dict(),
        warnings=warnings,
        onset_by_split=onset_by_split,
    )

    # Save JSON
    json_path = reports_dir / "data_quality_report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics.to_dict(), fh, indent=2)

    # Save Markdown
    md_path = reports_dir / "data_quality_report.md"
    md_content = _build_markdown_report(metrics)
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(md_content)

    return md_path, json_path, metrics


def _build_markdown_report(m: QualityMetrics) -> str:
    lines = [
        f"# AegisFlow Phase 2: Data Quality & Temporal Distribution Report",
        f"",
        f"**Dataset Key:** `{m.dataset_key}`  ",
        f"**Temporal Duration:** {m.temporal_range_start} to {m.temporal_range_end} ({m.temporal_duration_hours} hours)  ",
        f"**Unique Monitored Hosts:** {m.unique_hosts_count}  ",
        f"",
        f"---",
        f"",
        f"## 1. Flow Ingestion & Cleaning Summary",
        f"",
        f"| Metric | Count | Percentage |",
        f"|---|---|---|",
        f"| Raw Rows Loaded | {m.raw_rows_loaded:,} | 100.0% |",
        f"| Cleaned Rows Kept | {m.cleaned_rows_kept:,} | {(m.cleaned_rows_kept / max(1, m.raw_rows_loaded)):.2%} |",
        f"| Rows Dropped | {m.dropped_rows_total:,} | {(m.dropped_rows_total / max(1, m.raw_rows_loaded)):.2%} |",
        f"",
        f"### Dropped Rows Breakdown",
        f"",
        f"| Reason | Dropped Count |",
        f"|---|---|",
    ]
    for reason, count in sorted(m.dropped_by_reason.items(), key=lambda x: -x[1]):
        lines.append(f"| `{reason}` | {count:,} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Temporal Representation Layer",
        f"",
        f"| Representation Level | Total Count | Attack-Positive | Benign | Attack Ratio |",
        f"|---|---|---|---|---|",
        f"| Cleaned Flows | {m.cleaned_rows_kept:,} | {m.cleaned_rows_kept - m.flow_class_distribution.get('Benign', 0):,} | {m.flow_class_distribution.get('Benign', 0):,} | {((m.cleaned_rows_kept - m.flow_class_distribution.get('Benign', 0)) / max(1, m.cleaned_rows_kept)):.2%} |",
        f"| Host Windows (60s / 30s stride) | {m.total_host_windows:,} | {m.attack_host_windows:,} | {m.benign_host_windows:,} | {(m.attack_host_windows / max(1, m.total_host_windows)):.2%} |",
        f"| Sequences (Len=10, Horizon=1) | {m.total_sequences:,} | {m.target_attack_sequences:,} | {m.target_benign_sequences:,} | {(m.target_attack_sequences / max(1, m.total_sequences)):.2%} |",
        f"",
        f"**Target gap (target start - final input end):** min {m.target_gap_seconds.get('min')} s, median {m.target_gap_seconds.get('median')} s, max {m.target_gap_seconds.get('max')} s; nonpositive {m.target_gap_seconds.get('nonpositive_count')} sequences; overlaps {m.target_gap_seconds.get('overlap_count')}. Horizon counts eligible same-host windows whose start is strictly after the final input window end.",
        f"",
        f"---",
        f"",
        f"## 3. Class & Attack Stage Distribution Across Pipeline Tiers",
        f"",
        f"### Normalized Attack Classes",
        f"",
        f"| Class | Flow Count | Window Dominant Count | Sequence Target Count |",
        f"|---|---|---|---|",
    ])

    all_classes = sorted(set(list(m.flow_class_distribution.keys()) + list(m.window_dominant_class_distribution.keys()) + list(m.sequence_target_class_distribution.keys())))
    for c in all_classes:
        f_c = m.flow_class_distribution.get(c, 0)
        w_c = m.window_dominant_class_distribution.get(c, 0)
        s_c = m.sequence_target_class_distribution.get(c, 0)
        lines.append(f"| **{c}** | {f_c:,} | {w_c:,} | {s_c:,} |")

    lines.extend([
        f"",
        f"### Inferred Cyber Kill-Chain Stages (Proxy)",
        f"",
        f"| Kill-Chain Stage | Flow Count | Window Dominant Count | Sequence Target Count |",
        f"|---|---|---|---|",
    ])
    all_stages = sorted(set(list(m.flow_stage_distribution.keys()) + list(m.window_dominant_stage_distribution.keys()) + list(m.sequence_target_stage_distribution.keys())))
    for stg in all_stages:
        f_s = m.flow_stage_distribution.get(stg, 0)
        w_s = m.window_dominant_stage_distribution.get(stg, 0)
        s_s = m.sequence_target_stage_distribution.get(stg, 0)
        lines.append(f"| **{stg}** | {f_s:,} | {w_s:,} | {s_s:,} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Chronological Train / Val / Test Split Breakdown",
        f"",
        f"- **Val Cutoff Timestamp:** `{m.split_info.get('val_cutoff_time')}`",
        f"- **Test Cutoff Timestamp:** `{m.split_info.get('test_cutoff_time')}`",
        f"",
        f"| Split Partition | Sequence Count | Percentage |",
        f"|---|---|---|",
        f"| Training (`train`) | {m.split_info.get('train_sequences', 0):,} | {(m.split_info.get('train_sequences', 0) / max(1, m.total_sequences)):.2%} |",
        f"| Validation (`val`) | {m.split_info.get('val_sequences', 0):,} | {(m.split_info.get('val_sequences', 0) / max(1, m.total_sequences)):.2%} |",
        f"| Testing (`test`) | {m.split_info.get('test_sequences', 0):,} | {(m.split_info.get('test_sequences', 0) / max(1, m.total_sequences)):.2%} |",
        f"| Boundary Quarantined (`boundary_excluded`) | {m.split_info.get('boundary_excluded_sequences', 0):,} | {(m.split_info.get('boundary_excluded_sequences', 0) / max(1, m.total_sequences)):.2%} |",
        f"",
        f"### Positive Targets: Onset vs Continuation",
        f"",
        f"Onset = attack-positive target with no attack-present window among the sequence inputs; "
        f"continuation = attack-positive target whose inputs already contain attack traffic.",
        f"",
        f"| Split Partition | Positive Targets | Onset | Continuation |",
        f"|---|---|---|---|",
    ])
    for name, c in m.onset_by_split.items():
        lines.append(f"| `{name}` | {c['positive']:,} | {c['onset']:,} | {c['continuation']:,} |")
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. Structurally Unavailable Features",
        f"",
        f"The following features are not reported by CIC-IDS2017 sensor output and are explicitly marked as unavailable in the feature registry (never fabricated):",
        f"",
    ])
    for feat in m.unavailable_features:
        lines.append(f"- `{feat}` (retained as NA / omitted from numerical tensors)")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 6. Critical Observations & Data Warnings",
        f"",
    ])
    for w in m.warnings:
        lines.append(f"> [!WARNING]\n> {w}\n")

    return "\n".join(lines)
