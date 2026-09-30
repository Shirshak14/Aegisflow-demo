"""
Attack-stage label mapping: dataset_label -> normalized_attack_class -> attack_stage.

The mapping is loaded from configs/stages.yaml and is a documented PROXY
(see that file's header comment), never presented as ground truth. Every
distinct dataset_label seen in the data MUST have an entry in stages.yaml;
an unmapped label is a hard error (LabelMappingError), not a silent "Unknown".
"""
from __future__ import annotations

import pandas as pd

from ...config import AegisFlowConfig
from ...errors import LabelMappingError
from ...logging_setup import get_logger

log = get_logger("LABELS")


def apply_stage_mapping(df: pd.DataFrame, cfg: AegisFlowConfig, dataset_key: str) -> pd.DataFrame:
    """Add normalized_attack_class / attack_stage / label_confidence columns.

    Raises:
        LabelMappingError: if any dataset_label in ``df`` has no entry under
            ``configs/stages.yaml: <dataset_key>``.
    """
    if dataset_key not in cfg.stages:
        raise LabelMappingError(
            f"configs/stages.yaml has no mapping section for dataset '{dataset_key}'. "
            f"Add one before ingesting this dataset."
        )
    mapping = cfg.stages[dataset_key]

    observed_labels = set(df["dataset_label"].dropna().unique().tolist())
    unmapped = sorted(lbl for lbl in observed_labels if lbl not in mapping)
    if unmapped:
        raise LabelMappingError(
            f"{len(unmapped)} dataset_label value(s) have no entry in "
            f"configs/stages.yaml[{dataset_key}]: {unmapped}. "
            f"Add each one explicitly (with confidence: ground_truth|inferred) -- "
            f"AegisFlow refuses to guess attack-stage mappings silently."
        )

    normalized = df["dataset_label"].map(lambda lbl: mapping[lbl]["normalized_attack_class"] if pd.notna(lbl) else pd.NA)
    stage = df["dataset_label"].map(lambda lbl: mapping[lbl]["attack_stage"] if pd.notna(lbl) else pd.NA)
    confidence = df["dataset_label"].map(lambda lbl: mapping[lbl]["confidence"] if pd.notna(lbl) else pd.NA)

    out = df.copy()
    out["normalized_attack_class"] = normalized.astype("string")
    out["attack_stage"] = stage.astype("string")
    out["label_confidence"] = confidence.astype("string")

    stage_counts = out["attack_stage"].value_counts(dropna=False).to_dict()
    log.info("stage mapping applied", **{str(k): int(v) for k, v in stage_counts.items()})
    return out


def unknown_stage_order(cfg: AegisFlowConfig) -> list[str]:
    """Canonical stage ordering used for confusion matrices / severity lookups."""
    return list(cfg.stages.stages_order)
