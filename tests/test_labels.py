from __future__ import annotations

import pytest

from aegisflow.errors import LabelMappingError
from aegisflow.ml.datasets.cic_ids2017 import CicIds2017Adapter
from aegisflow.ml.preprocessing.labels import apply_stage_mapping


def test_apply_stage_mapping_known_labels(cfg, cic_ids2017_entry):
    df = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry).load_raw()
    out = apply_stage_mapping(df, cfg, "cic_ids2017")
    assert set(out["attack_stage"].unique()) <= set(cfg.stages.stages_order)
    benign_rows = out[out["dataset_label"] == "BENIGN"]
    assert (benign_rows["label_confidence"] == "ground_truth").all()
    attack_rows = out[out["dataset_label"] != "BENIGN"]
    assert (attack_rows["label_confidence"] == "inferred").all()


def test_apply_stage_mapping_unknown_label_raises(cfg, cic_ids2017_entry):
    df = CicIds2017Adapter(cfg=cfg, entry=cic_ids2017_entry).load_raw()
    df.loc[0, "dataset_label"] = "TotallyUnmappedLabel"
    with pytest.raises(LabelMappingError):
        apply_stage_mapping(df, cfg, "cic_ids2017")
