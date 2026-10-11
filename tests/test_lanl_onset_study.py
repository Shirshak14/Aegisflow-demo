from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lanl_onset_study.py"
spec = importlib.util.spec_from_file_location("lanl_onset_study", SCRIPT)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _bins(victim: str, onset: int) -> pd.DataFrame:
    feats = S.OWN + S.INB + S.CTX
    b = (onset - 1) // S.BIN - 1   # the last complete bin before onset
    row = {f: 0.0 for f in feats} | {"victim": victim, "bin": b, "own_flows": 5.0}
    return pd.DataFrame([row])


def test_anchors_stop_at_onset_and_label_horizon():
    onset = 10 * 86400 + 1234
    A = S.anchors(_bins("C1", onset), pd.Series({"C1": onset}))
    assert (A["t"] <= onset).all()
    assert len(A["t"]) == S.LOOKBACK // S.BIN          # one anchor per bin end in 7 days
    y = S.labels(A["t"], A["onset"], 15)
    assert y.sum() == 3                                  # three 5-min bin ends within 15 min before onset
    # the last complete pre-onset bin is the last input of the final anchor; the partial
    # bin that contains the onset is never used, so no anchor ends after the onset
    assert A["X"][-1, -1, A["features"].index("own_flows")] == 5.0


def test_bursts_chain_within_30_minutes():
    on = pd.Series({"a": 0, "b": 1000, "c": 2700, "d": 10_000})
    assert S.bursts(on).to_dict() == {"a": 0, "b": 0, "c": 0, "d": 1}


def test_time_features_are_relative_clock():
    t = np.array([0, 3600 * 25])
    tf = S.time_features(t)
    assert tf[1, 0] == 1.0 and tf[1, 1] == 1.0
