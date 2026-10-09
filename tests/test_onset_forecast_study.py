"""Onset target used by scripts/onset_forecast_study.py: eligibility, horizons, leakage."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from aegisflow.ml.modeling import MODEL_FEATURES

spec = importlib.util.spec_from_file_location(
    "onset_forecast_study", Path(__file__).resolve().parents[1] / "scripts/onset_forecast_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def _windows(host: str, n: int, attack_from: int | None, t0="2017-07-07 09:00:00") -> pd.DataFrame:
    start = pd.date_range(t0, periods=n, freq="30s")
    df = pd.DataFrame({name: 1.0 for name in MODEL_FEATURES}, index=range(n))
    df["host_id"] = host
    df["window_start"] = start
    df["window_end"] = start + pd.Timedelta("60s")
    df["attack_present"] = 0
    if attack_from is not None:
        df.loc[attack_from:, "attack_present"] = 1
    return df


def test_onset_labels_are_attack_free_and_horizon_bounded():
    w = pd.concat([_windows("a", 200, attack_from=150), _windows("b", 50, None)], ignore_index=True)
    w = w.sort_values(["host_id", "window_start"]).reset_index(drop=True)
    A = study.build_anchors(w)
    onset = w[(w.host_id == "a") & (w.attack_present == 1)].window_start.min().to_datetime64()
    a = A["host"] == "a"
    # no eligible anchor sees an attack window in its inputs
    assert w.attack_present.to_numpy()[A["win"]].max() == 0
    # every anchor ends before the onset; none after it is eligible (attack within the last 30 min)
    assert (A["t"][a] <= onset).all()
    for h in study.HORIZONS_MIN:
        lead = (onset - A["t"][a][A["y"][h][a] == 1]) / np.timedelta64(1, "m")
        assert len(lead) and (lead >= 0).all() and (lead < h).all()
    # a 60 s window ending exactly at the onset is the last positive anchor, with zero lead
    assert A["y"][1][a].sum() == 2
    assert A["y"][30][~a].sum() == 0


def test_shifted_labels_keep_positive_count_per_host():
    w = _windows("a", 400, attack_from=300)
    A = study.build_anchors(w)
    y = A["y"][15]
    rows = np.arange(len(y))
    ys = study.shifted_labels(A, y, rows, np.random.default_rng(0))
    assert ys.sum() == y.sum() and not np.array_equal(ys, y)
