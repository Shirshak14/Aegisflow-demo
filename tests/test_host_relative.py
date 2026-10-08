import numpy as np
import pandas as pd

from aegisflow.ml.modeling import ForecastDataset, host_relative_inputs
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

NF = len(STANDARD_NUMERIC_WINDOW_FEATURES)


def frame_for(levels: dict[str, float], n=40, L=3):
    """Each host sends constant traffic at its own level; sequences start 30 s apart."""
    rows = []
    t0 = pd.Timestamp("2017-07-03 09:00")
    for host, level in levels.items():
        for i in range(n):
            seq = np.full((L, NF), level, dtype=np.float32)
            rows.append({"sequence_id": f"{host}_{i}", "host_id": host,
                         "seq_start_time": t0 + pd.Timedelta(seconds=30 * i),
                         "sequence_features": seq, "target_features": seq[-1],
                         "target_attack_present": 0, "target_dominant_class": "Benign",
                         "target_dominant_stage": "Benign", "split": "train"})
    return pd.DataFrame(rows)


def test_constant_hosts_become_indistinguishable_regardless_of_level():
    f = frame_for({"quiet": 2.0, "loud": 5000.0})
    ds = ForecastDataset.from_frame(f, host_relative=True)
    assert ds.host_relative
    assert np.allclose(ds.X, 0.0)  # level is removed; only change vs own history remains


def test_spike_shows_up_as_deviation_from_own_past():
    f = frame_for({"a": 10.0})
    spike = f.index[-1]
    f.at[spike, "sequence_features"] = np.full((3, NF), 10000.0, dtype=np.float32)
    ds = ForecastDataset.from_frame(f, host_relative=True)
    assert np.abs(ds.X[spike]).max() > 5.0
    assert np.allclose(ds.X[:-1], 0.0)


def test_baseline_is_causal_future_changes_do_not_alter_past_rows():
    f = frame_for({"a": 10.0})
    base = ForecastDataset.from_frame(f, host_relative=True).X
    g = f.copy()
    for i in range(30, 40):
        g.at[i, "sequence_features"] = np.full((3, NF), 9999.0, dtype=np.float32)
    changed = ForecastDataset.from_frame(g, host_relative=True).X
    assert np.array_equal(base[:30], changed[:30])


def test_baseline_excludes_overlapping_windows_and_short_history_is_zero():
    f = frame_for({"a": 10.0}, n=12)
    x = np.log1p(np.full((12, 3, NF), 10.0))
    out = host_relative_inputs(f, np.full((12, 3, NF), 10.0, dtype=np.float32),
                               list(STANDARD_NUMERIC_WINDOW_FEATURES), window_seconds=60.0, min_history=5)
    # windows are 60 s long and start 30 s apart, so sequence i only sees sequences <= i-2
    assert np.allclose(out[:7], 0.0)
    assert out.shape == x.shape
