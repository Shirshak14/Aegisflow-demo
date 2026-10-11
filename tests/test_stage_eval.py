import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))
import stage_eval  # noqa: E402


def _frame(times, hosts):
    t = pd.to_datetime(times)
    return pd.DataFrame({"target_window_start": t, "seq_start_time": t - pd.Timedelta(minutes=5), "host_id": hosts})


def test_sequence_summary_shape():
    X = np.random.default_rng(0).random((4, 10, 3)).astype(np.float32)
    assert stage_eval.sequence_summary(X, ["a", "b", "c"]).shape == (4, 12)


def test_purge_drops_same_host_neighbours(monkeypatch):
    seen = []

    class Recorder:
        def fit(self, X, y):
            seen.append(set(X[:, 0].astype(int)))
            return self

        def predict(self, X):
            return np.array(["A"] * len(X), dtype=object)

    monkeypatch.setattr(stage_eval, "stage_classifier", Recorder)
    # rows 0,1: stage A at 10:00 and 10:20 (different 20-minute blocks), same host -> purged from each other
    # row 2: stage B at 13:00, far away -> always in training for the A folds
    f = _frame(["2017-07-07 10:00", "2017-07-07 10:20", "2017-07-07 13:00"], ["h", "h", "h"])
    feats = np.arange(3, dtype=float)[:, None]
    y = np.array(["A", "A", "B"], dtype=object)
    stage_eval.grouped_purged_oof(f, feats, y, np.arange(3))
    assert seen[0] == {2} and seen[1] == {2}
