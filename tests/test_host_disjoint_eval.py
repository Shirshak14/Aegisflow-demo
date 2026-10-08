"""Fold builder and resampling units of scripts/host_disjoint_eval.py (no data or training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from host_disjoint_eval import assign_host_fold, resampling_units, threshold_at_fpr  # noqa: E402


def _frame():
    t = pd.Timestamp("2017-07-07 10:00")
    rows = [  # host, attack, class, minutes after t
        ("a", 1, "Botnet", 0), ("a", 1, "Botnet", 2), ("a", 1, "Botnet", 90), ("a", 0, "Benign", 5),
        ("b", 1, "Botnet", 0), ("b", 0, "Benign", 40),
        ("c", 0, "Benign", 0), ("c", 1, "Denial of Service", 1), ("c", 1, "Botnet", 3),
    ]
    return pd.DataFrame({"host_id": [r[0] for r in rows], "target_attack_present": [r[1] for r in rows],
                         "target_dominant_class": [r[2] for r in rows],
                         "target_window_start": [t + pd.Timedelta(minutes=r[3]) for r in rows]})


def test_fold_is_host_disjoint_and_botnet_only():
    f = _frame()
    split = assign_host_fold(f, "a", "b")
    assert set(f.host_id[split == "test"]) == {"a"}
    assert set(f.host_id[split == "val"]) == {"b"}
    assert set(f.host_id[split == "train"]) == {"c"}
    kept = f[split != "dropped"]
    assert set(kept.target_dominant_class[kept.target_attack_present == 1]) == {"Botnet"}
    assert (split == "dropped").sum() == 1
    with pytest.raises(ValueError):
        assign_host_fold(f, "a", "a")


def test_units_split_episodes_on_30_minute_gaps_per_host():
    f = _frame()
    u = resampling_units(f, f.target_attack_present.to_numpy())
    assert u[0] == u[1] != u[2]          # 2 min apart: same episode; 88 min later: new one
    assert u[0] != u[4]                  # same time, different host
    assert u[3].startswith("a|neg|") and u[5] != u[3]


def test_threshold_keeps_fpr_at_or_below_target():
    neg = np.linspace(0, 1, 1000)
    t = threshold_at_fpr(neg, 0.01)
    assert (neg >= t).mean() <= 0.01
    assert (neg >= np.nextafter(t, -np.inf) - 0.002).mean() > 0.01 - 1e-9
