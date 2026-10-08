"""Platt calibration (B9): monotone, fitted on validation only, improves an inflated score."""
from __future__ import annotations

import numpy as np
import pytest

from aegisflow.ml.calibration import calibration_metrics, fit_platt


def _inflated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    true_p = np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 20, n))
    return y, np.sqrt(true_p)          # monotone distortion that overstates every probability


def test_platt_is_increasing_and_keeps_alert_decisions():
    y, p = _inflated()
    cal = fit_platt(y, p)
    assert cal.a > 0
    q = cal.transform(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= 0)
    t = 0.33
    assert np.array_equal(p >= t, q >= cal.transform(t))


def test_platt_reduces_calibration_error_on_held_out_scores():
    y, p = _inflated(seed=1)
    yt, pt = _inflated(seed=2)
    cal = fit_platt(y, p)
    before, after = calibration_metrics(yt, pt), calibration_metrics(yt, cal.transform(pt))
    assert after["ece"] < before["ece"] and after["brier"] < before["brier"]
    assert sum(b["n"] for b in after["reliability"]) == len(yt)


def test_platt_refuses_single_class_or_inverted_scores():
    y, p = _inflated()
    with pytest.raises(ValueError):
        fit_platt(np.zeros_like(y), p)
    with pytest.raises(ValueError):
        fit_platt(y, 1 - p)
