"""select_threshold (precision_recall_curve based) must match the original O(n^2) loop exactly."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from aegisflow.ml.modeling import select_threshold

MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "cic_ids2017"


def reference_select_threshold(y, p):
    """The implementation that produced the committed threshold, kept verbatim as the reference."""
    if len(y) == 0 or not np.any(y == 1):
        return 0.5, "validation set has no positive targets; default 0.5 retained"
    candidates = sorted(set([0.01, 0.05, 0.1, 0.2, 0.3, 0.5] + list(map(float, p))))
    scores = [(f1_score(y, p >= t, zero_division=0), t) for t in candidates]
    return float(max(scores, key=lambda z: (z[0], -z[1]))[1]), "validation F1 maximization"


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n,decimals", [(30, 1), (200, 2), (400, None)])
def test_matches_reference_on_random_data_with_ties(seed, n, decimals):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rng.choice([0.02, 0.1, 0.5])).astype(int)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.25), 0, 1).astype(np.float32)
    if decimals is not None:  # rounding creates many exactly-tied probabilities (and some equal to grid values)
        p = np.round(p, decimals).astype(np.float32)
    assert select_threshold(y, p) == reference_select_threshold(y, p)


def test_ties_resolve_to_the_lowest_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])  # every threshold in (0.2, 0.8] gives F1 = 1
    assert select_threshold(y, p)[0] == 0.3 == reference_select_threshold(y, p)[0]


@pytest.mark.parametrize("y,p", [
    (np.array([]), np.array([])),
    (np.zeros(5, dtype=int), np.linspace(0, 1, 5)),
    (np.array([1, 0, 1]), np.array([0.001, 0.002, 0.003])),  # all probabilities below the whole grid
    (np.array([1, 1]), np.array([0.99, 0.999])),             # all probabilities above the whole grid
])
def test_edge_cases_match_reference(y, p):
    assert select_threshold(y, p) == reference_select_threshold(y, p)


@pytest.mark.skipif(not (MODEL_DIR / "audit_predictions.csv").exists(), reason="committed model artifacts not present")
def test_reproduces_committed_threshold_on_committed_validation_scores():
    df = pd.read_csv(MODEL_DIR / "audit_predictions.csv")
    val = df[df["split"] == "val"]
    y, p = val["target_attack_present"].to_numpy(), val["attack_probability"].to_numpy().astype(np.float32)
    committed = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))["threshold"]
    got = select_threshold(y, p)[0]
    assert got == committed == reference_select_threshold(y, p)[0]
