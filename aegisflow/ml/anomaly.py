"""Benign-only isolation forest, saved for the replay as a label-free second opinion on LSTM alerts.

Same model and fit set as ``scripts/evaluation/anomaly_eval.py`` (see reports/anomaly_report.md):
300-tree IsolationForest on the flattened, preprocessed 10-window input sequence, fit on train-split
sequences whose target is benign AND whose input windows carry no attack flow. No attack label is used
to fit it, so it cannot learn the attacker-host shortcut from labels.

The flag threshold is the ``quantile`` (default 0.99) of the scores on that benign fit set: nominally a
1% false-alarm rate, chosen without looking at validation or test data. In the replay the LSTM alone
raises alerts; this model only says whether it also finds an alerted sequence unusual.

Files written to the model directory: isolation_forest.joblib, preprocessor.joblib, anomaly.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .modeling import ForecastDataset, SequencePreprocessor
from .temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

META_FILE = "anomaly.json"
MODEL_FILE = "isolation_forest.joblib"
PREP_FILE = "preprocessor.joblib"


def input_attack_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where any input window of the sequence carries an attack flow (used only to clean the fit set)."""
    col = list(STANDARD_NUMERIC_WINDOW_FEATURES).index("attack_flow_ratio")
    return np.array([np.nan_to_num(np.stack(v)[:, col]).max() > 0 for v in frame.sequence_features], dtype=bool)


def fit_indices(data: ForecastDataset) -> np.ndarray:
    """Train-split sequences with a benign target and no attack flow in any input window."""
    split = data.frame["split"].to_numpy()
    return np.flatnonzero((split == "train") & (data.attack == 0) & ~input_attack_mask(data.frame))


def train_anomaly(data: ForecastDataset, out_dir: str | Path, *, seed: int = 42, n_estimators: int = 300,
                  quantile: float = 0.99, dataset: str | None = None) -> dict[str, Any]:
    """Fit on the benign fit set, set the threshold at ``quantile`` of its scores, save and return the metadata."""
    from sklearn.ensemble import IsolationForest

    if not 0.5 <= quantile < 1:
        raise ValueError(f"quantile must be in [0.5, 1), got {quantile}")
    ix = fit_indices(data)
    if len(ix) == 0:
        raise ValueError("no benign train sequences without attack traffic in their inputs to fit on")
    prep = SequencePreprocessor(data.feature_names).fit(data.X[ix])
    flat = prep.transform(data.X[ix]).reshape(len(ix), -1)
    iso = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(flat)
    fit_scores = -iso.score_samples(flat)
    meta = {
        "model_type": "isolation_forest", "dataset": dataset, "labels_used_to_fit": False,
        "feature_names": list(data.feature_names), "fit_sequences": len(ix),
        "n_estimators": n_estimators, "seed": seed,
        "threshold": float(np.quantile(fit_scores, quantile)), "threshold_quantile": quantile,
        "threshold_rule": f"the {quantile * 100:g}th percentile of the scores on the benign fit set (no attack label used)",
        "score": "negated IsolationForest.score_samples; higher = more unusual",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, out / MODEL_FILE)
    joblib.dump(prep, out / PREP_FILE)
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class AnomalyScorer:
    """Loads a saved anomaly model; ``score`` takes raw [n, windows, features] inputs like the LSTM's."""

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        self.model = joblib.load(d / MODEL_FILE)
        self.prep = joblib.load(d / PREP_FILE)
        self.threshold = float(self.meta["threshold"])

    def score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return -self.model.score_samples(self.prep.transform(X).reshape(len(X), -1))

    def flag(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) >= self.threshold
