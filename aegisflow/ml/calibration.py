"""Probability calibration for a trained forecaster (B9).

The LSTM is trained with ``pos_weight`` = negatives/positives (about 132 on the committed
model), so its sigmoid outputs overstate the attack probability. Platt scaling maps a raw
probability p to sigmoid(a * logit(p) + b), with a and b fitted on the VALIDATION split only.

Platt scaling is strictly increasing when a > 0, so mapping the alert threshold through the
same function leaves every alert decision unchanged: ``raw >= t`` iff ``calibrated >= c(t)``.
Only the displayed confidence changes. ``fit_platt`` refuses a non-increasing fit.

Output: ``<model_dir>/calibration.json`` with a, b, the calibrated threshold, and Brier score
and expected calibration error (ECE) before and after on validation (fit) and test (held out).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_FILE = "calibration.json"
_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def transform(self, p):
        z = self.a * _logit(p) + self.b
        out = 1.0 / (1.0 + np.exp(-z))
        return float(out) if np.ndim(out) == 0 else out


def fit_platt(y: np.ndarray, p: np.ndarray) -> PlattCalibrator:
    """Fit on validation labels/scores. Raises ValueError when it cannot give a monotone increasing map."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("calibration needs both classes in the validation split")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if not a > 0:
        raise ValueError(f"Platt fit is not increasing (a={a:.4g}); scores do not rank validation positives higher")
    return PlattCalibrator(a, b)


def calibration_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Brier score, expected calibration error over equal-width bins, and the bins themselves."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            table.append({"bin": [float(edges[k]), float(edges[k + 1])], "n": int(m.sum()),
                          "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean())})
    return {"n": int(len(y)), "positives": int(y.sum()), "brier": float(np.mean((p - y) ** 2)),
            "ece": float(ece), "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
            "reliability": table}


def load_calibrator(model_dir: Path) -> PlattCalibrator | None:
    path = Path(model_dir) / CALIBRATION_FILE
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return PlattCalibrator(float(d["a"]), float(d["b"]))


def calibrate_model_dir(model_dir: Path, sequences_path: Path) -> dict[str, Any]:
    """Score val and test with the saved LSTM, fit Platt on val, write calibration.json; return its content."""
    import joblib
    import torch

    from .modeling import ForecastDataset, load_model
    model_dir = Path(model_dir)
    net, meta = load_model(model_dir)
    data = ForecastDataset.from_parquet(sequences_path, host_relative=bool(meta.get("host_relative", False)))
    prep = joblib.load(model_dir / "preprocessor.joblib")
    scores = {}
    for split in ("val", "test"):
        ix = data.indices(split)
        with torch.no_grad():
            scores[split] = (data.attack[ix], torch.sigmoid(net(torch.tensor(prep.transform(data.X[ix])))).numpy())
    cal = fit_platt(*scores["val"])
    threshold = float(meta["threshold"])
    out = {**asdict(cal), "method": "Platt scaling on the validation split (LSTM probability)",
           "raw_threshold": threshold, "calibrated_threshold": cal.transform(threshold),
           "note": "Monotone map: alert decisions are unchanged; only the displayed confidence changes.",
           "metrics": {}}
    for split, (y, p) in scores.items():
        out["metrics"][split] = {"raw": calibration_metrics(y, p), "calibrated": calibration_metrics(y, cal.transform(p)),
                                 "role": "fit" if split == "val" else "held out"}
    (model_dir / CALIBRATION_FILE).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
