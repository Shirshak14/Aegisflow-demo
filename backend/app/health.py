"""Model health and input drift for the replay, computed without labels.

Compares the sequences replayed so far with what the model was trained on, and the LSTM scores with the
validation scores the trainer wrote. Every signal is a reason to look closer, never evidence of an attack:
traffic on a new day differs from training for many benign reasons.

  ingestion          rows dropped while cleaning, and features this dataset cannot provide
                     (reports/data_quality_report.json; written at preprocessing time, not live)
  input values       NaN / infinite model inputs (the windowing step stores missing values as 0, so a gap is
                     indistinguishable from a real zero downstream; the share of zeros is compared instead)
  feature drift      population stability index (PSI) of each feature in the last input window, binned at the
                     training deciles; share of values outside the training min-max range
  traffic volume     sequences and flows per bucket of replay time, against the same per-bucket numbers in training
  score drift        histogram and PSI of LSTM probabilities against the validation split
                     (audit_predictions.csv in the model directory), and the alert rate on each
Processing lag is not reported: the replay scores the whole test split once at start-up, so there is no
per-sequence scoring delay to measure.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pandas as pd

from . import replay as _replay
from .info import REPORTS

PSI_WATCH, PSI_INVESTIGATE = 0.1, 0.25
MIN_SEQUENCES = 200
BUCKET = "15min"
SCORE_BINS = np.linspace(0.0, 1.0, 21)
_EPS = 1e-4
_lock = threading.Lock()
_ref: dict[str, Any] = {}


def _psi(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    p = np.clip(ref_counts / max(ref_counts.sum(), 1), _EPS, None)
    q = np.clip(cur_counts / max(cur_counts.sum(), 1), _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _edges(values: np.ndarray) -> np.ndarray:
    """Interior bin edges at the training deciles (duplicates collapse, e.g. for features that are mostly 0)."""
    return np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))


def _counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


def _level(psi: float | None) -> str:
    if psi is None:
        return "n/a"
    return "investigate" if psi >= PSI_INVESTIGATE else "watch" if psi >= PSI_WATCH else "stable"


def _reference() -> dict[str, Any]:
    """Training-split statistics, computed once from the sequence table (and validation scores, if present)."""
    with _lock:
        key = str(_replay.DATA_DIR), str(_replay.MODEL_DIR)
        if _ref.get("key") == key:
            return _ref
        from aegisflow.ml.modeling import MODEL_FEATURES
        from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES, feature_array
        raw = list(STANDARD_NUMERIC_WINDOW_FEATURES)
        train = pd.read_parquet(_replay.DATA_DIR / "sequences.parquet",
                                columns=["sequence_features", "target_window_start", "split"],
                                filters=[("split", "==", "train")])
        x = feature_array(train.sequence_features, n_features=len(raw))[:, -1, :]
        x = x[:, [raw.index(n) for n in MODEL_FEATURES]].astype(np.float64)
        feats = {}
        for i, name in enumerate(MODEL_FEATURES):
            v = x[:, i]
            e = _edges(v)
            feats[name] = {"edges": e, "counts": _counts(v, e), "min": float(v.min()), "max": float(v.max()),
                           "zero_share": float((v == 0).mean())}
        flows = x[:, MODEL_FEATURES.index("flow_count")]
        buckets = pd.DataFrame({"t": pd.to_datetime(train.target_window_start), "flows": flows}) \
            .groupby(pd.Grouper(key="t", freq=BUCKET)).agg(sequences=("flows", "size"), flows=("flows", "sum"))
        buckets = buckets[buckets.sequences > 0]
        volume = {k: {"median": float(buckets[k].median()), "p10": float(buckets[k].quantile(0.1)),
                      "p90": float(buckets[k].quantile(0.9))} for k in ("sequences", "flows")}
        val_scores = None
        audit = _replay.MODEL_DIR / "audit_predictions.csv"
        if audit.exists():
            a = pd.read_csv(audit, usecols=["split", "attack_probability"])
            val_scores = a.loc[a["split"] == "val", "attack_probability"].to_numpy(dtype=float)
        _ref.clear()
        _ref.update(key=key, names=list(MODEL_FEATURES), features=feats, train_sequences=int(len(x)),
                    volume=volume, val_scores=val_scores)
        return _ref


def _ingestion() -> dict[str, Any] | None:
    path = REPORTS / "data_quality_report.json"
    if not path.exists():
        return None
    q = json.loads(path.read_text(encoding="utf-8"))
    return {"raw_rows": q.get("raw_rows_loaded"), "kept_rows": q.get("cleaned_rows_kept"),
            "dropped_rows": q.get("dropped_rows_total"), "dropped_by_reason": q.get("dropped_by_reason") or {},
            "unavailable_features": q.get("unavailable_features") or [], "source": "reports/data_quality_report.json"}


def health(engine) -> dict[str, Any]:
    """Health and drift of the replayed sequences so far (current session)."""
    base: dict[str, Any] = {"ingestion": _ingestion(), "replayed": 0, "min_sequences": MIN_SEQUENCES,
                            "psi_levels": {"watch": PSI_WATCH, "investigate": PSI_INVESTIGATE},
                            "processing_lag": "not measured: the replay scores the whole test split once at start-up, "
                                              "so there is no per-sequence scoring delay"}
    if engine.events is None:
        return base | {"state": "model_not_ready"}
    n = int(engine.cursor)
    base["replayed"] = n
    if n < MIN_SEQUENCES:
        return base | {"state": "too_few_sequences"}
    ref = _reference()
    ev = engine.events.iloc[:n]
    x = engine._raw_X[:n, -1, :].astype(np.float64)
    finite = np.isfinite(engine._raw_X[:n])
    features = []
    for i, name in enumerate(ref["names"]):
        r, v = ref["features"][name], x[:, i]
        v = np.where(np.isfinite(v), v, 0.0)
        psi = _psi(r["counts"], _counts(v, r["edges"]))
        features.append({"feature": name, "psi": round(psi, 4), "level": _level(psi),
                         "outside_train_range": round(float(((v < r["min"]) | (v > r["max"])).mean()), 4),
                         "train_range": [r["min"], r["max"]],
                         "zero_share": round(float((v == 0).mean()), 4), "train_zero_share": round(r["zero_share"], 4)})
    features.sort(key=lambda f: -f["psi"])

    flows = x[:, ref["names"].index("flow_count")]
    b = pd.DataFrame({"t": ev["target_window_start"].to_numpy(), "flows": flows}) \
        .groupby(pd.Grouper(key="t", freq=BUCKET)).agg(sequences=("flows", "size"), flows=("flows", "sum"))
    volume = [{"t": t.isoformat(), "sequences": int(r.sequences), "flows": float(r.flows)} for t, r in b.iterrows()]

    p = ev["lstm_p"].to_numpy(dtype=float)
    thr = float(engine.lstm_threshold)
    scores = {"bins": SCORE_BINS.tolist(), "replay": np.histogram(p, SCORE_BINS)[0].tolist(),
              "replay_alert_rate": float((p >= thr).mean()), "threshold": thr, "validation": None, "psi": None}
    if ref["val_scores"] is not None and len(ref["val_scores"]):
        vs = ref["val_scores"]
        ve = _edges(vs)
        scores.update(validation=np.histogram(vs, SCORE_BINS)[0].tolist(), validation_alert_rate=float((vs >= thr).mean()),
                      psi=round(_psi(_counts(vs, ve), _counts(p, ve)), 4))
    scores["level"] = _level(scores["psi"])

    levels = [f["level"] for f in features]
    return base | {
        "state": "ok", "session_id": engine.session_id, "train_sequences": ref["train_sequences"],
        "window": [ev["target_window_start"].iloc[0].isoformat(), ev["target_window_start"].iloc[-1].isoformat()],
        "non_finite_inputs": int((~finite).sum()),
        "features": features,
        "features_by_level": {k: levels.count(k) for k in ("investigate", "watch", "stable")},
        "volume": {"bucket": BUCKET, "replay": volume, "train": ref["volume"]},
        "scores": scores,
    }
