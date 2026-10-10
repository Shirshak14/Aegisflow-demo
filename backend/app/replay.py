"""Replay engine: plays the held-out TEST split through the saved Phase 3 models.

Every score is a real model output on real held-out sequences:
  - LSTM (artifacts/models/cic_ids2017/best.pt) -- primary detector; an alert is
    written when its probability >= the validation-selected LSTM threshold.
  - Logistic regression (logistic.joblib) -- scored alongside, with its own threshold.
  - Reference rule "last input window under attack" -- reads the label-derived
    attack_flow_ratio; shown for comparison only, never used in the risk score.
Sequences are replayed in target_window_start order on a simulated clock that
advances `speed` simulated seconds per wall-clock second.

Risk score (0-100) = 100 * sum(weight_k * component_k), weights from config.yaml:
  attack_probability     LSTM probability
  stage_severity         severity of the predicted stage from config; the predicted stage is always
                         the "uncertain" label (no stage model, see info.STAGE_NOTE), so this is 0
  prediction_confidence  how far the LSTM probability is above its threshold, scaled to 0-1
  abnormality_score      share of the 28 inputs in the last window beyond 3 robust-scaled
                         units (scaler fitted on training data only); label-free
  recent_attack_history  min(1, this host's alerts in the previous 10 simulated minutes / 5)
Alert ``confidence`` is the Platt-calibrated LSTM probability when ``calibration.json`` exists in the
model directory (``python -m aegisflow calibrate``), else the raw probability. Calibration is monotone,
so it never changes which alerts fire; the risk score above still uses the raw probability.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aegisflow.config import load_config
from aegisflow.ml.calibration import CALIBRATION_FILE, load_calibrator
from aegisflow.ml.modeling import ForecastDataset, LSTMForecaster, score_batched
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

from .audit import Ledger
from .info import DATA_DIR, MODEL_DIR, ROOT, model_version

_PREPARE_COLUMNS = ("sequence_id", "host_id", "seq_end_time", "target_window_start", "target_window_end",
                    "sequence_features", "target_features", "target_attack_present", "target_dominant_class",
                    "target_dominant_stage", "split")

HISTORY_WINDOW = pd.Timedelta(minutes=10)

RISK_COMPONENTS = ("attack_probability", "stage_severity", "prediction_confidence",
                   "abnormality_score", "recent_attack_history")


def validate_risk_weights(weights: dict[str, Any]) -> dict[str, float]:
    """Return ``weights`` as floats, or raise ValueError if a component is missing/unknown or the sum != 1."""
    got = {str(k): float(v) for k, v in dict(weights).items()}
    if set(got) != set(RISK_COMPONENTS):
        raise ValueError(f"risk_scoring.weights must have exactly {list(RISK_COMPONENTS)}, got {sorted(got)}")
    if any(v < 0 for v in got.values()):
        raise ValueError(f"risk_scoring.weights must be non-negative, got {got}")
    total = sum(got.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"risk_scoring.weights must sum to 1.0 (so risk stays on a 0-100 scale), got {total:.6f}")
    return got


class ReplayEngine:
    def __init__(self, ledger: Ledger):
        cfg = load_config().config
        self.ledger = ledger
        self.weights = validate_risk_weights(cfg.risk_scoring.weights)
        self.levels = dict(cfg.risk_scoring.thresholds)
        self.stage_severity = {str(k): float(v) for k, v in cfg.risk_scoring.stage_severity.items()}
        self.conf_threshold = float(cfg.confidence.stage_prediction_threshold)
        self.uncertain = str(cfg.confidence.uncertain_label)
        # Default: no stage model, so the predicted stage is always the "uncertain" label. Opt-in:
        # replay.stage_model_dir names a `train-multitask` model whose stage head labels each alert
        # (stage kept only when its probability >= confidence.stage_prediction_threshold).
        self.predicted_stage = self.uncertain
        stage_dir = cfg.replay.get("stage_model_dir")
        self.stage_model_dir = (ROOT / stage_dir) if stage_dir else None
        self.allowed_speeds = [int(s) for s in cfg.replay.allowed_speeds]
        self.default_speed = int(cfg.replay.default_speed)
        self._lock = threading.RLock()
        self._prep_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.events: pd.DataFrame | None = None
        self._rows: list | None = None   # events as namedtuples, built once in _prepare for the hot loop
        self._starts: list | None = None  # events.target_window_start as Timestamps, same order
        self._raw_X: np.ndarray | None = None
        self._explainer = None
        self._explainer_lock = threading.Lock()
        self._stage_predictor = None
        self._attention_models: dict[str, Any] = {}
        self._reset_state()

    # ------------------------------------------------------------------ scoring
    def prepare(self) -> None:
        """Load the test split and score it once with the saved models (~seconds). Thread-safe."""
        with self._prep_lock:
            if self.events is None:
                self._prepare()

    def warm_up(self) -> None:
        """Score in the background at server start so the first Start click is instant."""
        threading.Thread(target=self._safe_prepare, name="warm-up", daemon=True).start()

    def _safe_prepare(self) -> None:
        try:
            self.prepare()
        except Exception as exc:
            self.error = f"model warm-up failed: {exc!r}"

    def _prepare(self) -> None:
        import torch
        # Only the test rows and the columns used below; the rest of the ~32 MB file is never materialised.
        frame = pd.read_parquet(DATA_DIR / "sequences.parquet", columns=list(_PREPARE_COLUMNS),
                                filters=[("split", "==", "test")])
        frame = frame.sort_values(["target_window_start", "host_id"]).reset_index(drop=True)
        data = ForecastDataset.from_frame(frame)
        meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
        prep = joblib.load(MODEL_DIR / "preprocessor.joblib")
        lr = joblib.load(MODEL_DIR / "logistic.joblib")
        tc = meta["training_config"]
        net = LSTMForecaster(len(meta["feature_names"]), tc["hidden_size"], tc["dropout"]).net
        net.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=True))
        net.eval()
        x = prep.transform(data.X)
        lstm_p = score_batched(lambda t: torch.sigmoid(net(t)), x).astype(float)
        lr_p = lr.predict_proba(x.reshape(len(x), -1))[:, 1].astype(float)
        # `python -m aegisflow calibrate` writes calibration.json; without it confidence is the raw probability
        calibrator = load_calibrator(MODEL_DIR)
        conf = calibrator.transform(lstm_p).astype(float) if calibrator else lstm_p
        std = list(STANDARD_NUMERIC_WINDOW_FEATURES)
        raw_last = np.stack([np.stack(v)[-1] for v in frame.sequence_features])
        if self.stage_model_dir is not None:
            from aegisflow.ml.multitask import MultiTaskPredictor
            sp = MultiTaskPredictor(self.stage_model_dir).predict(data.X)
            pred_stage = np.where(sp["stage_probability"] >= self.conf_threshold, sp["stage"], self.uncertain)
        self.lstm_threshold = float(metrics["threshold"])
        self.lr_threshold = float(metrics["models"]["logistic_regression"]["threshold"])
        self.events = pd.DataFrame({
            "sequence_id": frame.sequence_id.astype(str), "host_id": frame.host_id.astype(str),
            "predicted_at": pd.to_datetime(frame.seq_end_time),
            "target_window_start": pd.to_datetime(frame.target_window_start),
            "target_window_end": pd.to_datetime(frame.target_window_end),
            "lstm_p": lstm_p, "lr_p": lr_p, "conf": conf,
            "rule": (raw_last[:, std.index("attack_flow_ratio")] > 0).astype(int),
            "abnormality": (np.abs(x[:, -1, :]) > 3).mean(axis=1),
            "truth_attack": frame.target_attack_present.astype(int).to_numpy(),
            "truth_class": frame.target_dominant_class.astype(str), "truth_stage": frame.target_dominant_stage.astype(str),
        })
        self._raw_X = data.X
        if self.stage_model_dir is not None:
            self.events["pred_stage"] = pred_stage.astype(str)
        # The replay loop reads these per row; building them once avoids ev.iloc[i] (a Series per call).
        self._rows = list(self.events.itertuples(index=False))
        self._starts = [r.target_window_start for r in self._rows]
        self.model_version = model_version()

    def _row_stage(self, row) -> str:
        return str(getattr(row, "pred_stage", self.predicted_stage))

    # ------------------------------------------------------------------ explanations (opt-in API only)
    def explain(self, sequence_id: str, *, model: str = "lstm", method: str = "shap", top: int = 5) -> dict[str, Any] | None:
        """Attribute one test-split prediction to its inputs; None if the sequence is not in the replay.

        Background for SHAP / the IG baseline is a fixed sample of TRAINING sequences, loaded on first use.
        Does not touch replay state, alerts, or the ledger.
        """
        from aegisflow.ml.explain import ModelExplainer
        self.prepare()
        hits = np.flatnonzero(self.events.sequence_id.to_numpy() == sequence_id)
        if len(hits) == 0:
            return None
        with self._explainer_lock:
            if self._explainer is None:
                cols = ["sequence_features", "target_features", "target_attack_present",
                        "target_dominant_class", "target_dominant_stage", "split"]
                train = pd.read_parquet(DATA_DIR / "sequences.parquet", columns=cols)
                train = ForecastDataset.from_frame(train[train["split"] == "train"])
                self._explainer = ModelExplainer(MODEL_DIR, train.X, seed=int(load_config().config.random_seed))
        e = self._explainer.explain(self._raw_X[hits[:1]], [sequence_id], model=model, method=method)[0]
        return e.to_dict(k=top)

    def _sequence_inputs(self, sequence_id: str) -> np.ndarray | None:
        """Raw [1, 10, 28] inputs of one replayed test sequence, or None if it is not in the replay."""
        self.prepare()
        hits = np.flatnonzero(self.events.sequence_id.to_numpy() == sequence_id)
        return None if len(hits) == 0 else self._raw_X[hits[:1]]

    # ------------------------------------------------------------------ stage / future-state forecast (opt-in)
    def forecast(self, sequence_id: str) -> dict[str, Any] | None:
        """Multi-task model output for one sequence: stage distribution (+ MITRE), K-step future state.

        Raises LookupError when no stage model is configured (replay.stage_model_dir); None if unknown sequence.
        """
        if self.stage_model_dir is None:
            raise LookupError("no stage model configured (replay.stage_model_dir is null)")
        x = self._sequence_inputs(sequence_id)
        if x is None:
            return None
        import yaml

        from aegisflow.ml.multitask import MultiTaskPredictor
        with self._explainer_lock:
            if self._stage_predictor is None:
                self._stage_predictor = MultiTaskPredictor(self.stage_model_dir)
        mitre = yaml.safe_load((ROOT / "configs" / "mitre_mapping.yaml").read_text(encoding="utf-8"))
        row = self._stage_predictor.rows(x, [sequence_id], stage_threshold=self.conf_threshold,
                                         uncertain_label=self.uncertain, mitre_lookup=mitre.get)[0]
        return row | {"model_dir": str(self.stage_model_dir.relative_to(ROOT)), "horizons": self._stage_predictor.horizons}

    # ------------------------------------------------------------------ attention of an optional model (opt-in)
    def attention(self, sequence_id: str, model_dir) -> dict[str, Any] | None:
        """Per-window attention weights of a trained attention_lstm / transformer model for one sequence.

        Raises ValueError if ``model_dir`` is not an attention model; None if the sequence is unknown.
        """
        import torch

        from aegisflow.ml.modeling import load_model
        x = self._sequence_inputs(sequence_id)
        if x is None:
            return None
        key = str(model_dir)
        with self._explainer_lock:
            if key not in self._attention_models:
                net, meta = load_model(model_dir)
                mtype = meta["training_config"].get("model_type", "lstm")
                if not hasattr(net, "attention"):
                    raise ValueError(f"{model_dir.name} is a {mtype} model; it has no attention weights")
                metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
                self._attention_models[key] = (net.eval(), joblib.load(model_dir / "preprocessor.joblib"), mtype,
                                               float(metrics["threshold"]))
            net, prep, mtype, threshold = self._attention_models[key]
        xt = torch.tensor(prep.transform(x))
        with torch.no_grad():
            p = float(torch.sigmoid(net(xt))[0])
            w = net.attention(xt)[0].numpy().astype(float)
        return {"sequence_id": sequence_id, "model": model_dir.name, "model_type": mtype,
                "attack_probability": p, "threshold": threshold, "predicted_attack": p >= threshold,
                "window_weights": [round(v, 4) for v in w]}

    def _risk(self, row, recent_alerts: int) -> tuple[float, dict[str, float]]:
        p, t = float(row.lstm_p), self.lstm_threshold
        comp = {"attack_probability": p, "stage_severity": self._stage_term(self._row_stage(row)),
                "prediction_confidence": max(0.0, (p - t) / (1 - t)),
                "abnormality_score": float(row.abnormality),
                "recent_attack_history": min(1.0, recent_alerts / 5)}
        risk = 100 * sum(self.weights[k] * v for k, v in comp.items())
        return round(risk, 2), {k: round(v, 4) for k, v in comp.items()}

    def _stage_term(self, stage: str | None = None) -> float:
        """Severity (0-1) of the predicted stage. The uncertain label is not in the table, so 0."""
        return self.stage_severity.get(stage or self.predicted_stage, 0.0) / 100.0

    def level(self, risk: float) -> str:
        return "low" if risk < self.levels["low"] else "medium" if risk < self.levels["medium"] else "high"

    def risk_config(self) -> dict[str, Any]:
        """Risk/confidence settings the dashboard needs, so it never hard-codes copies of them."""
        top_stage = (max(self.stage_severity.values(), default=0.0) / 100.0 if self.stage_model_dir is not None
                     else self._stage_term())
        max_reachable = 100 * (sum(w for k, w in self.weights.items() if k != "stage_severity")
                               + self.weights["stage_severity"] * top_stage)
        return {"weights": dict(self.weights),
                "thresholds": {"low": float(self.levels["low"]), "medium": float(self.levels["medium"])},
                "max_reachable_risk": round(max_reachable, 2),
                "confidence_threshold": self.conf_threshold,
                "confidence_calibrated": (MODEL_DIR / CALIBRATION_FILE).exists(), "uncertain_label": self.uncertain,
                "default_speed": self.default_speed, "allowed_speeds": self.allowed_speeds}

    # ------------------------------------------------------------------ control
    def _reset_state(self) -> None:
        self.state, self.speed, self.session_id = "idle", self.default_speed, None
        self.cursor, self.alerts_emitted, self.alerts_attack_label, self.sim_time = 0, 0, 0, None
        self.started_at = self.finished_at = self.error = None
        self.hosts: dict[str, dict[str, Any]] = {}
        self._recent: dict[str, deque] = {}

    def start(self, speed: int | None = None, reset: bool = True) -> dict[str, Any]:
        speed = int(speed or self.default_speed)
        if speed not in self.allowed_speeds:
            raise ValueError(f"speed must be one of {self.allowed_speeds}")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("replay already running; stop it first")
            self.prepare()
            # `reset` only starts a fresh session (host state, cursor, new session_id). The ledger is kept, so
            # earlier sessions stay verifiable; the API scopes alerts to the current session by default.
            self._reset_state()
            self.state, self.speed, self.session_id = "running", speed, uuid.uuid4().hex[:8]
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="replay", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            if self.state == "running":
                self.state = "stopped"
        return self.status()

    def _run(self) -> None:
        try:
            rows, starts = self._rows, self._starts
            n = len(rows)
            t0_sim = starts[0]
            t0_wall = time.monotonic()
            while self.cursor < n and not self._stop.is_set():
                sim_now = t0_sim + pd.Timedelta(seconds=(time.monotonic() - t0_wall) * self.speed)
                while self.cursor < n and starts[self.cursor] <= sim_now:
                    self._process(rows[self.cursor])
                    self.cursor += 1
                with self._lock:
                    self.sim_time = min(sim_now, starts[-1])
                time.sleep(0.05)
            with self._lock:
                if self.cursor >= n:
                    self.state = "finished"
                    self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception as exc:  # surface to /replay/status instead of dying silently
            with self._lock:
                self.state, self.error = "error", repr(exc)

    def _process(self, row) -> None:
        recent = self._recent.setdefault(row.host_id, deque())
        while recent and recent[0] < row.target_window_start - HISTORY_WINDOW:
            recent.popleft()
        risk, comp = self._risk(row, len(recent))
        alerted = row.lstm_p >= self.lstm_threshold
        alert_id = None
        if alerted:
            conf = float(row.conf)
            rec = self.ledger.append({
                "session_id": self.session_id, "sequence_id": row.sequence_id, "host_id": row.host_id,
                "predicted_at": row.predicted_at.isoformat(), "target_window_start": row.target_window_start.isoformat(),
                "target_window_end": row.target_window_end.isoformat(),
                "lstm_probability": row.lstm_p, "lstm_threshold": self.lstm_threshold,
                "lr_probability": row.lr_p, "lr_threshold": self.lr_threshold,
                "lr_flag": int(row.lr_p >= self.lr_threshold), "reference_rule_flag": int(row.rule),
                "risk_score": risk, "risk_components": comp, "confidence": conf,
                "confidence_label": self.uncertain if conf < self.conf_threshold else "LIKELY_ATTACK",
                "predicted_stage": self._row_stage(row),
                "truth_attack": int(row.truth_attack), "truth_class": row.truth_class, "truth_stage": row.truth_stage,
                "model_version": self.model_version,
            })
            alert_id = rec["id"]
            recent.append(row.target_window_start)
        with self._lock:
            h = self.hosts.setdefault(row.host_id, {"host_id": row.host_id, "sequences": 0, "alerts": 0,
                                                    "max_risk": 0.0, "timeline": []})
            h["sequences"] += 1
            h["alerts"] += int(alerted)
            h["latest_risk"], h["latest_lstm_p"] = risk, round(float(row.lstm_p), 4)
            h["max_risk"] = max(h["max_risk"], risk)
            h["last_seen"] = row.target_window_start.isoformat()
            h["timeline"].append({"t": row.target_window_start.isoformat(), "risk": risk,
                                  "lstm_p": round(float(row.lstm_p), 4), "alert_id": alert_id})
            if alerted:
                self.alerts_emitted += 1
                self.alerts_attack_label += int(row.truth_attack)

    # ------------------------------------------------------------------ views
    def status(self) -> dict[str, Any]:
        with self._lock:
            total = 0 if self.events is None else len(self.events)
            return {"state": self.state, "model_ready": self.events is not None, "speed": self.speed, "allowed_speeds": self.allowed_speeds,
                    "session_id": self.session_id, "processed": self.cursor, "total": total,
                    "progress": round(self.cursor / total, 4) if total else 0.0,
                    "alerts_emitted": self.alerts_emitted, "alerts_matching_attack_label": self.alerts_attack_label,
                    "hosts_seen": len(self.hosts),
                    "sim_time": None if self.sim_time is None else self.sim_time.isoformat(),
                    "replay_window": None if self.events is None else
                    [self.events.target_window_start.iloc[0].isoformat(), self.events.target_window_start.iloc[-1].isoformat()],
                    "started_at": self.started_at, "finished_at": self.finished_at, "error": self.error,
                    "lstm_threshold": getattr(self, "lstm_threshold", None)}

    def host_list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [{k: v for k, v in h.items() if k != "timeline"} | {"level": self.level(h["latest_risk"])}
                    for h in self.hosts.values()]
        return sorted(rows, key=lambda r: (-r["alerts"], -r["latest_risk"]))

    def host(self, host_id: str) -> dict[str, Any] | None:
        with self._lock:
            h = self.hosts.get(host_id)
            return None if h is None else {**h, "timeline": list(h["timeline"]), "level": self.level(h["latest_risk"])}
