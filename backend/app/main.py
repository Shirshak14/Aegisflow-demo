"""AegisFlow demo backend (FastAPI). Run: uvicorn backend.app.main:app --port 8000"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from aegisflow.config import load_config

from . import info
from .audit import Ledger
from .replay import ReplayEngine

ROOT = Path(__file__).resolve().parents[2]


class ReplayStart(BaseModel):
    speed: int | None = None
    reset: bool = True


class StreamFlows(BaseModel):
    flows: list[dict[str, Any]]


class AlertIn(BaseModel):
    session_id: str | None = None
    sequence_id: str
    host_id: str
    predicted_at: str | None = None
    target_window_start: str | None = None
    target_window_end: str | None = None
    lstm_probability: float | None = None
    lstm_threshold: float | None = None
    lr_probability: float | None = None
    lr_threshold: float | None = None
    lr_flag: int | None = None
    reference_rule_flag: int | None = None
    risk_score: float | None = None
    risk_components: dict[str, Any] | None = None
    confidence: float | None = None
    confidence_label: str | None = None
    predicted_stage: str | None = None
    truth_attack: int | None = None
    truth_class: str | None = None
    truth_stage: str | None = None
    model_version: str | None = None


def default_db_path() -> Path:
    if os.environ.get("AEGISFLOW_DB"):
        return Path(os.environ["AEGISFLOW_DB"])
    url = str(load_config().config.database.url)
    return ROOT / url.replace("sqlite:///", "")


def create_app(db_path: str | Path | None = None, warm_up: bool = True) -> FastAPI:
    app = FastAPI(title="AegisFlow demo API", version="0.1.0")
    ledger = Ledger(db_path or default_db_path())
    engine = ReplayEngine(ledger)
    app.state.ledger, app.state.engine = ledger, engine

    @app.exception_handler(FileNotFoundError)
    def missing_artifact(_: Request, exc: FileNotFoundError) -> JSONResponse:
        # Dataset files and trained models are not in the git repo; tell a fresh checkout what to run.
        name = Path(exc.filename).name if exc.filename else "a required file"
        return JSONResponse(status_code=503, content={"detail": (
            f"Missing {name}: processed data / trained model not found. Run "
            "`python -m aegisflow preprocess --dataset cic_ids2017` then "
            "`python -m aegisflow train --dataset cic_ids2017` (see README).")})
    if warm_up:
        engine.warm_up()

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "alerts_in_ledger": ledger.count(), "replay_state": engine.state}

    @app.get("/dataset/status")
    def dataset_status() -> dict[str, Any]:
        return info.dataset_status()

    @app.get("/model/status")
    def model_status() -> dict[str, Any]:
        return info.model_status()

    @app.get("/risk/config")
    def risk_config() -> dict[str, Any]:
        return engine.risk_config()

    @app.get("/evaluation")
    def evaluation() -> dict[str, Any]:
        return info.evaluation()

    @app.post("/replay/start")
    def replay_start(req: ReplayStart | None = None) -> dict[str, Any]:
        req = req or ReplayStart()
        try:
            return engine.start(speed=req.speed, reset=req.reset)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/replay/stop")
    def replay_stop() -> dict[str, Any]:
        return engine.stop()

    @app.get("/replay/status")
    def replay_status() -> dict[str, Any]:
        return engine.status()

    @app.get("/hosts")
    def hosts() -> list[dict[str, Any]]:
        return engine.host_list()

    @app.get("/hosts/{host_id}")
    def host(host_id: str) -> dict[str, Any]:
        h = engine.host(host_id)
        if h is None:
            raise HTTPException(404, f"host {host_id} not seen in the current replay")
        return h | {"alerts_list": ledger.list(limit=200, host_id=host_id)}

    @app.get("/mitre/{stage}")
    def mitre(stage: str) -> dict[str, Any]:
        m = info.mitre(stage)
        if m is None:
            raise HTTPException(404, f"unknown stage '{stage}'; known: {info.mitre_stages()}")
        return m

    @app.post("/alerts", status_code=201)
    def post_alert(alert: AlertIn) -> dict[str, Any]:
        """Internal: used by the replay engine to append an alert to the hash chain."""
        return ledger.append(alert.model_dump())

    @app.get("/alerts")
    def list_alerts(limit: int = 100, offset: int = 0, host_id: str | None = None) -> list[dict[str, Any]]:
        return ledger.list(limit=min(limit, 1000), offset=offset, host_id=host_id)

    @app.get("/alerts/{alert_id}")
    def get_alert(alert_id: int) -> dict[str, Any]:
        rec = ledger.get(alert_id)
        if rec is None:
            raise HTTPException(404, f"alert {alert_id} not found")
        return rec

    # ---- streaming telemetry (opt-in; not used by the dashboard); see aegisflow/ml/streaming.py
    stream_lock = threading.Lock()

    def stream_scorer():
        with stream_lock:
            if getattr(app.state, "stream", None) is None:
                from aegisflow.ml.streaming import StreamScorer
                cfg = load_config()
                st = cfg.config.get("stream") or {}
                model_dir = ROOT / str(st.get("model_dir") or "artifacts/models/cic_ids2017")
                app.state.stream = StreamScorer(model_dir, cfg, allowed_lateness=float(st.get("allowed_lateness_seconds") or 0))
            return app.state.stream

    @app.post("/stream/flows")
    def stream_flows(body: StreamFlows) -> dict[str, Any]:
        """Push canonical flow records (JSON objects with the aegisflow.schema columns), oldest first.
        Returns the host sequences scored because a window closed. Nothing is written to the ledger."""
        import pandas as pd
        from aegisflow.schema import REQUIRED_COLUMNS, coerce_canonical_frame
        df = pd.DataFrame(body.flows)
        if df.empty:
            return {"scored": [], "status": stream_scorer().status()}
        if "dataset_label" not in df:
            df["dataset_label"] = "UNLABELED"
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise HTTPException(422, f"flow records missing required fields: {missing}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        scored = stream_scorer().push(coerce_canonical_frame(df))
        return {"scored": scored, "status": stream_scorer().status()}

    @app.post("/stream/flush")
    def stream_flush() -> dict[str, Any]:
        return {"scored": stream_scorer().flush(), "status": stream_scorer().status()}

    @app.get("/stream/status")
    def stream_status() -> dict[str, Any]:
        return stream_scorer().status()

    @app.get("/stream/results")
    def stream_results(limit: int = 100, alerts_only: bool = False) -> list[dict[str, Any]]:
        rows = [r for r in stream_scorer().results if r["predicted_attack"] or not alerts_only]
        return rows[::-1][:max(1, min(limit, 1000))]

    @app.post("/stream/reset")
    def stream_reset() -> dict[str, Any]:
        stream_scorer().reset()
        return stream_scorer().status()

    @app.get("/audit/verify")
    def verify() -> dict[str, Any]:
        return ledger.verify()

    return app


_default_app: FastAPI | None = None
_default_app_lock = threading.Lock()


def __getattr__(name: str) -> Any:
    """Build the default ``app`` on first access instead of at import time.

    ``uvicorn backend.app.main:app`` looks the attribute up with ``getattr``, so it still gets a
    fully built app. Merely importing this module (e.g. for ``create_app`` in tests) no longer
    opens the live ledger DB, starts the model warm-up thread, or loads torch.
    """
    global _default_app
    if name == "app":
        with _default_app_lock:
            if _default_app is None:
                _default_app = create_app()
            return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
