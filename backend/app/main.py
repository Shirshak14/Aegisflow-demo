"""AegisFlow demo backend (FastAPI). Run: uvicorn backend.app.main:app --port 8000"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aegisflow.config import load_config

from . import info
from .audit import Ledger
from .replay import ReplayEngine

ROOT = Path(__file__).resolve().parents[2]


class ReplayStart(BaseModel):
    speed: int | None = None
    reset: bool = True


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

    @app.get("/audit/verify")
    def verify() -> dict[str, Any]:
        return ledger.verify()

    return app


app = create_app()
