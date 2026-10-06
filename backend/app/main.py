"""AegisFlow demo backend (FastAPI). Run: uvicorn backend.app.main:app --port 8000"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aegisflow.config import load_config

from . import features, info
from .analyst import ActionError, AnalystLog
from .audit import Ledger
from .replay import ReplayEngine

ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = 200 * 2**20


class ReplayStart(BaseModel):
    speed: int | None = None
    reset: bool = True


class ActionIn(BaseModel):
    action: str
    analyst: str
    note: str | None = None
    response: str | None = None
    stage: str | None = None


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
    analyst = AnalystLog(ledger, list(load_config().stages.stages_order))
    app.state.ledger, app.state.engine, app.state.analyst = ledger, engine, analyst

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

    static = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.get("/classic", include_in_schema=False)
    def classic_dashboard() -> FileResponse:
        """The original single-file dashboard, kept as a fallback."""
        return FileResponse(static / "classic.html")

    # Dashboard assets, including the vendored React / htm builds (no CDN, so the demo works offline).
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/features")
    def features_status() -> dict[str, Any]:
        """Which opt-in features this install can use (SHAP, stage model, PCAP readers, streaming model)."""
        st = load_config().config.get("stream") or {}
        return features.capabilities(stage_model_dir=engine.stage_model_dir,
                                     stream_model_dir=ROOT / str(st.get("model_dir") or "artifacts/models/cic_ids2017")
                                     ) | {"stages": analyst.stages}

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        """Trained models under artifacts/models with the test metrics their trainer wrote (nothing recomputed)."""
        return features.list_models(demo_dir=info.MODEL_DIR)

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

    @app.get("/explain/{sequence_id}")
    def explain(sequence_id: str, model: str = "lstm", method: str = "shap", top: int = 5) -> dict[str, Any]:
        """Per-feature / per-time-step attribution of one test-split prediction."""
        if model not in {"lstm", "logistic_regression"} or method not in {"shap", "integrated_gradients"}:
            raise HTTPException(422, "model must be lstm|logistic_regression, method shap|integrated_gradients")
        try:
            out = engine.explain(sequence_id, model=model, method=method, top=max(1, min(top, 28)))
        except ImportError:
            raise HTTPException(501, "SHAP is not installed (pip install shap); use method=integrated_gradients")
        if out is None:
            raise HTTPException(404, f"sequence {sequence_id} is not in the replayed test split")
        return out

    @app.get("/forecast/{sequence_id}")
    def forecast(sequence_id: str) -> dict[str, Any]:
        """Stage distribution, MITRE tactic and K-step future state from the opt-in multi-task model."""
        try:
            out = engine.forecast(sequence_id)
        except LookupError as exc:
            raise HTTPException(409, f"{exc}; train one with `python -m aegisflow train-multitask`")
        if out is None:
            raise HTTPException(404, f"sequence {sequence_id} is not in the replayed test split")
        return out

    @app.get("/attention/{sequence_id}")
    def attention(sequence_id: str, model: str) -> dict[str, Any]:
        """Per-window attention weights of a trained attention_lstm / transformer model (see GET /models)."""
        model_dir = (features.MODELS_DIR / model).resolve()
        if model_dir.parent != features.MODELS_DIR.resolve() or not (model_dir / "best.pt").exists():
            raise HTTPException(404, f"no trained model named {model!r} in artifacts/models")
        try:
            out = engine.attention(sequence_id, model_dir)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        if out is None:
            raise HTTPException(404, f"sequence {sequence_id} is not in the replayed test split")
        return out

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
    def list_alerts(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                    host_id: str | None = None) -> list[dict[str, Any]]:
        return ledger.list(limit=limit, offset=offset, host_id=host_id)

    @app.get("/alerts/{alert_id}")
    def get_alert(alert_id: int) -> dict[str, Any]:
        rec = ledger.get(alert_id)
        if rec is None:
            raise HTTPException(404, f"alert {alert_id} not found")
        return rec

    # ---- streaming telemetry (opt-in); see aegisflow/ml/streaming.py
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

    @app.post("/ingest/upload")
    async def ingest_upload(file: UploadFile, kind: str = "auto", reset: bool = True) -> dict[str, Any]:
        """Upload a packet capture (kind=pcap) or NetFlow v5/v9/IPFIX export capture / nfdump CSV
        (kind=netflow), turn it into flows and play them through the streaming scorer.
        Results land in GET /stream/results; nothing is written to the alert ledger."""
        import tempfile

        suffix = Path(file.filename or "upload").suffix.lower()
        if kind == "auto":
            kind = "netflow" if suffix == ".csv" else "pcap"
        if kind not in {"pcap", "netflow"}:
            raise HTTPException(422, "kind must be auto, pcap or netflow")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 2**20} MB; use the CLI for big captures")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix or '.bin'}"
            path.write_bytes(data)
            try:
                if kind == "pcap":
                    from aegisflow.ml.ingestion.pcap import pcap_to_flows
                    flows = pcap_to_flows(path)
                else:
                    from aegisflow.ml.ingestion.netflow import netflow_to_flows
                    flows = netflow_to_flows(path)
            except ImportError as exc:
                raise HTTPException(501, f"PCAP reader not installed: {exc}")
            except Exception as exc:  # malformed capture: report it, never crash the server
                raise HTTPException(422, f"could not read {file.filename} as {kind}: {exc}")
        scorer = stream_scorer()
        if reset:
            scorer.reset()
        scored = scorer.push(flows.sort_values("timestamp")) if len(flows) else []
        scored += scorer.flush()
        return {"file": file.filename, "kind": kind, "flows": int(len(flows)),
                "source_hosts": int(flows["source_ip"].nunique()) if len(flows) else 0,
                "sequences_scored": len(scored), "alerts": sum(int(r["predicted_attack"]) for r in scored),
                "status": scorer.status()}

    @app.post("/stream/reset")
    def stream_reset() -> dict[str, Any]:
        stream_scorer().reset()
        return stream_scorer().status()

    @app.get("/export/alerts", response_class=PlainTextResponse)
    def export_alerts(format: str = "cef", since_id: int = 0, limit: int = 1000) -> str:
        """Alerts with id > since_id as CEF, RFC 5424 syslog or JSON lines (one event per line), for a SIEM."""
        from .siem import FORMATS, alerts_since, render
        if format not in FORMATS:
            raise HTTPException(422, f"format must be one of {FORMATS}")
        lines = render(alerts_since(ledger, since_id, max(1, min(limit, 10000))), format)
        return "\n".join(lines) + ("\n" if lines else "")

    @app.get("/audit/verify")
    def verify() -> dict[str, Any]:
        return ledger.verify()

    # ---- analyst review; see backend/app/analyst.py
    @app.post("/alerts/{alert_id}/actions", status_code=201)
    def add_action(alert_id: int, body: ActionIn) -> dict[str, Any]:
        """Record an analyst decision on an alert. Approving a response records it; nothing is executed."""
        try:
            return analyst.record(alert_id, body.action, body.analyst, note=body.note, response=body.response,
                                  stage=body.stage)
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        except ActionError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/alerts/{alert_id}/actions")
    def alert_actions(alert_id: int) -> dict[str, Any]:
        h = analyst.history(alert_id)
        if h is None:
            raise HTTPException(404, f"alert {alert_id} not found")
        return h

    @app.get("/triage/statuses")
    def triage_statuses() -> dict[int, str]:
        return analyst.statuses()

    @app.get("/triage/summary")
    def triage_summary() -> dict[str, int]:
        return analyst.summary()

    @app.get("/audit/verify-actions")
    def verify_actions() -> dict[str, Any]:
        return analyst.verify()

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
