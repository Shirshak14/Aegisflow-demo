"""Routes for the investigation features: temporal context, model health and drift, evidence export.

Kept out of main.py as one router; every route reads the replay engine, ledger and analyst log from
``request.app.state`` (set in ``create_app``) and none of them changes replay results or the ledger.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import evidence, health, temporal_context

router = APIRouter()


class EvidenceIn(BaseModel):
    alert_ids: list[int] = Field(..., min_length=1, max_length=evidence.MAX_ALERTS)
    include_ground_truth: bool = False


@router.get("/context/summary")
def context_summary() -> dict[str, Any]:
    """How variable input spans and forecast leads are across the test split."""
    return temporal_context.split_summary()


@router.get("/context/{sequence_id}")
def sequence_context(sequence_id: str) -> dict[str, Any]:
    """Input-history span, silent windows, gap before the last input window and lead to the target window."""
    out = temporal_context.context(sequence_id)
    if out is None:
        raise HTTPException(404, f"sequence {sequence_id} is not in the sequence table")
    return out


@router.get("/health/model")
def model_health(request: Request) -> dict[str, Any]:
    """Label-free health and drift of the sequences replayed so far against training (see health.py)."""
    return health.health(request.app.state.engine)


@router.post("/evidence")
def export_evidence(body: EvidenceIn, request: Request) -> JSONResponse:
    """Investigation bundle for the given alerts, as a JSON download. Ground truth only when asked for."""
    out = evidence.bundle(request.app.state, body.alert_ids, include_ground_truth=body.include_ground_truth)
    if not out["alerts"]:
        raise HTTPException(404, f"none of the alerts {body.alert_ids[:5]} are in the ledger")
    first = out["scope"]["alert_ids"][0]
    name = f"aegisflow-evidence-{first}{'-plus' + str(len(out['alerts']) - 1) if len(out['alerts']) > 1 else ''}.json"
    return JSONResponse(out, headers={"Content-Disposition": f'attachment; filename="{name}"'})
