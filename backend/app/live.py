"""Server-sent events for the dashboard (B2): one stream instead of five polled endpoints.

``GET /live`` sends a ``snapshot`` event (replay status, hosts, latest alerts, triage statuses and
summary: the same bodies as the five REST endpoints) whenever that snapshot changes, checked every
``interval`` seconds in-process. Nothing is sent while nothing changes, apart from a comment line
every ``heartbeat`` seconds so proxies keep the connection open. The REST endpoints stay as they were.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi.encoders import jsonable_encoder

INTERVAL_S = 0.5
HEARTBEAT_S = 15.0


def encode(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data), separators=(',', ':'))}\n\n"


async def snapshot_events(snapshot: Callable[[], dict[str, Any]], is_disconnected: Callable[[], Awaitable[bool]], *,
                          interval: float = INTERVAL_S, heartbeat: float = HEARTBEAT_S,
                          max_events: int | None = None) -> AsyncIterator[str]:
    """Yield SSE frames: a snapshot on connect and on every change, a heartbeat comment when idle."""
    last, sent, quiet_since = None, 0, time.monotonic()
    yield "retry: 2000\n\n"  # browser reconnect delay after a dropped connection
    while not await is_disconnected():
        try:
            body = encode("snapshot", snapshot())
        except Exception as exc:  # e.g. model still loading or a missing artifact: tell the page, keep the stream
            body = encode("error", {"detail": str(exc)})
        if body != last:
            last, quiet_since = body, time.monotonic()
            yield body
            sent += 1
            if max_events is not None and sent >= max_events:
                return
        elif time.monotonic() - quiet_since >= heartbeat:
            quiet_since = time.monotonic()
            yield ": keep-alive\n\n"
        await asyncio.sleep(interval)
