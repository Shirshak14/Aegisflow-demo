"""Structured logging: human-readable console lines ``[STAGE] message k=v`` plus optional JSONL file."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_NAME = "aegisflow"


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", "APP")
        kv = getattr(record, "kv", {}) or {}
        tail = "".join(f" {k}={v}" for k, v in kv.items())
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{ts} [{stage}] {record.getMessage()}{tail}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "stage": getattr(record, "stage", "APP"),
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "kv", {}) or {})
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_file: Path | None = None) -> None:
    """Configure the ``aegisflow`` logger once (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ConsoleFormatter())
    logger.addHandler(console)
    if json_file is not None:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(json_file, encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)


class StageLogger:
    """Logger bound to a pipeline stage tag such as INGEST, VALIDATE, EDA."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._log = logging.getLogger(_LOGGER_NAME)

    def _emit(self, level: int, msg: str, kv: dict[str, Any]) -> None:
        self._log.log(level, msg, extra={"stage": self.stage, "kv": kv})

    def debug(self, msg: str, **kv: Any) -> None:
        self._emit(logging.DEBUG, msg, kv)

    def info(self, msg: str, **kv: Any) -> None:
        self._emit(logging.INFO, msg, kv)

    def warning(self, msg: str, **kv: Any) -> None:
        self._emit(logging.WARNING, msg, kv)

    def error(self, msg: str, **kv: Any) -> None:
        self._emit(logging.ERROR, msg, kv)


def get_logger(stage: str) -> StageLogger:
    return StageLogger(stage)
