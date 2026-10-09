"""Discovery helpers for the dashboard: which opt-in features are usable here, and which trained models exist.

Read-only. Nothing here trains, scores or writes; metrics are copied from each model's own metrics.json.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "artifacts" / "models"
METRIC_KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall", "false_positive_rate")


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def model_type(meta: dict[str, Any]) -> str:
    """Architecture recorded by the trainer: metadata ``model_type`` (multi-task, GNN) or ``training_config.model_type``."""
    return str(meta.get("model_type") or meta.get("training_config", {}).get("model_type") or "lstm")


def _attack_metrics(m: Any) -> dict[str, float] | None:
    if isinstance(m, dict) and "roc_auc" in m:
        return {k: m[k] for k in METRIC_KEYS if isinstance(m.get(k), (int, float))}
    return None


def test_metrics(metrics: dict[str, Any], mtype: str) -> dict[str, float] | None:
    """Held-out test attack metrics as the trainer wrote them, or None.

    `train` and `train-gnn` write test metrics per model under ``models.<model type>``;
    `train-multitask` writes them under ``attack``. Nothing is recomputed.
    """
    models = metrics.get("models")
    if isinstance(models, dict):
        return _attack_metrics(models.get(mtype))
    return _attack_metrics(metrics.get("attack"))


def list_models(models_dir: Path = MODELS_DIR, demo_dir: Path | None = None) -> list[dict[str, Any]]:
    out = []
    for d in sorted(p for p in models_dir.glob("*") if (p / "metadata.json").exists()):
        meta, metrics = _read(d / "metadata.json"), _read(d / "metrics.json")
        mtype = model_type(meta)
        weights = d / "best.pt"
        extra: dict[str, Any] = {}
        test = (metrics.get("splits") or {}).get("test") or {}
        if mtype == "multitask_lstm":
            st = metrics.get("stage", {})
            extra = {"stage_accuracy": st.get("accuracy"), "stage_majority_baseline": st.get("majority_baseline", {}).get("accuracy"),
                     "attack_stage_accuracy": st.get("attack_stage_accuracy"), "horizons": meta.get("horizons")}
        out.append({
            "name": d.name, "model_type": mtype, "dataset": meta.get("dataset") or metrics.get("dataset"),
            "demo_model": demo_dir is not None and d.resolve() == demo_dir.resolve(),
            "trained_at": (datetime.fromtimestamp(weights.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                           if weights.exists() else None),
            "test_metrics": test_metrics(metrics, mtype), "threshold": metrics.get("threshold", meta.get("threshold")),
            # Split sizes travel with the metrics: models trained on different data are not comparable.
            "test_sequences": test.get("sequences"), "test_positive_targets": test.get("positive_targets"),
            "has_attention": mtype in {"attention_lstm", "transformer"}, **extra,
        })
    return out


def capabilities(*, stage_model_dir: Path | None, stream_model_dir: Path) -> dict[str, Any]:
    """What the dashboard can offer on this install, and how to enable what is missing."""
    return {
        "explanations": {"shap": _has("shap"), "integrated_gradients": True,
                         "hint": None if _has("shap") else "pip install shap (Integrated Gradients works without it)"},
        "stage_model": {"configured": stage_model_dir is not None,
                        "path": None if stage_model_dir is None else str(stage_model_dir.relative_to(ROOT)),
                        "available": stage_model_dir is not None and (stage_model_dir / "best.pt").exists(),
                        "hint": "python -m aegisflow train-multitask, then set replay.stage_model_dir in configs/config.yaml"},
        "pcap": {"scapy": _has("scapy"), "pyshark": _has("pyshark") and shutil.which("tshark") is not None},
        "netflow": {"available": True, "formats": ["NetFlow v5/v9/IPFIX export capture (.pcap)", "nfdump CSV (.csv)"]},
        "stream": {"model_dir": str(stream_model_dir.relative_to(ROOT)) if stream_model_dir.is_relative_to(ROOT)
                   else str(stream_model_dir), "available": (stream_model_dir / "best.pt").exists()},
        "siem_formats": ["cef", "syslog", "jsonl"],
    }
