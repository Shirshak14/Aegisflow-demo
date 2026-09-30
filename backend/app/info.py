"""Read-only status, MITRE lookup and evaluation numbers, all taken from existing artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/cic_ids2017"
MODEL_DIR = ROOT / "artifacts/models/cic_ids2017"
REPORTS = ROOT / "reports"

HOST_NOTE = ("Known limitation: on CIC-IDS2017 the model's detections are largely attacker-host recognition "
             "(172.16.0.1), not generalisable attack detection; unseen attacks (Botnet) are not detected.")
STAGE_NOTE = ("The Phase 3 model is a binary attack/benign detector. It does not predict a kill-chain stage, "
              "so predicted_stage is always UNCERTAIN and the stage-severity risk term is 0.")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mtime(path: Path) -> str | None:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None


def dataset_status() -> dict[str, Any]:
    split = _read(DATA_DIR / "split_metadata.json")
    q = _read(REPORTS / "data_quality_report.json")
    return {
        "dataset": "CIC-IDS2017 (GeneratedLabelledFlows)",
        "sample": "500k-row seeded spread-out sample (same fraction of every day-file); not the full 2.83M rows",
        "raw_rows_sampled": q["raw_rows_loaded"], "cleaned_flows": q["cleaned_rows_kept"],
        "time_range": [q["temporal_range_start"], q["temporal_range_end"]],
        "hosts": q["unique_hosts_count"], "host_windows": q["total_host_windows"],
        "sequences": q["total_sequences"], "positive_sequences": q["target_attack_sequences"],
        "split": {k: split[k] for k in ("val_cutoff_time", "test_cutoff_time", "train_sequences", "val_sequences",
                                        "test_sequences", "boundary_excluded_sequences")},
        "positives_by_split": q.get("onset_by_split"),
        "files": {p.name: _mtime(p) for p in (DATA_DIR / "sequences.parquet", DATA_DIR / "split_metadata.json")},
    }


@lru_cache(maxsize=1)
def model_version() -> str:
    digest = hashlib.sha256((MODEL_DIR / "best.pt").read_bytes()).hexdigest()[:12]
    return f"phase3-chronological-lstm-best.pt-{digest}"


def model_status() -> dict[str, Any]:
    meta = _read(MODEL_DIR / "metadata.json")
    metrics = _read(MODEL_DIR / "metrics.json")
    return {
        "model": "Phase 3 chronological-split baselines (LSTM primary, logistic regression alongside)",
        "version": model_version(),
        "lodo_pilot_models_used": False,
        "input_shape": meta["input_shape"], "feature_names": meta["feature_names"],
        "lstm_threshold": metrics["threshold"],
        "lr_threshold": metrics["models"]["logistic_regression"]["threshold"],
        "threshold_selection": metrics["threshold_selection"] + " (validation split only)",
        "training": metrics["training"],
        "test_metrics": {m: {k: v for k, v in x.items() if k != "threshold"} for m, x in metrics["models"].items()},
        "stage_prediction": "not available", "stage_note": STAGE_NOTE, "limitation": HOST_NOTE,
        "artifacts": {p: _mtime(MODEL_DIR / p) for p in ("best.pt", "logistic.joblib", "preprocessor.joblib",
                                                         "metadata.json", "metrics.json")},
    }


@lru_cache(maxsize=1)
def _mitre_table() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/mitre_mapping.yaml").read_text(encoding="utf-8"))


def mitre(stage: str) -> dict[str, Any] | None:
    table = _mitre_table()
    key = next((k for k in table if k.lower() == stage.strip().lower()), None)
    if key is None:
        return None
    return {"stage": key, **table[key],
            "source": "configs/mitre_mapping.yaml (static lookup; no model involved)"}


def mitre_stages() -> list[str]:
    return list(_mitre_table())


def _pick(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "confusion_matrix")
    return {k: m.get(k) for k in keys}


def evaluation() -> dict[str, Any]:
    """Real numbers from the Phase 3 reports; nothing recomputed or rounded for effect."""
    b = _read(REPORTS / "phase3_baselines_onset.json")
    names = {"majority": "Majority baseline", "logistic_regression": "Logistic regression",
             "lstm_existing_checkpoint": "LSTM (demo model)",
             "last_window_under_attack_rule": "Reference: 'last window under attack' (uses labels)"}
    table = {}
    for variant in ("any", "onset"):
        models = b["variants"][variant]["models"]
        table[variant] = [{"model": names[m], "val": _pick(models[m]["val"]), "test": _pick(models[m]["test"])}
                          for m in names if m in models]
    host = b["variants"]["any"]["models"]["lstm_existing_checkpoint"]["host_check"]["test"]["shortcut_host_vs_rest"]
    lodo = _read(REPORTS / "lodo_pilot_results.json")["folds"]
    dos = {}
    for fold, label in (("Wed_2017-07-05", "Fri DDoS -> Wed DoS"), ("Fri_2017-07-07", "Wed DoS -> Fri DDoS")):
        c = lodo[fold]["classes"]["Denial of Service"]["models"]
        dos[label] = {m: {"detected": c[m]["detected"], "positives": lodo[fold]["classes"]["Denial of Service"]["positives"],
                          "roc_auc_vs_attacker_host_negatives": c[m]["roc_auc_vs_172.16.0.1_negatives"]}
                      for m in ("logistic_regression", "lstm", "ref_host_identity")}
    return {
        "source": ["reports/phase3_baselines_onset.json", "reports/lodo_pilot_results.json"],
        "targets": {"any": "attack present in the next window", "onset": "attack starts with no attack in the 10 input windows"},
        "table": table,
        "host_check_test_lstm": host,
        "lodo_dos_cross_day": dos,
        "honest_note": HOST_NOTE,
    }
