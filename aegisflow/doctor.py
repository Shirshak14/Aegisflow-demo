"""``python -m aegisflow doctor``: list which files the demo needs that are missing, and how to create each."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATASET = "cic_ids2017"
_PREPROCESS = f"python -m aegisflow preprocess --dataset {DATASET}"
_TRAIN = f"python -m aegisflow train --dataset {DATASET}"


@dataclass(frozen=True)
class Check:
    group: str
    label: str
    path: str  # relative to the repo root; a glob when it contains '*'
    fix: str
    needed_for: str


CHECKS: tuple[Check, ...] = (
    Check("raw data", "CIC-IDS2017 CSVs", f"data/raw/{DATASET}/*/*.pcap_ISCX.csv",
          f"python scripts/download_data.py --dataset {DATASET}  (prints where to download, then unpack under "
          f"data/raw/{DATASET}/)", "preprocess (only if data/processed is missing)"),
    Check("processed data", "sequences", f"data/processed/{DATASET}/sequences.parquet", _PREPROCESS,
          "replay, /dataset/status, training"),
    Check("processed data", "split metadata", f"data/processed/{DATASET}/split_metadata.json", _PREPROCESS,
          "/dataset/status, /evaluation"),
    Check("model", "LSTM weights", f"artifacts/models/{DATASET}/best.pt", _TRAIN, "replay, /model/status"),
    Check("model", "logistic regression", f"artifacts/models/{DATASET}/logistic.joblib", _TRAIN, "replay"),
    Check("model", "preprocessor", f"artifacts/models/{DATASET}/preprocessor.joblib", _TRAIN, "replay"),
    Check("model", "model metadata", f"artifacts/models/{DATASET}/metadata.json", _TRAIN, "replay, /model/status"),
    Check("model", "model metrics", f"artifacts/models/{DATASET}/metrics.json", _TRAIN, "replay, /evaluation"),
    Check("reports", "data quality report", "reports/data_quality_report.json", _PREPROCESS, "/dataset/status"),
    Check("reports", "onset baselines", "reports/phase3_baselines_onset.json",
          "python scripts/phase3_baselines.py  (or git checkout the committed copy)", "/evaluation"),
    Check("reports", "LODO pilot results", "reports/lodo_pilot_results.json",
          "python scripts/lodo_pilot.py  (or git checkout the committed copy)", "/evaluation"),
)


def _exists(root: Path, pattern: str) -> bool:
    if "*" in pattern:
        return next(root.glob(pattern), None) is not None
    return (root / pattern).exists()


def run_checks(root: Path) -> list[tuple[Check, bool]]:
    return [(c, _exists(root, c.path)) for c in CHECKS]


def format_report(results: list[tuple[Check, bool]]) -> str:
    lines: list[str] = []
    for check, ok in results:
        lines.append(f"[{'ok' if ok else 'MISSING':>7}] {check.group}: {check.label}  ({check.path})")
        if not ok:
            lines.append(f"          create with: {check.fix}")
            lines.append(f"          needed for : {check.needed_for}")
    missing = [c for c, ok in results if not ok]
    if not missing:
        lines.append("\nAll required files are present. Start the demo: uvicorn backend.app.main:app --port 8000")
    else:
        blocking = [c for c in missing if c.group != "raw data"]
        lines.append(f"\n{len(missing)} file(s) missing, {len(blocking)} of them block the demo "
                     "(the raw CSVs are only needed to re-run preprocess).")
    return "\n".join(lines)


def demo_ready(results: list[tuple[Check, bool]]) -> bool:
    """True when everything the server reads exists (raw CSVs are optional once data is processed)."""
    return all(ok for c, ok in results if c.group != "raw data")
