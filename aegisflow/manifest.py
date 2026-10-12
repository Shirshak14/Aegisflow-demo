"""Model/artifact manifest: what exactly a trained model directory contains, and how it was made.

``build_manifest`` records, for one model directory: SHA-256 and size of every artifact file, the feature schema
(names, order, input shape, and a hash over them), the alert threshold and how it was selected, the training
configuration (hyperparameters, seed, epochs), the windowing configuration, the processed-data split it was trained
on, and the library versions. ``verify_manifest`` recomputes the hashes and reports any file that changed, vanished or
was added. The manifest describes a model; it does not prove the model is good (see the evaluation reports).

Weight files are not bit-for-bit reproducible across machines and torch builds, so a rebuilt model gets a NEW
manifest rather than matching an old hash. Use the manifest to check that the files you run are the files you recorded.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
ARTIFACT_FILES = ("best.pt", "last.pt", "logistic.joblib", "preprocessor.joblib", "metadata.json", "metrics.json",
                  "calibration.json", "anomaly.joblib")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for mod in ("torch", "numpy", "pandas", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def build_manifest(model_dir: Path, *, root: Path | None = None, config: Any = None,
                   split_metadata: Path | None = None) -> dict[str, Any]:
    """Describe ``model_dir``. ``config`` is an ``AegisFlowConfig`` (for the windowing section), optional."""
    model_dir = Path(model_dir)
    meta, metrics = _read_json(model_dir / "metadata.json"), _read_json(model_dir / "metrics.json")
    files = {p: {"sha256": sha256_file(model_dir / p), "bytes": (model_dir / p).stat().st_size}
             for p in ARTIFACT_FILES if (model_dir / p).exists()}
    names = list(meta["feature_names"])
    schema_hash = hashlib.sha256(json.dumps([names, list(meta["input_shape"])]).encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": model_dir.name,
        "files": files,
        "feature_schema": {"feature_names": names, "input_shape": list(meta["input_shape"]),
                           "n_features": len(names), "schema_sha256": schema_hash},
        "threshold": {"lstm": metrics.get("threshold"),
                      "logistic_regression": metrics.get("models", {}).get("logistic_regression", {}).get("threshold"),
                      "selection": metrics.get("threshold_selection")},
        "training": metrics.get("training"),
        "test_metrics": {m: {k: v for k, v in x.items() if k != "threshold"}
                         for m, x in metrics.get("models", {}).items()},
        "software": {**_versions(), "git_commit": _git_commit(root) if root else None},
    }
    if config is not None:
        manifest["windowing"] = dict(config.config.windowing)
        manifest["random_seed"] = config.config.get("random_seed")
    if split_metadata is not None and Path(split_metadata).exists():
        s = _read_json(Path(split_metadata))
        manifest["data_split"] = {k: s.get(k) for k in ("train_sequences", "val_sequences", "test_sequences",
                                                        "total_sequences", "val_cutoff_time", "test_cutoff_time")}
    return manifest


def write_manifest(model_dir: Path, **kw: Any) -> Path:
    path = Path(model_dir) / MANIFEST_NAME
    path.write_text(json.dumps(build_manifest(model_dir, **kw), indent=2), encoding="utf-8")
    return path


def verify_manifest(model_dir: Path) -> dict[str, Any]:
    """Recompute artifact hashes against ``manifest.json``: ``{"ok", "changed", "missing", "unlisted"}``."""
    model_dir = Path(model_dir)
    path = model_dir / MANIFEST_NAME
    if not path.exists():
        return {"ok": False, "error": f"no {MANIFEST_NAME} in {model_dir}; run `python -m aegisflow manifest`"}
    listed = _read_json(path)["files"]
    changed = [f for f, v in listed.items() if (model_dir / f).exists() and sha256_file(model_dir / f) != v["sha256"]]
    missing = [f for f in listed if not (model_dir / f).exists()]
    unlisted = [f for f in ARTIFACT_FILES if (model_dir / f).exists() and f not in listed]
    return {"ok": not (changed or missing or unlisted), "changed": changed, "missing": missing, "unlisted": unlisted}


BUNDLE_NOTICE = """AegisFlow model bundle

Contains: trained model files, preprocessor, metadata/metrics and manifest.json (hashes, feature schema, threshold,
training configuration). Contains NO dataset records: no flows, no sequences, no raw or processed CIC-IDS2017 data.

The model was trained on CIC-IDS2017 (Sharafaldin, Lashkari, Ghorbani, ICISSP 2018). The dataset page names no licence
and does not address redistribution or derived works, only asks for citation; whether publishing weights trained on it
is acceptable is the publisher's decision. Cite the paper if you use or share this bundle.

Verify after unpacking:  python -m aegisflow manifest --model-dir <unpacked dir> --verify
"""


def make_bundle(model_dir: Path, out_zip: Path) -> Path:
    """Zip the model files plus manifest.json (written first if missing). The data directories are never included."""
    import zipfile
    model_dir, out_zip = Path(model_dir), Path(out_zip)
    if not (model_dir / MANIFEST_NAME).exists():
        write_manifest(model_dir)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in (*ARTIFACT_FILES, MANIFEST_NAME):
            if (model_dir / name).exists():
                z.write(model_dir / name, f"model/{name}")
        z.writestr("NOTICE.txt", BUNDLE_NOTICE)
    return out_zip


def verify_bundle(bundle: Path, workdir: Path) -> dict[str, Any]:
    """Unpack ``bundle`` into the (empty) ``workdir`` and verify it against its own manifest."""
    import zipfile
    with zipfile.ZipFile(bundle) as z:
        z.extractall(workdir)
    return verify_manifest(Path(workdir) / "model")
