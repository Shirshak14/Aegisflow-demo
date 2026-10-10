"""Golden-file harness for CIC-IDS2017 ingestion (B5).

Runs the real ``run_ingestion`` with its output redirected to a scratch directory (the committed
``data/interim`` is never touched), and records what it produced and what it cost:

    python scripts/golden_ingest.py --out <dir> [--sample-size N] [--only-file SUBSTR]
    python scripts/golden_ingest.py --out <new_dir> ... --compare <golden_dir>

``--out`` receives ``cic_ids2017.parquet``, ``cic_ids2017.ingest.json`` and ``golden_record.json``
(SHA-256 of both outputs, row counts, schema, wall time, peak memory, git commit).
``--compare`` then checks the new output against a golden directory:
  * ``pandas.testing.assert_frame_equal(check_exact=True)`` (values, dtypes, column and row order),
  * identical Arrow schema,
  * identical ``ingest.json`` (row counts and drops by reason),
and reports whether the Parquet bytes are identical too (informational: the frame comparison is the
pass/fail criterion, because Parquet metadata is allowed to differ).

Run each configuration in its own process so the peak-memory figure is that run's alone.
Peak memory is the process peak working set (Windows) or sampled RSS (elsewhere), via psutil.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = "cic_ids2017"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _PeakMemory:
    """Peak resident memory of this process in MiB."""

    def __init__(self) -> None:
        import psutil

        self._proc, self._peak, self._stop = psutil.Process(), 0, threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def mib(self) -> float:
        info = self._proc.memory_info()
        peak = max(self._peak, getattr(info, "peak_wset", 0) or 0, info.rss)  # peak_wset exists on Windows only
        return round(peak / 2**20, 1)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(out: Path, sample_size: int | None, only_file: str | None) -> dict:
    import pyarrow.parquet as pq

    from aegisflow.config import load_config
    from aegisflow.ml.datasets.cic_ids2017 import CicIds2017Adapter
    from aegisflow.ml.ingestion.pipeline import run_ingestion

    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg.config.paths.data_interim = str(out.resolve())  # absolute path: cfg.path(...) then resolves to it

    if only_file:  # restrict discovery to the matching CSV(s), e.g. the smallest and the largest day-file
        original = CicIds2017Adapter.discover

        def discover(self):
            report = original(self)
            report.files = [f for f in report.files if only_file.lower() in f.name.lower()]
            if not report.files:
                raise SystemExit(f"--only-file {only_file!r} matched no CSV")
            return report

        CicIds2017Adapter.discover = discover

    t0 = time.perf_counter()
    with _PeakMemory() as mem:
        result = run_ingestion(cfg, DATASET, sample_size=sample_size)
    wall = time.perf_counter() - t0

    parquet, summary = out / f"{DATASET}.parquet", out / f"{DATASET}.ingest.json"
    record = {
        "git_commit": _git_commit(), "sample_size": sample_size, "only_file": only_file,
        "rows_raw": result.rows_raw, "rows_clean": result.cleaning.rows_out,
        "wall_seconds": round(wall, 2), "peak_memory_mib": mem.mib,
        "parquet_sha256": _sha256(parquet), "ingest_json_sha256": _sha256(summary),
        "parquet_bytes": parquet.stat().st_size, "arrow_schema": str(pq.read_schema(parquet)),
    }
    (out / "golden_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def compare(new: Path, golden: Path) -> bool:
    import pandas as pd
    import pyarrow.parquet as pq

    ok = True
    a, b = golden / f"{DATASET}.parquet", new / f"{DATASET}.parquet"
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)
        print("frame values, dtypes, column and row order: IDENTICAL (assert_frame_equal, check_exact=True)")
    except AssertionError as exc:
        ok = False
        print(f"FRAME DIFFERS: {str(exc)[:600]}")
    same_schema = pq.read_schema(a).equals(pq.read_schema(b), check_metadata=False)
    print(f"Arrow schema (columns and types): {'IDENTICAL' if same_schema else 'DIFFERS'}")
    ok &= same_schema
    same_json = json.loads((golden / f"{DATASET}.ingest.json").read_text()) == json.loads(
        (new / f"{DATASET}.ingest.json").read_text())
    print(f"ingest.json counts: {'IDENTICAL' if same_json else 'DIFFERS'}")
    ok &= same_json
    print(f"Parquet bytes (informational): {'IDENTICAL' if _sha256(a) == _sha256(b) else 'differ'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="scratch directory for this run's outputs")
    ap.add_argument("--sample-size", type=int, default=None, help="spread-out row sample (default: all rows)")
    ap.add_argument("--only-file", default=None, help="only read CSV files whose name contains this text")
    ap.add_argument("--compare", type=Path, default=None, help="golden directory to compare the new output against")
    args = ap.parse_args()
    if (ROOT / "data") in [args.out.resolve(), *args.out.resolve().parents]:
        raise SystemExit("refusing to write under the repo's data/ directory; use a scratch directory")
    record = run(args.out, args.sample_size, args.only_file)
    print(json.dumps({k: v for k, v in record.items() if k != "arrow_schema"}, indent=2))
    return 0 if args.compare is None or compare(args.out, args.compare) else 1


if __name__ == "__main__":
    raise SystemExit(main())
