#!/usr/bin/env python3
"""
Run the full Phase-1 data preparation pipeline for one dataset:
discover -> load -> clean -> label -> write data/interim/<dataset>.parquet -> EDA.

Usage:
    python scripts/prepare_data.py --dataset cic_ids2017
    python scripts/prepare_data.py --dataset cic_ids2017 --sample-size 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegisflow.config import load_config
from aegisflow.errors import AegisFlowError
from aegisflow.logging_setup import configure_logging
from aegisflow.ml.eda import run_eda
from aegisflow.ml.ingestion.pipeline import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    configure_logging(level=cfg.config.logging.level, json_file=cfg.path(cfg.config.logging.json_file))

    try:
        result = run_ingestion(cfg, args.dataset, sample_size=args.sample_size)
    except AegisFlowError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    print(f"\nIngestion complete: {result.output_path} ({result.cleaning.rows_out} rows)")

    if not args.skip_eda:
        summary = run_eda(cfg, args.dataset)
        print(f"EDA complete: {summary['n_rows']} rows summarized, "
              f"figures in reports/figures/, JSON in reports/eda/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
