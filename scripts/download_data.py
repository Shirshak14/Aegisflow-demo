#!/usr/bin/env python3
"""
Print exact, official download instructions for a configured dataset.

Usage:
    python scripts/download_data.py --dataset cic_ids2017
    python scripts/download_data.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegisflow.config import load_config
from aegisflow.ml.datasets import get_adapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Key from configs/datasets.yaml")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if not args.dataset and not args.all:
        parser.error("pass --dataset <key> or --all")

    keys = list(cfg.datasets.keys()) if args.all else [args.dataset]
    for key in keys:
        if key not in cfg.datasets:
            print(f"Unknown dataset '{key}'. Known: {list(cfg.datasets.keys())}", file=sys.stderr)
            return 1
        entry = cfg.datasets[key]
        print("=" * 70)
        print(f"{entry.display_name}  (status: {entry.status})")
        print("=" * 70)
        if entry.status != "primary" and entry.adapter not in _implemented_adapters():
            print(f"Adapter '{entry.adapter}' is not implemented yet (planned). "
                  f"Official source for when it lands: {entry.official_url}\n")
            continue
        adapter = get_adapter(entry.adapter, cfg)
        print(adapter.download_instructions())
        print()
    return 0


def _implemented_adapters() -> set[str]:
    return {"cic_ids2017"}


if __name__ == "__main__":
    raise SystemExit(main())
