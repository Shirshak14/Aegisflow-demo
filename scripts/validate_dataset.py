#!/usr/bin/env python3
"""Thin wrapper: python scripts/validate_dataset.py --dataset cic_ids2017"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegisflow.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate-dataset", *sys.argv[1:]]))
