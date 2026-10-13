"""The synthetic end-to-end smoke test (aegisflow/smoke.py) must pass without CIC-IDS2017 or any artifact."""
import pytest

pytest.importorskip("torch")


def test_synthetic_end_to_end_smoke(tmp_path):
    from aegisflow.smoke import run_smoke
    report = run_smoke(tmp_path / "work")
    failed = [c for c in report["checks"] if not c["ok"]]
    assert report["ok"], failed
    names = [c["check"] for c in report["checks"]]
    assert "ledger hash chain verifies" in names and "manifest detects a modified file" in names
    per_split = next(c for c in report["checks"] if c["check"].startswith("windows"))["detail"]["per_split"]
    assert all(per_split[s]["positives"] > 0 for s in ("train", "val", "test"))
