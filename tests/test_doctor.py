from aegisflow.doctor import CHECKS, demo_ready, format_report, run_checks
from aegisflow.cli import main


def _touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_empty_root_reports_every_file_with_a_fix(tmp_path):
    results = run_checks(tmp_path)
    assert not any(ok for _, ok in results)
    assert not demo_ready(results)
    out = format_report(results)
    assert out.count("MISSING") == len(CHECKS)
    assert "python -m aegisflow preprocess --dataset cic_ids2017" in out
    assert "python -m aegisflow train --dataset cic_ids2017" in out


def test_ready_without_raw_csvs(tmp_path):
    for c in CHECKS:
        if c.group != "raw data":
            _touch(tmp_path, c.path)
    results = run_checks(tmp_path)
    assert demo_ready(results)
    assert "MISSING" in format_report(results)  # raw CSVs still reported, but not blocking


def test_glob_check_and_cli_exit_code(tmp_path):
    _touch(tmp_path, "data/raw/cic_ids2017/day/a.pcap_ISCX.csv")
    raw = [ok for c, ok in run_checks(tmp_path) if c.group == "raw data"]
    assert raw == [True]
    assert main(["doctor"]) in (0, 1)
