"""The evaluation tables are static report files; the dashboard must warn when they were computed on
different data/model than the live replay uses. The comparison lives in info.evaluation_consistency."""
import copy
import json

import pytest

from backend.app import info

SPLIT = {"train_sequences": 1000, "val_sequences": 200, "test_sequences": 300, "total_sequences": 1520}
THRESHOLD = 0.3307546377182007


def _reports(split=SPLIT, threshold=THRESHOLD, purged=20):
    """Minimal report structures with exactly the keys evaluation_consistency reads."""
    baselines = {"variants": {"any": {
        "split_counts": {s: {"total": split[f"{s}_sequences"]} for s in ("train", "val", "test")},
        "models": {"lstm_existing_checkpoint": {"threshold": threshold}}}}}
    counts = {s: {"total": split[f"{s}_sequences"]} for s in ("train", "val", "test")}
    counts["purged"] = {"total": purged}
    lodo = {"folds": {"skipped_fold": {"note": "no counts here"}, "Wed": {"counts": counts}}}
    return baselines, lodo


@pytest.fixture()
def live(tmp_path, monkeypatch):
    """Point info.DATA_DIR / MODEL_DIR at a temp 'live' dataset + model; returns a setter for its values."""
    data_dir, model_dir = tmp_path / "data", tmp_path / "model"
    data_dir.mkdir()
    model_dir.mkdir()
    monkeypatch.setattr(info, "DATA_DIR", data_dir)
    monkeypatch.setattr(info, "MODEL_DIR", model_dir)

    def set_live(split=SPLIT, threshold=THRESHOLD):
        (data_dir / "split_metadata.json").write_text(json.dumps(split), encoding="utf-8")
        (model_dir / "metrics.json").write_text(json.dumps({"threshold": threshold}), encoding="utf-8")

    set_live()
    return set_live


def test_matching_inputs_report_no_warning(live):
    out = info.evaluation_consistency(*_reports())
    assert out["consistent"] is True and out["mismatches"] == []
    assert out["reports"] == out["live"] == {**SPLIT, "lstm_threshold": THRESHOLD}


def test_threshold_equal_within_1e6_is_not_a_mismatch(live):
    assert info.evaluation_consistency(*_reports(threshold=THRESHOLD + 5e-7))["consistent"] is True
    assert info.evaluation_consistency(*_reports(threshold=THRESHOLD + 5e-6))["consistent"] is False


@pytest.mark.parametrize("key", ["train_sequences", "val_sequences", "test_sequences"])
def test_split_size_mismatch_is_reported(live, key):
    other = {**SPLIT, key: SPLIT[key] + 1}
    out = info.evaluation_consistency(*_reports(split=other))
    assert out["consistent"] is False
    assert any(m.startswith(f"{key}: evaluation reports {other[key]} vs live model/data {SPLIT[key]}") for m in out["mismatches"])


def test_total_sequences_mismatch_is_reported(live):
    out = info.evaluation_consistency(*_reports(purged=21))  # reports now total 1521, live says 1520
    assert out["consistent"] is False
    assert [m.split(":")[0] for m in out["mismatches"]] == ["total_sequences"]


def test_threshold_mismatch_is_reported(live):
    out = info.evaluation_consistency(*_reports(threshold=0.5))
    assert out["consistent"] is False
    assert out["mismatches"] == [f"lstm_threshold: evaluation reports 0.5 vs live model/data {THRESHOLD}"]


def test_live_data_changing_after_the_reports_were_made_is_reported(live):
    reports = _reports()
    live(split={**SPLIT, "test_sequences": 301, "total_sequences": 1521}, threshold=0.41)
    out = info.evaluation_consistency(*reports)
    assert out["consistent"] is False
    assert sorted(m.split(":")[0] for m in out["mismatches"]) == ["lstm_threshold", "test_sequences", "total_sequences"]


@pytest.mark.parametrize("break_it", [
    lambda b, l: b["variants"]["any"].pop("split_counts"),
    lambda b, l: b["variants"]["any"]["models"].pop("lstm_existing_checkpoint"),
    lambda b, l: l.update(folds={"only": {"note": "no fold has counts"}}),
    lambda b, l: b.update(variants=None),
])
def test_unreadable_reports_are_reported_not_silently_consistent(live, break_it):
    baselines, lodo = copy.deepcopy(_reports())
    break_it(baselines, lodo)
    out = info.evaluation_consistency(baselines, lodo)
    assert out["consistent"] is False
    assert out["mismatches"][0].startswith("could not read the evaluation reports' fingerprint")


def test_evaluation_endpoint_carries_the_warning(tmp_path, live):
    """End to end through /evaluation (the JSON the dashboard renders). Uses the committed reports."""
    if not (info.REPORTS / "phase3_baselines_onset.json").exists():
        pytest.skip("committed evaluation reports not present")
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    reports = json.loads((info.REPORTS / "phase3_baselines_onset.json").read_text(encoding="utf-8"))
    lodo = json.loads((info.REPORTS / "lodo_pilot_results.json").read_text(encoding="utf-8"))
    fp = info.evaluation_consistency(reports, lodo)["reports"]
    client = TestClient(create_app(tmp_path / "e.db", warm_up=False))

    live(split={k: fp[k] for k in SPLIT}, threshold=fp["lstm_threshold"])  # live == what the reports describe
    ok = client.get("/evaluation").json()["consistency"]
    assert ok["consistent"] is True and ok["mismatches"] == []

    live(split={k: fp[k] for k in SPLIT}, threshold=fp["lstm_threshold"] + 0.1)  # model retrained, reports stale
    bad = client.get("/evaluation").json()["consistency"]
    assert bad["consistent"] is False and bad["mismatches"][0].startswith("lstm_threshold:")


def test_committed_artifacts_are_consistent():
    """The committed reports, split metadata and model must agree (the dashboard shows no warning)."""
    needed = [info.REPORTS / "phase3_baselines_onset.json", info.REPORTS / "lodo_pilot_results.json",
              info.DATA_DIR / "split_metadata.json", info.MODEL_DIR / "metrics.json"]
    if not all(p.exists() for p in needed):
        pytest.skip("committed artifacts not present")
    out = info.evaluation()["consistency"]
    assert out["consistent"] is True, out["mismatches"]
