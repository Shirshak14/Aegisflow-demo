import json

from aegisflow.manifest import make_bundle, verify_bundle, verify_manifest, write_manifest


def _model_dir(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "best.pt").write_bytes(b"weights")
    (d / "preprocessor.joblib").write_bytes(b"prep")
    (d / "metadata.json").write_text(json.dumps({"feature_names": ["a", "b"], "input_shape": [10, 2]}))
    (d / "metrics.json").write_text(json.dumps({"threshold": 0.33, "threshold_selection": "val F1",
                                                "training": {"epochs_run": 3}, "models": {"lstm": {"threshold": 0.33, "f1": 0.2}}}))
    return d


def test_manifest_records_schema_threshold_and_detects_changes(tmp_path):
    d = _model_dir(tmp_path)
    m = json.loads(write_manifest(d).read_text())
    assert m["feature_schema"]["n_features"] == 2 and m["threshold"]["lstm"] == 0.33 and set(m["files"]) >= {"best.pt", "metadata.json"}
    assert verify_manifest(d)["ok"]
    (d / "best.pt").write_bytes(b"tampered")
    (d / "calibration.json").write_text("{}")
    r = verify_manifest(d)
    assert not r["ok"] and r["changed"] == ["best.pt"] and r["unlisted"] == ["calibration.json"]
    (d / "preprocessor.joblib").unlink()
    assert verify_manifest(d)["missing"] == ["preprocessor.joblib"]


def test_bundle_round_trip_contains_notice_and_no_data(tmp_path):
    d = _model_dir(tmp_path)
    (d / "sequences.parquet").write_bytes(b"data")  # must never be bundled
    z = make_bundle(d, tmp_path / "b.zip")
    import zipfile
    names = zipfile.ZipFile(z).namelist()
    assert "NOTICE.txt" in names and not any("parquet" in n for n in names)
    assert verify_bundle(z, tmp_path / "out")["ok"]
