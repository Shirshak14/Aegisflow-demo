"""Routes the React dashboard uses: page and vendored assets, feature discovery, model listing,
opt-in forecast / attention guards, triage statuses and capture upload."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import features
from backend.app.info import MODEL_DIR
from backend.app.main import create_app

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "netflow"


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path / "d.db", warm_up=False))


def _alert(i):
    return {"sequence_id": f"h_seq_{i}", "host_id": "10.0.0.1", "lstm_probability": 0.9,
            "predicted_stage": "UNCERTAIN", "risk_score": 50.0}


def test_react_page_classic_fallback_and_vendored_assets(client):
    page = client.get("/").text
    for src in ("/static/vendor/react.production.min.js", "/static/vendor/react-dom.production.min.js",
                "/static/vendor/htm.umd.js", "/static/app.js", "/static/app.css"):
        assert src in page
        assert client.get(src).status_code == 200, src
    assert "https://" not in page  # no CDN: the demo must work offline
    assert "<title>AegisFlow Demo</title>" in client.get("/classic").text
    assert client.get("/static/../main.py").status_code == 404


def test_features_report_defaults(client):
    f = client.get("/features").json()
    assert f["stage_model"]["configured"] is False and f["stage_model"]["available"] is False
    assert f["explanations"]["integrated_gradients"] is True
    assert f["siem_formats"] == ["cef", "syslog", "jsonl"]
    assert "Impact" in f["stages"] and "Benign" in f["stages"]


def _model(root, name, meta, metrics):
    d = root / name
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(meta))
    (d / "metrics.json").write_text(json.dumps(metrics))
    (d / "best.pt").write_bytes(b"")
    return d


def test_list_models_copies_each_trainers_test_metrics(tmp_path):
    m = {"roc_auc": 0.7, "pr_auc": 0.2, "f1": 0.3, "precision": 0.4, "recall": 0.25, "false_positive_rate": 0.01}
    split = {"test": {"sequences": 100, "positive_targets": 7}}
    demo = _model(tmp_path, "a_demo", {"training_config": {"hidden_size": 16}},
                  {"splits": split, "threshold": 0.3, "models": {"lstm": m, "logistic_regression": {"roc_auc": 0.1}}})
    _model(tmp_path, "b_tf", {"training_config": {"model_type": "transformer"}},
           {"splits": split, "models": {"transformer": m | {"roc_auc": 0.8}}})
    _model(tmp_path, "c_mt", {"model_type": "multitask_lstm", "horizons": 3},
           {"splits": split, "attack": m | {"roc_auc": 0.6},
            "stage": {"accuracy": 0.5, "attack_stage_accuracy": 0.4, "majority_baseline": {"accuracy": 0.9}}})
    _model(tmp_path, "d_gnn", {"model_type": "temporal_gnn"}, {"splits": split, "models": {"temporal_gnn": m | {"roc_auc": 0.55}}})
    (tmp_path / "not_a_model").mkdir()
    rows = {r["name"]: r for r in features.list_models(tmp_path, demo_dir=demo)}
    assert list(rows) == ["a_demo", "b_tf", "c_mt", "d_gnn"]
    assert rows["a_demo"]["demo_model"] and rows["a_demo"]["test_metrics"]["roc_auc"] == 0.7
    assert rows["b_tf"]["model_type"] == "transformer" and rows["b_tf"]["has_attention"] and rows["b_tf"]["test_metrics"]["roc_auc"] == 0.8
    assert rows["c_mt"]["test_metrics"]["roc_auc"] == 0.6 and rows["c_mt"]["stage_majority_baseline"] == 0.9
    assert rows["d_gnn"]["test_metrics"]["roc_auc"] == 0.55 and not rows["d_gnn"]["has_attention"]
    assert rows["d_gnn"]["test_sequences"] == 100 and rows["d_gnn"]["test_positive_targets"] == 7


def test_forecast_needs_stage_model_and_attention_needs_known_model(client):
    r = client.get("/forecast/x_seq_1")
    assert r.status_code == 409 and "train-multitask" in r.json()["detail"]
    assert client.get("/attention/x_seq_1", params={"model": "no_such_model"}).status_code == 404
    assert client.get("/attention/x_seq_1", params={"model": "../../backend"}).status_code == 404


def test_triage_statuses(client):
    for i in range(3):
        client.post("/alerts", json=_alert(i))
    client.post("/alerts/1/actions", json={"action": "acknowledge", "analyst": "a"})
    client.post("/alerts/2/actions", json={"action": "dismiss", "analyst": "a"})
    client.post("/alerts/2/actions", json={"action": "comment", "analyst": "a", "note": "backup job"})
    assert client.get("/triage/statuses").json() == {"1": "acknowledged", "2": "dismissed"}


def test_upload_rejects_bad_kind_and_unreadable_file(client):
    files = {"file": ("x.pcap", b"not a capture", "application/octet-stream")}
    assert client.post("/ingest/upload", params={"kind": "zip"}, files=files).status_code == 422
    r = client.post("/ingest/upload", params={"kind": "netflow"}, files=files)
    assert r.status_code == 422 and "could not read" in r.json()["detail"]


@pytest.mark.skipif(not (MODEL_DIR / "best.pt").exists(), reason="needs the trained demo model (gitignored)")
@pytest.mark.parametrize("name,kind", [("softflowd_v9.pcap", "netflow"), ("nfdump_v9.csv", "auto"),
                                       ("source_traffic.pcap", "pcap")])
def test_upload_scores_capture_through_stream(client, name, kind):
    pytest.importorskip("scapy")
    r = client.post("/ingest/upload", params={"kind": kind}, files={"file": (name, (FIXTURES / name).read_bytes())})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["flows"] > 0 and out["sequences_scored"] > 0
    assert out["sequences_scored"] == out["status"]["sequences_scored"]  # reset=True: counts are this file's
    assert len(client.get("/stream/results", params={"limit": 1000}).json()) == out["sequences_scored"]
