"""SIEM export: CEF escaping/severity, syslog framing, JSON lines, incremental polling, UDP send."""
import json
import socket

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import AUTH
from backend.app.audit import Ledger
from backend.app.main import create_app
from backend.app.siem import alerts_since, render, send_syslog_udp, severity, to_cef, to_jsonl, to_syslog


def _alert(i, **kw):
    return {"sequence_id": f"172.16.0.1_seq_{i}", "host_id": "172.16.0.1", "lstm_probability": 0.8,
            "lstm_threshold": 0.33, "risk_score": 10.0 * i, "predicted_stage": "UNCERTAIN",
            "predicted_at": "2017-07-07T15:00:00", "target_window_start": "2017-07-07T15:00:30",
            "target_window_end": "2017-07-07T15:01:30", "risk_components": {"attack_probability": 0.8},
            "model_version": "lstm-abc", **kw}


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "a.db")
    for i in range(1, 6):
        led.append(_alert(i))
    return led


def test_cef_fields_and_escaping(ledger):
    a = alerts_since(ledger)[2]
    cef = to_cef(a)
    head = cef.split("|")
    assert head[:7] == ["CEF:0", "AegisFlow", "AegisFlow Forecaster", "0.1.0", "aegisflow-forecast",
                        "Predicted attack in next window", "3"]
    assert "src=172.16.0.1" in cef and f"cs2={a['hash']}" in cef and "externalId=3" in cef
    assert "rt=1499439600000" in cef  # 2017-07-07T15:00:00Z in ms
    tricky = to_cef({**a, "sequence_id": "a=b\\c", "predicted_stage": "x|y"})
    assert "cs3=a\\=b\\\\c" in tricky and "cs1=x|y" in tricky   # '|' is only escaped in the header


def test_severity_bounds():
    assert [severity(x) for x in (None, 0, 34, 75, 100, 250)] == [0, 0, 3, 8, 10, 10]


def test_syslog_and_jsonl(ledger):
    a = alerts_since(ledger)[4]
    line = to_syslog(a)
    # risk 50 -> CEF severity 5 -> syslog warning (4); facility local4 (20): PRI = 20*8 + 4
    assert line.startswith("<164>1 ") and " aegisflow aegisflow - alert - CEF:0|" in line
    assert to_syslog({**a, "risk_score": 90}).startswith("<162>1 ")  # crit
    doc = json.loads(to_jsonl(a))
    assert doc["source"]["ip"] == "172.16.0.1" and doc["event"]["id"] == 5 and doc["event"]["hash"] == a["hash"]
    assert doc["aegisflow"]["risk_components"] == {"attack_probability": 0.8}


def test_incremental_export_and_bad_format(ledger):
    assert [a["id"] for a in alerts_since(ledger, since_id=3)] == [4, 5]
    assert len(render(alerts_since(ledger, limit=2), "jsonl")) == 2
    with pytest.raises(ValueError):
        render([], "xml")


def test_udp_syslog_is_received(ledger):
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); srv.bind(("127.0.0.1", 0)); srv.settimeout(2)
    port = srv.getsockname()[1]
    assert send_syslog_udp(render(alerts_since(ledger), "syslog"), "127.0.0.1", port) == 5
    got = [srv.recv(65535).decode() for _ in range(5)]
    srv.close()
    assert all("CEF:0|AegisFlow" in g for g in got)


def test_api_and_cli(tmp_path, capsys):
    db = tmp_path / "a.db"
    client = TestClient(create_app(db, warm_up=False))
    for i in range(1, 4):
        client.post("/alerts", headers=AUTH, json=_alert(i))
    r = client.get("/export/alerts", params={"format": "jsonl", "since_id": 1})
    assert r.status_code == 200 and [json.loads(l)["event"]["id"] for l in r.text.splitlines()] == [2, 3]
    assert client.get("/export/alerts", params={"format": "xml"}).status_code == 422
    assert client.get("/export/alerts", params={"since_id": 99}).text == ""
    assert client.get("/audit/verify").json()["status"] == "VERIFIED"   # exporting never changes the chain
    from aegisflow.cli import main
    assert main(["export-alerts", "--db", str(db), "--format", "cef"]) == 0
    assert capsys.readouterr().out.count("CEF:0|AegisFlow") == 3
