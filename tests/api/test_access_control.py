"""Access control (backend/app/security.py): local-only default, signed tokens, roles, limits, fail-closed config."""
import asyncio
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.audit import Ledger
from backend.app.main import create_app
from backend.app.security import (BoundedRunner, ConfigError, Settings, issue_token, parse_limits, verify_token)

SECRET = b"s" * 40


def _client(tmp_path, n_alerts=2, **settings):
    db = tmp_path / "a.db"
    led = Ledger(db)
    for i in range(n_alerts):
        led.append({"sequence_id": f"seq_{i}", "host_id": "172.16.0.1", "lstm_probability": 0.9, "risk_score": 50.0})
    return TestClient(create_app(db, warm_up=False, settings=Settings(auth=True, secret=SECRET, **settings)))


def _bearer(user, role):
    return {"Authorization": f"Bearer {issue_token(SECRET, user, role)}"}


# ---- configuration
def test_default_is_local_mode():
    s = Settings.from_env({})
    assert not s.auth and not s.production and s.secret is None


@pytest.mark.parametrize("env,msg", [
    ({"AEGISFLOW_AUTH": "on"}, "AEGISFLOW_AUTH_SECRET"),
    ({"AEGISFLOW_AUTH": "on", "AEGISFLOW_AUTH_SECRET": "short"}, "AEGISFLOW_AUTH_SECRET"),
    ({"AEGISFLOW_ENV": "production", "AEGISFLOW_AUTH_SECRET": "x" * 40}, "AEGISFLOW_LEDGER_KEY"),
    ({"AEGISFLOW_ENV": "production", "AEGISFLOW_LEDGER_KEY": "k"}, "AEGISFLOW_AUTH_SECRET"),
    ({"AEGISFLOW_ENV": "production", "AEGISFLOW_AUTH": "off", "AEGISFLOW_AUTH_SECRET": "x" * 40,
      "AEGISFLOW_LEDGER_KEY": "k"}, "not allowed"),
    ({"AEGISFLOW_AUTH": "yes"}, "must be 'on' or 'off'"),
    ({"AEGISFLOW_RATE_LIMITS": "upload=lots"}, "bad AEGISFLOW_RATE_LIMITS"),
])
def test_incomplete_configuration_fails_closed(env, msg):
    with pytest.raises(ConfigError, match=msg):
        Settings.from_env(env)


def test_production_turns_auth_on_and_app_refuses_to_start(tmp_path, monkeypatch):
    s = Settings.from_env({"AEGISFLOW_ENV": "production", "AEGISFLOW_AUTH_SECRET": "x" * 40, "AEGISFLOW_LEDGER_KEY": "k"})
    assert s.auth and s.production
    monkeypatch.setenv("AEGISFLOW_ENV", "production")
    with pytest.raises(ConfigError):
        create_app(tmp_path / "a.db", warm_up=False)
    assert not (tmp_path / "a.db").exists()  # failed before opening the ledger


def test_parse_limits():
    assert parse_limits("upload=2, write=0")["upload"] == 2 and parse_limits("")["inference"] == 60


# ---- tokens
def test_token_roundtrip_tamper_and_expiry():
    tok = issue_token(SECRET, "asha", "analyst", hours=1, now=1000)
    p, exp = verify_token(SECRET, tok, now=1001)
    assert (p.user, p.role, exp) == ("asha", "analyst", 1000 + 3600)
    with pytest.raises(HTTPException, match="expired"):
        verify_token(SECRET, tok, now=1000 + 3600)
    with pytest.raises(HTTPException, match="invalid"):
        verify_token(b"other" * 8, tok, now=1001)
    v, body, sig = tok.split(".")
    forged = issue_token(SECRET, "asha", "admin", hours=1, now=1000).split(".")[1]
    with pytest.raises(HTTPException, match="invalid"):
        verify_token(SECRET, f"{v}.{forged}.{sig}", now=1001)  # role swapped, old signature
    with pytest.raises(ValueError):
        issue_token(SECRET, "asha", "root")


# ---- local mode
def test_local_mode_serves_this_machine_only(tmp_path):
    app = create_app(tmp_path / "a.db", warm_up=False)
    assert TestClient(app).get("/hosts").status_code == 200
    assert TestClient(app).get("/auth/me").json() == {"auth_enabled": False, "user": None, "role": "admin"}
    assert TestClient(app, client=("10.1.2.3", 5000)).get("/hosts").status_code == 403
    assert TestClient(app, client=("127.0.0.1", 5000)).get("/hosts").status_code == 200
    # a reverse proxy on this machine connects from loopback but forwards someone else's request
    r = TestClient(app, client=("127.0.0.1", 5000)).get("/hosts", headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 403 and "AEGISFLOW_AUTH=on" in r.json()["detail"]


def test_local_mode_keeps_self_declared_analyst_name(tmp_path):
    Ledger(tmp_path / "a.db").append({"sequence_id": "s", "host_id": "h"})
    client = TestClient(create_app(tmp_path / "a.db", warm_up=False))
    r = client.post("/alerts/1/actions", json={"action": "acknowledge", "analyst": "asha"})
    assert r.status_code == 201 and r.json()["analyst"] == "asha"


# ---- authenticated mode
def test_reads_need_a_token_and_public_paths_stay_open(tmp_path):
    client = _client(tmp_path)
    assert client.get("/alerts").status_code == 401
    assert client.get("/live").status_code == 401
    assert client.get("/alerts", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert client.get("/alerts", headers={"Authorization": "Basic abc"}).status_code == 401
    for path in ("/", "/login", "/health", "/auth/me", "/static/app.css"):
        assert client.get(path).status_code == 200, path
    assert client.get("/auth/me").json() == {"auth_enabled": True, "user": None, "role": None}
    assert len(client.get("/alerts", headers=_bearer("vik", "viewer")).json()) == 2


def test_roles_gate_state_changing_endpoints(tmp_path):
    client = _client(tmp_path)
    viewer, analyst, admin = _bearer("vik", "viewer"), _bearer("asha", "analyst"), _bearer("root", "admin")
    for path in ("/replay/stop", "/stream/reset"):
        assert client.post(path, headers=viewer).status_code == 403
        assert client.post(path, headers=analyst).status_code == 403
    assert client.post("/replay/stop", headers=admin).status_code == 200
    act = {"action": "acknowledge"}
    assert client.post("/alerts/1/actions", json=act, headers=viewer).status_code == 403
    assert client.post("/alerts/1/actions", json=act, headers=analyst).status_code == 201
    upload = {"file": ("x.csv", b"not,a,capture\n")}
    assert client.post("/ingest/upload", files=upload, headers=viewer).status_code == 403
    # POST /alerts keeps its own X-API-Key check (machine producers have no user token)
    assert client.post("/alerts", json={"sequence_id": "s", "host_id": "h"}).status_code == 401
    assert client.post("/alerts", json={"sequence_id": "s", "host_id": "h"},
                       headers={"X-API-Key": "test-key"}).status_code == 201


def test_analyst_identity_comes_from_the_token(tmp_path):
    client = _client(tmp_path)
    r = client.post("/alerts/1/actions", json={"action": "comment", "analyst": "mallory", "note": "looks like a scan"},
                    headers=_bearer("asha", "analyst"))
    assert r.status_code == 201 and r.json()["analyst"] == "asha"
    hist = client.get("/alerts/1/actions", headers=_bearer("vik", "viewer")).json()
    assert [a["analyst"] for a in hist["actions"]] == ["asha"]


def test_cookie_session_and_cross_origin_refusal(tmp_path):
    client = _client(tmp_path)
    assert client.post("/auth/session", json={"token": "bad.token.here"}).status_code == 401
    r = client.post("/auth/session", json={"token": issue_token(SECRET, "asha", "analyst")})
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    assert "samesite=strict" in r.headers["set-cookie"].lower()
    assert client.get("/auth/me").json() == {"auth_enabled": True, "user": "asha", "role": "analyst"}
    assert client.get("/alerts").status_code == 200
    act = {"action": "acknowledge"}
    assert client.post("/alerts/1/actions", json=act, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/alerts/1/actions", json=act, headers={"Origin": "http://testserver"}).status_code == 201
    client.post("/auth/logout")
    assert client.get("/alerts").status_code == 401


def test_rate_limit_per_user(tmp_path):
    client = _client(tmp_path, limits=parse_limits("write=2"))
    asha, ravi = _bearer("asha", "analyst"), _bearer("ravi", "analyst")
    note = {"action": "comment", "note": "n"}
    assert [client.post("/alerts/1/actions", json=note, headers=asha).status_code for _ in range(3)] == [201, 201, 429]
    assert client.post("/alerts/1/actions", json=note, headers=ravi).status_code == 201
    r = client.post("/alerts/1/actions", json=note, headers=asha)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_request_body_size_limit(tmp_path):
    client = _client(tmp_path)
    big = {"action": "comment", "note": "x" * (2 * 2**20)}
    assert client.post("/alerts/1/actions", json=big, headers=_bearer("asha", "analyst")).status_code == 413

    def chunks():  # no Content-Length: the limit is enforced on the bytes received
        for _ in range(3):
            yield b" " * 2**19
    r = client.post("/alerts/1/actions", content=chunks(),
                    headers={**_bearer("asha", "analyst"), "content-type": "application/json"})
    assert r.status_code == 413


# ---- bounded work
def test_bounded_runner_times_out_and_refuses_when_busy():
    async def scenario():
        runner = BoundedRunner("test", workers=1, timeout_s=0.2)
        with pytest.raises(HTTPException) as slow:
            await runner.run(time.sleep, 0.6)
        assert slow.value.status_code == 504
        with pytest.raises(HTTPException) as busy:  # the abandoned job still holds the only slot
            await runner.run(lambda: 1)
        assert busy.value.status_code == 429
        await asyncio.sleep(0.6)
        assert await runner.run(lambda: 42) == 42
    asyncio.run(scenario())
