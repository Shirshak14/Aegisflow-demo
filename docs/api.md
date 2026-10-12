# API documentation

FastAPI app in `backend/app/main.py`. Start with `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`; interactive docs at `/docs`. The React dashboard (`backend/app/static/index.html` + `app.js`, React 18 and htm vendored under `static/vendor/`, no build step, works offline) is served at `/`; the original single-file dashboard stays at `/classic`.

Until `preprocess` and `train` have been run, endpoints that need data or the model return HTTP 503 with instructions.

Every endpoint below is subject to the access-control rules in [Access control](#access-control). The **Role** column is the minimum role in authenticated mode; in local mode (the default) every request from this machine is allowed and every other request gets 403.

| Method | Path | Role | Purpose |
|---|---|---|---|
| GET | `/` | public | Dashboard page (React) |
| GET | `/classic` | public | Original single-file dashboard (fallback) |
| GET | `/login` | public | Sign-in page (authenticated mode): paste a token |
| GET | `/auth/me` | public | `{auth_enabled, user, role}` for the caller |
| POST | `/auth/session` | public | Body `{"token"}`: verify the token and set the session cookie. 401 for a bad or expired token, 409 in local mode |
| POST | `/auth/logout` | public | Clear the session cookie |
| GET | `/features` | viewer | What this install can use: SHAP installed, stage model configured/trained, PCAP readers, streaming model, SIEM formats, stage list |
| GET | `/models` | viewer | Trained models under `artifacts/models/` with type, training time, test split size and the test metrics from each model's own `metrics.json` (nothing recomputed) |
| GET | `/health` | public | Status, alerts in the whole ledger, alerts in the current session, session id, replay state |
| GET | `/dataset/status` | viewer | Dataset summary and split (from `reports/data_quality_report.json`, `split_metadata.json`) |
| GET | `/model/status` | viewer | Model version, features, thresholds, training config, test metrics, limitations, `confidence_calibration` (Platt parameters and test Brier/ECE, or `none`) |
| GET | `/risk/config` | viewer | Risk weights, low/medium bands, max reachable risk, confidence cutoff, `confidence_calibrated`, uncertain label, replay speeds. Read from `configs/config.yaml`; the dashboard uses it instead of hard-coded copies |
| GET | `/evaluation` | viewer | Baseline tables (any/onset targets), host check, leave-one-day-out DoS summary |
| POST | `/replay/start` | admin | Body `{"speed": 1\|10\|100\|1000, "reset": true}`. `reset: true` starts a fresh session; the ledger is kept, earlier sessions stay in it and keep verifying. 409 if already running, 422 for a bad speed |
| POST | `/replay/stop` | admin | Stop the replay |
| GET | `/replay/status` | viewer | State, progress, sim clock, alert counts. `anomaly` is null unless `replay.anomaly_model_dir` is set; then it holds the anomaly threshold and how many alerts both models flag (and how many of those match an attack label) |
| GET | `/live` | viewer | Server-sent events for the dashboard: a `snapshot` event (`st`, `hosts`, `alerts`, `statuses`, `triage`, the bodies of `/replay/status`, `/hosts`, `/alerts?limit=50`, `/triage/statuses`, `/triage/summary`) whenever that data changes, checked every 0.5 s; a keep-alive comment every 15 s otherwise; an `error` event if a snapshot fails. `max_events` closes the stream after N events (tests) |
| GET | `/hosts` | viewer | Hosts seen in the current replay, sorted by alerts then risk |
| GET | `/hosts/{host_id}` | viewer | One host: risk timeline and its alerts |
| GET | `/explain/{sequence_id}` | viewer | Feature / time-step attribution of one test-split prediction. Query: `model=lstm\|logistic_regression`, `method=shap\|integrated_gradients`, `top` (1-28). 404 if the sequence is not in the test split, 501 if `shap` is not installed. Not used by the dashboard; never writes to the ledger |
| GET | `/forecast/{sequence_id}` | viewer | Opt-in multi-task model output for one replayed sequence: stage distribution, predicted stage (or `UNCERTAIN` below the confidence cutoff), MITRE tactic, K-step future state. 409 if `replay.stage_model_dir` is not set |
| GET | `/attention/{sequence_id}` | viewer | Query `model=<directory name from /models>`: that attention_lstm / transformer model's probability and per-window attention weights for the sequence. 404 for an unknown model, 422 for a model without attention |
| GET | `/mitre/{stage}` | viewer | Static lookup from `configs/mitre_mapping.yaml` (no model involved) |
| POST | `/alerts` | API key | Append an alert to the hash chain (external producers only; needs the `X-API-Key` header and is disabled unless `AEGISFLOW_API_KEY` is set; the replay engine writes directly) |
| GET | `/alerts` | viewer | List alerts, newest first (`limit` max 1000, `offset`, `host_id`, `session_id`: omitted = the current replay session, or everything if none has started; `all` = every session) |
| GET | `/alerts/{id}` | viewer | One alert with `prev_hash` and `hash`. With `replay.anomaly_model_dir` set, alerts from `/alerts`, `/alerts/{id}`, `/hosts/{host_id}` and `/live` also carry `anomaly_score`, `anomaly_threshold` and `anomaly_flag` (not part of the hashed record) |
| POST | `/alerts/{id}/actions` | analyst | Analyst decision on an alert. Body `{"action", "analyst"?, "note"?, "response"?, "stage"?}`. With authentication on, the recorded analyst is the signed-in user and `analyst` is ignored; in local mode `analyst` is required and self-declared; actions `acknowledge`, `approve_response` (needs `response`), `dismiss`, `reopen`, `override_stage` (needs `stage` from `stages.yaml`), `comment` (needs `note`). 422 for a missing field or a transition the current status forbids, 404 for an unknown alert. Approving a response only records it; AegisFlow never executes it |
| GET | `/alerts/{id}/actions` | viewer | Alert's review status (`open`, `acknowledged`, `response_approved`, `dismissed`, `reopened`), model stage vs analyst-effective stage, approved response, full action history |
| GET | `/triage/statuses` | viewer | `{alert_id: status}` for every alert with a review action (others are `open`) |
| GET | `/triage/summary` | viewer | Number of alerts in each review status (same `session_id` scoping as `/alerts`) |
| GET | `/audit/verify-actions` | viewer | Recompute the analyst-action hash chain. Each action's hash also covers the hash of its alert, so changing the alert after a decision is detected |
| POST | `/stream/flows` | analyst | Streaming telemetry: body `{"flows": [canonical flow records, oldest first]}`. Windows are scored as soon as they close; returns the newly scored host sequences and stream status. 422 if a record lacks a required schema field. Never writes to the ledger |
| POST | `/stream/flush` | analyst | End of stream: close and score every window that still holds flows |
| GET | `/stream/status` | viewer | Watermark, flows seen / late / buffered, windows closed, sequences scored, alerts |
| GET | `/stream/results` | viewer | Most recent scored sequences (`limit`, `alerts_only`) |
| POST | `/ingest/upload` | analyst | Multipart `file` (up to 200 MB; one upload is processed at a time, others get 429; parsing and scoring run off the event loop and give up with 504 after `AEGISFLOW_UPLOAD_TIMEOUT_S`, default 300 s) plus `kind=auto\|pcap\|netflow` and `reset` (default true): a packet capture, a NetFlow v5/v9/IPFIX export capture or an nfdump CSV is turned into flows and played through the streaming scorer. Returns flow / host / scored-sequence / alert counts; results appear in `/stream/results`, not in the alert ledger. 422 for an unreadable file, 501 if Scapy is missing |
| POST | `/stream/reset` | admin | Clear stream state |
| GET | `/export/alerts` | viewer | SIEM export, one event per line: `format=cef` (default), `syslog` (RFC 5424, facility local4, CEF message) or `jsonl` (ECS-style fields); `since_id` for incremental polling, `limit` up to 10000. Each event carries the alert's ledger id and chain hash. Read-only. CLI: `python -m aegisflow export-alerts --format cef [--since-id N] [--syslog-host H]` |
| GET | `/audit/verify` | viewer | Recompute the whole chain across all sessions and check the head anchor: `VERIFIED` (with `head_hash` and `anchor` state) or `CORRUPTED` with the first bad record or the anchor problem |
| GET | `/audit/sessions` | viewer | Sessions in the ledger with alert count and first/last record id |

Notes: the replay plays the held-out test split through pre-scored LSTM outputs on a simulated clock. Risk score is 0-100 but capped at 75 in practice (stage term is 0). `predicted_stage` is `UNCERTAIN` unless `replay.stage_model_dir` names a trained multi-task model. The dashboard uses every endpoint above except `/explain`. Analyst actions are kept with their alerts across sessions. See `docs/codebase_walkthrough.md` for details and limits of the ledger.

## Access control

Implemented in `backend/app/security.py`; settings are read from environment variables when the app starts.

**Local mode (default).** No sign-in. The server answers only requests from this machine: a loopback client address and no `Forwarded` / `X-Forwarded-For` / `X-Real-IP` header (a reverse proxy on the same machine connects from loopback, so proxied requests count as remote). Anything else gets 403. Starting uvicorn with `--host 0.0.0.0` therefore does not expose the API. Analyst names are typed in the dashboard and self-declared. No rate limits.

**Authenticated mode** (`AEGISFLOW_AUTH=on`, or always with `AEGISFLOW_ENV=production`). Every request outside the public paths (`/`, `/classic`, `/login`, `/static/*`, `/health`, `/auth/*`) needs a token, sent as `Authorization: Bearer <token>` or as the session cookie. Missing or invalid token: 401; role too low: 403.

| Role | Can |
|---|---|
| viewer | Read every page and endpoint, `/live`, SIEM export, chain verification, explanations |
| analyst | Viewer, plus analyst decisions, capture upload, `POST /stream/flows` and `/stream/flush` |
| admin | Analyst, plus replay start / stop and `POST /stream/reset` (shared state every viewer sees) |

Tokens are HMAC-SHA256 signed with `AEGISFLOW_AUTH_SECRET` and carry the user name, role and expiry (default 12 h, at most 7 days). The analyst recorded on a decision is the token's user; a name in the request body is ignored.

```bash
export AEGISFLOW_AUTH_SECRET=$(python -m backend.app.security generate-secret)
python -m backend.app.security issue-token --user asha --role analyst --hours 12
AEGISFLOW_AUTH=on uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

In the dashboard, `/login` takes the token and exchanges it (`POST /auth/session`) for an HttpOnly, `SameSite=Strict` cookie (`Secure` in production). State-changing requests authenticated by the cookie are refused (403) when their `Origin` is another site. `GET /auth/me` returns `{auth_enabled, user, role}`; `POST /auth/logout` clears the cookie. There is no revocation list: keep tokens short, and rotate `AEGISFLOW_AUTH_SECRET` to invalidate all of them.

**Production** (`AEGISFLOW_ENV=production`) fails closed: the app refuses to start unless `AEGISFLOW_AUTH_SECRET` has at least 32 characters, `AEGISFLOW_AUTH` is not `off`, and `AEGISFLOW_LEDGER_KEY` is set (signed ledger head anchor). `AEGISFLOW_AUTH=on` without a valid secret also refuses to start.

**Limits.**

| What | Limit |
|---|---|
| Request body | 1 MB; `POST /stream/flows` 20 MB (and at most 50,000 flows); `POST /ingest/upload` 200 MB. Over the limit: 413, checked by `Content-Length` and by counting bytes received |
| Rate, per user per minute (authenticated mode) | `write` 120 (decisions, replay and stream controls), `inference` 60 (`/explain`, `/forecast`, `/attention`, `/stream/flows`), `upload` 6, `login` 10 per client address. Over the limit: 429 with `Retry-After`. Override with `AEGISFLOW_RATE_LIMITS="upload=6,inference=60,write=120,login=10"` (0 turns one off) |
| Concurrency and time | One capture upload at a time; two explanation / forecast / attention jobs at a time. Extra requests get 429. A job over `AEGISFLOW_UPLOAD_TIMEOUT_S` (300 s) or `AEGISFLOW_INFERENCE_TIMEOUT_S` (120 s) returns 504; its thread cannot be killed, so it finishes in the background and keeps its slot until then |

`POST /alerts` is separate: it is for machine producers, needs `X-API-Key` matching `AEGISFLOW_API_KEY` in both modes, and is disabled (503) when that variable is unset.

Not covered: single sign-on, a user database, per-token revocation, TLS (terminate it at a reverse proxy, and run that proxy with authentication on).

