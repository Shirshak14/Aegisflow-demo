# API documentation

FastAPI app in `backend/app/main.py`. Start with `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`; interactive docs at `/docs`. The React dashboard (`backend/app/static/index.html` + `app.js`, React 18 and htm vendored under `static/vendor/`, no build step, works offline) is served at `/`; the original single-file dashboard stays at `/classic`.

Until `preprocess` and `train` have been run, endpoints that need data or the model return HTTP 503 with instructions.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard page (React) |
| GET | `/classic` | Original single-file dashboard (fallback) |
| GET | `/features` | What this install can use: SHAP installed, stage model configured/trained, PCAP readers, streaming model, SIEM formats, stage list |
| GET | `/models` | Trained models under `artifacts/models/` with type, training time, test split size and the test metrics from each model's own `metrics.json` (nothing recomputed) |
| GET | `/health` | Status, alerts in the whole ledger, alerts in the current session, session id, replay state |
| GET | `/dataset/status` | Dataset summary and split (from `reports/data_quality_report.json`, `split_metadata.json`) |
| GET | `/model/status` | Model version, features, thresholds, training config, test metrics, limitations, `confidence_calibration` (Platt parameters and test Brier/ECE, or `none`) |
| GET | `/risk/config` | Risk weights, low/medium bands, max reachable risk, confidence cutoff, `confidence_calibrated`, uncertain label, replay speeds. Read from `configs/config.yaml`; the dashboard uses it instead of hard-coded copies |
| GET | `/evaluation` | Baseline tables (any/onset targets), host check, leave-one-day-out DoS summary |
| POST | `/replay/start` | Body `{"speed": 1\|10\|100\|1000, "reset": true}`. `reset: true` starts a fresh session; the ledger is kept, earlier sessions stay in it and keep verifying. 409 if already running, 422 for a bad speed |
| POST | `/replay/stop` | Stop the replay |
| GET | `/replay/status` | State, progress, sim clock, alert counts |
| GET | `/live` | Server-sent events for the dashboard: a `snapshot` event (`st`, `hosts`, `alerts`, `statuses`, `triage`, the bodies of `/replay/status`, `/hosts`, `/alerts?limit=50`, `/triage/statuses`, `/triage/summary`) whenever that data changes, checked every 0.5 s; a keep-alive comment every 15 s otherwise; an `error` event if a snapshot fails. `max_events` closes the stream after N events (tests) |
| GET | `/hosts` | Hosts seen in the current replay, sorted by alerts then risk |
| GET | `/hosts/{host_id}` | One host: risk timeline and its alerts |
| GET | `/explain/{sequence_id}` | Feature / time-step attribution of one test-split prediction. Query: `model=lstm\|logistic_regression`, `method=shap\|integrated_gradients`, `top` (1-28). 404 if the sequence is not in the test split, 501 if `shap` is not installed. Not used by the dashboard; never writes to the ledger |
| GET | `/forecast/{sequence_id}` | Opt-in multi-task model output for one replayed sequence: stage distribution, predicted stage (or `UNCERTAIN` below the confidence cutoff), MITRE tactic, K-step future state. 409 if `replay.stage_model_dir` is not set |
| GET | `/attention/{sequence_id}` | Query `model=<directory name from /models>`: that attention_lstm / transformer model's probability and per-window attention weights for the sequence. 404 for an unknown model, 422 for a model without attention |
| GET | `/mitre/{stage}` | Static lookup from `configs/mitre_mapping.yaml` (no model involved) |
| POST | `/alerts` | Append an alert to the hash chain (external producers only; needs the `X-API-Key` header and is disabled unless `AEGISFLOW_API_KEY` is set; the replay engine writes directly) |
| GET | `/alerts` | List alerts, newest first (`limit` max 1000, `offset`, `host_id`, `session_id`: omitted = the current replay session, or everything if none has started; `all` = every session) |
| GET | `/alerts/{id}` | One alert with `prev_hash` and `hash` |
| POST | `/alerts/{id}/actions` | Analyst decision on an alert. Body `{"action", "analyst", "note"?, "response"?, "stage"?}`; actions `acknowledge`, `approve_response` (needs `response`), `dismiss`, `reopen`, `override_stage` (needs `stage` from `stages.yaml`), `comment` (needs `note`). 422 for a missing field or a transition the current status forbids, 404 for an unknown alert. Approving a response only records it; AegisFlow never executes it |
| GET | `/alerts/{id}/actions` | Alert's review status (`open`, `acknowledged`, `response_approved`, `dismissed`, `reopened`), model stage vs analyst-effective stage, approved response, full action history |
| GET | `/triage/statuses` | `{alert_id: status}` for every alert with a review action (others are `open`) |
| GET | `/triage/summary` | Number of alerts in each review status (same `session_id` scoping as `/alerts`) |
| GET | `/audit/verify-actions` | Recompute the analyst-action hash chain. Each action's hash also covers the hash of its alert, so changing the alert after a decision is detected |
| POST | `/stream/flows` | Streaming telemetry: body `{"flows": [canonical flow records, oldest first]}`. Windows are scored as soon as they close; returns the newly scored host sequences and stream status. 422 if a record lacks a required schema field. Never writes to the ledger |
| POST | `/stream/flush` | End of stream: close and score every window that still holds flows |
| GET | `/stream/status` | Watermark, flows seen / late / buffered, windows closed, sequences scored, alerts |
| GET | `/stream/results` | Most recent scored sequences (`limit`, `alerts_only`) |
| POST | `/ingest/upload` | Multipart `file` (up to 200 MB) plus `kind=auto\|pcap\|netflow` and `reset` (default true): a packet capture, a NetFlow v5/v9/IPFIX export capture or an nfdump CSV is turned into flows and played through the streaming scorer. Returns flow / host / scored-sequence / alert counts; results appear in `/stream/results`, not in the alert ledger. 422 for an unreadable file, 501 if Scapy is missing |
| POST | `/stream/reset` | Clear stream state |
| GET | `/export/alerts` | SIEM export, one event per line: `format=cef` (default), `syslog` (RFC 5424, facility local4, CEF message) or `jsonl` (ECS-style fields); `since_id` for incremental polling, `limit` up to 10000. Each event carries the alert's ledger id and chain hash. Read-only. CLI: `python -m aegisflow export-alerts --format cef [--since-id N] [--syslog-host H]` |
| GET | `/audit/verify` | Recompute the whole chain across all sessions and check the head anchor: `VERIFIED` (with `head_hash` and `anchor` state) or `CORRUPTED` with the first bad record or the anchor problem |
| GET | `/audit/sessions` | Sessions in the ledger with alert count and first/last record id |

Notes: the replay plays the held-out test split through pre-scored LSTM outputs on a simulated clock. Risk score is 0-100 but capped at 75 in practice (stage term is 0). `predicted_stage` is `UNCERTAIN` unless `replay.stage_model_dir` names a trained multi-task model. The dashboard uses every endpoint above. The analyst endpoints have no authentication, so `analyst` names are self-declared. Analyst actions are kept with their alerts across sessions. See `docs/codebase_walkthrough.md` for details and limits of the ledger.
