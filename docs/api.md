# API documentation

FastAPI app in `backend/app/main.py`. Start with `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`; interactive docs at `/docs`. The dashboard (`backend/app/static/index.html`) is served at `/`.

Until `preprocess` and `train` have been run, endpoints that need data or the model return HTTP 503 with instructions.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard page |
| GET | `/health` | Status, alert count in the ledger, replay state |
| GET | `/dataset/status` | Dataset summary and split (from `reports/data_quality_report.json`, `split_metadata.json`) |
| GET | `/model/status` | Model version, features, thresholds, training config, test metrics, limitations |
| GET | `/risk/config` | Risk weights, low/medium bands, max reachable risk, confidence cutoff, uncertain label, replay speeds. Read from `configs/config.yaml`; the dashboard uses it instead of hard-coded copies |
| GET | `/evaluation` | Baseline tables (any/onset targets), host check, leave-one-day-out DoS summary |
| POST | `/replay/start` | Body `{"speed": 1\|10\|100\|1000, "reset": true}`. `reset: true` clears the ledger. 409 if already running, 422 for a bad speed |
| POST | `/replay/stop` | Stop the replay |
| GET | `/replay/status` | State, progress, sim clock, alert counts |
| GET | `/hosts` | Hosts seen in the current replay, sorted by alerts then risk |
| GET | `/hosts/{host_id}` | One host: risk timeline and its alerts |
| GET | `/mitre/{stage}` | Static lookup from `configs/mitre_mapping.yaml` (no model involved) |
| POST | `/alerts` | Append an alert to the hash chain (used by the replay engine; no authentication) |
| GET | `/alerts` | List alerts, newest first (`limit` max 1000, `offset`, `host_id`) |
| GET | `/alerts/{id}` | One alert with `prev_hash` and `hash` |
| POST | `/alerts/{id}/actions` | Analyst decision on an alert. Body `{"action", "analyst", "note"?, "response"?, "stage"?}`; actions `acknowledge`, `approve_response` (needs `response`), `dismiss`, `reopen`, `override_stage` (needs `stage` from `stages.yaml`), `comment` (needs `note`). 422 for a missing field or a transition the current status forbids, 404 for an unknown alert. Approving a response only records it; AegisFlow never executes it |
| GET | `/alerts/{id}/actions` | Alert's review status (`open`, `acknowledged`, `response_approved`, `dismissed`, `reopened`), model stage vs analyst-effective stage, approved response, full action history |
| GET | `/triage/summary` | Number of alerts in each review status |
| GET | `/audit/verify-actions` | Recompute the analyst-action hash chain. Each action's hash also covers the hash of its alert, so changing the alert after a decision is detected |
| GET | `/audit/verify` | Recompute the chain: `VERIFIED` (with `head_hash`) or `CORRUPTED` with the first bad record |

Notes: the replay plays the held-out test split through pre-scored LSTM outputs on a simulated clock. Risk score is 0-100 but capped at 75 in practice (stage term is 0). `predicted_stage` is always `UNCERTAIN`. The analyst endpoints are not used by the dashboard. They have no authentication, so `analyst` names are self-declared. `POST /replay/start` with `reset: true` clears the actions together with the alerts. See `docs/codebase_walkthrough.md` for details and limits of the ledger.
