# AegisFlow — Optimization Opportunities (what is still open)

Status as of the B8 commit on branch `section-b`. Items that have been done are listed once in "Already done" and removed from the tables below. Risk and effort figures are estimates from reading the code, not measurements.

Rule used for sorting: if touching an item could plausibly break the demo path (`backend/app/*`, `index.html`, the saved artifacts, or the Parquet files the server reads), it is in Section B even if the fix is trivial.

## Already done

| Item | What was done | Where |
|---|---|---|
| A1 | Local backups: `aegisflow-demo-backup.zip` (processed data, models, reports, consistent ledger copy, `requirements.lock.txt`) and `aegisflow-full-backup.zip` (whole project except `.venv`). Uploading a copy to Drive/USB is still open (see A1 below). | Desktop, outside the repo |
| A2 | Replaced the "PLANNED" stubs in `docs/model.md`, `api.md`, `demo.md`; added a current-state box to `architecture.md`. | `df66590` |
| A3 | Added `httpx` to `requirements.txt`. | `4368ac6` |
| A4 | "SUPERSEDED" banners on `phase3_evaluation_audit.md`, `phase3_temporal_alignment_audit.md`, `phase3_split_audit.md`. | `df66590` |
| A6 | Comments on unused/changed config keys. (Later superseded by B8, which made `model:` real.) | `4368ac6`, `0719aab` |
| A7 | Unused packages (`xgboost`, `shap`, `sqlalchemy`, `pyshark`, `scapy`) moved into a commented "planned, not used" block in `requirements.txt`. | `4368ac6` |
| A8 | Removed unused imports in `eda.py`, `modeling.py`, `temporal/pipeline.py`. | `4368ac6` |
| A9 | Rewrote the stale `cli.py` module docstring (the `--help` text). | `4368ac6` |
| A10 | Reworded the stale `stage_proxy.status` text in `modeling.py`. | `4368ac6` |
| A12 | Added `requirements.lock.txt` (`pip freeze` of the tested environment). | `4368ac6` |
| B10 | Default FastAPI app is built lazily (module `__getattr__`), so importing `backend.app.main` no longer opens the live DB, starts the warm-up thread or loads torch. Test run 151 s → 49 s. | `af532d1` (on `main`) |
| B8 | `train` reads `config.model` (CLI overrides; non-`lstm` type rejected) and `model:` now equals the committed model; risk weights validated to sum to 1; `stage_severity` read for the predicted stage; new `GET /risk/config`; dashboard reads weights, bands, max risk, confidence cutoff and default speed from it; `replay.default_speed` set to 1000 to keep the old UI default. 13 new tests, 71 total pass. | `0719aab` (on `section-b`, not yet merged to `main`) |

---

## Section A — Still open, safe to do

| # | File : line | Issue | Fix | Risk |
|---|---|---|---|---|
| A1 (remainder) | Off-repo | The backup zips are on the same laptop as the demo. A dead laptop still means no demo. | Copy `aegisflow-demo-backup.zip` to Drive/USB (keep the Drive folder private: the processed data is derived from licensed CIC-IDS2017). Also record a screen capture of one full replay as a fallback. | **None.** |
| A5 | `README.md` "Model Evaluation" (about lines 395-415) | The README mentions limitations generically ("class imbalance, temporal attack distribution…") but not what the dashboard itself shows: host recognition (`172.16.0.1`), 0/409 Botnet, predicted stage always `UNCERTAIN`, risk capped at 75, ledger limits. A judge who reads the README first and the dashboard second sees an inconsistency. | Add a short "Known limitations" section with those five bullets, linking `reports/lodo_pilot_report.md` and `reports/phase3_training_report.md`. Keep wording consistent with `docs/codebase_walkthrough.md`. | **None** (Markdown only). |
| A10 (remainder) | `aegisflow/ml/modeling.py`, `forecast_lead_time.interpretation` string in `train_experiment` | Conditional wording ("not leakage-free forecasts") that no longer applies: all targets now start strictly after the inputs end. Only lives in the gitignored `metrics.json`. | Reword to describe the measurement only. Do not re-train just to regenerate `metrics.json`. | **Low, value low.** Optional. |
| A11 | Repo root: nine `.pytest_*` folders (`.pytest_audit_tmp`, `.pytest_phase2_*`, `.pytest_phase3_*`) | Leftover per-run temp folders. Gitignored, so not in git, but they clutter the folder. Windows denies access to them (could not list or delete them from my session). | Delete them yourself in Explorer; if that is also denied, take ownership first (right-click → Properties → Security) or ignore them. Keep `.pytest_cache` and `.venv`. | **None** (local only). |

Tip for new test runs: use `-p no:cacheprovider --basetemp=<some folder outside the repo>` so no new locked `.pytest_*` folders appear, and set `AEGISFLOW_DB` to a temp path so the live ledger is never touched.

### Things that look trivial but belong in B

- Collapsing the duplicate `seq_end` assignment and the redundant `input_ends.max()` guard in `aegisflow/ml/temporal/sequences.py:175-185, 191-200`. Harmless duplication, but this is the leakage-critical function. Leave it.
- Removing the hard-coded benchmark branches in `aegisflow/ml/temporal/pipeline.py:118-121, 214-216`. They are never taken for the current data, so removal would be behaviour-identical, but it edits the preprocessing path (see B14). Just don't run `rebuild-temporal` (see walkthrough Appendix B).

---

## Section B — Worth doing later

Effort is a rough single-developer estimate. "Why not now" is the specific risk to the committed demo.

| # | Improvement | What it fixes | Effort | Why not now |
|---|---|---|---|---|
| B1 | **Per-host normalisation / host-controlled representation** — *experiment done, not adopted.* Opt-in `host_relative=True` in `ForecastDataset` (causal rolling per-host baseline); results in `reports/hostrel_report.md`, run with `scripts/hostrel_eval.py`. | Weakens the host shortcut (within-host ROC on 172.16.0.1 0.77 to 0.55) but does not fix detection: Botnet 3/409, 29/29 benign still flagged on 172.16.0.1, PR-AUC 0.177 to 0.112, logistic regression threshold collapses. Not wired into `train`, the backend or the dashboard. | Remaining: a lodo rerun, MAD scaling, other lookbacks | Adopting it would change every dashboard number for no demonstrated gain. Needs more hosts/campaigns first (B13). |
| B2 | **Push updates (WebSocket or Server-Sent Events) instead of 1 s polling** (`poll()` in `index.html` hits three endpoints every second; `docs/architecture.md` promised WebSocket) | 3 requests/s per open tab, one-second lag, polling continues after stop. | 0.5-1 day | Touches `main.py`, `replay.py` and the whole UI loop; a subtle bug shows up as a frozen dashboard during the live demo. |
| B3 | **Real streaming path** (flows → windows → sequences online, scoring as data arrives) | `ReplayEngine._prepare` scores the entire test split in one batch at startup; the replay only paces pre-computed scores. "Live prediction" is a simulation. | 1-2 weeks | New code on the hot path; needs incremental windowing and a state store. |
| B4 | **Ledger hardening** (`backend/app/audit.py`, `main.py` `POST /alerts`) — (a) per-session chains instead of `reset()` wiping everything on every Start; (b) authenticate `POST /alerts`; (c) HMAC or signed head hash, or periodically write `head_hash` somewhere else, to catch tail deletion and full recomputation; (d) reuse one connection instead of a new connection + `PRAGMA journal_mode=WAL` on every call (may also emit `ResourceWarning`s on newer Pythons, unverified); (e) the dashboard's alert panel shows only the newest 50 alerts by design (`/alerts?limit=50`), so consider a "load more" | Closes "can't someone recompute the chain?" and "is it really append-only?". | 1-2 days with tests | Changes the hashed fields or schema, which invalidates every existing ledger and the four `test_audit.py` tests; a regression shows up on stage as `CORRUPTED`. |
| B5 | **Read the ingest in chunks with `usecols`** (`cic_ids2017.py:192-205`) | Each CSV is read with all ~80 columns, `low_memory=False`, then all frames are concatenated and copied again in `_to_canonical`. Fine at 500k rows; heavy at 2.83M (1.1 GB of CSV). | 0.5-1 day | Must produce byte-identical Parquet to keep every downstream number valid; needs a golden-file comparison. A change to the parse path is also how the 12-hour-clock bug got in. |
| B6 | **Vectorise windowing** (`windowing.py:113` Python list comprehension of `np.arange` per flow; `:255-280` row-wise `.apply` for dominant class/stage/distribution over every window) | Preprocessing time scales poorly; the slowest Python-level loop in the pipeline. | 1 day | Output must match exactly (same labels per window). One off-by-one changes `attack_present`. Needs parity tests against the current Parquet. |
| B7 | **Store sequences as arrays, not nested Python lists** (`sequences.py:187-188` `.tolist()`; `modeling.py:51-52` converts back with a double `np.asarray(...tolist())`); also the per-sequence `np.flatnonzero` over the rest of the host's windows (`sequences.py:179`) is quadratic per host | `sequences.parquet` is ~32 MB for 85k rows; loading goes through slow object arrays. | 1 day | Touches the leakage-critical builder and the training loader; would invalidate saved artifacts unless parity is proven. |
| B9 | **Calibrate probabilities** (Platt/isotonic on a larger validation set) and use PR-based thresholds | `confidence` and `prediction_confidence` are raw sigmoid outputs of a model trained with `pos_weight` 132, so they are inflated; alerts at p 0.33-0.5 show UNCERTAIN. | 1 day | Changes thresholds, alert counts (349 / 79 asserted in `backend/tests/test_replay.py:23`) and every dashboard number. |
| B11 | **SQLAlchemy migration** (the ledger is stdlib `sqlite3`; `sqlalchemy` is only in the commented "planned" block of `requirements.txt`) | Only worthwhile if you want a non-SQLite database. | 1 day | Hashes must stay byte-identical (types, rounding in `_normalize`) or every verify breaks. I would not do this unless a real need appears. |
| B12 | **A real stage model** (multi-class head or per-stage binary heads); then the `stage_severity` risk term becomes non-zero (B8 already reads the config table for the predicted stage) | "Predicted Stage" is always UNCERTAIN; risk is capped at 75. | Days, plus more data | Train positives cover only Initial Access (241) and Impact (141); test has C2 (409), Impact (44), Recon (34), so there is nothing valid to train or score on. Needs more attack days and hosts first. |
| B13 | **Better data and evaluation design**: full 2.83M ingest, episode- or campaign-grouped CV, a second dataset (adapters for CSE-CIC-IDS2018 / UNSW-NB15 are placeholders in `configs/datasets.yaml`) | Thin validation (127 positives, 2 episodes) makes thresholds fragile; one attacker host makes everything confounded. `lodo_pilot_report.md` argues rows alone won't help; you need more hosts and campaigns. | 1-3 weeks. **Scoped, see the data-limitation section of `reports/hostrel_report.md`**: no checked source has many attacker hosts per class together with usable IPs (full CIC-IDS2017: one attacker host per class; CSE-CIC-IDS2018 CSVs have no source IP; CTU-13 labels are one-to-one with infected IPs; UNSW-NB15 unverified). | Everything downstream changes. |
| B14 | **Report generator that reads real counts** (`temporal/pipeline.py:106-123, 214-216`) | `raw_count` and cleaning drops are re-derived from hard-coded numbers when the interim file has exactly 181,905 rows; otherwise `rebuild-temporal` writes `raw = cleaned`, zero drops. Persist the ingestion result (e.g. `ingestion_summary.json`) and read it. | 2-3 hours | Changes the preprocess path and the file the dashboard reads (`data_quality_report.json`). |

### Suggested order

B14 → B4 (cheap hygiene and the ledger questions judges will ask), then B1 (the real scientific fix), then B9/B13 as the evaluation matures. B2 and B3 only if the product direction needs them. Merge `section-b` into `main` once you have re-verified a full replay on it.
