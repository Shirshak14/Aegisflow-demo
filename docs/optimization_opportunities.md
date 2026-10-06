# AegisFlow — Optimization Opportunities (report only; nothing here has been applied)

This file lists what is still open. Items from the earlier A and B series that are finished have been removed; the new series is **C**. You decide what, if anything, to touch.

Method and limits: read-only review of source, config, tests and docs. I did not run the pipeline, the tests, or the server, so each "risk" is an estimate from reading the code, not a measurement. Line numbers are as of this pass.

Rule applied: if touching an item could plausibly break the demo path (`backend/app/*`, `index.html`, the saved artifacts, or the Parquet files the server reads), it is not marked safe, even if the fix is trivial.

## Removed because they are done

- **A1** (off-laptop backup and replay recording, done by you), **A2–A4, A6–A10, A12:** stale docs and report banners, `httpx` in `requirements.txt`, config comments, unused dependencies annotated as planned, unused imports, CLI docstring, stale metrics string, `requirements.lock.txt`.
- **B8** (config wired to behaviour) and **B10** (lazy backend app): both are on `main`.
- **A5** (README "Known limitations" section): tried, then reverted at your request. The README keeps its original generic wording.

---

## Section A — Still open (safe before the deadline)

| # | Item | Status / what is left | Risk |
|---|---|---|---|
| A11 | Nine empty `.pytest_*` directories at the repo root (`.pytest_audit_tmp`, `.pytest_phase2_*`, `.pytest_phase3_*`). | Gitignored and empty, but Windows denies access, so a normal delete fails. Take ownership first (`takeown /f <dir> /r`, then `rmdir /s /q <dir>`) or delete from an administrator shell. Keep `.pytest_cache`. | None |

---

## Section B — Worth doing AFTER the deadline

| # | Improvement | What it fixes | Effort | Why post-deadline |
|---|---|---|---|---|
| B1 | **Per-host normalisation / host-controlled representation** (z-score each host's windows against its own history, or key windows by victim host; see `reports/lodo_pilot_report.md` and `docs/host_disjoint_evaluation_design.md`) | The headline weakness: the LSTM acts as an attacker-host recogniser (29/29 benign flagged on `172.16.0.1`, and the host rule matches it). | 2-4 days including re-running preprocess, train, `phase3_baselines.py`, `lodo_pilot.py`, and rewriting the reports | Changes features, tensors, thresholds and every number on the dashboard and in every report. It may make the headline metrics worse before it makes them honest. |
| B2 | **Push updates (WebSocket or SSE) instead of 1 s polling** (`index.html` polls three endpoints every second) | 3 requests/s per open tab, one-second lag, polling continues after stop. | 0.5-1 day | Touches `main.py`, `replay.py` and the whole UI loop; a subtle bug shows up as a frozen dashboard live. |
| B3 | **Real streaming path** (flows → windows → sequences online, scoring as data arrives) | `ReplayEngine._prepare` scores the whole test split in one batch at startup; the replay only paces pre-computed scores. | 1-2 weeks | New code on the hot path; needs incremental windowing and a state store. |
| B4 | **Ledger hardening** (`backend/app/audit.py`) — (a) per-session chains instead of `reset()` wiping everything; (b) authenticate `POST /alerts`; (c) HMAC or signed head hash, or write `head_hash` to a second location, to catch tail deletion and full recomputation; (d) reuse one connection instead of a new connection plus `PRAGMA journal_mode=WAL` on every call (`audit.py:_connect`); note `with sqlite3.connect()` commits but does not close the connection | Closes "can't someone recompute the chain?" and "is it really append-only?". | 1-2 days with tests | Changes the hashed fields or schema, which invalidates existing ledgers and the `test_audit.py` tests. |
| B5 | **Read the ingest in chunks with `usecols`** (`cic_ids2017.py`) | Each CSV is read with all ~80 columns, `low_memory=False`, then frames are concatenated and copied again in `_to_canonical`. Fine at 500k rows; heavy at 2.83M. | 0.5-1 day | Must produce byte-identical Parquet; needs a golden-file comparison. |
| B6 | **Vectorise windowing** (`windowing.py`: per-flow `np.arange` list comprehension; row-wise `.apply` for dominant class/stage/distribution) | Slowest Python-level loop in the pipeline. | 1 day | Output must match exactly; one off-by-one changes `attack_present`. |
| B7 | **Store sequences as arrays, not nested Python lists** (`sequences.py` `.tolist()`; `modeling.py` `ForecastDataset.from_frame` converts back with a double `np.asarray(...tolist())`; per-sequence `np.flatnonzero` over a host's windows is quadratic per host) | `sequences.parquet` is ~32 MB for 85k rows; loading goes through slow object arrays. | 1 day | Touches the leakage-critical builder and the training loader. |
| B9 | **Calibrate probabilities** (Platt/isotonic on a larger validation set) and use PR-based thresholds | `confidence` and `prediction_confidence` are raw sigmoid outputs of a model trained with `pos_weight` 132, so they are inflated. | 1 day | Changes thresholds, alert counts (349 / 79 asserted in `backend/tests/test_replay.py`) and every dashboard number. |
| B11 | **SQLAlchemy migration** | Only worthwhile for a non-SQLite database; otherwise leave the dependency commented out. | 1 day | Hashes must stay byte-identical. Not recommended without a real need. |
| B12 | **A real stage model** (multi-class head or per-stage binary heads) and a non-zero `stage_severity` term | "Predicted Stage" is always UNCERTAIN; risk is capped at 75. | Days, plus more data | Train positives cover only Initial Access and Impact; test has C2, Impact and Recon. Needs more attack days and hosts first. |
| B13 | **Better data and evaluation design**: full 2.83M ingest, episode- or campaign-grouped CV, a second dataset (CSE-CIC-IDS2018 / UNSW-NB15 adapters are placeholders in `configs/datasets.yaml`) | Thin validation (127 positives, 2 episodes) makes thresholds fragile; one attacker host confounds everything. | 1-3 weeks | Everything downstream changes. |
| B14 | **Report generator that reads real counts** (`temporal/pipeline.py`) | `raw_count` and cleaning drops are re-derived from hard-coded numbers when the interim file has exactly 181,905 rows. Persist the ingestion result and read it. | 2-3 hours | Changes the preprocess path and the file the dashboard reads. |

---

## Section C — New findings (not in the earlier A/B lists)

C1-C5 and C7 are done; C6 is partly done; C8-C17 are for after the deadline.

### C-low: done

C1 (`artifacts/experiments/**` gitignored), C2 (Makefile: `preprocess`, `train`, `serve` targets, `test` runs plain `pytest`, `lint` removed), C3 (`GET /alerts` now validates `limit` 1-1000 and `offset` >= 0, returns 422 otherwise), C4 (`pyproject.toml`: dependencies read from `requirements.txt`, `aegisflow` console script, minimal ruff section), C5 (`modeling.py` defaults equal `config.yaml`: hidden 16, epochs 8, batch 256, patience 3), C7 (global seeding / deterministic-algorithms side effect documented in the `train_experiment` docstring). All 71 tests pass.

Still open from this group:

| # | File | Issue | Fix | Risk |
|---|---|---|---|---|
| C6 | `aegisflow/ml/modeling.py` `train_experiment` | The `;`-packed statements in the training loop were split onto separate lines (behaviour unchanged). The function is still one ~100-line block doing preprocessing, three models, metrics and file output, and the rest of the file (e.g. `predict_sequences`) keeps the dense style. | Run a formatter (none is installed in `.venv`), then split into `fit_lstm`, `evaluate`, `save_artifacts`. | Low for formatting; the split is a post-deadline refactor |

### C-later: worth doing after the deadline

| # | File | Issue | Fix | Effort |
|---|---|---|---|---|
| C8 | `.github/` (does not exist) | No CI. Nothing runs the tests on push, so a broken commit is found by hand. | Add a GitHub Actions workflow: install `requirements.txt`, run `pytest`. See C9 for why the demo path is not covered. | 1-2 hours |
| C9 | `backend/tests/test_replay.py:10` | The replay test is skipped when `data/processed` and `artifacts/models` are absent, which is always the case on a fresh clone or in CI. So the demo path (scoring, risk, ledger, 349/79 counts) is only tested on this laptop. | Commit a tiny synthetic `sequences.parquet` and a small trained model as test fixtures (or generate them in a fixture), and assert on those. | 0.5-1 day |
| C10 | `backend/app/replay.py` `_run`/`_process` | The hot loop does `ev.target_window_start.iloc[self.cursor]` and `ev.iloc[self.cursor]` per row. Each pandas `iloc` builds Series/objects, so a 10,000× replay spends most of its time in pandas overhead. | Convert the columns to NumPy arrays or a list of tuples once in `_prepare`, and iterate over those. | 2-3 hours (demo path; verify counts unchanged) |
| C11 | `backend/app/replay.py:_prepare`, `modeling.py` `predict_sequences` | The LSTM scores the whole split in one `net(torch.tensor(x))` call, and `train_experiment` does the same for validation and test. Fine at 11k sequences; memory grows linearly, so the 2.83M ingest (B13) could run out of RAM. | Score in batches (e.g. 4,096) under `torch.no_grad()`. | 1-2 hours |
| C12 | `backend/app/replay.py:_prepare` | Reads all of `sequences.parquet` (~32 MB) and then filters to `split == "test"`. `info._sequence_host_counts` reads it again. Slower server start-up than necessary. | Use `pd.read_parquet(path, filters=[("split", "==", "test")])` and only the columns needed. | 1 hour |
| C13 | `aegisflow/ml/modeling.py:select_threshold` | Tries every unique validation probability as a candidate and calls `f1_score` for each, which is O(n²). It is instant at 1.5k rows but slow at larger validation sets. Ties also resolve to the lowest threshold. | Use `sklearn.metrics.precision_recall_curve` and compute F1 for all thresholds at once. | 1 hour; must reproduce the committed threshold exactly |
| C14 | `backend/app/main.py` `AlertIn` | Fields such as `lstm_probability`, `risk_score` and `confidence` accept any float, and `truth_attack`/`lr_flag` accept any integer. Combined with the unauthenticated `POST /alerts` (B4b), anyone can write out-of-range data into the ledger. | Add `Field(ge=0, le=1)` bounds, `Literal[0, 1]` for the flags, and length limits on strings. | 1 hour; do together with B4 |
| C15 | `backend/app/replay.py` `hosts` state | Each host's `timeline` list grows by one entry per sequence for the whole replay and `/hosts/{id}` copies it on every call. Fine for 11k sequences; unbounded for a longer run. | Cap the timeline (e.g. last 500 points) or downsample. | 1 hour |
| C16 | `backend/app/info.py:evaluation` and `index.html` | The dashboard compares evaluation tables against the live replay at runtime and shows a warning when they differ. That check is only exercised when someone changes the model, and has no test. | Add a test that builds mismatching inputs and asserts the warning is raised. | 1-2 hours |
| C17 | `README.md` | The quickstart assumes the local data and artifacts exist. A fresh clone still gets HTTP 503 on three endpoints (see A1). The README names this, but there is no one-command check. | Add a `python -m aegisflow doctor` (or `make check`) that lists which required files are missing and the command to create each. | 2-3 hours |

---

## Suggested order

- **Before the deadline:** nothing further is needed (A11 is optional clutter cleanup).
- **After the deadline:** B1 (the real scientific fix) first, then C8 and C9 (CI and a test that runs without local data), B4 together with C14, then C10 to C13 before any larger ingest (B13). B2 and B3 only if the product direction needs them.
