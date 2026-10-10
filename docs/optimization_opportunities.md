# AegisFlow — Optimization Opportunities (report only; nothing here has been applied)

This file lists what is still open. Items from the earlier A and B series that are finished have been removed; the new series is **C**. You decide what, if anything, to touch.

Method and limits: read-only review of source, config, tests and docs. I did not run the pipeline, the tests, or the server, so each "risk" is an estimate from reading the code, not a measurement. Line numbers are as of this pass.

Rule applied: if touching an item could plausibly break the demo path (`backend/app/*`, `index.html`, the saved artifacts, or the Parquet files the server reads), it is not marked safe, even if the fix is trivial.

## Removed because they are done

- **A1** (off-laptop backup and replay recording, done by you), **A2–A4, A6–A10, A12:** stale docs and report banners, `httpx` in `requirements.txt`, config comments, unused dependencies annotated as planned, unused imports, CLI docstring, stale metrics string, `requirements.lock.txt`.
- **B8** (config wired to behaviour) and **B10** (lazy backend app): both are on `main`.
- **C1–C5, C7:** `artifacts/experiments/**` gitignored; Makefile targets; `GET /alerts` validation; `pyproject.toml` cleanup; `modeling.py` defaults match `config.yaml`; seeding side effect documented.
- **C17** (`python -m aegisflow doctor` / `make check` lists missing data, model and report files with the fix command for each), **C8** (`.github/workflows/tests.yml` runs `pytest` on every push and pull request), **C9** (`backend/tests/conftest.py` trains a tiny model on synthetic sequences, so `test_replay.py` runs without local data), **C14** (`AlertIn` range, flag and length limits).
- **B4:** `POST /alerts` needs `X-API-Key` (disabled unless `AEGISFLOW_API_KEY` is set); a head anchor file `<db>.head`, HMAC-signed when `AEGISFLOW_LEDGER_KEY` is set, catches tail deletion and full chain recomputation; one reused SQLite connection per thread instead of a new connection and `PRAGMA` on every call. Replay Start no longer wipes the ledger: sessions accumulate in one verifiable chain and reads default to the current session (`session_id=all` for everything).
- **B14** (real ingestion counts): `run_ingestion` writes `data/interim/<dataset>.ingest.json` (raw rows, cleaned rows, drops by reason); `preprocess` and `rebuild-temporal` read it and no longer contain the 181,905-row hard-coded numbers. Without the file, the report says the counts were not recorded.
- **B9** (calibration, first part): `python -m aegisflow calibrate` fits Platt scaling on the validation split and writes `calibration.json`; replay `confidence` uses it when present. It is monotone, so the 349 alerts / 79 true positives are unchanged. Committed model: a = 1.869, b = -1.743; test Brier 0.050 -> 0.039, ECE 0.096 -> 0.032; mean confidence of the 349 alerts 0.46 -> 0.18 (their precision is 0.23); LIKELY_ATTACK labels 80 -> 41. The risk score still uses the raw probability. Isotonic was not tried (too few validation positives).
- **B2** (push updates): `GET /live` streams server-sent events; the dashboard opens one `EventSource` instead of five requests per second, gets a snapshot only when it changes, and goes quiet when the replay stops. Checked in the browser against the real model: one `/live` request for a whole replay, status and counts updated live, Stop reflected. Not changed: the capture panel's 3 s `/stream/*` polling and the `/classic` page.
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
| B1 | **Per-host normalisation / host-controlled representation** — *experiment done, not adopted.* Opt-in `host_relative=True` in `ForecastDataset` (causal rolling per-host baseline); results in `reports/hostrel_report.md`, run with `scripts/hostrel_eval.py`. | Weakens the host shortcut (within-host ROC on 172.16.0.1 0.77 to 0.55) but does not fix detection: Botnet 3/409, 29/29 benign still flagged on 172.16.0.1, PR-AUC 0.177 to 0.112, logistic regression threshold collapses. Not wired into `train`, the backend or the dashboard. | Remaining: a lodo rerun, MAD scaling, other lookbacks | Adopting it would change every dashboard number for no demonstrated gain. Needs more hosts/campaigns first (B13). |
| B3 | **Real streaming path** (flows → windows → sequences online, scoring as data arrives) | `ReplayEngine._prepare` scores the whole test split in one batch at startup; the replay only paces pre-computed scores. | 1-2 weeks | New code on the hot path; needs incremental windowing and a state store. |
| B5 | **Read the ingest in chunks with `usecols`** (`cic_ids2017.py`) — *implemented, awaiting review.* | `_read_used_columns` parses only the 26 columns `_to_canonical` uses (of ~85), in 250,000-row chunks concatenated once; output checked identical to golden files with `scripts/golden_ingest.py`. Largest CSV: peak memory 1.65 GB to 0.92 GB. `_to_canonical` still copies the frame once. Not yet run on the full 2.83M rows. | 0.5-1 day | Must produce byte-identical Parquet; needs a golden-file comparison. |
| B6 | **Vectorise windowing** (`windowing.py`: per-flow `np.arange` list comprehension; row-wise `.apply` for dominant class/stage/distribution) | Slowest Python-level loop in the pipeline. | 1 day | Output must match exactly; one off-by-one changes `attack_present`. |
| B7 | **Store sequences as arrays, not nested Python lists** (`sequences.py` `.tolist()`; `modeling.py` `ForecastDataset.from_frame` converts back with a double `np.asarray(...tolist())`; per-sequence `np.flatnonzero` over a host's windows is quadratic per host) | `sequences.parquet` is ~32 MB for 85k rows; loading goes through slow object arrays. | 1 day | Touches the leakage-critical builder and the training loader. |
| B9 (rest) | **PR-based alert threshold** | The alert threshold is still the validation F1 maximum over 127 positives from 2 episodes. Moving it changes alert counts (349 / 79 in `backend/tests/test_replay.py`). | 0.5 day | A decision about the demo's alert numbers, not a code fix. |
| B11 | **SQLAlchemy migration** | Only worthwhile for a non-SQLite database; otherwise leave the dependency commented out. | 1 day | Hashes must stay byte-identical. Not recommended without a real need. |
| B12 | **A real stage model** (multi-class head or per-stage binary heads) and a non-zero `stage_severity` term | "Predicted Stage" is always UNCERTAIN; risk is capped at 75. | Days, plus more data | Train positives cover only Initial Access and Impact; test has C2, Impact and Recon. Needs more attack days and hosts first. |
| B13 | **Better data and evaluation design**: full 2.83M ingest, episode- or campaign-grouped CV, a second dataset (adapters for CSE-CIC-IDS2018 / UNSW-NB15 are placeholders in `configs/datasets.yaml`) | Thin validation (127 positives, 2 episodes) makes thresholds fragile; one attacker host makes everything confounded. `lodo_pilot_report.md` argues rows alone won't help; you need more hosts and campaigns. | 1-3 weeks. **Scoped, see the data-limitation section of `reports/hostrel_report.md`**: no checked source has many attacker hosts per class together with usable IPs (full CIC-IDS2017: one attacker host per class; CSE-CIC-IDS2018 CSVs have no source IP; CTU-13 labels are one-to-one with infected IPs; UNSW-NB15 unverified). | Everything downstream changes. |

---

## Section C — New findings (not in the earlier A/B lists)

C6 is partly done; C10-C13, C15 and C16 are for after the deadline. C1-C5, C7, C8, C9, C14 and C17 are done and removed.

### C-low: still open

| # | File | Issue | Fix | Risk |
|---|---|---|---|---|
| C6 | `aegisflow/ml/modeling.py` `train_experiment` | The `;`-packed statements in the training loop were split onto separate lines (behaviour unchanged). The function is still one ~100-line block doing preprocessing, three models, metrics and file output, and the rest of the file (e.g. `predict_sequences`) keeps the dense style. | Run a formatter (none is installed in `.venv`), then split into `fit_lstm`, `evaluate`, `save_artifacts`. | Low for formatting; the split is a post-deadline refactor |

### C-later: worth doing after the deadline

| # | File | Issue | Fix | Effort |
|---|---|---|---|---|
| C10 | `backend/app/replay.py` `_run`/`_process` | The hot loop does `ev.target_window_start.iloc[self.cursor]` and `ev.iloc[self.cursor]` per row. Each pandas `iloc` builds Series/objects, so a 10,000× replay spends most of its time in pandas overhead. | Convert the columns to NumPy arrays or a list of tuples once in `_prepare`, and iterate over those. | 2-3 hours (demo path; verify counts unchanged) |
| C11 | `backend/app/replay.py:_prepare`, `modeling.py` `predict_sequences` | The LSTM scores the whole split in one `net(torch.tensor(x))` call, and `train_experiment` does the same for validation and test. Fine at 11k sequences; memory grows linearly, so the 2.83M ingest (B13) could run out of RAM. | Score in batches (e.g. 4,096) under `torch.no_grad()`. | 1-2 hours |
| C12 | `backend/app/replay.py:_prepare` | Reads all of `sequences.parquet` (~32 MB) and then filters to `split == "test"`. `info._sequence_host_counts` reads it again. Slower server start-up than necessary. | Use `pd.read_parquet(path, filters=[("split", "==", "test")])` and only the columns needed. | 1 hour |
| C13 | `aegisflow/ml/modeling.py:select_threshold` | Tries every unique validation probability as a candidate and calls `f1_score` for each, which is O(n²). It is instant at 1.5k rows but slow at larger validation sets. Ties also resolve to the lowest threshold. | Use `sklearn.metrics.precision_recall_curve` and compute F1 for all thresholds at once. | 1 hour; must reproduce the committed threshold exactly |
| C15 | `backend/app/replay.py` `hosts` state | Each host's `timeline` list grows by one entry per sequence for the whole replay and `/hosts/{id}` copies it on every call. Fine for 11k sequences; unbounded for a longer run. | Cap the timeline (e.g. last 500 points) or downsample. | 1 hour |
| C16 | `backend/app/info.py:evaluation` and `index.html` | The dashboard compares evaluation tables against the live replay at runtime and shows a warning when they differ. That check is only exercised when someone changes the model, and has no test. | Add a test that builds mismatching inputs and asserts the warning is raised. | 1-2 hours |

---

## Suggested order

- **Before the deadline:** nothing further is needed (A11 is optional clutter cleanup).
- **After the deadline:** B1 (the real scientific fix) first, C10 to C13 before any larger ingest (B13). B2 and B3 only if the product direction needs them.
