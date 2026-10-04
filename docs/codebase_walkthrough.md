# AegisFlow — Codebase Walkthrough (viva prep)

Purpose: let any team member explain and defend the code live. Every claim below points to a real file and function, or to a number in a committed report.

How this was produced: read-only pass over the repo. I did **not** run the pipeline, the tests, or the server. Numbers are quoted from `reports/*.md`, `reports/*.json`, `artifacts/models/cic_ids2017/*.json`, and a read-only look at `data/processed/cic_ids2017/*.parquet`. Line numbers are as of this pass; if someone edits a file, they drift.

Contents: 1 Data flow · 2 The three hard pieces · 3 Judge Q&A · 4 Config glossary · 5 Things that look fancier than they are · Appendix (numbers cheat sheet, do-not-run list)

---

## 1. End-to-end data flow (one hop per line)

Raw CSV → demo screen. `python -m aegisflow <cmd>` is wired in `aegisflow/__main__.py` → `aegisflow/cli.py:main`.

| # | Hop | Function (file:line) | What it does |
|---|---|---|---|
| 0 | Load config | `load_config` (`aegisflow/config.py:103`) | Merges `configs/config.yaml`, `datasets.yaml`, `stages.yaml`, `mitre_mapping.yaml` into dot-access objects; applies `--set a.b=c` overrides. |
| 1 | Read raw CSVs | `CicIds2017Adapter.load_raw` (`aegisflow/ml/datasets/cic_ids2017.py:169`) | Reads each of the 8 day-files with `pd.read_csv(..., encoding="cp1252")`, drops re-embedded header rows, concatenates. With `--sample-size`, `spread_sample_plan` (`:91`) picks the same fraction of rows uniformly from the whole of every file, seeded by `random_seed`. |
| 2 | Map to canonical schema | `CicIds2017Adapter._to_canonical` (`:217`) then `coerce_canonical_frame` / `validate_canonical_frame` (`aegisflow/schema.py:83`, `:65`) | Renames CICFlowMeter columns via `_COLUMN_MAP`, converts duration µs→s, parses timestamps (12-hour clock, hours < 8 get +12 h: `:260-261`), sums fwd+bwd packets/bytes. TTL and retransmissions stay NA, never invented. |
| 3 | Clean | `clean_canonical_frame` (`aegisflow/ml/preprocessing/cleaning.py:36`) | Drops rows with missing timestamp / IPs, negative duration, inf values, exact duplicate (timestamp + 5-tuple). Every drop is counted by reason. |
| 4 | Label → class → stage | `apply_stage_mapping` (`aegisflow/ml/preprocessing/labels.py:20`) | Looks up each dataset label in `configs/stages.yaml`. An unmapped label raises `LabelMappingError`. Adds `normalized_attack_class`, `attack_stage`, `label_confidence`. |
| 5 | Write interim | `run_ingestion` (`aegisflow/ml/ingestion/pipeline.py:42`) | Chains steps 1-4, sorts by timestamp, writes `data/interim/cic_ids2017.parquet`. |
| 6 | Flow features | `compute_flow_features` (`aegisflow/ml/features/flow_features.py:21`) | Adds packets/s, bytes/s, bytes/packet and five TCP flag ratios, with safe denominators. Written to `data/processed/cic_ids2017/flows.parquet` by `run_preprocessing` (`aegisflow/ml/temporal/pipeline.py:72`). |
| 7 | Build windows | `aggregate_host_windows` (`aegisflow/ml/temporal/windowing.py:42`) | Groups flows by `source_ip` into 60 s windows with 30 s stride (each flow lands in 2 windows). Aggregates counts, byte/packet sums, flag counts and ratios, unique dst IPs/ports, destination-port entropy (`:190-194`), connection burst (`:197-199`). Also derives label columns (`attack_present`, `dominant_class`, `dominant_stage`) at `:229-293`. Output `host_windows.parquet`. |
| 8 | Build sequences | `build_host_sequences` (`aegisflow/ml/temporal/sequences.py:81`) | Per host, takes 10 consecutive observed windows as input; the target is the `forecast_horizon`-th same-host window whose start is strictly after the last input window's end (`:179-182`). Stores features as nested lists, plus target labels and `target_is_onset` (`:221`). |
| 9 | Chronological split | `compute_temporal_splits` (`aegisflow/ml/temporal/split.py:55`) | Cutoffs = 60 % / 80 % quantiles of `target_window_end` (`:119-120`). Assigns train/val/test or `boundary_excluded` (`:127-136`). Output `sequences.parquet`, `split_metadata.json`. |
| 10 | Quality report | `generate_data_quality_report` (`aegisflow/ml/temporal/reporting.py:53`) | Writes `reports/data_quality_report.{md,json}`. The dashboard's dataset line reads this JSON. |
| 11 | Build tensors | `ForecastDataset.from_parquet` → `from_frame` (`aegisflow/ml/modeling.py:38`, `:42`) | Stacks `sequence_features` to `[N,10,30]`, then selects the 28 `MODEL_FEATURES` (drops the two label-derived columns, `:23-24`, `:55`). |
| 12 | Train | `train_experiment` (`aegisflow/ml/modeling.py:152`), called from the `train` branch of `_dispatch` (`aegisflow/cli.py:155`) | `SequencePreprocessor.fit` on train only (`:160`), majority + logistic-regression baselines, a small LSTM with `pos_weight`, early stopping on val loss, thresholds from validation F1 (`select_threshold`, `:130`). Saves `best.pt`, `preprocessor.joblib`, `logistic.joblib`, `metadata.json`, `metrics.json` to `artifacts/models/cic_ids2017/`. |
| 13 | Offline evaluation | `scripts/audit_phase3.py:run`, `scripts/phase3_baselines.py:run`, `scripts/lodo_pilot.py:run` | Integrity audit, any-vs-onset baselines with per-host check, leave-one-day-out. Outputs in `reports/` (see §2.2). |
| 14 | Start server | `create_app` (`backend/app/main.py:57`) → `ReplayEngine.warm_up` (`backend/app/replay.py:68`) | `uvicorn backend.app.main:app`. A background thread runs `_prepare` (`:78`): loads the **test split** of `sequences.parquet`, scores every sequence once with LSTM and logistic regression, builds the events table. |
| 15 | "Live" prediction | `ReplayEngine.start` (`:132`) → `_run` (`:159`) → `_process` (`:180`) | A thread releases the pre-scored events in `target_window_start` order on a simulated clock (`speed` sim-seconds per wall-second). `_process` computes the risk score (`_risk`, `:112`), and if LSTM p ≥ threshold, appends an alert. |
| 16 | Audit ledger | `Ledger.append` (`backend/app/audit.py:85`) | Writes the alert to SQLite (`data/processed/aegisflow.db`) with `prev_hash` and `hash`. |
| 17 | Dashboard | `backend/app/static/index.html` (`poll`, line 169) | Polls `/replay/status`, `/hosts`, `/alerts` every second; `Verify chain` calls `GET /audit/verify` → `Ledger.verify` (`audit.py:119`). |

**Say this plainly in the viva:** the demo does not run the model on live traffic. Scoring happens in one batch at startup (`_prepare`); the replay only paces the release of those precomputed scores and applies the threshold. Those scores are genuine model outputs on genuine held-out sequences (a test in `backend/tests/test_replay.py` checks that the replay reproduces the offline confusion matrix `[[10622, 270], [408, 79]]`).

Commands, in order: `validate-dataset`, `preprocess --reingest --sample-size 500000`, `train --epochs 8 --batch-size 256 --learning-rate 0.001 --hidden-size 16 --dropout 0.2 --patience 3` (the committed model used exactly these; since B8 they are also the defaults in `configs/config.yaml → model:`, so plain `train` reproduces it and any flag overrides one value), `uvicorn backend.app.main:app`.

---

## 2. The three hard pieces

### 2.1 How temporal leakage is prevented

Plain language: the model may only see the past, the "answer" must happen strictly after the inputs end, the exam (test) period must come after the study (train) period, and nothing computed from the exam may be used to prepare the study material.

Layers, each with its code:

1. **Windows are time-slices of one host's own flows.** `aggregate_host_windows` sorts by timestamp (`windowing.py:90`) and assigns each flow only to windows `[start, end)` that contain its timestamp (`:103-122`). A window never contains a later flow.
2. **Sequences never mix hosts.** `build_host_sequences` loops `for host_id, host_group in clean_df.groupby("host_id")` (`sequences.py:150`) and builds inputs and target from that host only.
3. **Target is strictly in the future of the inputs.** Candidate targets are windows with `window_start > seq_end` (`:179`); the `forecast_horizon`-th such window is picked (`:182`). Three guards raise `ValueError` if violated (`:184-185`, `:196-200`, plus the monotonic-order check at `:169-170`). Tests: `tests/test_sequences.py::test_regression_old_minus_30_second_overlap_is_rejected`, `test_every_target_is_strict_future_nonoverlapping_and_same_host`, `test_no_future_to_past_leakage`, `test_no_cross_host_sequence_leakage`.
4. **Split is by time, never random.** `compute_temporal_splits` takes quantiles of `target_window_end` (`split.py:119-120`). Train = target ends at or before the val cutoff; val = starts at/after val cutoff and ends at/before test cutoff; test = starts at/after test cutoff (`:127-131`). Anything that straddles a cutoff gets `boundary_excluded` (`:134`): 10,635 sequences (12.5 %). Tests: `tests/test_temporal_split.py` (four tests, including "label independent" and "deterministic").
5. **Scaler / imputer fit on train only.** In `train_experiment`: `prep = SequencePreprocessor(...).fit(data.X[train_i])` (`modeling.py:160`), then `.transform` on val and test only. `fit` (`:88-94`) computes train medians for missing values, signed log1p for skewed count features, and a `RobustScaler`. The class weight `negatives/positives` is also train-only (`positive_class_weight`, `:109`).
6. **Label-derived columns are not model inputs.** `attack_flow_ratio` and `benign_flow_ratio` are computed from labels, so `LABEL_DERIVED_FEATURES` (`modeling.py:23`) removes them: 30 → 28 inputs.
7. **Selection uses validation, not test.** Checkpoint = lowest validation BCE loss (`:191-192`). Threshold = max validation F1 (`select_threshold`).
8. **There is an audit script, not just a claim.** `scripts/audit_phase3.py` (a) refits the preprocessor on train and checks it equals the saved one (`:38-41`), (b) flips every target label and zeroes `target_features`, then asserts the input tensor `X` is byte-identical (`:53-58`), (c) re-derives the class weight and best epoch.

Bugs the team found and fixed in earlier phases (these are strengths; know the story):

- Label-derived features were inside the input (`reports/phase3_evaluation_audit.md`). Removed.
- Target window overlapped the input by 30 s because the old code used "row i + L" (`reports/phase3_temporal_alignment_audit.md`). Fixed by the timestamp-based `eligible` rule in `sequences.py:179`.
- CIC-IDS2017 timestamps have a 12-hour clock with no AM/PM, so afternoon traffic sat 12 h early (`reports/phase3_split_audit.md`). Fixed at `cic_ids2017.py:77` and `:260-261`; `test_afternoon_12_hour_timestamps_are_parsed_as_pm` is now a normal test.
- Sampling originally read the head of each file. Replaced with `spread_sample_plan` (`reports/phase3_onset_baseline_audit.md` §1).

What the split does **not** guarantee (be ready for this):

- Validation is thin: 127 positives from 2 episodes (`reports/phase3_training_report.md`). The thresholds picked on it do not transfer to test (LSTM F1 0.304 → 0.189).
- Sequences are "10 consecutive *observed* windows" of a host, not 10 consecutive time slots. Median input span is 630 s (contiguous would be 330 s) and the 90th percentile is about 85,000 s (~23.6 h). A "next window" can be hours or days later for sparse hosts. This is why `boundary_excluded` is so large. (Measured read-only from `sequences.parquet`.)

### 2.2 How the host-identity confound was discovered

The finding: almost every attack in the data comes from one source host, `172.16.0.1`, so a model can score well just by recognising that host's traffic fingerprint.

Order of discovery (taken from the evidence trail and file timestamps; git history was not readable from my session, so confirm the order with whoever ran it):

1. **First sign — `reports/phase3_training_report.md`, "Diagnostics" item 1.** All 382 training positives sit on `172.16.0.1`. On test, the LSTM flags 29 of 29 *benign* sequences from that host plus 75 of 76 attack sequences, but detects only 4 of 411 attacks on every other host (AUC 0.472). I could not find a standalone script that produced these first numbers; the same numbers are reproduced by step 2.
2. **Formalised — `scripts/phase3_baselines.py`.** `per_host` (`:43`) with `SHORTCUT_HOST = "172.16.0.1"` (`:40`) compares "shortcut host vs all other hosts" at each model's own threshold and computes per-host ROC-AUC. Output: `reports/phase3_baselines_onset.json` → `variants.any.models.<model>.host_check`. Written up in `reports/phase3_onset_baseline_audit.md` §5. The dashboard shows it ("Host check" line, from `backend/app/info.py:evaluation`, `:108`).
3. **Decisive — `scripts/lodo_pilot.py`.** It adds a *non-model* reference score, `ref_host_identity` = 1 if host is `172.16.0.1` else 0 (`evaluate_fold`, `:113`), and compares it with the LSTM and logistic regression on leave-one-day-out folds (`assign_fold`, `:57`). It also computes `roc_auc_vs_172.16.0.1_negatives` (`:142`): the class's attack sequences versus that same host's own benign sequences. Conclusion in `reports/lodo_pilot_report.md`: Wed→Fri DoS, both models catch exactly the 42 sequences the host rule catches, and the LSTM flags 29/29 benign sequences from that host. Fri→Wed DoS fails outright (LSTM 12/129 at 38 % FPR; logistic regression 0/129).

Key comparison to quote: **a one-line rule that has never seen a flow ("is it 172.16.0.1?") matches the LSTM.**

Honest nuance: the host ID is *not* a model feature (`MODEL_FEATURES`; `tests/test_modeling.py::test_threshold_evaluation_uses_requested_threshold_and_no_ids_as_features`). So the model cannot memorise the string. The evidence says its behavioural inputs (volume, fan-out, port counts) act as a fingerprint of that host. *Which* features carry the fingerprint was not analysed in the repo; say "we believe" if asked.

The dashboard states this limit openly: `HOST_NOTE` in `backend/app/info.py:18`, shown in the "Honest finding" banner (`index.html:140`).

### 2.3 How the hash-chain audit ledger proves tampering

Code: `backend/app/audit.py`. In plain terms:

- Each alert is a row with 22 fields (`HASH_FIELDS`, `:24`: host, probabilities, thresholds, risk score and components, confidence, stage, truth labels, model version, `created_at`).
- The row's fingerprint is `hash = SHA-256( canonical_json(those 22 fields) + previous_row.hash )` (`canonical` `:44`, `compute_hash` `:49`). "Canonical JSON" = keys sorted, no spaces, so the same data always gives the same text. The first row's "previous hash" is 64 zeros (`GENESIS`, `:21`).
- Floats are rounded to 6 decimals *before* hashing (`_normalize`, `:53`) so what is hashed is exactly what is stored and displayed.
- `Ledger.verify` (`:119`) walks rows in id order and applies two checks to each:
  1. Does this row's stored `prev_hash` equal the previous row's stored `hash`? If not: "chain link broken" (a row was deleted, reordered, or a hash rewritten).
  2. Recompute the hash from the row's **stored columns** and compare to its stored `hash`. If not: "record content does not match its stored hash" (a field was edited).
- Because each hash includes the one before it, changing row 5 changes row 5's hash, which would have to be rewritten into row 6, which changes row 6's hash, and so on to the end.

Tests that prove it (`backend/tests/test_audit.py`): editing a field directly in SQLite is caught and the record is named (`test_api_detects_direct_db_tampering_and_names_record`); editing a record *and* recomputing its hash is caught one row later (`test_rewriting_a_hash_breaks_the_next_link`); deleting a middle record is caught (`test_deleting_a_middle_record_is_detected`).

What it does **not** prove (say this before a judge does; the module docstring already admits the first one):

- Deleting the newest record(s) leaves a shorter, valid chain. Detecting that requires recording `head_hash` somewhere else (`verify` returns it).
- Someone with write access to the DB who edits a row and recomputes *every later hash* produces a valid chain. There is no secret key (no HMAC), no signature, no external anchor.
- `Ledger.reset()` (`:114`) deletes all rows, and `ReplayEngine.start(reset=True)` calls it on every Start click (`replay.py:140-141`). So it is append-only within a replay session, not across sessions.
- `POST /alerts` (`backend/app/main.py:130`) has no authentication.
- It is a single-node SHA-256 hash chain in SQLite. The README says so ("not a decentralized blockchain network"). Use the words "tamper-evident", never "tamper-proof" or "blockchain".

Live tamper demo (only on a throwaway copy of `aegisflow.db`, with the server pointed at it via `AEGISFLOW_DB`): `UPDATE alerts SET risk_score = 0 WHERE id = 3;` then press **Verify chain** → `CORRUPTED`, record #3. Not run by me.

---

## 3. Likely judge questions and where the answer lives

| # | Question | Short honest answer | Backing in code / reports |
|---|---|---|---|
| 1 | **Why does "Predicted Stage" always say UNCERTAIN?** | The model is a binary attack/benign detector. No stage model exists, so the field is hard-coded to the `uncertain_label`. We chose not to invent stages. | `backend/app/replay.py:198` (`"predicted_stage": self.uncertain`); explanation text `STAGE_NOTE` in `backend/app/info.py:20`, shown on the dashboard banner; asserted in `backend/tests/test_replay.py:24`. Why not trained: train positives have only Initial Access (241) and Impact (141); the test has C2 (409), Impact (44), Recon (34), so stage skill could not even be measured. `reports/phase3_training_report.md` "Split class composition". |
| 2 | **Why can risk never exceed 75?** | Risk = 100 × Σ weight × component. The `stage_severity` component (weight 0.25) is fixed at 0 because there's no stage prediction. Remaining weights sum to 0.40 + 0.15 + 0.10 + 0.10 = 0.75. | Weights `configs/config.yaml:50-56`; stage term from `ReplayEngine._stage_term` (the predicted stage is the `UNCERTAIN` label, which is not in the severity table, so 0); formula in `_risk`; asserted `0 ≤ risk ≤ 75` in `test_replay.py:25`; UI shows `max_reachable_risk` from `GET /risk/config` (75). |
| 3 | **How do you know the LSTM didn't just memorise the attacker's IP?** | It can't memorise the IP (no ID features), but we showed it behaves like an IP recogniser: it flags 29/29 benign sequences from that host, and a host-only rule matches it. We report that as a limitation. | `MODEL_FEATURES` (`modeling.py:24`); `scripts/phase3_baselines.py:per_host`; `scripts/lodo_pilot.py` (`ref_host_identity`, `:113`); `reports/lodo_pilot_report.md`; §2.2 above. |
| 4 | **Your F1 is 0.19 and ROC-AUC 0.55. Is it useful?** | Not as a general detector, and we say so. On this split it's indistinguishable from logistic regression (F1 0.183) and below a label-using rule (F1 0.55). The contribution is the leak-free pipeline, the audit trail, and an honest evaluation. | `artifacts/models/cic_ids2017/metrics.json`; `reports/phase3_training_report.md`; dashboard "Model evaluation" table (`info.evaluation`). |
| 5 | **Is this really forecasting? How far ahead?** | The target is "attack present in the host's next observed window after the inputs". Median lead on test is 30 s (90th percentile 630 s, max 18,090 s). 418 of 487 test positives are *continuations* of an attack already visible in the inputs; only 69 are onsets. Onset ROC-AUC for the LSTM is 0.324 (below chance). | `build_host_sequences` (`sequences.py:221` onset flag); `reports/data_quality_report.md` "Onset vs Continuation"; `reports/phase3_onset_baseline_audit.md` §3-4; `metrics.json → forecast_lead_time_seconds`. |
| 6 | **How do you prevent temporal leakage?** | See §2.1. Time-ordered split on target end, strictly-future targets with guards, train-only scaler, no label-derived features, and an audit script that mutates labels and checks the inputs don't change. | Functions and tests in §2.1. |
| 7 | **Is the ledger a blockchain? Can't someone just recompute the hashes?** | No. It's a tamper-evident hash chain. Yes, an attacker with DB write access who recomputes all later hashes wins unless the head hash was stored elsewhere. Tail deletion is also undetected. | `audit.py` docstring lines 8-9; §2.3 limits; `README.md` "Audit Ledger". |
| 8 | **Is the replay "real"? Is that live traffic?** | No. It replays the held-out Friday test split (from 2017-07-07 09:15) through pre-scored model outputs on a simulated clock. Scores are real; the traffic is recorded. | `replay.py` module docstring; `_prepare` (`:78`); `split_metadata.json` test cutoff `2017-07-07 09:15:22`. |
| 9 | **Why does Botnet get 0 of 409?** | Botnet never appears in training. CIC-IDS2017 runs one attack family per day, so a chronological split puts Botnet (Friday morning), Recon and Infiltration after the training period. It's a true zero-shot test, and the model does not transfer (LSTM 4/409; ROC-AUC 0.47). | `reports/phase3_onset_baseline_audit.md` §1; `reports/lodo_pilot_report.md` zero-shot table; `metrics.json`. |
| 10 | **Are the attack stages and MITRE tactics ground truth?** | No. Stages are an inferred proxy from dataset labels; MITRE is a static YAML lookup. The MITRE panel on an alert is shown for the **dataset label**, not for a model prediction. | `configs/stages.yaml` header comment; `label_confidence` column; `backend/app/info.py:mitre` (`:79`, "static lookup; no model involved"); `index.html:210` uses `a.truth_stage`. |
| 11 | **Why a 500k sample, not all 2.83M rows?** | Speed. The LODO pilot argues more rows wouldn't help: the full set adds rows, not days, campaigns or attacker hosts. | `spread_sample_plan`; `reports/lodo_pilot_report.md` "Implication for the full 2.83M run"; `info.dataset_status` sample text (`info.py:37`). |
| 12 | **How was the alert threshold (0.33) chosen? What's the false-alarm cost?** | Maximum F1 on the validation split, among a candidate set that includes every validation score. At that threshold the replay raises 349 alerts, 79 matching an attack label (precision 22.6 %), test FPR 2.5 %. Because validation is thin, the threshold is itself fragile. | `select_threshold` (`modeling.py:130`); `metadata.json → threshold 0.3308`; `test_replay.py:23`; dashboard "match an attack label". |
| 13 | **Why is "confidence" UNCERTAIN on an alert that fired?** | Two different thresholds. Alert fires at LSTM p ≥ 0.3308 (validation-selected). The confidence label says LIKELY_ATTACK only if p ≥ 0.5 (`confidence.stage_prediction_threshold`). Alerts with p in [0.33, 0.5) show UNCERTAIN confidence. Separate from the always-UNCERTAIN *stage*. | `replay.py:197`; `index.html:222`. |
| 14 | **What model is this? (Older config said GRU, hidden 64.)** | An LSTM (hidden 16, one layer). Hyperparameters come from `config.yaml → model:` (which now matches the committed model), overridable by CLI flags; `model.type` other than `lstm` raises `ConfigError`. | `LSTMForecaster` (`modeling.py:138`); `train_hyperparameters` in `aegisflow/cli.py`; `metadata.json → training_config`. |
| 15 | **Does it generalise to other days/datasets?** | Not shown. Leave-one-day-out collapses once host is controlled; only one dataset adapter exists. | `reports/lodo_pilot_report.md` verdict; `configs/datasets.yaml` (others `status: planned`). |

---

## 4. Config glossary (what changing each key does)

Rule of thumb: windowing/split keys require re-running `preprocess` (or `rebuild-temporal`, see warning in the Appendix) **and** retraining. Replay keys take effect at server restart.

### 4.1 Read at runtime — pipeline (`configs/config.yaml`)

| Key (current) | Read in | If you change it |
|---|---|---|
| `windowing.window_size_seconds` (60) | `WindowingConfig.from_config`, `windowing.py:31` | Longer = fewer, smoother windows with more flows each; shorter = noisier. Also changes the minimum forecast lead: lead = first multiple of stride greater than the window size, minus the window size (60/30 → 30 s; 60/60 → 60 s). Requires rebuild + retrain; all committed metrics become stale. |
| `windowing.stride_seconds` (30) | same | Smaller = more overlap, more near-duplicate windows and sequences (each flow sits in window_size/stride windows). Larger = fewer windows. Same rebuild + retrain need. |
| `windowing.sequence_length` (10) | `SequenceConfig.from_config`, `sequences.py:72` | Longer history needs more windows per host: fewer hosts qualify (a host needs ≥ L+1 windows), shorter spans. Changes the model input shape, so a saved model stops loading. |
| `windowing.forecast_horizon` (1) | same | H-th strictly-future window of the same host. Higher = target further away, fewer sequences, harder task. Counts windows, not seconds. |
| `windowing.min_flows_per_window` (1) | `windowing.py:36`, `:184` | Raising it drops sparse windows (fewer windows, fewer hosts, fewer chances of "attack with a handful of flows"). |
| `windowing.group_by` ("source_ip") | `windowing.py:38`, `:85` | The entity being monitored. Any flow column works mechanically. `destination_ip` would key on victims (the idea behind the post-deadline fix), but the features and labels were designed for source hosts. Untested. |
| `split.train_fraction` / `val_fraction` / `test_fraction` (0.6/0.2/0.2) | `split.py:99-101` | Must sum to 1.0 ± 0.01. Moves the cutoffs. `reports/phase3_split_audit.md` shows other fractions can leave 0 positives in val or test. |
| `random_seed` (42) | `cli.py:159`, `cic_ids2017.py:182` | Changes which rows `--sample-size` picks and the training seed. Different data and different model. |
| `paths.data_interim`, `paths.data_processed`, `paths.reports` | `ingestion/pipeline.py:56`, `temporal/pipeline.py:98-101`, `eda.py` | Where Parquet and reports go. Note `info.py` hard-codes `data/processed/cic_ids2017` and `reports/`, so moving them breaks the dashboard. |
| `logging.json_file` | `cli.py:120` | Location of the JSONL log. |

### 4.2 Read at runtime — replay/dashboard

| Key (current) | Read in | If you change it |
|---|---|---|
| `risk_scoring.weights.attack_probability` (0.40) | `ReplayEngine.__init__`, `replay.py:48`; used `:118` | Weight on the LSTM probability. Weights are not validated to sum to 1. If they don't, scores can exceed the stated scale. |
| `…weights.stage_severity` (0.25) | same | Multiplies the severity of the *predicted* stage. The predicted stage is always `UNCERTAIN` (not in the severity table), so the component is 0 and raising this weight changes nothing today. |
| `risk_scoring.stage_severity` table | `ReplayEngine._stage_term` | Severity (0-100) per stage name; only matters once a stage model predicts a real stage. |
| `…weights.prediction_confidence` (0.15) | same | Weight on `(p − t)/(1 − t)`. This is monotone in p, so it largely double-counts attack_probability. |
| `…weights.abnormality_score` (0.10) | same | Weight on the share of the 28 inputs of the last window beyond 3 robust-scaled units. Label-free. |
| `…weights.recent_attack_history` (0.10) | same | Weight on min(1, this host's alerts in the previous 10 simulated minutes / 5). |
| `risk_scoring.thresholds.low` / `medium` (33 / 66) | `replay.py:49`, `level` `:121` | Low/medium/high banding on the server side. The dashboard now reads the same bands from `GET /risk/config`, so they cannot disagree. `thresholds.high` (100) is not read. |
| `confidence.stage_prediction_threshold` (0.5) | `replay.py:50`, `:197` | Cutoff on the *attack probability* for LIKELY_ATTACK vs UNCERTAIN confidence. It does not change which alerts fire and has nothing to do with stages despite its name. |
| `confidence.uncertain_label` ("UNCERTAIN") | `replay.py:51` | Text used for both `confidence_label` and `predicted_stage`. |
| `replay.default_speed` (1000) / `allowed_speeds` ([1,10,100,1000]) | `ReplayEngine.__init__`; `start` validates | Which speeds the API accepts and the default when `POST /replay/start` omits `speed`. The dashboard dropdown is built from these and pre-selects `default_speed`. |
| `database.url` | `main.py:default_db_path` (`:50`) | SQLite file for the ledger; env var `AEGISFLOW_DB` overrides. Pointing at a new file starts a fresh chain. |

### 4.3 In config but NOT read by any runtime code (don't claim they do anything)

| Key | Reality |
|---|---|
| `api.host`, `api.port` | Never read. uvicorn is started from the command line. |
| `sample_size` | Never read. The CLI flag `--sample-size` is used. |
| `active_dataset` | Only `AegisFlowConfig.active_dataset_entry`, which only tests call. Commands take `--dataset`. |
| `paths.models`, `paths.configs` | Never read. Models go to `artifacts/models/<dataset>`. |
| `logging.level` | Only `scripts/prepare_data.py`. The CLI uses `--log-level`. |
| `stages.yaml → stages_order` | Used only by `unknown_stage_order` (`labels.py:58`), which nothing calls. |

### 4.4 Other YAML files in use

- `configs/stages.yaml` (per-label class + stage + `confidence`): read by `apply_stage_mapping`. Adding a dataset label without an entry makes ingestion fail on purpose.
- `configs/mitre_mapping.yaml`: read by `info.mitre`; powers `GET /mitre/{stage}`. Editing it changes the dashboard's MITRE text, not any model output.
- `configs/datasets.yaml`: `raw_dir` and `adapter` are used; `expected_files_glob`, `citation`, `official_url` appear only in help text; `cic_ids2018`, `unsw_nb15`, `ctu_13` are placeholders with no adapter.

### 4.5 Knobs that are hard-coded in code, not config

| Constant | Where | Meaning |
|---|---|---|
| `HISTORY_WINDOW = 10 min` | `replay.py:41` | Look-back for `recent_attack_history`. |
| `/ 5`, `> 3` | `replay.py:116`, `:106` | Alert count that saturates the history term; the |robust z| cutoff for the abnormality share. |
| `_FIRST_MORNING_HOUR = 8` | `cic_ids2017.py:77` | Hours below 8 are treated as PM. |
| `LOG_FEATURES` | `modeling.py:66` | Which 12 features get signed log1p. |
| LSTM architecture: 1 layer, last hidden state, dropout on hidden, `Linear(hidden,1)` | `modeling.py:138-149` | About 3k parameters at hidden 16. |
| Candidate thresholds `[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]` + every val score | `modeling.py:133` | Threshold search space. |
| Training defaults | `configs/config.yaml → model:` via `train_hyperparameters` (`cli.py`) | Epochs 8, batch 256, lr 0.001, hidden 16, dropout 0.2, patience 3 (the committed model). The code-level defaults of `train_experiment` (hidden 32, epochs 20, ...) only apply to programmatic callers. |

---

## 5. Things that look more sophisticated than they are

Do not overclaim these live.

| Where | What it suggests | What it actually is | Safe phrasing |
|---|---|---|---|
| `configs/config.yaml → model:` (the `type` field suggests alternatives); `docs/architecture.md` §5 (three heads, attention, SHAP, Transformer, XGBoost, temporal GNN) | A configurable multi-model, multi-head forecaster | One small LSTM with one binary output, plus logistic regression and a majority baseline. `model.type` must be `lstm`; other values are rejected. No attention, XGBoost, Transformer or stage head exists. Per-prediction SHAP / Integrated Gradients explanations exist (`aegisflow/ml/explain.py`). | "A binary LSTM baseline with a logistic-regression comparison." |
| `risk_scoring` block; `ReplayEngine._risk` | A multi-factor risk engine | A linear blend: 0.40·p + 0.15·(p−t)/(1−t) + 0.10·abnormality + 0.10·history. Stage term is 0 (predicted stage is always UNCERTAIN). Two of the terms are functions of p. It isn't a calibrated probability. | "A transparent, label-free risk score built mostly from the LSTM probability." |
| `confidence.stage_prediction_threshold` | A stage-prediction confidence cut | A cutoff on the *attack probability* (0.5). | "Confidence label on the attack probability." |
| `predicted_stage` | A kill-chain forecast | The literal string `UNCERTAIN`. | "Stage forecasting is not implemented; we show UNCERTAIN." |
| `attack_stage` / `stages.yaml`, Exfiltration in `stages_order` and the severity table | A reliable kill-chain mapping | Proxy from dataset labels. No CIC-IDS2017 label maps to Exfiltration, so the stage never occurs. Heartbleed/Infiltration → Lateral Movement is flagged "weak" in the YAML. | "Inferred proxy stages, not ground truth." |
| MITRE panel in alert detail | MITRE-mapped prediction | A YAML lookup keyed on the **dataset label's** stage. | "Static ATT&CK mapping of the label." |
| "Network Traffic Replay" | Flows pushed through the pipeline in real time | Pre-scored held-out sequences released on a simulated clock. | "Replay of recorded held-out data." |
| "Hash-chain / blockchain" (SIH theme) | Immutable ledger | Single-node SQLite chain; tail deletion undetected; wiped on each Start; unauthenticated `POST /alerts`; no keys. | "Tamper-evident hash chain." |
| "Forecasting / early warning" | Minutes-ahead prediction | Median 30 s lead; 86 % of test positives are continuations; a rule that reads labels beats the models. | "Next-window prediction; we measure onset separately." |
| `SequencePreprocessor` + `RobustScaler` + seeds + `torch.use_deterministic_algorithms(True, warn_only=True)` | Fully reproducible training | Seeded and mostly deterministic on one machine. `warn_only=True` means non-deterministic ops only warn. | "Seeded; same machine reproduces." |
| The LSTM "best" checkpoint | A trained model | Epoch 1 of 4: validation loss was 1.68 at epoch 1 and rose after (2.30, 2.10, 2.75). ~3k parameters, 382 training positives, all from one host. | "A small baseline that overfits quickly." |
| `train` threshold "validation F1 maximization" | Principled tuning | Searches every validation score on 127 positives from 2 episodes; it does not transfer to test. | "Validation-selected; fragile." |
| `LSTMForecaster` | A model class | A thin wrapper that builds `.net`; callers use `LSTMForecaster(...).net`. | Describe `.net`, not "the Forecaster". |
| `dataset-agnostic adapter pattern` (`datasets/base.py`, `datasets.yaml`) | Multi-dataset platform | One adapter (CIC-IDS2017) with dataset-specific hacks (12-hour clock, header fixes). Three other entries are `planned`. | "Adapter interface, one implementation." |
| `validate-dataset` | Dataset validation | File discovery and a 200-row load (`validate.py`); not a statistical validation. | "File and schema check." |
| `reports/data_quality_report.md` "Unique Monitored Hosts: 9,971" | 9,971 monitored hosts | Count of distinct source IPs in flows. Only 1,390 hosts have enough windows to form sequences; 207 appear in the test replay. | "9,971 source hosts, 1,390 with usable sequences." |
| `temporal/pipeline.py:118-121`, `:214-216` (`181905`, `200000`, `18088`, `7`) | Computed cleaning stats | Hard-coded numbers from an old sample run that apply only if the interim file has exactly 181,905 rows. Not triggered by current data, but re-running `rebuild-temporal` writes a report with `raw rows = cleaned rows` and zero drops. | Don't point judges to this code. |
| `modeling.py:213` `stage_proxy.status` | "Four attack-positive test targets" | Stale text from an older run. The current test has 487 positives. | Ignore; not shown in the UI. |
| `docs/model.md`, `docs/api.md`, `docs/demo.md` | Documentation | Each says "PLANNED"; the features exist. `docs/architecture.md` phase table is also stale. | Don't open these in front of judges until updated. |
| `reports/phase3_evaluation_audit.md`, `phase3_temporal_alignment_audit.md`, `phase3_split_audit.md` | Current results | Superseded. `phase3_training_report.md` says so for the first two; the split audit's "blocking" clock bug has since been fixed. Numbers in them (4 test positives, ROC-AUC 0.08) are from the old data. | Cite only `phase3_training_report.md`, `phase3_onset_baseline_audit.md`, `lodo_pilot_report.md`, `data_quality_report.md`. |

---

## Appendix A — Numbers to have in your head

Data: 500,001 sampled lines → 446,137 cleaned flows (46,256 blank rows, 7,589 duplicates, 19 negative durations dropped) → 132,460 host windows (60 s / 30 s) → 85,077 sequences (length 10, horizon 1) from 1,390 hosts (of 9,971 source IPs).

Split (cutoffs 2017-07-06 09:20:22 and 2017-07-07 09:15:22):

| Split | Sequences | Positives | Onsets | Hosts |
|---|---:|---:|---:|---:|
| train | 51,093 | 382 | 9 | 915 |
| val | 11,970 | 127 | 6 | 215 |
| test | 11,379 | 487 | 69 | 207 |
| boundary_excluded | 10,635 | 16 | 2 | 1,257 |

All 382 training positives are on `172.16.0.1`. In test, 76 of 487 positives are on that host (DoS and Reconnaissance); the other 411 are on 8 other hosts (the 409 Botnet sequences plus 2 DoS).

Test results (`artifacts/models/cic_ids2017/metrics.json`):

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.5 | 0 | 0 | 0 | 0.500 | 0.043 | 0 |
| Logistic regression | 0.99868 | 0.642 | 0.107 | 0.183 | 0.473 | 0.143 | 0.0027 |
| LSTM | 0.33075 | 0.226 | 0.162 | 0.189 | 0.553 | 0.177 | 0.0248 |
| Reference: "last window under attack" (uses labels) | – | 0.547 | 0.554 | 0.550 | 0.767 | 0.322 | 0.0206 |

Prevalence (PR-AUC baseline) is 4.28 %. LSTM confusion matrix `[[10622, 270], [408, 79]]`.

Host check (LSTM, test): on `172.16.0.1`, 29/29 benign flagged and 75/76 attacks detected; on all other hosts 4/411 detected, AUC 0.472.

LODO DoS (host-controlled): Fri→Wed, LSTM 12/129, logistic regression 0/129; Wed→Fri, LSTM 42/44 = the host rule's 42/44; LSTM flags 29/29 of that host's benign sequences.

Model: LSTM(28→16) + dropout 0.2 + Linear(16→1), `pos_weight` 132.75, Adam 1e-3, batch 256, best epoch 1 of 4 run.

Replay: 11,379 events, 349 alerts, 79 matching an attack label.

## Appendix B — Do **not** run before the deadline unless you mean to

- `python -m aegisflow rebuild-temporal` / `preprocess` without `--reingest`: overwrites `reports/data_quality_report.{md,json}` and the processed Parquet. The hard-coded benchmark branch won't fire for the current 446,137-row interim file, so the regenerated report will show `raw rows loaded = 446,137` and zero dropped. The dashboard's dataset line reads that JSON. `preprocess --sample-size` without `--reingest` also truncates to the first N rows.
- `python -m aegisflow train ...` (overwrites `artifacts/models/cic_ids2017/`) and `scripts/audit_phase3.py` (overwrites `evaluation_audit.json`, `audit_predictions.csv`).
- Clicking **Start** in the dashboard wipes the alert ledger (`reset=True`).
- `data/`, `artifacts/models/**` and `*.db` are in `.gitignore`: the demo works only on a machine that has the local files. A fresh clone returns HTTP 503 from `/dataset/status`, `/model/status`, `/replay/start` until preprocess + train are run.
