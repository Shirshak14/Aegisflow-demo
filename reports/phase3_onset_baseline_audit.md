# Phase 3 follow-up: sampling hypothesis, onset targets, baselines, host shortcut

No deep model was retrained. No split, architecture, or Phase 4 change was made. Reproduce the baseline numbers with `python scripts/phase3_baselines.py`; the full output is in `reports/phase3_baselines_onset.json`.

## 1. Truncation hypothesis: refuted

`load_raw()` no longer reads a per-file prefix. That was replaced earlier by `spread_sample_plan`: one global fraction applied to every file, with seeded uniform draws across each whole file.

**Per-file coverage.** For each file, the sampled rows cover the file's full time range at 0.160 of its rows. Friday-WorkingHours-Morning: raw 08:59–12:59, sample 09:00–12:59.

**Bot coverage.** Bot in the raw file is 1,966 flows (09:34–12:59). The sample has 349 Bot rows, 10:04–12:59. After deduplication, 348 Botnet flows from 7 hosts are in the training interim. Heartbleed (11 flows) is the only label absent.

**Why Botnet has no training positives.** All Bot, PortScan (Reconnaissance) and Infiltration traffic falls after the validation or test cutoffs:

- Bot: Friday 09:34–12:59
- PortScan: Friday 13:05–15:23
- Infiltration: Thursday 14:19–15:45
- Test cutoff: 2017-07-07 09:15:22

The cause is the chronological split combined with CIC-IDS2017's one-attack-type-per-day schedule. No sampling change can put these classes into train under this split.

**Row counts reconcile.** 500,001 sampled lines (sized from raw line counts) − 46,256 blank padding rows = 453,745. Then 453,745 − 7,589 duplicates − 19 negative durations = 446,137 cleaned flows.

- All 46,256 blank rows are in Thursday-WebAttacks, lines 170,373–458,966. Timestamp and Label are both empty strings, and every such line is exactly 84 commas.
- The verification sample matches the interim row for row: all 446,137 interim rows are in it, and the 7,608 extra rows are the cleaning drops.
- The rebuilt `sequences.parquet` matches the training copy in all 17 original columns, and `split_metadata.json` is identical.

## 2. Adapter

Unchanged. The requested regression test `test_sampling_does_not_exclude_late_file_attack_burst` was added: a late 10% attack burst is kept, and over 50 seeds the tail keeps roughly its 10% share.

## 3. Onset flag

`sequences.py` gains `target_is_onset`: true when the target is attack-present and none of the `sequence_length` input windows is attack-present. `target_attack_present` is unchanged. The data quality report now shows positives split into onset and continuation per split.

| Split | Positive | Onset | Continuation | Onset classes |
|---|---:|---:|---:|---|
| Train | 382 | **9** | 373 | DoS 6, Web Attack 3 |
| Validation | 127 | **6** | 121 | Infiltration 6 |
| Test | 487 | **69** | 418 | Botnet 67, DoS 2 |
| Boundary excluded | 16 | 2 | 14 | |

## 4. Baselines

Thresholds were chosen on validation only. The "last window under attack" rule reads the label-derived `attack_flow_ratio` and is a reference score, not a model. For onset targets, continuation positives are removed from every split. The LSTM row scores the existing checkpoint, which was not retrained.

### Any-positive targets

| Model | Split | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Majority | val | 0.5 | 0 | 0 | 0 | 0.500 | 0.011 | 0 | [[11843, 0], [127, 0]] |
| Majority | test | 0.5 | 0 | 0 | 0 | 0.500 | 0.043 | 0 | [[10892, 0], [487, 0]] |
| Logistic regression | val | 0.99868 | 0.793 | 0.362 | 0.497 | 0.552 | 0.355 | 0.0010 | [[11831, 12], [81, 46]] |
| Logistic regression | test | 0.99868 | 0.642 | 0.107 | 0.183 | 0.473 | 0.143 | 0.0027 | [[10863, 29], [435, 52]] |
| Last-window rule | val | – | 0.915 | 0.929 | **0.922** | 0.964 | 0.851 | 0.0009 | [[11832, 11], [9, 118]] |
| Last-window rule | test | – | 0.547 | 0.554 | **0.550** | 0.767 | 0.322 | 0.0206 | [[10668, 224], [217, 270]] |
| LSTM (existing) | val | 0.33075 | 0.194 | 0.709 | 0.304 | 0.942 | 0.151 | 0.0317 | [[11468, 375], [37, 90]] |
| LSTM (existing) | test | 0.33075 | 0.226 | 0.162 | 0.189 | 0.553 | 0.177 | 0.0248 | [[10622, 270], [408, 79]] |

Logistic regression reproduces the saved `logistic.joblib` predictions exactly.

### Onset-only targets

| Model | Split | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Majority | val | 0.5 | 0 | 0 | 0 | 0.500 | 0.001 | 0 | [[11843, 0], [6, 0]] |
| Majority | test | 0.5 | 0 | 0 | 0 | 0.500 | 0.006 | 0 | [[10892, 0], [69, 0]] |
| Logistic regression | val | ≈0 | 0.001 | 1.000 | 0.001 | 0.253 | 0.0005 | 1.000 | [[0, 11843], [0, 6]] |
| Logistic regression | test | ≈0 | 0.006 | 1.000 | 0.013 | 0.284 | 0.004 | 1.000 | [[0, 10892], [0, 69]] |
| Last-window rule | val | – | 0 | 0 | 0 | 0.500 | 0.001 | 0.0009 | [[11832, 11], [6, 0]] |
| Last-window rule | test | – | 0 | 0 | 0 | 0.490 | 0.006 | 0.0206 | [[10668, 224], [69, 0]] |
| LSTM (existing) | val | 0.33075 | 0.005 | 0.333 | 0.010 | 0.670 | 0.008 | 0.0317 | [[11468, 375], [4, 2]] |
| LSTM (existing) | test | 0.33075 | 0 | 0 | 0 | 0.324 | 0.004 | 0.0248 | [[10622, 270], [69, 0]] |

With 9 training onsets, logistic regression's scores rank below chance. Its validation-F1 threshold collapses to about 0, so it flags every sequence. The last-window rule gets zero recall by construction, because onsets have no attack in their input windows.

## 5. Host-shortcut check: persists

Test set, any-positive targets, at each model's selected threshold:

| Model | 172.16.0.1: negatives flagged | 172.16.0.1: positives detected | 172.16.0.1: within-host ROC-AUC | Other hosts: positives detected | Other hosts: ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 28 / 29 | 52 / 76 | 0.487 | 0 / 411 | 0.380 |
| LSTM (existing) | 29 / 29 | 75 / 76 | 0.772 | 4 / 411 | 0.472 |

Both models score the attacker NAT host high whether or not it is being attacked. Per-host ROC-AUC on the 7 Botnet victim hosts ranges from 0.023 to 0.677.

For onset targets, 172.16.0.1 has no onsets in validation or test. Every onset is on another host, and no model ranks those above chance. Per the scope, no fix was attempted.
