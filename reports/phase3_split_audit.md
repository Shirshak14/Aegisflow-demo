> **SUPERSEDED — historical record.** The 12-hour-clock bug described here was fixed (`aegisflow/ml/datasets/cic_ids2017.py`, PM shift for hours < 8) and the data was regenerated. Current results: `phase3_training_report.md`, `phase3_onset_baseline_audit.md`, `lodo_pilot_report.md`, `data_quality_report.md`.

# Phase 3 split audit — no revised split created

**Decision:** The chronological split was not changed. `sequences.parquet` and `split_metadata.json` are unchanged. The existing 60/20/20 split is still the only split. It is not fit for model selection, and no revised development split exists. The LSTM was not retrained.

## Blocking finding: the time axis is wrong upstream of the split

The raw CIC-IDS2017 `TrafficLabelling` CSVs use a 12-hour clock with no AM/PM marker. The ingestion adapter (`aegisflow/ml/datasets/cic_ids2017.py`) parses `7/7/2017 3:30` as 03:30 when it means 15:30. The raw files show this directly:

| Raw file | Hours, in order of first appearance |
|---|---|
| Monday-WorkingHours | 8, 9, 10, 11, 12, 1, 2, 3, 4, 5 |
| Friday-…-Afternoon-PortScan | 1, 2, 3 |
| Friday-…-Afternoon-DDos | 3, 4, 5 |
| Thursday-…-Afternoon-Infilteration | 1, 2, 3, 4, 5 |

`flows.parquet` contains only hours 1–3 and 8–10, so every afternoon flow sits 12 hours earlier than it really happened. Sequence construction and splitting were correct *with respect to the parsed timestamps*. Measured against real time, the current `sequences.parquet` has:

- 722 of 15,668 sequences whose target is not in the future of the input end
- 996 sequences whose input start is after its input end
- 1,718 violating sequences in total (train 667, test 237, boundary-excluded 814), 11 of them with positive targets

Example: `172.16.0.1`, input 2017-07-06 01:15–01:24 (really 13:15–13:24), target at 09:14 on the same day. The target is four hours in the **past**.

Any split cutoff placed on this axis is not chronological.

## Second limitation: the flows are a sample from the start of each file

`flows.parquet` holds 181,905 flows, against about 2.83M raw rows. They come from a sampled ingest that reads the first `nrows` of each file, so each day covers only about its first hour of each capture file. Attack coverage is almost gone: DDoS 2,793 flows, FTP-Patator 1,175, Web Brute Force 280, PortScan 4, Bot 2, slowloris 1. SSH-Patator, Infiltration, Heartbleed, DoS Hulk and DoS GoldenEye are absent.

## What a chronological split could look like on corrected time

This was simulated in memory only; nothing was written. PM hours were shifted by +12 h, then the unchanged `aggregate_host_windows`, `build_host_sequences` and `compute_temporal_splits` were run. Result: 15,667 sequences, 61 positives. Those positives form only **6 attack episodes**, and 5 of them are on the single host `172.16.0.1`:

| # | Class | Host | Real target time | Sequences | Distinct target windows |
|---|---|---|---|---:|---:|
| 1 | Brute Force | 172.16.0.1 | Tue 10:09–10:30 | 25 | 23 |
| 2 | Denial of Service | 172.16.0.1 | Wed 14:23–14:24 | 2 | 1 |
| 3 | Web Attack | 172.16.0.1 | Thu 09:14–09:29 | 23 | 18 |
| 4 | Botnet | 192.168.10.12 | Fri 09:33–09:36 | 4 | 4 |
| 5 | Reconnaissance | 172.16.0.1 | Fri 13:04–13:07 | 3 | 2 |
| 6 | Denial of Service | 172.16.0.1 | Fri 15:55–15:58 | 4 | 2 |

All 32 train/val/test fraction combinations tried (train 0.40–0.70, val 0.10–0.30, test ≥ 0.10) give at most **one** positive episode in validation. Most give either zero validation positives or zero test positives. Examples:

| Fractions | Train (pos) | Val (pos) | Test (pos) | Val cutoff | Test cutoff |
|---|---|---|---|---|---|
| 0.60/0.20/0.20 (current config) | 9,406 (50) | 1,507 (4, Botnet) | 1,314 (**0**) | Fri 09:04:28 | Fri 13:21:28 |
| 0.40/0.10/0.50 | 6,287 (29) | 597 (10, Web Attack) | 4,909 (11) | Thu 09:15 | Thu 13:13 |
| 0.50/0.25/0.25 | 7,870 (50) | 1,959 (6, Botnet + Recon) | 1,895 (**0**) | Thu 13:13 | Fri 13:06 |

In the 0.40/0.10/0.50 case, the Web Attack episode is cut across the train/val boundary, so validation would score a continuation of an attack the model trained on.

Validation positives would be one or two short episodes of one class, so choosing a threshold on them would mean fitting that single episode. That is not a statistically meaningful validation set, so no revised split was forced.

## Current (unchanged) split, parsed-time axis

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| Train | 9,439 | 50 | 9,389 |
| Validation | 1,424 | 0 | 1,424 |
| Test | 1,330 | 4 | 1,326 |
| Boundary excluded | 3,475 | 8 | 3,467 |

Cutoffs, taken from `target_window_end` quantiles on the mis-parsed clock: global min 2017-07-03 08:55:58, validation cutoff 2017-07-07 01:04:28, test cutoff 2017-07-07 03:42:58, global max 2017-07-07 09:36:58.

## Required before any split revision or retraining

1. Fix PM parsing in the CIC-IDS2017 adapter. `tests/test_cic_ids2017_adapter.py::test_afternoon_12_hour_timestamps_are_parsed_as_pm` is a strict xfail that will flag the fix.
2. Re-ingest, preferably the full dataset or at least a sample that isn't taken from the start of each file, then rebuild windows, sequences and the temporal audit.
3. Only then re-evaluate cutoffs, keeping the original 60/20/20 split metadata alongside any revised development split.
