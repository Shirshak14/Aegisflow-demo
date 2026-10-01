> **SUPERSEDED — historical record.** Numbers below come from an earlier, mis-clocked dataset (4 test positives). Kept as an audit trail of bugs found and fixed. Current results: `phase3_training_report.md`, `phase3_onset_baseline_audit.md`, `lodo_pilot_report.md`, `data_quality_report.md`.

# Phase 3 temporal alignment repair and model rerun

## Previous bug

The old sequence builder selected target row `i + sequence_length - 1 + forecast_horizon`. With 60-second windows, 30-second stride, sequence length 10, and horizon 1, the tenth input window ended at `start + 60s`, while the next dataframe row started only 30 seconds after the tenth window's start. Its target therefore began **30 seconds before the input ended**. Four old test positives had overlapping input/target time intervals, invalidating their forecasting interpretation.

## Code change and temporal rule

The builder still forms consecutive, chronological input windows independently within each host. It now selects the target by timestamp, not by assuming the next row is in the future:

1. Set `seq_end_time` to the end of the final input window.
2. Find same-host windows with `window_start > seq_end_time`.
3. Select the one-based `forecast_horizon`-th eligible window. Horizon 1 is the first strictly future, non-overlapping target; horizon 2 is the second such window.
4. Reject any target that starts at or before the latest input-window end.

This strict `>` rule avoids ambiguity at touching boundaries. `forecast_horizon` counts eligible strictly future windows, not raw dataframe rows or seconds. The rule is documented in code and `configs/config.yaml`. A regression test reproduces the former −30-second case and now asserts a positive gap. Tests also check horizon changes, chronological input values, same-host target, no overlap, and existing split quarantine behavior.

## Regenerated Phase 2 data

Rebuilt from existing `data/processed/cic_ids2017/flows.parquet`; no raw files were read or modified, and no download or ingestion was run. `flows.parquet` remains unchanged. The regenerated outputs are:

- `data/processed/cic_ids2017/host_windows.parquet` — 36,573 rows, 4,616 monitored hosts
- `data/processed/cic_ids2017/sequences.parquet` — 15,668 rows, 552 hosts with qualifying sequences
- `data/processed/cic_ids2017/split_metadata.json`
- `reports/data_quality_report.md`
- `reports/data_quality_report.json`

For all **15,668** sequences, target gap (`target_window_start - seq_end_time`) is:

| Minimum | Median | Maximum | Gap <= 0 | Overlap |
|---:|---:|---:|---:|---:|
| 30 seconds | 30 seconds | 316,770 seconds | 0 | 0 |

The large maximum is due to host-specific gaps between observed windows; horizon 1 selects the next eligible observed window for that host, so elapsed forecast time varies.

## New distributions

| Partition | Sequences | Positive | Negative |
|---|---:|---:|---:|
| Train | 9,439 | 50 | 9,389 |
| Validation | 1,424 | 0 | 1,424 |
| Test | 1,330 | 4 | 1,326 |
| Boundary excluded | 3,475 | 8 | 3,467 |

Target dominant classes:

| Partition | Class counts |
|---|---|
| Train | Benign 9,389; Brute Force 25; Web Attack 23; Denial of Service 2 |
| Validation | Benign 1,424 |
| Test | Benign 1,326; Botnet 4 |
| Boundary excluded | Benign 3,467; Reconnaissance 4; Denial of Service 4 |

Target dominant stages (inferred proxies):

| Partition | Stage counts |
|---|---|
| Train | Benign 9,389; Initial Access 48; Impact 2 |
| Validation | Benign 1,424 |
| Test | Benign 1,326; Command and Control 4 |
| Boundary excluded | Benign 3,467; Reconnaissance 4; Impact 4 |

The train/validation/test split remains chronological with its configured 60/20/20 time cutoffs; the 3,475 boundary sequences remain quarantined. Boundary integrity checks passed. Validation still has zero positive targets.

## Phase 3 input and retraining

The prior Phase 3 feature leakage correction remains active. `attack_flow_ratio` and `benign_flow_ratio` are label-derived and excluded from `X`; the model has 28 numeric traffic features in the same deterministic order across all splits. The tensor shape is `[batch, 10, 28]`. A test mutating all target columns leaves the model input unchanged. Preprocessing is fitted on training inputs only, and positive weight is calculated only from training labels: `9,389 / 50 = 187.78`.

Retrained the majority/prior baseline, logistic regression, and the unchanged small LSTM (seed 42, epochs 8, batch 256, hidden 16, learning rate .001, dropout .2, patience 3). Best checkpoint: epoch 6 by validation BCE loss. No threshold was tuned on test data.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Predicted positives | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority/prior | 0 | 0 | 0 | 0.5000 | 0.003008 | 0 | 0 |
| Logistic regression | 0 | 0 | 0 | 0.3767 | 0.002951 | 30 | 0 |
| LSTM | 0 | 0 | 0 | 0.0767 | 0.001965 | 20 | 0 |

All metrics use the same 1,330-row chronological test set (4 positive, 1,326 negative) at threshold 0.5. LSTM confusion matrix is `[[1306, 20], [4, 0]]`. Test prevalence baseline PR-AUC is `4/1330 = 0.003008`.

Validation has no positives, so validation ROC-AUC/PR-AUC and positive-class threshold optimization are not estimable. Validation score-quantile thresholds are reported in `artifacts/models/cic_ids2017/evaluation_audit.json` only as alert-volume operating points; none detected a test positive. No test-based tuning was performed.

## Conclusion and limitations

The temporal target construction now guarantees strictly future, non-overlapping targets; all regenerated sequences pass the timestamp check. The Phase 2 suite passed, the processed files and reports were regenerated, and the existing Phase 3 models were retrained. The Phase 3 feature input remains free of direct label-derived features.

The repaired temporal alignment makes the experiment a valid future-window setup, but the Phase 3 results do **not** establish a useful predictor: test support is only four attacks, validation contains none, and the LSTM's observed ranking is poor (ROC-AUC 0.0767; PR-AUC below prevalence baseline). Its reversed-label diagnostic ROC-AUC is 0.9233, while the stored label mapping and training path are correct; this indicates inverse ranking on these four test positives, not a label-encoding fix. The four examples are too few for a stable generalization estimate. Proxy stage labels are not ground truth and were not separately scored.

**Phase 3 is temporally repaired and reproducible, but its model evidence is not strong enough to claim reliable attack forecasting.** No Phase 4 work was started.
