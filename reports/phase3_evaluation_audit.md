> **SUPERSEDED — historical record.** Numbers below come from an earlier, mis-clocked dataset (4 test positives). Kept as an audit trail of bugs found and fixed. Current results: `phase3_training_report.md`, `phase3_onset_baseline_audit.md`, `lodo_pilot_report.md`, `data_quality_report.md`.

# Phase 3 evaluation audit

Audit run after excluding label-derived model inputs and retraining the same LSTM configuration. No split rows were moved, removed, oversampled, or used for tuning. Test labels were not consulted when choosing thresholds.

## Split distribution

| Split | Rows | Positive (`target_attack_present=1`) | Negative (`=0`) |
|---|---:|---:|---:|
| Train | 9,821 | 52 | 9,769 |
| Validation | 1,574 | 0 | 1,574 |
| Test | 1,411 | 4 | 1,407 |

The training-only positive weight is `9769 / 52 = 187.8653846`.

## Implementation checks

- **Label encoding/orientation:** `target_attack_present` is consumed as stored (`0=benign`, `1=attack`); metadata agrees. BCEWithLogitsLoss receives those labels, and sigmoid probabilities are interpreted as attack probabilities. The low test ROC-AUC is not caused by a reversed label mapping. Reversing the labels would yield ROC-AUC 0.9124, which describes the observed inverse ranking, not a corrected result.
- **Sequence order:** Phase 2 sorts each host's windows by `window_start` before stacking. Sequence table rows are monotonic by `seq_start_time` within train, validation, and test.
- **Feature order and shape:** all splits use the same registered feature ordering. The audited LSTM input is `[batch, 10, 28]` (test tensor `[1411, 10, 28]`).
- **Preprocessing isolation:** independently refitting the saved imputer/scaler on training sequences reproduces the stored parameters. The fit used only the 9,821 training sequences; the same transform was applied to validation and test.
- **Class weighting:** independently recomputed from training labels only; counts and weight match the saved run.
- **Target leakage:** X is read only from `sequence_features`; mutating attack/class/stage/future-state target columns leaves X unchanged. The audit found and fixed two label-derived historical columns that had been inside `sequence_features`: `attack_flow_ratio` and `benign_flow_ratio`. They are excluded from the model feature list now. This reduced inputs from 30 to 28. They were not future target columns, but using ground-truth label ratios as inputs would be unavailable and misleading for an unlabeled operational predictor.
- **Checkpoint:** evaluation loads `best.pt`. It is the checkpoint from epoch 6, the minimum validation BCE loss (`0.16465664`); epoch 7 and 8 losses were higher. The saved metadata and audit agree.

## Validation results

Validation has no positive examples. ROC-AUC and PR-AUC are therefore **not estimable**; no validation F1-based threshold can be selected. At threshold 0.5, the model flagged 124 of 1,574 benign sequences (FPR 7.88%). Validation probability quantiles were: min 0.02805, median 0.04591, 90th percentile 0.31332, 95th percentile 0.69024, 99th percentile 0.82492, max 0.87258. Validation metrics at alert-rate thresholds are in `evaluation_audit.json`; as all validation targets are benign, they show alert volume/FPR only.

## Test results after the input fix

At threshold 0.5:

| Metric | Result |
|---|---:|
| Precision / recall / F1 | 0 / 0 / 0 |
| ROC-AUC | 0.087598 |
| PR-AUC | 0.001874 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[1383, 24], [4, 0]]` |
| Actual positives / negatives | 4 / 1,407 |
| Predicted positives | 24 |

The PR-AUC baseline from prevalence is `4/1411 = 0.002835`; the LSTM is below it. Its four positive probabilities are `0.0282414, 0.0305980, 0.0307473, 0.0379578`, versus a negative probability median of `0.044696`. The positive samples rank at ascending positions 7, 42, 47, and 407 among the test scores. Thus the probabilities are not inverted by code; on this sample, the model assigns low scores to the positives. ROC-AUC is sensitive to these four observations.

The output is not a single collapsed constant: test probabilities range from 0.02673 to 0.80906 (median 0.04468; 99th percentile 0.67790). Most scores are low, with a high-scoring tail. The four positives fall in the low-score cluster.

## Validation-derived threshold analysis

Because validation contains no positives, these thresholds use only validation **score quantiles** to set alert volume. They do not optimize a positive-class metric. Test results below are reporting-only after fixing each threshold from validation.

| Validation score cutoff | Threshold | Validation alerts | Test alerts | Test TP | Test FP |
|---|---:|---:|---:|---:|---:|
| 90th percentile (upper 10%) | 0.313318 | 158 | 40 | 0 | 40 |
| 95th percentile (upper 5%) | 0.690236 | 79 | 13 | 0 | 13 |
| 99th percentile (upper 1%) | 0.824923 | 16 | 0 | 0 | 0 |
| 99.9th percentile (upper 0.1%) | 0.869851 | 2 | 0 | 0 | 0 |

At fixed 0.5, validation produced 124 alerts and test produced 24. None of the thresholds detected a test positive. No threshold can be properly optimized for recall or F1 on the current validation split.

## Audit conclusion and remaining data issue

The LSTM's low ROC-AUC is a real result of this checkpoint on the supplied test rows, not a label-orientation or checkpoint-loading bug. It cannot be called a reliable estimate of general performance: there are only four test positives, no validation positives, and ROC-AUC is highly unstable at this support. The reversed-label diagnostic and positive score values show inverse ranking for these particular examples; that should be reported as model failure on this sample, not dismissed solely as metric noise.

Separately, the Phase 2 target-window timestamps still overlap the input interval for most test rows. The median `target_window_start - seq_end_time` is -30 seconds; all four positive test targets have -30 second separation. Only 150 test rows have a positive time gap, and none is positive. Thus these metrics do not establish leakage-free future attack forecasting, even after the model-input label features were removed. Phase 2 temporal alignment and a regenerated dataset are still required before Phase 3 can be declared forecasting-ready.

## Artifacts

- `artifacts/models/cic_ids2017/evaluation_audit.json`: structured audit checks, validation/test summaries, and validation-derived threshold analysis.
- `artifacts/models/cic_ids2017/audit_predictions.csv`: raw attack probabilities for every train, validation, and test sample, with labels, split, and timestamps; filter `split=test` to inspect all 1,411 test rows.
- `artifacts/models/cic_ids2017/metrics.json`: post-fix training/evaluation metrics and checkpoint history.

The Phase 3 suite passes: **42 passed**. No Phase 4 work was started.
