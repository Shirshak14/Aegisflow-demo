# Escalation forecasting, round 2: per-host baselines and per-scenario thresholds (design)

Status: written and committed before any round-2 model was trained. Round 1 (`docs/escalation_forecasting_report.md`) failed E2 and E4: the signal did not hold up against random escalation placements, and a threshold fitted on one botnet scenario gave 60 to 98% false positives on others. This round tests one stated fix for that cross-scenario failure. Results go in `docs/escalation_forecasting_report.md` §6; the code is the `normalised` part of `scripts/escalation_forecast_study.py`. Nothing here changes the demo model, its threshold, the default dataset or the backend.

## 1. What changes

Only two things change from `docs/escalation_forecasting_design.md`:

1. **Inputs relative to the host's own baseline.** For an anchor at time t, every one of its 10 input windows (25 features plus log gap, signed-log as in `SequencePreprocessor`) has subtracted from it the per-feature median of **all** windows of the same host (one IP in one scenario) that ended at or before t. This is causal (nothing after t), always uses at least 10 windows, and removes each bot family's and capture's level, so the models see only change relative to the host itself.
2. **Per-scenario rank threshold for E4.** In the held-out scenario, alert on the anchors whose score is in the top 1% of that scenario's own anchor scores. This uses no labels, but it does use the whole capture's unlabelled scores, so it is not causal; that limitation is stated in advance. The round-1 threshold (1% FPR on the validation scenario) is still reported for comparison.

## 2. What stays the same

Data, anchors, horizons (1, 5, 15, 30 minutes), leave-one-scenario-out folds and the validation rule, models (logistic regression on flattened inputs, gradient boosting on last/mean/max/min/slope summaries, same hyperparameters), the time-only control (hour, minutes since capture start, minutes since infection; inputs not normalised), the pseudo-onset null (200 draws) and the shifted-label retraining control. Nothing is tuned on held-out scenarios; there is no hyperparameter search.

## 3. Bars (unchanged)

| # | Bar |
|---|---|
| E1 | Median held-out scenario AUC ≥ 0.70 |
| E2 | Pseudo-onset p < 0.05 on at least 5 folds |
| E3 | Median AUC beats the time-only control by ≥ 0.10 and the shifted-label control by ≥ 0.15 |
| E4 | With the per-scenario top-1% rank threshold, at least half of the held-out escalation episodes get an alert within H before they start, with every test scenario's FPR ≤ 2% |

8 more variants are scored (2 models × 4 horizons), 16 in all across both rounds. A pass still needs a second seed before anything is proposed, and would remain an opt-in research result.

## 4. Expected limit

Normalisation can fix the threshold problem (E4) but not the sample size: there are still about 6 independent escalations, and E2 is a test of how often the real start beats random starts on the same hosts. If E2 still fails, the answer is that this data cannot show escalation forecasting, whatever the model.
