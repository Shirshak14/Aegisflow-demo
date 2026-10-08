# Onset forecasting study: design and pre-stated bars

Status: written and committed before any model in this study was trained. Results go in `docs/onset_forecasting_report.md`; the script is `scripts/onset_forecast_study.py`. Nothing here changes the demo model, its threshold or the backend.

## 1. Question

Can anything trained on the local CIC-IDS2017 data forecast that a host is **about to start** attacking (or being used in an attack) from that host's traffic while it is still attack-free? Scoring a window that already contains attack traffic, or predicting the next beacon of an ongoing Botnet episode, does not count.

## 2. Why the existing target cannot answer it

- `target_attack_present` is "attack in the host's next observed window". 926 of 1,012 positive targets (92%) are continuations: attack traffic is already in the inputs.
- `target_is_onset` (86 sequences) only requires the last 10 observed windows to be attack-free. Most of those are mid-episode gaps between Botnet beacons, not episode starts.
- "Next observed window" has no fixed lead time: windows exist only when a host sends traffic, so the target can be 30 seconds or 18 hours after the inputs.

## 3. Target used here

Anchors are host windows (60 s, 30 s stride). For anchor time t (the end of the anchor window) and horizon H:

- **Inputs:** the host's last 10 windows ending at or before t, the 28 model features plus the log gap to the previous window.
- **Eligible:** the host has no attack window starting in [t - 30 min, t) and no attack in its 10 input windows. 30 minutes is the episode gap rule used in earlier reports.
- **Positive:** the host has an attack window starting in [t, t + H). Horizons H = 1, 5, 15, 30 minutes, fixed in advance.

An episode is a run of attack windows on one host with no gap over 30 minutes. A host that is silent before its onset has no anchors there and cannot be forecast from its own traffic; that is reported, not hidden.

## 4. Folds

Leave one host out over every host that has an episode onset with eligible anchors before it (expected: 192.168.10.5, .8, .9, .14, .15, .17, .50). Test is every eligible anchor of the held-out host. Validation (early stopping and the alert threshold) is the next host in that list. Training is every other host. 172.16.0.1 and 205.174.165.73 stay in training only. Training negatives may be subsampled to 30% for the neural models (test is never subsampled).

## 5. Variants

- Features: **own** (host features), **own+net** (plus per-window network context: active hosts, total flows, total bytes, mean destination ports, all computed without labels), **time** (hour and day of week only; a control, not a candidate).
- Models: logistic regression, gradient boosting on per-feature summaries (last, mean, max, slope), LSTM, Transformer, multi-task LSTM (attack head plus next-3-window state head), temporal GNN (own+neighbour features). Classical models run at all four horizons; neural models at H = 5 and 15 minutes only, to bound runtime.

## 6. Controls

1. **Pseudo-onset null (no retraining).** For each held-out host, 200 random pseudo-onset times on that host, labelled with the same horizon rule, scored with the same trained model. Per-fold p = (1 + count of pseudo AUC >= real AUC) / 201.
2. **Shifted-label retraining.** Every training episode's onset is moved to a random time on its own host, the model is retrained, and scored against the true test labels. Should sit at chance.
3. **Time-only model.** Gradient boosting on hour and day of week. Botnet starts on every host between 10:03 and 10:36 on Friday, so a clock can look like a forecast.
4. **Clock-matched AUC.** Pre-onset anchors against the same host's eligible anchors within 60 minutes of the same clock time on other days.

## 7. Bars (all must hold for a variant to count as forecasting)

| # | Bar |
|---|---|
| P1 | Median held-out within-host AUC >= 0.70 |
| P2 | Pseudo-onset p < 0.05 on at least 5 of the folds |
| P3 | Median clock-matched AUC >= 0.65 |
| P4 | Median AUC beats the time-only control by >= 0.10 and the shifted-label control by >= 0.15 |
| P5 | At the threshold giving 1% FPR on the validation host, at least half of the held-out episodes get an alert within H before onset, with test-host FPR <= 2% |

About 40 variants are tried, so a variant that passes must also pass P1 to P5 on a second seed before anything is proposed. If one does, it would be offered as a separate opt-in model; the demo stays as it is either way.

## 8. Expected limit

The local data has 18 attack episodes in total. Only about 8 start on a host that has benign traffic beforehand, and 6 of those are one Botnet campaign that begins within 33 minutes on Friday morning. Even a pass would be evidence about one campaign, not about attacks in general.
