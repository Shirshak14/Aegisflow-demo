# Escalation forecasting study (CTU-13): design and pre-stated bars

Status: written and committed after counting the data (the `diagnose` part) and before any model in this study was trained. Results go in `docs/escalation_forecasting_report.md`; the script is `scripts/escalation_forecast_study.py`. Nothing here changes the demo model, its threshold, the default dataset or the backend.

Background: `docs/combined_datasets_feasibility.md` found that CTU-13 has no benign-to-attack onsets, because every flow an infected host sends is labelled Botnet. The only within-host events left to forecast are **escalations**: an infected host moving from C2, DNS and connection-attempt traffic to spam, click fraud or ICMP flooding.

## 1. Question

Can a model trained on some CTU-13 scenarios forecast, in a held-out scenario, that an infected host is **about to escalate**, using only that host's own recent traffic while it has not escalated yet?

## 2. Data

- All 13 CTU-13 scenarios, only the 35 infected hosts (10 IPs; one host = one IP in one scenario, `sNN:ip`, so captures are never merged), only their own (source) Botnet flows.
- Host windows from the existing pipeline: 60 s, 30 s stride, 25 features (the 28 model features minus the 3 that CTU-13 lacks) plus the log gap to the previous window.
- A flow is **escalation activity** if its CTU label matches `SPAM|SMTP|HTTP-Ad|ICMP`. A window is an escalation window if it has any such flow.

## 3. Target

Same construction as `docs/onset_forecasting_design.md` §3, with "escalation" in place of "attack":

- **Anchor:** a host window with 10 windows of history before it; no escalation window in those 10 windows and none starting in the 30 minutes before the anchor time t (the anchor window's end).
- **Positive at horizon H:** an escalation window starts on the same host in [t, t + H). H = 1, 5, 15, 30 minutes.
- An escalation episode is a run of escalation windows on one host with no gap over 30 minutes.

## 4. Counts (diagnose, before training)

11,037 windows, 6,007 of them escalation windows, 62 escalation episodes, 3,512 eligible anchors. Most escalations happen too soon after the host's first traffic to have 10 escalation-free windows before them. Episodes with at least one positive anchor:

| Scenario | 3 | 4 | 8 | 9 | 10 | 12 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| H = 1 min | 4 | 0 | 2 | 3 | 1 | 0 | 10 |
| H = 5 min | 4 | 1 | 2 | 4 | 3 | 1 | 15 |
| H = 15 / 30 min | 4 | 1 | 2 | 4 | 3 | 1 | 15 |

Scenarios 1, 2, 5, 6, 7, 11 and 13 have no eligible anchor before any escalation and are used for training only. Scenario 9's 4 episodes have only 7 anchors in total. In scenarios 9 and 10 the bots are commanded together, so their episodes are not independent.

## 5. Folds

Leave one scenario out over the scenarios that have positive anchors at that horizon (6 at H = 5, 15, 30; 4 at H = 1). Test is every eligible anchor of the held-out scenario. Validation (alert threshold) is the next scenario in that list. Training is every other scenario, including those without positives. AUC is pooled over the held-out scenario's hosts.

## 6. Variants and controls

Models: logistic regression on the flattened 10-window input, gradient boosting on per-feature summaries (last, mean, max, min, last minus first). No neural models: they did not beat the classical ones in the onset study, and the data here is smaller.

Controls:

1. **Time-only model (the simple baseline).** Gradient boosting on hour of day, minutes since capture start and minutes since the host's first botnet flow. Botnet operators trigger spam and DDoS on a schedule, so "time since infection" can look like a forecast.
2. **Pseudo-onset null (no retraining).** 200 draws; in each, every host that really escalates in the held-out scenario gets one random pseudo-onset among its negative anchors, labelled with the same horizon rule. Per-fold p = (1 + count of null AUC ≥ real AUC) / 201.
3. **Shifted-label retraining.** Each training host's labels rolled by a random offset (3 retrains for logistic regression, 1 for gradient boosting), scored against the true test labels. Should sit near 0.5.

## 7. Bars (all must hold)

| # | Bar |
|---|---|
| E1 | Median held-out scenario AUC ≥ 0.70 |
| E2 | Pseudo-onset p < 0.05 on at least 5 folds |
| E3 | Median AUC beats the time-only control by ≥ 0.10 and the shifted-label control by ≥ 0.15 |
| E4 | At the threshold giving 1% FPR on the validation scenario, at least half of the held-out escalation episodes get an alert within H before they start, with every test scenario's FPR ≤ 2% |

E2 needs 5 of 6 folds at H = 5, 15 and 30 minutes, and cannot be met at H = 1 (4 folds); that is accepted in advance. 8 variants are scored (2 models × 4 horizons). A variant that passes must pass again on a second seed before anything is proposed, and even then it would be an opt-in research result: 15 escalations from 6 scenarios are evidence about these botnets, not about attacks in general.

## 8. Expected limit

With 1 to 4 episodes per fold, a single fold's AUC is close to one event, and the pseudo-onset null will be wide. A failure here would mean "not shown on this data", not "impossible".
