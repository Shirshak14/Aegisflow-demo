# Escalation forecasting study (CTU-13): results (2026-10-09)

Implements `docs/escalation_forecasting_design.md`, which was committed and pushed (9e2fa19) before any model in this study was trained. Reproduce with `.venv/Scripts/python.exe scripts/escalation_forecast_study.py` (a few minutes on CPU; needs the 13 files in `data/raw/ctu_13/`). Raw output: `reports/escalation_forecast_study.json`. The demo model, its threshold, the default dataset and `backend/` are unchanged.

## Answer

**No.** None of the 8 variants passes the bars. The best one, gradient boosting at H = 5 minutes, reaches a median held-out AUC of 0.74 and beats both the time-only control (0.54) and its shifted-label control (0.37). But it is significant against the pseudo-onset null in only 1 of 6 held-out scenarios. At the threshold set for 1% false positives on the validation scenario, it fires on 60 to 98% of the non-escalating windows in three of the six held-out scenarios. Fold AUCs range from 0.16 to 1.0, so the median rests on very few events.

What can be said: on these captures, an infected host's own traffic carries **some** signal that spam, click fraud or flooding is about to start, more than "time since infection" does. That signal is not stable across botnets, and no single threshold works across scenarios.

## 1. Data actually used

- 35 infected hosts, 11,037 host windows of their own botnet traffic, 6,007 of them escalation windows, 62 escalation episodes.
- 3,512 eligible anchors (no escalation in the 10 input windows or the 30 minutes before).
- Scored escalation episodes per host (a host-level count; the design's table counted distinct start windows, which merged bots that escalate in the same window in scenarios 9 and 10): 18 at H = 1, 30 at H = 5, 26 at H = 15 and 30.
- They come from 6 scenarios (3, 4, 8, 9, 10, 12). Scenario 10 alone has 18 host episodes, all from one commanded ICMP flood. Scenario 9 has 7 anchors in total, all positive at H ≥ 15, so it cannot be scored there.

## 2. Results (leave one scenario out)

"p<.05" counts held-out scenarios where the real AUC beats the pseudo-onset null. "Alerted" is escalation episodes with an alert inside the horizon, at the validation scenario's 1% FPR threshold. "Worst FPR" is the worst held-out scenario's false-positive rate at that threshold.

| Variant | H (min) | Folds | Median AUC | Time-only AUC | Shifted-label AUC | p<.05 | Alerted | Worst FPR | Bars passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic | 1 | 4 | 0.635 | 0.473 | 0.575 | 1/4 | 0/18 | 0.52 | none |
| Gradient boosting | 1 | 4 | 0.693 | 0.473 | 0.526 | 1/4 | 11/18 | 1.00 | E3 |
| Logistic | 5 | 6 | 0.590 | 0.538 | 0.667 | 2/6 | 4/30 | 1.00 | none |
| **Gradient boosting** | **5** | 6 | **0.736** | 0.538 | 0.372 | 1/6 | 23/30 | 0.98 | **E1, E3** |
| Logistic | 15 | 5 | 0.575 | 0.417 | 0.486 | 2/5 | 18/26 | 1.00 | none |
| Gradient boosting | 15 | 5 | 0.597 | 0.417 | 0.619 | 1/5 | 24/26 | 1.00 | none |
| Logistic | 30 | 5 | 0.697 | 0.366 | 0.500 | 1/5 | 23/26 | 1.00 | E3 |
| Gradient boosting | 30 | 5 | 0.556 | 0.366 | 0.554 | 0/5 | 23/26 | 1.00 | none |

| Bar | Variants passing (of 8) |
|---|---:|
| E1 median AUC ≥ 0.70 | 1 |
| E2 pseudo-onset p < 0.05 on ≥ 5 folds | **0** (best: 2) |
| E3 beats time-only by ≥ 0.10 and shifted-label by ≥ 0.15 | 3 |
| E4 half the episodes alerted at test FPR ≤ 2% | **0** |
| All four | **0** |

Nothing passed on the first seed, so the second-seed step did not run.

Per held-out scenario, gradient boosting at H = 5: scenario 3 0.71, 4 1.00, 8 0.76, 9 0.92, 10 0.65, 12 0.16.

## 3. What the numbers mean

- **Unlike the CIC onset study, the clock does not explain this.** The time-only control (hour, minutes since capture start, minutes since infection) sits at 0.37 to 0.54. Escalations in CTU-13 are not scheduled at a fixed offset the way the CIC Botnet starts were.
- **But one event per fold cannot be significant.** Scenarios 4 and 12 have one escalation each, scenario 8 two. For gradient boosting at H = 5, the pseudo-onset null's 95th percentile is 0.86 to 0.98 in scenarios 3, 4 and 8, so even a 0.76 is p ≈ 0.11. The only small p values come from scenarios with very few anchors, where a single lucky score is enough.
- **Thresholds do not transfer between botnets.** A threshold fitted on one scenario's negatives gives 0% FPR on some held-out scenarios and 60 to 100% on others. Window features such as bytes per packet and flows per window depend on the bot family (Neris, Rbot, Murlo, NSIS), so the "normal" level differs per scenario. Any deployment would need per-host or per-scenario calibration, which this design did not allow.
- **Scenario 10 dominates the episode count.** Its 18 escalations are one ICMP flood started by one command on ten bots. Counting them as 18 events overstates the evidence; the effective sample is closer to 6 independent escalations, one per scenario.

## 4. Deviations, recorded honestly

The first run stopped with an error at H = 5, after the H = 1 results had printed. Two fixes were made, neither of which changes a model, a feature or a score:

1. When a held-out scenario has fewer than 2 negative anchors, the pseudo-onset null cannot be built. That fold now gets no p value and does not count towards E2 (scenario 9 at H = 5).
2. When the validation scenario has no negative anchors, the 1% FPR threshold cannot be set. Validation now moves to the next scenario in the list that has negatives (scenario 9 is skipped as a validation scenario at H ≥ 15).

The design's counts table under-counted episodes by merging bots that escalate in the same window (section 1). The bars are unaffected.

## 5. What this means for AegisFlow

- Nothing is added to the product, not even as an opt-in model. The demo model and its threshold are unchanged.
- Together with `docs/onset_forecasting_report.md` and `docs/combined_datasets_feasibility.md`: neither CIC-IDS2017 nor CTU-13 supports a claim that AegisFlow forecasts attack starts or escalations. The defensible wording stays "next-window risk", i.e. early detection of activity that is already under way.
- The escalation signal (median 0.74, beating the clock) is worth recording as a lead, not a result. Testing it properly needs many independent escalations from many botnet families, and per-host calibration in the design from the start.

## 6. Round 2: per-host baselines and per-scenario thresholds

Design: `docs/escalation_forecasting_round2_design.md`, pushed (b522495) before any round-2 model was trained. Each anchor's inputs are expressed relative to the median of the same host's own windows up to the anchor time, and E4 uses a label-free top-1% rank threshold inside each held-out scenario. Everything else, including the bars, is unchanged. Raw output: the `normalised` key of `reports/escalation_forecast_study.json`.

**Answer: still no, and the round-1 signal disappears.** No variant passes E1, E2 or E4.

| Variant | H (min) | Median AUC | Round 1 AUC | Time-only | Shifted-label | p<.05 | Alerted (rank) | Worst FPR (rank) | Bars passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic | 1 | 0.442 | 0.635 | 0.473 | 0.498 | 0/4 | 1/18 | 0.52 | none |
| Gradient boosting | 1 | 0.637 | 0.693 | 0.473 | 0.468 | 2/4 | 1/18 | 0.07 | E3 |
| Logistic | 5 | 0.663 | 0.590 | 0.538 | 0.506 | 1/6 | 2/30 | 0.08 | E3 |
| Gradient boosting | 5 | 0.518 | **0.736** | 0.538 | 0.476 | 0/6 | 2/30 | 0.03 | none |
| Logistic | 15 | 0.538 | 0.575 | 0.417 | 0.519 | 1/5 | 20/26 | 1.00 | none |
| Gradient boosting | 15 | 0.560 | 0.597 | 0.417 | 0.583 | 2/5 | 8/26 | 0.11 | none |
| Logistic | 30 | 0.500 | 0.697 | 0.366 | 0.500 | 1/5 | 19/26 | 1.00 | none |
| Gradient boosting | 30 | 0.492 | 0.556 | 0.366 | 0.476 | 2/5 | 4/26 | 0.13 | none |

What it means:

- **The round-1 signal was mostly the host's level, not a change before escalation.** Once each host is compared with its own past, the best round-1 variant (gradient boosting, 5 minutes) falls from 0.74 to 0.52, which is chance. A model that knows "this looks like a Neris or Rbot capture" can rank that capture's anchors above another's; it cannot tell when, inside one host's timeline, the escalation is about to come.
- **The threshold problem is mostly fixed, but there is nothing left to alert on.** With the per-scenario rank threshold, the worst false-positive rate drops to 3 to 13% for gradient boosting, against 98 to 100% in round 1. At that rate it catches 2 of 30 escalations at 5 minutes. The remaining 100% rows are logistic models that give one constant score across scenario 10, so every anchor ties at the threshold.
- **E2 is no closer.** At most 2 folds beat random placements of the escalation on the same hosts, against the 5 required.

Conclusion across both rounds (16 variants): CTU-13 does not show that an infected host's own traffic forecasts its escalation. With about 6 independent escalations it could not show it reliably even if the effect were real. Further rounds on this data would amount to searching for a variant that passes, so I recommend stopping here.
