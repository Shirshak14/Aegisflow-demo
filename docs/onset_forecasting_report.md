# Onset forecasting study: results (2026-10-09)

Implements `docs/onset_forecasting_design.md`. The design, the bars P1 to P5 and the one recorded deviation were committed before any model in this study was trained. Reproduce with `.venv/Scripts/python.exe scripts/onset_forecast_study.py`, which takes about 2.5 hours on CPU. Raw output is in `reports/onset_forecast_study.json`. The demo model, its threshold, `backend/` and the Parquet files are unchanged.

## Answer

**No.** None of the 24 candidate variants passes the bars, and none comes close to passing all of them. At every horizon, a control that knows only the clock time and day of week forecasts the held-out attack starts **better** than every traffic model except one Transformer run, which ties it and is not stable. On this data, what looks like forecasting is "Friday, about 10 a.m.": that is when the Botnet campaign began, and it begins on six hosts within 33 minutes.

The data cannot support a forecasting claim either way. It holds 18 attack episodes in a week. Only 8 of them begin on a host that had normal traffic beforehand, and 6 of those 8 are one Botnet campaign. Each held-out host contributes one start, so a single host's result is one event. The pseudo-onset null shows that random times on the same host reach a 95th-percentile AUC of about 0.9.

## 1. Why the current onset numbers fail

| Finding | Number | Source |
|---|---|---|
| Positive sequence targets that are continuations (attack already in the inputs) | 926 of 1,012 (92%) | `sequences.parquet` |
| Sequences flagged `target_is_onset` | 86 | same |
| ...of which are the first window of an attack episode | 17 | episodes rule: 30-minute gap |
| Lead time of "next observed window", median / 90th / 99th percentile | 1 min / 128 min / 1,512 min | `target_window_start - seq_end_time` |
| Attack episodes in the whole dataset | 18 | `host_windows.parquet` |
| Episodes on a host with traffic in the 10 minutes before the start | 9 (8 usable in folds) | same |

So the existing "onset" label mostly means "the next Botnet beacon after a quiet gap", and the existing horizon ranges from 30 seconds to a day. Neither is a forecasting target. The rest of this study replaces both (design section 3): anchors are attack-free host windows, and positives are attack starts within a fixed 1, 5, 15 or 30 minutes.

Not forecastable from a host's own traffic, by construction:

- **172.16.0.1** sends traffic only while attacking. 8 of its 9 episode starts have no window from that host in the 10 minutes before. It stays in training only.
- **205.174.165.73** has no windows before its Botnet start.

## 2. Results (leave one host out, 7 folds, 8 held-out episode starts)

Median within-host ROC-AUC across the 7 held-out hosts. "Clock" is pre-onset anchors against the same host's anchors at the same clock time (±60 min) on other days. "p<.05" counts the folds where the real start beats the host's own pseudo-onset null. "Alerted" counts the starts that got an alert inside the horizon at the validation 1% FPR threshold, and "test FPR" is the worst held-out host's false-positive rate at that threshold.

| Variant | H (min) | AUC | Clock | p<.05 folds | Alerted | Worst test FPR | Shifted-label AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Time-only control** (gradient boosting, hour + weekday) | 1 | **0.845** | 1.000 | 2/7 | 0/8 | 0.051 | n/a |
| | 5 | **0.930** | 0.991 | 4/7 | 2/8 | 0.026 | n/a |
| | 15 | **0.972** | 1.000 | 4/7 | 4/8 | 0.026 | n/a |
| | 30 | **0.960** | 1.000 | 4/7 | 5/8 | 0.023 | n/a |
| Logistic, own | 1 / 5 / 15 / 30 | 0.574 / 0.758 / 0.690 / 0.732 | 0.60 / 0.75 / 0.65 / 0.71 | ≤2/7 | 0 to 3/8 | 0.036 to 0.058 | 0.28 to 0.57 |
| Logistic, own+net | 1 / 5 / 15 / 30 | 0.625 / 0.693 / 0.531 / 0.718 | 0.58 / 0.71 / 0.54 / 0.71 | ≤1/7 | 0 to 3/8 | 0.038 to 0.051 | 0.39 to 0.61 |
| Gradient boosting, own | 1 / 5 / 15 / 30 | 0.793 / 0.851 / 0.721 / 0.731 | 0.73 / 0.87 / 0.75 / 0.74 | ≤3/7 | 1 to 5/8 | 0.028 to 0.139 | 0.34 to 0.60 |
| Gradient boosting, own+net | 1 / 5 / 15 / 30 | 0.793 / 0.815 / 0.678 / 0.701 | 0.73 / 0.85 / 0.68 / 0.70 | ≤3/7 | 1 to 5/8 | 0.041 to 0.091 | 0.46 to 0.68 |
| LSTM, own+net | 5 / 15 | 0.660 / 0.717 | 0.64 / 0.77 | 2/7, 2/7 | 2/8, 4/8 | 0.226, 0.281 | 0.50 / 0.67 |
| Transformer, own+net | 5 / 15 | **0.932** / 0.468 | 0.93 / 0.48 | 4/7, 2/7 | 2/8, 1/8 | 0.075, 0.057 | 0.50 / 0.67 |
| Multi-task LSTM (attack + next-3-window state), own+net | 5 / 15 | 0.758 / 0.726 | 0.82 / 0.70 | 3/7, 1/7 | 2/8, 1/8 | 0.175, 0.040 | 0.50 / 0.67 |
| Temporal GNN, own + neighbours | 5 / 15 | 0.664 / 0.647 | 0.64 / 0.69 | 3/7, 2/7 | 2/8, 5/8 | 0.070, 0.156 | 0.50 / 0.67 |

The shifted-label column for the neural rows is one shifted LSTM per horizon, shared by all four neural models.

### Bars

| Bar | Variants passing (of 24) |
|---|---:|
| P1 median AUC ≥ 0.70 | 14 |
| P2 pseudo-onset p < 0.05 on ≥ 5 of 7 folds | **0** (best: 4) |
| P3 clock-matched AUC ≥ 0.65 | 18 |
| P4 beats the time-only control by ≥ 0.10 and the shifted-label control by ≥ 0.15 | **0** |
| P5 half the starts alerted at test FPR ≤ 2% | **0** |
| All five | **0** |

No variant passed on the first seed, so the second-seed confirmation step did not run.

## 3. What the numbers mean

- **The time-only control beats almost everything.** It has never seen traffic. It learned from the other hosts that attacks start on Friday late morning, and that transfers to every held-out Botnet host. Gradient boosting on traffic reaches 0.85 at 5 minutes, but the clock alone reaches 0.93. The Transformer's 0.93 at 5 minutes falls to 0.47 at 15 minutes with the same setup, so it is fold noise and not a stable signal. Per host, it scores 0.94 to 0.99 on .5, .14, .15 and .50, and about 0.5 on .8, .9 and .17.
- **My clock-matched control is weaker than intended.** It compares Friday 10 a.m. with 10 a.m. on other days. A model that recognises "it is Friday" from that day's traffic level passes it. P3 therefore does not rule out the calendar shortcut, which is why P4 compares against the time-only model directly.
- **One event per host cannot be significant.** With about 10 positive anchors from one start, random pseudo-onsets on the same host reach AUC above 0.87 to 0.97 at their 95th percentile. Even a 0.97 is only p ≈ 0.005 to 0.02 on that host. P2 needed 5 of 7 hosts, and no variant got there.
- **The shifted-label controls behave.** Retraining with every training onset moved to a random time gives a median AUC of 0.28 to 0.68 (typically about 0.5). The real models do sit above their own shifted controls in most rows, so the traffic models learned something. That something is matched or beaten by the clock.
- **Operationally, there is no useful early warning.** At a threshold set for 1% false positives on the validation host, at most 5 of 8 starts get any alert in the 30 minutes before. In that configuration the worst held-out host has 9 to 14% false positives. At ≤ 2% FPR, it is 0 to 2 starts.

## 4. What this means for AegisFlow

- The demo model, its threshold and default behaviour are unchanged. No variant beat the controls, so nothing new was added, not even as opt-in.
- The demo model scores whether the next window of a host contains attack traffic. Since 92% of those positives are continuations, that is closer to **early detection of ongoing activity** than to forecasting a new attack. The dashboard can keep its "forecasting" title, but any written claim should say "next-window risk", not "predicts attacks before they start".
- The anchor target and controls in `scripts/onset_forecast_study.py` are reusable: pointed at a dataset with more episodes, they answer the same question without changes to the method.

## 5. What data would be needed

Forecasting needs many **independent** attack starts on hosts with normal traffic before them, at different times of day and in different campaigns, so that a clock or a single campaign cannot explain the result. As a rough bar: at least 30 episode starts across at least 3 campaigns, with no single start time shared by most of them. Candidates, not yet checked for size or licence:

- **LANL "Comprehensive, Multi-Source Cyber-Security Events" (2015):** 58 days of real enterprise authentication and flow data with labelled red-team events. It has many lateral-movement starts on hosts with long benign histories.
- **DARPA OpTC:** a multi-day APT exercise on about 1,000 hosts. Its host telemetry has a benign lead-up for each step.
- **CTU-13:** 13 separate botnet captures, giving 13 independent campaigns for Botnet onset.
- **CSE-CIC-IDS2018:** the route already deferred in the host-disjoint design. It is the same lab and the attacks are scheduled in the same way, so it would add volume but probably not independence.

## 6. Not done

Single seed for every variant (nothing passed, so the confirmation seed was not triggered). No hyperparameter search. The neural models ran only at 5 and 15 minutes, as stated in advance. The clock-matched control compares other days, not the same day (section 3).
