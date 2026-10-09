# LANL 2015 short-lead login study: results (2026-10-09)

Implements `docs/lanl_short_lead_design.md`. The design and bars F1 to F5 and D1 to D3 were committed before any model was trained, and neither they nor the script were changed for this run. Reproduce with:

    python scripts/lanl_short_lead_study.py bins    # one pass over flows.txt.gz, about 33 min
    python scripts/lanl_short_lead_study.py study   # about 9.5 h on CPU sequentially

Raw output: `reports/lanl_short_lead_study.json`. The demo model, its threshold, the default dataset and `backend/` are unchanged.

## Answer

**Forecasting: no.** None of the 16 forecast candidates passes a single bar. Trained and scored only on anchors where the attacker has not yet touched the victim, every model sits at 0.47 to 0.50 median fold AUC at every horizon (2, 5, 10, 15 min), while the time-only control on the same rows reaches 0.73 to 0.76. Pseudo-onset p < 0.05 on at most 8 of 31 folds (the bar needs 16). Without day 9 nothing changes. This is what §7 of the design predicted.

**Detection: no, under the pre-stated bars.** None of the 8 detection candidates passes D1 (best median fold AUC 0.567 against a bar of 0.85) or D2 (every one is 0.16 to 0.27 **below** time-only). One candidate, gradient boosting at H = 15, passes D3 on its own: all 35 contacted victims alerted within 15 min of the logon at 1.8% median test FPR, median alert lead 19.5 min. Because it fails D1 and D2, the design's wording rule (§6) allows **no detection claim** from this study.

**What the data does show, descriptively:** for the 35 of 188 victims the red-team machine contacts in the router flows, its first flow lands **at least 19 minutes** before the logon (median 19.4 min). An alarm on "a red-team source talked to this host" therefore fires about 19 minutes ahead of the compromise for those 35 hosts, and nothing like it exists for the other 153.

## 1. What was run

| Item | Value |
|---|---|
| Victims | 188 (same as the 5-minute study), 35 with visible attacker contact |
| Anchors | 1,895,040 (1-minute bins, 7 days before each onset, input = last 5 bins); 1,871,185 pre-contact |
| Folds | 34 onset bursts, leave one burst out (31 have both classes for the forecast rows) |
| Forecast candidates | logistic, gboost × own / own+inbound × H = 2, 5, 10, 15 (16), pre-contact anchors only |
| Detection candidates | logistic, gboost × own+inbound+attacker × H = 2, 5, 10, 15 (8), all anchors |
| Controls | time-only gboost on the same rows; shifted-label retrain; 200 pseudo-onsets per test victim |
| Seed | 42. Nothing passed all bars, so the seed-7 confirmation was not triggered. |

Run note: to fit the run into one evening, the study was executed as two processes (H = 2, 5 and H = 10, 15), each calling the committed `study()` unchanged with its own output file, then merged. Each process starts its own seed-42 random stream, so control draws for H = 5, 10, 15 differ from what a single sequential run would draw. H = 2 numbers match a partial sequential run exactly. Bins took about 33 min, not the 11 min of the 5-minute study, because the 1-minute pass keeps every bin until the last onset.

## 2. Lead of contact (onset minus first red-team flow to the victim)

| | Value |
|---|---|
| Contacted victims | 35 of 188 |
| Minimum | 18.9 min |
| 10th / 25th / 50th / 75th percentile | 19.0 / 19.1 / 19.4 / 30.3 min |
| 90th percentile / maximum | 16.7 h / 165.8 h |
| Leads ≥ 2, 5, 10, 15 min | 35, 35, 35, 35 |
| Leads ≥ 30 min | 9 |

The design (§2) feared the contact would mostly arrive under 2 minutes ahead. It is the opposite: the attacker's first packet to a contacted victim always arrives about 19 minutes or more before the logon, so at every horizon tested the contact is already visible. Consequence: **all** pre-contact positives come from the 153 victims with no visible contact. The forecast rows are, in effect, "predict a logon on a host the attacker has not touched in the flows".

## 3. Forecast results (pre-contact anchors)

Median fold AUC. Time-only control on the same rows: 0.762 / 0.754 / 0.751 / 0.728 at H = 2 / 5 / 10 / 15. Pooled positives: 306 / 765 / 1,530 / 2,295, so every candidate is testable.

| Features | Model | H2 | H5 | H10 | H15 | Best no-day-9 | Pseudo p<.05 folds (best) |
|---|---|---:|---:|---:|---:|---:|---:|
| own | logistic | 0.500 | 0.501 | 0.500 | 0.499 | 0.503 | 5/31 |
| own | gboost | 0.500 | 0.480 | 0.494 | 0.500 | 0.492 | 4/31 |
| own+inbound | logistic | 0.499 | 0.500 | 0.495 | 0.498 | 0.500 | 8/31 |
| own+inbound | gboost | 0.500 | 0.497 | 0.473 | 0.500 | 0.500 | 5/31 |

Shifted-label controls: 0.43 to 0.52.

| Bar | Passing (of 16) |
|---|---:|
| F1 median AUC ≥ 0.70 | 0 (best 0.501) |
| F2 ≥ 0.10 above time-only | 0 (every candidate 0.23 to 0.28 **below**) |
| F3 pseudo-onset p < 0.05 on ≥ half the folds | 0 (best 8 of 31) |
| F4 ≥ 0.15 above shifted-label | 0 (best +0.07) |
| F5 without day 9 | 0 (best 0.503 vs time-only 0.75) |
| All five | **0** |

## 4. Detection results (all anchors, attacker features included)

Time-only control on all anchors: 0.749 / 0.730 / 0.751 / 0.751. "Alert lead" = onset minus earliest alert in the 60 min before onset, median over victims with an alert.

| Model | H | Median AUC | Contacted alerted | Uncontacted alerted | Median test FPR | Median alert lead | D1 | D2 | D3 |
|---|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|
| logistic | 2 | 0.507 | 7/35 | 13/153 | 1.4% | 17.9 min | – | – | – |
| gboost | 2 | 0.483 | 4/35 | 10/153 | 1.3% | 45.5 min | – | – | – |
| logistic | 5 | 0.567 | 13/35 | 18/153 | 1.8% | 19.1 min | – | – | – |
| gboost | 5 | 0.478 | 9/35 | 22/153 | 2.3% | 19.5 min | – | – | – |
| logistic | 10 | 0.564 | 17/35 | 19/153 | 2.1% | 19.1 min | – | – | – |
| gboost | 10 | 0.499 | 17/35 | 29/153 | 2.7% | 28.4 min | – | – | – |
| logistic | 15 | 0.526 | 34/35 | 22/153 | 2.1% | 18.9 min | – | – | – |
| gboost | 15 | 0.501 | **35/35** | 31/153 | **1.8%** | **19.5 min** | – | – | **pass** |

| Bar | Passing (of 8) |
|---|---:|
| D1 median AUC ≥ 0.85 | 0 (best 0.567) |
| D2 ≥ 0.10 above time-only and ≥ 0.15 above shifted-label | 0 |
| D3 half of contacted alerted, FPR ≤ 2%, lead ≥ 2 min | 1 (gboost H15) |
| All three | **0** |

### Why D3 passes while D1 fails

- **D1 is a median over folds, and most folds have no attacker signal.** Only 13 of 34 folds hold a contacted victim; the other 21 contain only victims the attacker never visibly touches, where the attacker features are zero and the model is at chance. Even in the 13 contacted folds the median AUC is only 0.50 to 0.59.
- **The attacker signal does not line up with the label.** The label is "logon within H", but the attacker's flows start about 19 minutes before the logon. Anchors 16 to 19 minutes out carry the same attacker evidence as the positives but are labelled negative, so the model either alarms on them (hurting AUC) or misses the positives. The median alert lead of about 19 min for every candidate is the time of first contact, not a learned "logon is imminent" signal.
- **D3 only asks whether any alert falls inside the horizon.** Once the attacker is in contact, the alarm stays up through the logon, so at H = 15 nearly every contacted victim gets one. 35 of 35 contacted against 31 of 153 uncontacted (20%, roughly what a 2% per-minute false-alarm rate over 15 one-minute anchors gives by chance) is a real difference, but it is a difference the pre-registered bars were not built to credit on their own.

## 5. Forecast vs detection split

| | Forecast (attacker not yet visible) | Detection (attacker visible) |
|---|---|---|
| Rows | 1,871,185 pre-contact anchors, positives from the 153 uncontacted victims only | all 1,895,040 anchors |
| Best median fold AUC | 0.501 | 0.567 |
| Time-only on the same rows | 0.73 to 0.76 | 0.73 to 0.75 |
| Bars passed | none | D3 only, one candidate |
| Verdict (§6) | **no forecasting claim** | **no detection claim** |

## 6. What the deck can honestly say

- **Do not say** AegisFlow forecasts or predicts a compromise before it starts. The LANL 5-minute onset study, this short-lead study, CIC-IDS2017 and the escalation rounds all fail against a clock-only baseline.
- **Do not say** the LANL short-lead model detects intrusions minutes ahead. It fails the pre-stated detection bars.
- **Can say, as an observation about the data, not a model result:** "In the LANL 2015 red-team data, for the 35 compromised hosts the attacker visibly contacts over the network, the first attacker flow arrives at least 19 minutes (median 19 min) before the compromising logon. Watching for traffic from known-bad sources therefore gives up to about 19 minutes of warning for those hosts. Most victims (153 of 188) show no such contact in the flows." That is an argument for the demo's existing framing, "early warning of attacks in progress", and nothing stronger.

## 7. Not done

Seed 7 (nothing passed all bars). No hyperparameter search, no sequence models. Detection with a label aligned to first contact ("attacker in contact now") instead of "logon within H" would likely score far higher, but it is a different question and would need its own pre-registration. Victims without router flows (113 of 301) are not covered.
