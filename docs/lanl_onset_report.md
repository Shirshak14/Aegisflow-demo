# LANL 2015 onset forecasting study: results (2026-10-09)

Implements `docs/lanl_onset_design.md`. The design, the bars L1 to L5 and the one recorded deviation (negatives subsampled to 25%) were committed before any model was trained. Reproduce with:

    python scripts/lanl_onset_study.py bins    # one pass over flows.txt.gz, about 11 min
    python scripts/lanl_onset_study.py study   # about 2.3 h on CPU

Raw output: `reports/lanl_onset_study.json`. The demo model, its threshold, the default dataset and `backend/` are unchanged.

## Answer

**No.** None of the 24 candidates passes; none passes even one bar. Every flow-based model sits between 0.46 and 0.58 median AUC, which is chance, while a model that knows only the (relative) time of day and day reaches 0.73 to 0.77. Taking out day 9 changes nothing.

There is one real signal, and it is not forecasting. For the 35 victims that the red-team machine contacts in the flows shortly before logging in, inbound-traffic models separate the last 5 to 15 minutes before compromise almost perfectly (per-victim AUC 0.98 to 1.0). Those are the attacker's own first packets to the victim. That is early **detection** of an intrusion in progress, minutes ahead of the logon, for 35 of 188 victims, and it does not survive into the fold-level numbers.

This matches what the design predicted (§7): the attacker chooses the victim, so the victim's own traffic carries no warning.

## 1. What was run

| Item | Value |
|---|---|
| Victims | 188 (of 301 compromised computers; the rest have no router flows in the 7 days before onset) |
| Anchors | 379,008 (5-minute bins, 7 days before each onset, inputs = last 30 min) |
| Folds | 34 onset bursts (onsets < 30 min apart), leave one burst out; folds fall on days 2, 3, 6, 7, 8, 9, 13, 14, 15, 16 |
| Candidates | logistic regression and gradient boosting × own / own+inbound / own+inbound+context × H = 5, 15, 30, 60 min |
| Controls | time-only gradient boosting (relative hour, day mod 7); shifted-label retrain per fold; 200 pseudo-onsets per test victim |

## 2. Results

Median fold AUC. "No day 9" drops the 3 folds from day 9. Time-only control: 0.768 / 0.753 / 0.754 / 0.735 at H = 5 / 15 / 30 / 60 (no day 9: 0.783 / 0.754 / 0.756 / 0.735).

| Features | Model | H5 | H15 | H30 | H60 | Best no-day-9 | Pseudo p<.05 folds (best) |
|---|---|---:|---:|---:|---:|---:|---:|
| own | logistic | 0.516 | 0.576 | 0.575 | 0.517 | 0.586 | 12/34 |
| own | gboost | 0.474 | 0.456 | 0.489 | 0.493 | 0.496 | 4/34 |
| own+inbound | logistic | 0.507 | 0.535 | 0.570 | 0.557 | 0.631 | 12/34 |
| own+inbound | gboost | 0.501 | 0.517 | 0.520 | 0.510 | 0.505 | 9/34 |
| own+inbound+context | logistic | 0.527 | 0.536 | 0.545 | 0.570 | 0.598 | 13/34 |
| own+inbound+context | gboost | 0.501 | 0.501 | 0.566 | 0.532 | 0.568 | 13/34 |

Shifted-label controls sit at 0.37 to 0.53, as they should.

### Bars

| Bar | Candidates passing (of 24) |
|---|---:|
| L1 median AUC ≥ 0.70 | 0 (best 0.576) |
| L2 pseudo-onset p < 0.05 on ≥ half the folds | 0 (best 13 of 34) |
| L3 beats time-only by ≥ 0.10 and shifted-label by ≥ 0.15 | 0 (every candidate is 0.17 to 0.31 **below** time-only) |
| L4 without day 9: AUC ≥ 0.70 and ≥ 0.10 above time-only | 0 (best 0.631) |
| L5 half of onsets alerted at ≤ 2% test FPR | 0 (best 55 of 188 alerted, at 2.1% FPR) |
| All five | **0** |

No candidate passed on the first seed, so the second-seed confirmation did not run.

### Victims with visible attacker contact

Median per-victim AUC, 35 contacted victims vs 153 others:

| Features | H5 | H15 | H30 | H60 |
|---|---|---|---|---|
| own+inbound, logistic | 0.999 vs 0.495 | 1.000 vs 0.497 | 0.777 vs 0.497 | 0.667 vs 0.495 |
| own+inbound+context, logistic | 1.000 vs 0.495 | 1.000 vs 0.497 | 0.829 vs 0.497 | 0.767 vs 0.497 |
| own only, logistic | 0.693 vs 0.495 | 0.619 vs 0.498 | 0.619 vs 0.497 | 0.582 vs 0.496 |

The jump comes from inbound flows from the red-team machine in the minutes before its logon, which the feasibility census found for these 35 victims only, all within the final hour. With 1 to 12 positive anchors per victim, single-victim AUCs are noisy (one gradient-boosting run scores 0.004 for the same group, i.e. the direction flipped), so treat this as a pattern, not a measured detector.

## 3. What the numbers mean

- **The clock beats every traffic model by a wide margin.** The red team works in bursts at particular hours, so "it is this time of day" is a better predictor than anything in the flows. This is the same failure as the CIC study, now with 188 victims and 34 independent bursts instead of 8 onsets, so it cannot be blamed on sample size.
- **Victims' own traffic carries no warning.** own-only models sit at 0.46 to 0.58. A computer does not behave differently before an attacker picks it.
- **Network context helps only where the attacker is already touching the victim.** The "known compromised" context feature is near zero for almost every victim before onset: earlier victims and the red-team machines rarely talk to the next victim except during the intrusion itself.
- **Operational value is nil as forecasting.** At a 1% validation FPR threshold, 10 to 55 of 188 onsets get an alert in the horizon before onset, with test FPR 1.3 to 2.3%.

## 4. Across all four datasets

| Dataset | Usable onsets | Result |
|---|---:|---|
| CIC-IDS2017 | 8 | No variant beats the clock (`onset_forecasting_report.md`) |
| CTU-13 | 0 | Cannot be tested (`combined_datasets_feasibility.md`) |
| DARPA OpTC | 29 onsets, 11 independent | Has history, too few independent onsets (`optc_lanl_feasibility.md`) |
| LANL 2015 | 188 victims, 34 bursts | **Tested: no variant beats the clock** (this report) |

The honest summary for AegisFlow: **forecasting that a host is about to be attacked, from flow data, is not supported by any of the four public datasets we can run.** What does work is early detection: scoring windows that already contain attack traffic (the demo's next-window risk) and, on LANL, flagging the attacker's first contact a few minutes before the compromise is complete. Claims should say "early warning of attacks in progress", not "predicts attacks before they start".

## 5. What would still be worth trying

- **Attacker-side forecasting.** Predict the *next victim* of an attacker already known to be inside (which computer C17693 logs into next), instead of whether a given computer is about to be hit. That uses the lateral-movement graph the red team actually follows and is a different, pre-registrable question.
- **Authentication features.** auth.txt.gz (7.2 GB, not downloaded) has the logons the red team uses. Failed or unusual logons to a victim before the red-team logon might be a precursor that flows cannot see.
- **OpTC as a replication** of the "first contact" detection result above, on its 11 onset moments.

## 6. Not done

Single seed (nothing passed, so the confirmation run was not triggered). No hyperparameter search. No neural models: with every classical model at chance and the time control at 0.75, a sequence model would have to find a signal the summaries miss entirely, and the design did not include them. Victims with no router flows in their 7-day window (113 of 301) are not covered.
