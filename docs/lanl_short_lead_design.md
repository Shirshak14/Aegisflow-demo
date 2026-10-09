# LANL short-lead login study: design and pre-stated bars

Status: written and committed **before** any model in this study is trained. Script: `scripts/lanl_short_lead_study.py`. Results go in `docs/lanl_short_lead_report.md`. Nothing here changes the demo model, its threshold, the default dataset or the backend.

Background: `docs/lanl_onset_report.md`. The 5-minute study found that for the 35 victims the red-team machine contacts before logging in, inbound-traffic models separate the last 5 to 15 minutes almost perfectly. That is **detection** (the attacker's own packets are already in the input). This study asks how much of it, if any, is **forecasting**: warning before the attacker's first visible connection to the victim.

## 1. Questions

- **Q-forecast:** can a model predict a victim's first red-team logon from traffic seen *before the attacker has sent that victim any flow*, at horizons of 2, 5, 10 and 15 minutes, better than a model that only knows the time?
- **Q-detect:** once the attacker's connection is visible, how early before the logon does a model alert, at a low false-alarm rate?

## 2. Definitions (fixed here)

- Same victims, onsets (first red-team authentication) and red-team source computers as the 5-minute study.
- **Bins:** 1 minute. **Anchor:** every bin end t in the 7 days before onset (t <= onset). **Input:** the 5 bins ending at t.
- **Positive at horizon H:** onset in [t, t + H). H = 2, 5, 10, 15 min.
- **Attacker contact:** a flow whose source is a red-team source computer and whose destination is the victim, before onset. `c` = time of the victim's first such flow. The contact is **visible at anchor t** iff the end of the bin containing c is <= t (the same rule that puts the flow into the input features). Victims with no contact have c = infinity.
- **Pre-contact anchor:** contact not visible at t. This is a causal property (a defender at time t knows it). Everything on a pre-contact anchor is free of red-team-source flows to that victim, by construction.
- **Lead of contact** = onset - c, reported as a distribution. It bounds what forecasting can mean: a positive anchor is pre-contact only when the horizon reaches back before c.
- **Honesty flag:** pre-contact positives mostly come from the 153 victims with no visible contact at all, where the "forecast" can only come from the victim's own and general inbound traffic.

## 3. Features

Per victim and bin: own (flows, packets, bytes, distinct dsts, distinct dst ports, mean duration), inbound (flows, distinct srcs, bytes), context (inbound flows from computers already compromised at that moment), **attacker** (flows and bytes from red-team source computers to the victim). Summaries per feature over the 5 bins: last, mean, max, slope, log1p-scaled.

Feature sets: **own**, **own+inbound** (forecast models; attacker and context columns are excluded), **own+inbound+attacker** (detection model).

## 4. Folds, models, controls

Folds as in the 5-minute study: onset bursts (< 30 min apart), leave one burst out, validation = next burst, training anchors within 6 hours of a test onset dropped. Report with and without day 9. Logistic regression and `HistGradientBoostingClassifier` (max_iter 100, seed 42). Training negatives subsampled to 5% (positives all kept; recorded here before any run).

- **Forecast candidates:** {own, own+inbound} x {logistic, gboost} x 4 horizons = 16. Trained and evaluated on **pre-contact anchors only**.
- **Detection candidates:** own+inbound+attacker x {logistic, gboost} x 4 horizons = 8. Trained on all anchors, evaluated on all anchors.
- **Time-only control** (relative hour of day, day mod 7; gboost), trained and evaluated on the same anchor sets as the model it is compared to.
- **Shifted-label control:** training onsets moved to a random time in each victim's range, retrained, scored on true test labels.
- **Pseudo-onset null:** 200 random pseudo-onsets per test victim, same evaluation rows, fold p = (1 + count(pseudo AUC >= real)) / 201.

## 5. Bars (pre-registered)

**Forecasting claim** (a candidate must pass all of F1 to F5; the evaluation rows are pre-contact anchors):

| # | Bar |
|---|---|
| F1 | Median fold AUC >= 0.70 (folds with both classes only; at least 10 such folds and at least 30 pooled positives, else the candidate is "untestable") |
| F2 | Median fold AUC beats the time-only control by >= 0.10 |
| F3 | Pseudo-onset p < 0.05 on at least half of the valid folds |
| F4 | Beats the shifted-label control by >= 0.15 |
| F5 | Without day 9: AUC >= 0.70 and >= 0.10 above time-only |

**Detection claim** (candidate must pass D1 to D3):

| # | Bar |
|---|---|
| D1 | Median fold AUC over all anchors >= 0.85 |
| D2 | Beats time-only by >= 0.10 and shifted-label by >= 0.15 |
| D3 | At the threshold giving 1% FPR on the validation burst: at least half of the **contacted** victims' onsets are alerted within H before onset, median test-fold FPR <= 2%, and the median alert lead (onset minus the earliest alert in the 60 minutes before onset) is >= 2 minutes |

Passing candidates must repeat on seed 7 before anything is claimed.

## 6. How the result is worded

- F1 to F5 pass for some candidate: "short-horizon forecast of the login before the attacker's first connection". Otherwise **no forecasting claim**.
- D1 to D3 pass but not F: "early warning: alert minutes before the logon, once the attacker's connection is visible" and nothing stronger.
- Neither: no claim beyond what the 5-minute study said.
- A detection pass is also limited to the contacted victims (about 35 of 188) and one red team in one network.

## 7. Expectations stated in advance

Forecast candidates should sit at chance: the attacker picks the victim, and with contact not yet visible the victim's own traffic has no reason to change. Detection should pass D1 if the 5-minute result holds at 1-minute resolution; D3 depends on how many minutes elapse between the attacker's first flow and the logon. If the lead of contact is mostly under 2 minutes, D3 fails even though D1 passes.
