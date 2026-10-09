# LANL 2015 onset forecasting study: design and pre-stated bars

Status: written and committed **before** any model in this study was trained. Results go in `docs/lanl_onset_report.md`; the script is `scripts/lanl_onset_study.py`. Nothing here changes the demo model, its threshold, the default dataset or the backend.

Background: `docs/optc_lanl_feasibility.md` §2. LANL 2015 has 301 computers the red team compromised; 209 appear in router flows and 102 sent their own traffic in the 10 minutes before compromise.

## 1. Question

Can a model forecast that a computer is **about to be compromised** (its first red-team logon) from flow data available before that moment, better than a model that only knows the time? Scoring a window that already contains red-team traffic does not count.

## 2. Units, windows and target

- **Victims:** the 209 compromised computers that appear in `flows.txt.gz`. Onset = first red-team authentication to that computer (`redteam.txt.gz`). The other 92 have no flow data and are left out (reported, not hidden).
- **Windows:** fixed 5-minute bins of relative time (seconds since the dataset start). Bins with no flows are kept with zero counts: silence is information.
- **Anchors:** for each victim, every bin end t in the 7 days before its onset (t < onset). Inputs are the victim's 6 bins ending at t (30 minutes).
- **Positive:** onset in [t, t + H). Horizons H = 5, 15, 30, 60 minutes, fixed in advance. Anchors are never taken at or after onset, so no input contains post-compromise traffic.

## 3. Features (all computed from flows, never from labels except where stated)

Per victim and bin:

- **own** (victim as source): flow count, packets, bytes, distinct destinations, distinct destination ports, mean duration.
- **inbound** (victim as destination): flow count, distinct sources, bytes.
- **context**: inbound flows from computers **already known to be compromised at that moment** (the four red-team source computers, plus victims whose onset is earlier than the flow). This uses the label history, i.e. it assumes a defender learns of each compromise when it happens. That is optimistic, and it is the only route by which a victim's compromise is plausibly predictable, since the attacker chooses the victim.

Bins are aggregated while streaming the time-sorted file in chunks. Distinct counts in the one bin that straddles a chunk boundary are summed across the two halves (slight over-count, about one bin per 4 M rows).

Models get per-feature summaries over the 6 input bins (last, mean, max, slope), log1p-scaled.

## 4. Folds

Victims are grouped into **onset bursts**: onsets chained when less than 30 minutes apart (23 to 35 bursts depending on which victims have flows). Leave one burst out:

- **Test:** every anchor of the victims in the held-out burst.
- **Validation** (threshold only): the next burst in time order.
- **Train:** all other victims, excluding any training anchor within 6 hours of a test onset, so that contemporaneous network state cannot leak.

Results are reported per burst, and separately **without day 9**, which holds 170 of 301 onsets.

## 5. Variants and controls

Candidates, each at all four horizons: logistic regression and gradient boosting (sklearn `HistGradientBoostingClassifier`, defaults, `random_state`=seed), on feature sets **own**, **own+inbound**, **own+inbound+context**. 24 candidates.

Controls:

1. **Time-only:** gradient boosting on relative hour of day ((t mod 86,400) / 3,600) and day mod 7, the stand-in for hour and weekday. The real clock is hidden, but daily periodicity survives.
2. **Shifted-label retraining:** every training victim's onset moved to a random time in its own 7-day anchor range, model retrained, scored against the true test labels. Should sit near 0.5. One retrain per fold.
3. **Pseudo-onset null (no retraining):** for each test fold, 200 random pseudo-onset times per test victim inside its anchor range, labelled with the same horizon rule and scored with the same predictions. Fold p = (1 + count of pseudo AUC ≥ real AUC) / 201.

Metric: per fold, ROC-AUC over all test anchors of that burst (pooled across its victims); the headline is the median across folds. A fold needs at least one positive and one negative anchor.

## 6. Bars (all must hold for a variant to count as forecasting)

| # | Bar |
|---|---|
| L1 | Median fold AUC ≥ 0.70 |
| L2 | Pseudo-onset p < 0.05 on at least half of the folds |
| L3 | Median fold AUC beats the time-only control by ≥ 0.10 and the shifted-label control by ≥ 0.15 |
| L4 | Without day-9 folds: median AUC ≥ 0.70 and still ≥ 0.10 above the time-only control |
| L5 | At the threshold giving 1% FPR on the validation burst, at least half of the test onsets get an alert within H before onset, with median test-fold FPR ≤ 2% |

24 candidates are tried, so a variant that passes must pass L1 to L5 again on a second seed before anything is claimed. A pass would be offered as a separate opt-in model; the demo stays as it is either way.

## 7. Expectations stated in advance

- **own** and **own+inbound** should sit near chance: a victim's own traffic has no reason to change before an attacker picks it. If they pass, suspect a time or burst artefact first.
- **context** can only help the victims where earlier-compromised computers or the red-team sources talk to the victim beforehand; the feasibility census found attacker contact before onset for only 35 victims, all within the last hour. So any pass is likely at H = 30 to 60 minutes and driven by a minority of victims. The report will break results down by whether attacker contact was visible.
- Whatever the outcome, it is one red team in one network.
