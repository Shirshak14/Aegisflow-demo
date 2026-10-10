# Synthetic-lab onset forecasting: results (2026-10-09)

Implements `docs/synthetic_onset_design.md` (bars and conditions committed before any run). Reproduce:

    python scripts/synth_onset_study.py CONDITION --seed 11      # about 30 min per condition on one core
    python scripts/synth_checks.py                                # sanity and identity checks, condition B3

Raw output: `reports/synthetic_onset_<condition>_s<seed>.json`, `reports/synthetic_checks_B3_s11.json`. The demo model, its threshold and `backend/` are unchanged.

**All data here is synthetic.** It comes from my own simulator (60 hosts, 4 days, 28 host-window features, about 400 independent scripted attack runs), not from a network. The results say what this evaluation can detect when the truth is known. They are not evidence about real attackers.

## Answer

**With 400 independent onsets and no host-local precursor, nothing forecasts, as it should (condition A, chance). A precursor that is a 1 benign-standard-deviation shift ramping over the 10 minutes before the attack is not detectable (best AUC 0.64, fails every bar). The bars are only met when the precursor is about 4 standard deviations, and then only at 5 to 15 minute horizons (condition B5, repeated on a second seed).**

So the earlier real-data failures were not only a shortage of onsets. Even with plenty of onsets, a plain feature-based forecaster needs a strong host-local warning sign, and I have no evidence that real attacks leave one. The honest AegisFlow claim stays "early warning of attacks in progress", not "predicts attacks before they start".

## 1. Results (median held-out AUC of the best variant per horizon; 10 host-disjoint folds)

| Condition | Seed | Precursor | Best AUC H1 | H5 | H15 | H30 | Time-only AUC (H5) | Variants passing S1-S5 |
|---|---|---|---:|---:|---:|---:|---:|---|
| A | 11 | none | 0.502 | 0.508 | 0.511 | 0.524 | 0.483 | none |
| B1 | 11 | 0.25 sd | 0.523 | 0.518 | 0.515 | 0.519 | 0.483 | none |
| B2 | 11 | 0.5 sd | 0.557 | 0.541 | 0.523 | 0.525 | 0.483 | none |
| B3 | 11 | 1.0 sd | 0.638 | 0.597 | 0.545 | 0.537 | 0.483 | none |
| B3 | 12 | 1.0 sd | 0.628 | 0.601 | 0.544 | 0.518 | 0.503 | none |
| B4 | 11 | 2.0 sd (post-hoc) | 0.753 | 0.717 | 0.597 | 0.562 | 0.483 | none |
| B5 | 11 | 4.0 sd (post-hoc) | 0.929 | 0.919 | 0.706 | 0.608 | 0.483 | own|gboost|H5, own+net|gboost|H5, own|gboost|H15, own+net|gboost|H15 |
| B5 | 12 | 4.0 sd (post-hoc) | 0.928 | 0.900 | 0.703 | 0.604 | 0.503 | own|logistic|H5, own+net|gboost|H5, own+net|logistic|H15 |
| C | 11 | none, clock-concentrated | 0.957 | 0.957 | 0.953 | 0.944 | 0.979 | none |
| D | 11 | network-level only | 0.879 | 0.816 | 0.609 | 0.539 | 0.483 | none |

Notes on the table:

- A (no precursor) sits at 0.50 to 0.52 for every model: the guard against a generator or pipeline leak holds.
- B1 to B3 are the pre-registered dose-response conditions. **My pre-stated expectation was that B3 would pass for horizons up to 15 minutes. It did not** (best 0.64 at H1, S1 fails). Design section 9 says that outcome would mean the pipeline is too weak for 1 sd precursors; that is what it shows.
- B4 and B5 were **added after seeing B3 fail**, to find where the pipeline starts to work. They are exploratory and labelled post-hoc. B4 (2 sd) reaches S1 to S4 at H1 and H5 but fails S5 (at 1% false positives only 10 to 21% of episodes alert). B5 (4 sd) passes S1 to S5 at H5 and H15 for some variants; at seed 12 the passing set differs (own+net gradient boosting at H5 passes on both seeds, so that is the only variant that clears S6; other variants pass on one seed only, near the 0.70 line at H15).
- C (onsets in a 07:00 to 09:00 window, no precursor, 240 runs because the window limits non-overlapping runs): the time-only control scores 0.98. Models with network features score 0.92 to 0.96, which is the clock leaking through the summed network totals, not forecasting; they lose S4 to the time control. Own-host features stay at chance. This reproduces the real-data lesson that a clock beats traffic models.
- D (no host precursor, but 5 minutes before each onset a third of other hosts shift): network features reach 0.82 to 0.88 at H1 to H5, own-host features stay at chance. It fails S2 and S5 (the signal is not specific to the host that is about to be attacked), so it is not a forecast for that host.
- Shifted-label retraining sits at 0.46 to 0.60 everywhere and does not follow the real AUC.

## 2. Pipeline checks (condition B3, seed 11)

| Check | Result | Bar | Verdict |
|---|---|---|---|
| Detector sanity: same features, target = attack in progress | median fold AUC 0.934 | >= 0.95 | **just below the bar**; the features can see the attack, but not perfectly |
| Host identity: predict host from features, split by time | 99.9% accuracy (chance 1.7%) | AUC about 0.5 from host recognition | Hosts are trivially identifiable, as in real data. Every host has the same onset rate, so this gives no forecasting skill; A at chance confirms it |
| Seed transfer | B3 on seed 12 matches seed 11 within 0.02 AUC; B5 seed 12 within 0.02 at H1 to H15 | within 0.05 | holds |
| Grouped-by-run re-split | **not run** | | Host-disjoint folds already keep every run in one fold, which is stricter. Recorded as a deviation |

## 3. What it does and does not say

- It does say: the evaluation pipeline has no leak (A at chance), copes with the clock trap (C), and with 400 onsets it can detect a 4 sd host-local precursor 5 to 15 minutes ahead, but not a 1 sd one. A pass in real data would have needed a precursor on that scale.
- It does not say whether real attacks have a precursor, how large, or what shape. I wrote both the attack and the precursor.
- The 4 sd threshold depends on my choices (10 minute ramp, decoy bursts of 1.5 to 3.5 sd that look like attacks, AR(1) noise, summary features). A different simulator would move it. Treat it as order of magnitude.
- A stronger forecaster (sequence model, different features) might lower the threshold. I did not try neural models, as the design said only to add one after a classical pass; B5 passed, so one LSTM check would be the next step if wanted.

## 4. Deviations (all recorded; items 1 to 4 were committed before the runs)

1. Gradient boosting trains on 10% of negatives, 60 trees, depth 4.
2. Network features are four aggregates of the host windows themselves.
3. Host-identity and detector-sanity ran on B3 only; the grouped-by-run re-split was not run.
4. S6 (second seed) was run for B3 and B5 only, not every condition, because no pre-registered condition passed S1 to S5 so there was nothing else to confirm.
5. After B3 failed, conditions B4 and B5 were added (post-hoc, exploratory). The seed 12 run of B5 is therefore a confirmation of an exploratory finding, not of a pre-registered one.
6. The first launch of condition B2 produced no output (a process was killed by mistake); it was rerun unchanged.
