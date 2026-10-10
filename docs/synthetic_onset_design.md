# Synthetic-lab onset forecasting: design and pre-stated bars

Status: written and committed before the generator was run and before any model was trained. Results go in `docs/synthetic_onset_report.md`. Code: `scripts/synth_lab.py` (generator), `scripts/synth_onset_study.py` (evaluation). Nothing here touches the demo model, its threshold or `backend/`.

**The data is synthetic.** It comes from a simulator I wrote, not from a network. Every result below is about what the evaluation pipeline can and cannot detect when the ground truth is known. It is not evidence about real attackers.

## 1. Why this study

CIC-IDS2017, CTU-13, OpTC and LANL all failed the onset bars (docs/onset_forecasting_report.md, escalation_forecasting_report.md, host_disjoint_evaluation_report.md, and docs/lanl_onset_report.md on section-c). Two explanations are mixed together in those results:

1. **Too few independent onsets** (18 episodes in CIC, 8 usable), so even a real signal could not pass.
2. **No information about the onset in the host's own traffic**, so nothing could pass.

A lab with hundreds of independent attack starts removes explanation 1. The question then becomes whether the information exists, and the simulator lets us set that by hand: a condition with no host-local precursor (must fail) and conditions with a faint precursor of known strength (should pass once it is strong enough). This tells us how large a precursor the pipeline can detect, and gives a yardstick for "how faint would a real precursor have to be to be missed by our real-data studies".

## 2. Why a flow-level simulator and not a live attack lab

A live lab (Caldera, Atomic Red Team, Docker hosts running scans or brute force) needs offensive tooling, a capture setup and a long time to get hundreds of runs. The questions here are about onset timing and statistical power, not packet fidelity, so the simulator generates the same 28 host-window features the demo model uses (`aegisflow.ml.modeling.MODEL_FEATURES`, 60 s windows, 30 s stride) directly. No attack traffic is sent anywhere. The cost is that realism is limited to what I put in; that is stated in the report.

## 3. Generator (`scripts/synth_lab.py`)

- **Hosts:** 60 hosts, 4 days (96 h), windows every 30 s with 60 s span. Each host gets a role profile (workstation, server, db) with its own baseline levels for the 28 features, a diurnal curve with a random phase and amplitude per host, AR(1) noise, and heavy-tailed bursts of benign activity (backups, updates, port-scan-like benign monitoring) that look like precursors but are unrelated to attacks.
- **Attack runs:** about 400 independent runs. Each picks a host uniformly at random (every host has the same expected rate), a type (port scan, brute force, DoS flood, exfiltration, lateral movement), a duration (5 to 90 min) and an onset time. Runs on one host are at least 2 h apart so episodes are separate (the 30 min episode rule from earlier studies is kept for eligibility).
- **Onset timing:** onsets are drawn so the clock is **not** a shortcut in the main conditions: uniform over the 96 h, independent of hour of day. One extra condition (C below) concentrates onsets in a morning window, like the CIC Botnet campaign, to show the time control winning.
- **Attack signature:** during the run the type-specific features shift (e.g. scan: destination port entropy and unique ports up, small packets; flood: flow_count, syn_count up; brute force: many short flows to one port with rst/fin; exfil: byte_count_sum up). This is what detection uses; it is not what forecasting may use.
- **Precursor (the knob):** in the `L` minutes before an onset the host's features are pulled toward its attack signature by a factor ramping from 0 to `s` (strength, in units of the host's own benign standard deviation at the end of the ramp). Conditions:
  - **A, no precursor:** `s = 0`. The attacker's choice of host and time is independent of the host's traffic. Expected result: fail.
  - **B1, B2, B3, faint to clear precursor:** `s = 0.25, 0.5, 1.0`, `L = 10 min`.
  - **C, clock-concentrated, no precursor:** `s = 0`, onsets in a 07:00 to 09:00 window each day. Expected result: the time-only control passes and traffic models do not beat it.
  - **D, shared precursor (network-level):** `s = 0` on the host, but 5 min before each onset the *network-wide* context (active hosts, total flows) is perturbed, as when an attacker's campaign starts. Tests whether the `own+net` features pick up a context signal that `own` cannot.
- Fixed generator seeds per condition: 11 for training/tuning-free evaluation, a second independent seed 12 for the confirmation run. No generator parameter is tuned after seeing a result.

## 4. Target, eligibility, horizons

Identical to docs/onset_forecasting_design.md section 3: anchor at window end `t`, inputs the last 10 windows, eligible if no attack window starts in `[t - 30 min, t)` and none in the inputs, positive if an attack window starts in `[t, t + H)`. Horizons H = 1, 5, 15, 30 min.

## 5. Folds

Leave-hosts-out: 10 folds of 6 hosts, host assignment to folds fixed by seed 42. The next fold is validation (early stopping, alert threshold). Because every run belongs to one host and runs are at least 2 h apart, **a run is never split across train and test**. A second check, "grouped by run", re-splits with GroupKFold over run id and must give the same answer within 0.02 AUC median. Training negatives are subsampled to 30% for every model as before; test is never subsampled.

## 6. Models and features

Features: **own**, **own+net** (as before), **time** (hour and day of week; control). Models: logistic regression and gradient boosting on the same per-feature summaries (last, mean, max, slope) as `onset_forecast_study.py`. No neural models: classical models already pass the earlier bar structure, and the neural ones took hours for no gain on real data. If a classical model passes, one LSTM is added as a confirmation, not before.

## 7. Controls

1. **Pseudo-onset null** (no retraining), as before.
2. **Shifted-label retraining**, as before.
3. **Time-only model** on hour and day of week.
4. **Clock-matched AUC**, as before.
5. **Dataset-identity check.** (a) Host identity: a model must not be able to predict the host from the features alone well enough to explain AUC; every host has the same onset rate, so host recognition must give AUC about 0.5 on held-out hosts. (b) Run-seed identity: train on seed 11 and test on seed 12 with new hosts and new profiles; a model whose AUC drops by more than 0.05 relative to within-seed folds was relying on generator artefacts, not the precursor. (c) A "detector sanity" run: the same pipeline with the target moved to "attack in progress" (positives = windows inside an attack) must score AUC >= 0.95, to show the features and models can see the attack signature, so a failure on forecasting is not a pipeline bug.

## 8. Bars (all must hold for a variant to count as forecasting in a given condition)

| # | Bar |
|---|---|
| S1 | Median held-out fold AUC >= 0.70 |
| S2 | Pseudo-onset p < 0.05 on at least 7 of 10 folds |
| S3 | Median clock-matched AUC >= 0.65 |
| S4 | Median AUC beats the time-only control by >= 0.10 and the shifted-label control by >= 0.15 |
| S5 | At the threshold giving 1% FPR on the validation fold, at least half of held-out episodes get an alert within H before onset, test FPR <= 2% |
| S6 | Passes S1 to S5 again on seed 12 (independent generator seed) |

## 9. Pre-stated expectations (so a result cannot be spun afterwards)

- Condition A and C: no traffic model passes S1 to S5. In C the time-only control passes S1 and traffic models fail S4.
- Condition B: pass/fail is a function of `s`. I expect B1 (0.25) to fail, B3 (1.0) to pass for horizons up to 15 min, and B2 (0.5) to be borderline. If B3 fails, the pipeline is broken or too weak and the real-data failures are not informative; that outcome would be reported as a pipeline problem.
- Condition D: `own+net` passes where `own` fails. If not, network context features carry nothing here.
- If A passes, there is a leak in the generator or the pipeline and every other result is void until it is fixed. A is the guard.

## 10. What a result can and cannot say

- Conditions A and C failing and B3 passing shows the pipeline has power and does not hallucinate onsets. It says nothing about whether real attackers leave a precursor.
- A smallest passing `s` gives the size of precursor the pipeline can detect, in benign standard deviations. The real-data studies can then be read as "no precursor larger than about that was present", with the caveat that the real precursor, if any, may have a different shape.
- Because I author both the attack and the precursor, the headline cannot be "AegisFlow predicts attacks". It can be "the evaluation is sound and had enough independent onsets; here is the smallest detectable precursor".

## 11. Deviations

Recorded 2026-10-09, before the full runs and before any fold-level result was read. One timing smoke test of a single fold (condition B3, H = 5 min, fold 0) was run to size the compute; its AUCs (about 0.65 logistic, 0.59 gradient boosting) were seen and nothing was changed because of them.

1. Gradient boosting trains on 10% of negatives (logistic keeps 30%), 60 trees, depth 4, to keep the run to minutes. Test anchors are never subsampled.
2. Network features are four aggregates computed from the host windows themselves (summed flows, summed bytes, mean destination ports, summed destination IPs), last and mean over the 10 windows, because the simulator has a fixed host population.
3. The grouped-by-run re-split, the host-identity classifier and the detector sanity run from section 7 are run once on condition B3 only, not on every condition.
4. S6 (second seed) is run for every condition, so a pass or fail in any condition is checked on seed 12.
