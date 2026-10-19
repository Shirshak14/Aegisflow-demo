# Own-lab simulation v2: results (2026-10-19)

Design and bars: [DESIGN.md](DESIGN.md), committed before any run. Reproduce from the repo root:

    python research/synth_lab_v2/study.py MAIN --seed 1 --leave-chain-out   # about 20 min on one core
    python research/synth_lab_v2/study.py MAIN --seed 2
    python research/synth_lab_v2/study.py NULL   # likewise SUBTLE, EASY
    python research/synth_lab_v2/summarize.py    # prints the tables below

Raw output: `results/<CONDITION>_s<seed>.json`. The demo model, its threshold, `aegisflow/`, `backend/` and the
dashboard are unchanged.

**Everything here is synthetic.** I wrote the simulator, the attacks and the precursors. A pass would show the
pipeline can find a precursor when one exists. It cannot show that real attacks leave one.

## Answer

**MAIN fails the pre-registered bars at every horizon, on both seeds.** With realistic mess (partial chains, stealthy
attacks, 2 to 4 aborted chains per real attack, and 15% to 20% of attacks with no precursor at all), the learned
forecaster ranks future attack windows clearly better than every control at 30 minutes and 2 hours (PR-AUC 2x to 20x
the best control, winning in 4 to 6 of 6 held-out organisations). But at the fixed alert budget it warns only
**16% to 22% of attacks** ahead of time, below the 25% bar. At 12 hours it is no better than the controls.

Only **EASY** passes everything: no aborted chains, no skipped stages and doubled precursor strength. Then the model warns
53% to 60% of attacks, 2 to 4 times more than the best control, with a median lead of 2 hours at H = 2 h.

The leak guards hold. In NULL (precursors removed), the models sit at the controls' level (PR-AUC ratio 0.88 to 1.04).
Attacks with no precursor (Z) are warned 0 times in MAIN, SUBTLE and EASY.

So, even in a lab where I put the precursors in myself, forecasting only clears a usable bar when the precursors are
strong, complete and rarely faked by aborted attempts. The safe AegisFlow claim stays "early warning of attacks in
progress".

## 1. Bars per run (leave-one-organisation-out, 6 folds)

| Run | H | Best model PR-AUC | Best control (PR-AUC) | Ratio | Orgs won | ROC-AUC model / best ctrl | Recall model | Best control recall | Lead (min) | P1 | P2 | P3 | P4 | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NULL_s1 | 30m | M-LR 0.0010 | C-CTX 0.0009 | 1.04 | 3/6 | 0.78 / 0.80 | 0.02 | C-ALERT+CTX 0.01 | 20 | no | yes | no | yes | **no** |
| NULL_s1 | 2h | M-LR 0.0036 | C-ALERT+CTX 0.0037 | 0.96 | 1/6 | 0.79 / 0.80 | 0.02 | C-TIME 0.03 | 80 | no | yes | no | yes | **no** |
| NULL_s1 | 12h | M-LR 0.0208 | C-CTX 0.0237 | 0.88 | 1/6 | 0.79 / 0.80 | 0.02 | C-SCAN 0.09 | 370 | no | yes | no | yes | **no** |
| MAIN_s1 | 30m | M-LR 0.0465 | C-ALERT+CTX 0.0021 | 21.84 | 5/6 | 0.84 / 0.82 | 0.19 | C-ALERT+CTX 0.03 | 30 | yes | yes | no | yes | **no** |
| MAIN_s1 | 2h | M-LR 0.0317 | C-ALERT+CTX 0.0090 | 3.52 | 6/6 | 0.84 / 0.82 | 0.22 | C-ALERT+CTX 0.05 | 90 | yes | yes | no | yes | **no** |
| MAIN_s1 | 12h | M-LR 0.0275 | C-ALERT+CTX 0.0260 | 1.06 | 3/6 | 0.79 / 0.81 | 0.17 | C-SCAN 0.14 | 165 | no | yes | no | yes | **no** |
| MAIN_s2 | 30m | M-GB 0.0150 | C-ALERT 0.0072 | 2.08 | 4/6 | 0.83 / 0.65 | 0.17 | C-ALERT 0.05 | 30 | no | yes | no | yes | **no** |
| MAIN_s2 | 2h | M-GB 0.0352 | C-ALERT 0.0174 | 2.02 | 6/6 | 0.82 / 0.66 | 0.20 | C-SCAN 0.06 | 80 | yes | yes | no | yes | **no** |
| MAIN_s2 | 12h | M-LR 0.0351 | C-ALERT+CTX 0.0251 | 1.40 | 6/6 | 0.81 / 0.80 | 0.16 | C-SCAN 0.16 | 185 | no | yes | no | yes | **no** |
| SUBTLE_s1 | 30m | M-GB 0.0213 | C-CTX 0.0013 | 16.10 | 6/6 | 0.86 / 0.80 | 0.19 | C-TIME 0.01 | 30 | yes | yes | no | yes | **no** |
| SUBTLE_s1 | 2h | M-LR 0.0331 | C-CTX 0.0052 | 6.39 | 6/6 | 0.85 / 0.81 | 0.19 | C-SCAN 0.05 | 80 | yes | yes | no | yes | **no** |
| SUBTLE_s1 | 12h | M-GB 0.0323 | C-CTX 0.0271 | 1.19 | 3/6 | 0.81 / 0.82 | 0.10 | C-SCAN 0.12 | 245 | no | yes | no | yes | **no** |
| EASY_s1 | 30m | M-GB 0.2040 | C-ALERT 0.0422 | 4.84 | 6/6 | 0.90 / 0.72 | 0.55 | C-ALERT 0.14 | 30 | yes | yes | yes | yes | **yes** |
| EASY_s1 | 2h | M-GB 0.2978 | C-ALERT 0.1031 | 2.89 | 6/6 | 0.89 / 0.72 | 0.60 | C-ALERT+CTX 0.20 | 120 | yes | yes | yes | yes | **yes** |
| EASY_s1 | 12h | M-GB 0.1983 | C-ALERT+CTX 0.0919 | 2.16 | 6/6 | 0.81 / 0.81 | 0.53 | C-ALERT+CTX 0.27 | 270 | yes | yes | yes | yes | **yes** |

PR-AUC is low in absolute terms because positives are rare (base rate 0.02% to 0.5% of host-windows). Read it as a
ratio against the controls. P6 (same verdict on seed 2) holds: MAIN fails on both seeds at every horizon.

## 2. What the model catches and what it costs (H = 2 h, best model)

| Run | R ransomware | X data theft | W web exploit | I insider | M miner | Z no precursor | False flags per 100 host-days | Share near aborted chains |
|---|---|---|---|---|---|---|---|---|
| NULL | 0/67 | 3/64 | 0/44 | 0/37 | 0/42 | 3/51 | 20.7 | 22% |
| MAIN s1 | 25/52 | 12/66 | 9/45 | 1/46 | 20/42 | 0/54 | 16.8 | 68% |
| MAIN s2 | 29/63 | 3/59 | 13/44 | 4/44 | 11/51 | 0/44 | 15.3 | 77% |
| SUBTLE | 27/63 | 11/48 | 2/53 | 3/50 | 16/45 | 0/46 | 16.7 | 51% |
| EASY | 67/71 | 30/48 | 39/46 | 4/48 | 42/49 | 0/43 | 16.0 | 0% (no aborted chains) |

* The noisy chains (ransomware, miner) are where the warnings come from. Quiet chains (insider, most data theft) are
  almost never warned, even in EASY.
* In MAIN, **two thirds to three quarters of false warnings land on hosts where an attack started and was abandoned.**
  Those precursors are real attacker activity, so no model can tell them apart from the ones that go on to impact.
  That is the main reason recall at a fixed budget stays low. Each attack warned in time costs about 60 falsely
  flagged 10-minute windows.
* **Leave-chain-type-out** (gradient boosting, 2 h, MAIN seed 1): a model that never saw a chain type in training
  catches almost none of it: ransomware 7/52 (22/52 when that type is in training), data theft 1/66 (7/66),
  web exploit 0/45 (6/45), miner 0/42 (17/42). The forecaster learns specific chain signatures, not a general
  "something is about to happen" signal, so a new kind of attack would not be forecast.
* **SUBTLE** (no IDS alert on any precursor): the alert-count rules drop to near zero, while the models keep their
  recall (19%). Telemetry carries the signal, not alert counts. Even so, the recall bar fails.
* At 12 h, the "scan in the last 20 minutes" rule (C-SCAN) catches 35 of 45 web exploits in MAIN and ties the models
  on recall. Long-horizon warnings mostly come from scans that are already running, which matches the AIT-ADS
  finding.

## 3. Deviations and caveats

1. **Bug fixed before the reported run.** In the first run the calibration scores for the learned controls were not
   jittered, while test scores were. For the clock control (only 48 distinct values) this flagged 1.2% of windows
   instead of 0.1%, which inflated its recall. I fixed it and reran all five runs unchanged otherwise. The first-run
   output is kept in `results_v0/`. Model verdicts did not change (MAIN failed P3 everywhere in both runs).
2. **P2 (ROC-AUC >= 0.75) is not informative.** Attack targets depend on host role, so the role + clock control
   (C-CTX) alone reaches ROC-AUC 0.80 in every run, including NULL. P2 should have been relative to the controls. It
   decides nothing here because P3 fails.
3. In NULL, aborted chains leave no trace (an aborted chain is only precursors), so its decoys are the benign ones
   only.
4. The attacker's foothold activity on other hosts (for example, the scanning host itself) is not simulated. Every
   precursor is visible on the target host. This makes the task easier than reality, not harder.
5. The lead-time medians, stealth spread, abort ratio and IDS detection rates are my choices. A different simulator
   would move the numbers. The direction is the robust part: forecasting works when precursors are strong, complete
   and rarely faked, and degrades fast when they are not.

## 4. What it means for AegisFlow

* It shows the pipeline has no leak (NULL at control level, zero warnings on no-precursor attacks) and that it can
  forecast 30 minutes to 2 hours ahead when a clear, unambiguous precursor exists (EASY).
* It also shows that with a realistic mix the same pipeline warns about 1 in 5 attacks at a usable alarm budget,
  does not generalise to attack types it has not seen, and pays for most false alarms with abandoned attempts.
* It does not show that real attacks have precursors on this scale. The public-data studies (CIC-IDS2017, CTU-13,
  LANL, AIT-ADS) found none.
* Safe deck wording stays: early warning of attacks in progress. If forecasting is mentioned, it should be as
  "tested in simulation: works when precursors are strong, not shown on real data".
