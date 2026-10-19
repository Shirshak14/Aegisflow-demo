# Own-lab simulation v2: multi-stage forecasting test (design, written before any run)

Date: 2026-10-19. Branch: `claude/project-thread-ylw8xk`. Nothing here touches the demo, the dashboard,
`aegisflow/` or `backend/`.

**Everything in this study is synthetic.** I write both the attacks and the precursors. A pass shows that the
pipeline can find a precursor when one exists in data that looks like this. It is not evidence that real attacks
behave this way.

## Why a v2

The first own-lab study (section-d, 2026-10-09) used one generic precursor (a feature shift before every attack)
and only passed when that shift was about 4 standard deviations. Every public dataset since then failed against
simple controls (time of day, "scan already running"). v2 asks a narrower and more realistic question:

> When attacks come as multi-stage chains (recon, odd logins, lateral movement, staging, impact) with variable lead
> times, mixed with attacks that have no precursor and with precursor-looking activity that never becomes an
> attack, does a learned forecaster beat the simple rules a SOC already has?

## Simulator (`synth_lab_v2.py`)

* **6 scenarios** = 6 separate simulated organisations (120 to 260 hosts each, about 1,100 hosts in total),
  24 days each at 10-minute windows (first 3 days are burn-in and never scored).
* **Host roles**: workstation, app server, web server (internet-facing), file server, domain controller, dev box.
  Every host has its own per-channel activity multiplier, so hosts are recognisable as in real data.
* **Benign rhythm**: workday curve in each organisation's local hours (offsets differ), quiet nights and
  weekends; servers have a flatter curve and nightly backups.
* **19 channels per host-window.** 6 are IDS alerts (scan in, scan out, brute force, exploit, misc policy,
  impact). 13 are telemetry that never raises an alert: logins ok / failed / from a new source / privileged,
  remote admin tool runs, inbound connections, outbound connections, new internal destinations, bytes out
  internal, bytes out external, mass file reads, mass file writes, CPU saturation.
* **Six attack chains** (targets drawn by role):
  * R ransomware: recon scan, credential spraying, privileged lateral login + admin tools, file staging, encryption.
  * X data theft: new-source off-hours login, light recon, staging, large outbound transfer.
  * W web exploit: external scanning, exploit/brute attempts, webshell traffic out.
  * I insider: odd-hours logins by a normal account, small staging, outbound transfer. No IDS alert before impact.
  * M miner: brute force, new-source login, CPU saturation and pool connections.
  * Z no precursor at all (worm / zero-day style): impact only.
* **Lead times** between stages are log-normal (medians from 20 minutes to 1 day), so the gap from the first
  precursor to impact ranges from under an hour to several days.
* **Per-attack stealth** (log-normal multiplier on every precursor signal) and **skipped stages** (each precursor
  stage independently dropped with probability 0.3) make many chains partial.
* **Decoys that are not followed by attacks**:
  * aborted chains: exactly the same chain generator, stopped before impact (2 to 4 aborted per completed attack,
    depending on the organisation);
  * weekly vulnerability scanner hitting every host; one organisation has a pentest week;
  * nightly backups (staging-like transfers), monthly patch night (admin tools + privileged logins on all hosts),
    travelling users (new-source logins), password-typo lockout bursts, background internet noise on web servers.
* **Impact timing**: uniform in some organisations, skewed to nights/weekends in others, so the clock has some
  genuine signal, as in reality.
* IDS alerts for precursor activity are only raised with a per-organisation detection probability (0.4 to 0.9).

## Conditions

| Condition | What changes | Purpose |
|---|---|---|
| NULL | precursor stages removed from every chain (aborted chains and benign decoys kept) | leak guard: models must sit with the controls |
| MAIN | as above | the real test |
| SUBTLE | IDS never alerts on precursor stages (telemetry only) | does telemetry add anything a SOC alert count misses |
| EASY | no aborted chains, no skipped stages, stealth x2 | upper bound: pipeline can find an obvious precursor |

MAIN is also rerun on a second generator seed.

## Task and splits

* Row = (host, 10-minute window). Target at horizon H: an impact starts on this host within the next H.
  H = 30 minutes, 2 hours, 12 hours.
* A row is scored only if the host has had no impact in the previous 24 hours and the full horizon is inside
  the data. Rows during an attack are therefore never scored: this is forecasting, not detection.
* **Leave-one-scenario-out**: train on 5 organisations, test on the 6th, all 6 rotations.
* Features: per channel, log sums over the last 10 min, 1 h, 6 h and 24 h; the 6-hour sum relative to the host's
  own trailing 7-day level; hour of day (sin/cos), weekend flag, role. No feature uses labels, run ids or chain
  types.

## Models and controls

* Models: logistic regression and gradient boosting (sklearn HistGradientBoosting) on all features.
  Training keeps all positives and 5% of negatives.
* Controls (same splits):
  * **C-TIME**: hour of day + weekend (logistic regression).
  * **C-CTX**: role + hour + weekend (logistic regression).
  * **C-ALERT**: number of IDS alerts on the host in the last W, W picked on the training organisations from
    {10 min, 1 h, 6 h, 24 h}.
  * **C-SCAN**: scan alerts (in or out) on the host in the last 20 minutes ("scan already running").
  * **C-ALERT+CTX**: logistic regression on the four alert-count windows + role + time (a learned alert rule).
* Operating point: threshold set on the training organisations so 0.1% of negative windows are flagged
  (about 14 flagged windows per 100 host-days). An onset counts as warned if any window in
  [onset - H, onset - 10 min] is flagged; lead = onset minus the first such flag.

## Bars (fixed now; a horizon passes only if P1 to P4 all hold in MAIN)

* **P1 ranking**: median held-out PR-AUC of the best model >= 1.5x the best control's median, and the model
  beats that control in at least 5 of 6 organisations.
* **P2**: median held-out ROC-AUC of that model >= 0.75.
* **P3 operating point**: pooled onset recall at the 0.1% budget >= 25% and >= 1.5x the best control's recall
  at the same budget.
* **P4 lead**: for H >= 2 h, median lead of warned onsets >= 30 minutes (otherwise it is detection of a stage
  already running, not a forecast).
* **P5 leak guard** (must hold or the whole study is void): in NULL, the best model's median PR-AUC is
  <= 1.2x the best control; in MAIN, recall on no-precursor (Z) attacks is no more than the larger of 3x the
  random-flag expectation and the best control's Z recall + 5 points.
* **P6**: the MAIN verdict per horizon is the same on the second seed.

## What I expect before running

* NULL at control level; EASY passes everywhere.
* MAIN: I expect a pass at 2 h, and I am unsure at 30 min (the "scan already running" rule is strong there)
  and at 12 h (many aborted chains look the same). If MAIN fails, that is the result to report.
* SUBTLE: models beat C-ALERT by a wide margin, because the alert rule only sees decoys there.
* Most false warnings will land on hosts hit by aborted chains. That is the decoy cost and will be reported.

## Also reported (exploratory, not bars)

Recall by chain type, lead-time distribution, share of false warnings on aborted-chain hosts, and a
leave-chain-type-out check (train without one chain type's attacks, test on that type) for gradient boosting at
H = 2 h.
