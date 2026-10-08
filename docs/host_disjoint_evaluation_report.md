# Host-disjoint evaluation: results (2026-10-08)

Implements `docs/host_disjoint_evaluation_design.md` (D1 to D7) on the local CIC-IDS2017 sequence table (85,077 sequences, 1,390 hosts). Every number below comes from one run. Reproduce: `.venv/Scripts/python.exe scripts/host_disjoint_eval.py`, which takes about 25 minutes on CPU. Raw output is in `reports/host_disjoint_eval.json`. Fold models go to `artifacts/experiments/cic_ids2017_host_disjoint/`. The demo model, `backend/` and the Parquet files are untouched.

## Answer

**Formally, the pre-registered bars pass. I do not think the result supports a claim of attack detection beyond host recognition.** With the five infected Botnet hosts held out one at a time, the LSTM ranks attack above benign on all five unseen hosts: within-host ROC-AUC 0.59 to 0.73, median 0.655, against 0.5 for the identity rule. Three findings make this weak evidence:

1. **The intervals are not trustworthy.** Each host's Botnet positives form a single 30-minute episode. Episode-level resampling therefore has nothing to resample on the positive side, and the intervals only reflect benign-side noise. The label-shuffled control shows the real spread: shuffled models score between 0.32 and 0.59 per host, and their "95% intervals" exclude 0.5 on 4 of 5 hosts even though they learned nothing.
2. **Inside the Botnet hours the model barely separates attack from benign on the same host.** Matched in time (same host, same Friday hours), the LSTM AUC is 0.51 to 0.58.
3. **What transfers is "Botnet traffic is already in the inputs", not forecasting.** Benign target windows whose inputs contain Botnet traffic score above clean benign windows (AUC 0.56 to 0.69). The model recognises ongoing Botnet activity on a host it never saw. It does not predict the next window.

The folds are host-disjoint but **not campaign-disjoint**: all hosts are infected in one Friday campaign with one command-and-control server. Brute Force, Web Attack, Reconnaissance, DoS and Infiltration cannot be evaluated this way at all (section 5).

## 1. Setup

- **Folds (D3).** One fold per internal Botnet host with at least 20 attack targets: 192.168.10.5, .8, .9, .14, .15. Test is every sequence of the held-out host. Validation is every sequence of the next host in that list, used for checkpoint selection and the 1% false-positive threshold. Training is every sequence of every other host: 1,388 hosts, about 78,000 sequences and 260 to 286 Botnet positives per fold. Hosts .17 (2 attack targets) and 205.174.165.73 (2 benign targets) are always in training.
- **Positives.** Botnet only. Sequences whose target is another attack class are dropped everywhere: 603 sequences, 590 from 172.16.0.1, 11 Infiltration on .8 and 2 DoS on .50.
- **Models.** The existing `train_experiment` with the committed demo hyperparameters (LSTM hidden 16, 8 epochs, patience 3, seed 42) and the logistic regression it trains alongside.
- **Intervals (D5).** 2,000 bootstrap resamples, stratified by host. Positives are resampled by attack episode (under 30 minutes apart, as in `lodo_pilot.py`), benign targets in 30-minute clock blocks per host. Benign windows on one host form one continuous run, so the episode rule cannot split them.
- **Fixed before running:** at least 20 per label, 95% episode-level intervals, and the four acceptance criteria. Nothing was changed after seeing results. The time-matched control in section 3 was **added afterwards** and is labelled as such.

## 2. Leave-one-Botnet-host-out (D1, D2, D3, D6)

Each held-out host has 56 to 80 Botnet positives and 2,981 to 3,160 benign targets, spread over all five days.

| Held-out host | Pos / neg | LSTM AUC [95% CI] | LSTM, Friday only | LR AUC [95% CI] | Identity-only | LSTM recall at 1% val FPR (test FPR) |
|---|---|---|---:|---|---:|---|
| 192.168.10.5 | 69 / 3,124 | 0.616 [0.578, 0.651] | 0.566 | 0.597 [0.553, 0.640] | 0.500 | 0/69 (1.0%) |
| 192.168.10.8 | 62 / 2,981 | 0.655 [0.628, 0.682] | 0.616 | 0.603 [0.557, 0.646] | 0.500 | 1/62 (1.1%) |
| 192.168.10.9 | 67 / 3,069 | 0.591 [0.559, 0.619] | 0.563 | 0.596 [0.563, 0.628] | 0.500 | 0/67 (1.1%) |
| 192.168.10.14 | 56 / 3,129 | 0.685 [0.658, 0.714] | 0.597 | 0.587 [0.542, 0.634] | 0.500 | 4/56 (0.2%) |
| 192.168.10.15 | 80 / 3,160 | 0.734 [0.708, 0.760] | 0.679 | 0.701 [0.674, 0.726] | 0.500 | 2/80 (0.5%) |
| **Median / gap** | | **0.655** [0.629, 0.680]; gap vs identity **+0.156** [0.143, 0.169] | | 0.597 [0.577, 0.627]; gap +0.117 [0.099, 0.134] | | |

- "Friday only" keeps the held-out host's Friday benign targets only (about 570 to 650), which removes day-of-week differences. AUC drops by 0.04 to 0.09.
- The identity-only model (D6) is a logistic regression on a one-hot host ID. It scores exactly 0.5 on every held-out host, as it must, so the within-host metric is not contaminated by host identity.
- Recall at a 1% false-positive rate is 0 to 7%. At an operating point an analyst could use, the model catches almost nothing on unseen hosts.

### Label-shuffled control (D6)

The same five folds, retrained with training and validation labels shuffled, and scored against the true labels:

| Held-out host | .5 | .8 | .9 | .14 | .15 | Median [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| LSTM, shuffled labels | 0.500 | 0.325 | 0.515 | 0.357 | 0.593 | 0.500 [0.479, 0.517] |
| LR, shuffled labels | 0.528 | 0.442 | 0.487 | 0.557 | 0.502 | 0.502 [0.489, 0.518] |

The median is at chance, as required. The per-host values are not: a model that learned nothing ranges from 0.32 to 0.59 on single hosts. This is the realistic noise level for one host with one episode. Every real fold beats its own shuffled fold (5 of 5), and 4 of 5 real LSTM values exceed the highest shuffled one (0.593).

## 3. Time-matched control (added after the first results, not pre-registered)

Botnet positives on these hosts all fall between 10:03 and 12:59 on Friday. A model could score "Friday late morning" instead of Botnet behaviour. This control re-scores the saved real fold models on the held-out host only, Friday only:

| Held-out host | Benign in span (clean inputs) | Attack vs same-host benign in span | Clean benign: in span vs outside | Benign with Botnet traffic in inputs vs clean outside |
|---|---|---:|---:|---:|
| 192.168.10.5 | 177 (74) | 0.576 | 0.399 | 0.555 |
| 192.168.10.8 | 162 (49) | 0.552 | 0.419 | 0.673 |
| 192.168.10.9 | 202 (49) | 0.515 | 0.351 | 0.620 |
| 192.168.10.14 | 170 (75) | 0.510 | 0.620 | 0.651 |
| 192.168.10.15 | 162 (31) | 0.564 | 0.614 | 0.692 |

All values are LSTM. Logistic regression values are in the JSON and are similar or weaker.

- **Attack vs same-host benign in the same hours: 0.51 to 0.58.** Most of the separation in section 2 comes from contrasting the infected hours with the rest of the host's week, not from picking out the attack target windows.
- **The span itself is not what is scored.** Clean benign windows inside the span are not consistently above those outside it (0.35 to 0.62, three hosts below 0.5), so this is not just a clock effect.
- **What the score follows is Botnet traffic already in the inputs.** Benign targets whose input windows contain Botnet flows score above clean benign windows (0.56 to 0.69). Botnet beaconing is periodic, so "traffic in the last 5 minutes" and "traffic in the next window" largely coincide. The model recognises ongoing activity on an unseen host. That is a weaker claim than forecasting.

## 4. Existing time split, committed demo model (D1, D2, D4)

No retraining: the committed model is scored on the existing test split (Friday, 11,379 sequences, 487 positives). Botnet never appears in its training data, so for the Botnet hosts this is already host- and class-unseen.

| Host | Pos / neg | LSTM within-host AUC [95% CI] | LR AUC |
|---|---|---|---:|
| 192.168.10.14 | 56 / 576 | 0.677 [0.595, 0.737] | 0.577 |
| 192.168.10.15 | 80 / 630 | 0.624 [0.549, 0.694] | 0.586 |
| 192.168.10.8 | 62 / 548 | 0.615 [0.491, 0.718] | 0.593 |
| 192.168.10.5 | 69 / 599 | 0.504 [0.461, 0.545] | 0.428 |
| 192.168.10.9 | 67 / 583 | 0.442 [0.387, 0.489] | 0.367 |
| 172.16.0.1 (D4) | 76 / 29 | 0.772 [0.063, 1.000] | 0.487 |

- **Pooled test set, model vs identity rule** (score 1 for hosts with attacks in training, which is only 172.16.0.1): LSTM 0.553 vs 0.577, gap −0.024 [−0.134, 0.073]. Logistic regression 0.473, gap −0.104 [−0.176, −0.035]. On the split the demo uses, the committed model does not beat "is this 172.16.0.1".
- **172.16.0.1 within-host control (D4):** 0.772, but the interval spans 0.06 to 1.00. Its 76 test positives come from 3 episodes and it has 29 benign test targets. This is a sanity check with no weight, as the design expected.
- Hosts with under 20 targets of either label are not scored: .17 (2 positives), .50 (2), 205.174.165.73 (2 benign).

## 5. Acceptance criteria (D7)

| # | Criterion (fixed before running) | Result | Met? |
|---|---|---|---|
| 1 | Within-host AUC lower bound above 0.5 on at least 4 of 5 Botnet hosts | LSTM 5 of 5, LR 5 of 5 | Yes, but see note |
| 2 | Shortcut gap positive, interval excludes 0 | LSTM +0.156 [0.143, 0.169]; LR +0.117 [0.099, 0.134] | Yes, but see note |
| 3 | Label-shuffled control at chance (interval contains 0.5) | Median LSTM 0.500 [0.479, 0.517], LR 0.502 [0.489, 0.518] | Yes |
| 4 | Every class that cannot be tested this way listed as not evaluable | Below | Yes |

**Note on 1 and 2.** These bars assumed several episodes per host. There is exactly one per host, so the intervals understate uncertainty. The shuffled control shows that a single host can sit 0.18 from chance by noise alone (section 2). At host level the evidence is: 5 of 5 hosts above 0.5 (one-sided sign test p = 0.031), 5 of 5 above their own shuffled control, and a time-matched AUC of 0.51 to 0.58.

**Not evaluable for generalisation** (one attacker host each, so no host-disjoint test exists):

| Class | Attacker hosts | Positive sequences |
|---|---|---:|
| Denial of Service | 172.16.0.1 (+ 2 sequences from 192.168.10.50) | 185 |
| Brute Force | 172.16.0.1 | 238 |
| Web Attack | 172.16.0.1 | 130 |
| Reconnaissance | 172.16.0.1 | 39 |
| Infiltration | 192.168.10.8 | 11 |

## 6. What this changes

- The B1 finding stands: on the split the demo uses, the committed model is not better than host identity (section 4).
- When Botnet is trained on other hosts, the LSTM picks up a modest, real signal on unseen hosts: 0.59 to 0.73 within-host, beating every shuffled control. That signal is "Botnet traffic is active on this host", and it is campaign-specific. It does not support "forecasts attacks" or any claim about other classes.
- More evidence needs more independent episodes and campaigns, not more Friday rows. The deferred CSE-CIC-IDS2018 route (design section 7) remains the only candidate found.

Not done: no hyperparameter search, a single seed, no other input representation (the B1 host-relative inputs were not combined with these folds), and no campaign-disjoint test, which this data cannot provide.
