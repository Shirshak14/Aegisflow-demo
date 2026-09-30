# LODO pilot on the 500k sample: result

**Verdict:** The Wednesday↔Friday DoS cross-day result does **not** hold once host is controlled for.
- **Friday → Wednesday:** fails outright.
- **Wednesday → Friday:** matches a rule that flags host `172.16.0.1`, with no more detections than that rule, and the only within-host evidence comes from a small, confounded set of benign sequences.

The full 2.83M-row run is **not** justified by this result.

Reproduce with `python scripts/lodo_pilot.py`. The full output is in `reports/lodo_pilot_results.json`, and per-fold artifacts are in `artifacts/models/cic_ids2017_lodo_pilot/<fold>/`. The Phase 3 artifacts are untouched.

## Protocol

Each fold calls the existing `train_experiment`: train-only preprocessing, existing class weighting, validation-only checkpoint and threshold selection. Phase 3 hyperparameters: seed 42, 8 epochs, batch 256, learning rate 0.001, hidden size 16, dropout 0.2, patience 3.

Folds are **day-disjoint**:
- **Test:** sequences entirely on the held-out day.
- **Validation:** sequences entirely on the previous capture day. Monday is attack-free and never used for validation, so the Monday and Tuesday folds use the next day.
- **Purged:** sequences that cross into the test or validation day.
- **Train:** everything else.

A dry run confirmed that no training row touches its test or validation day.

A per-day 80/20 time split was tried first and rejected in the dry run. Friday's DDoS is the last 20 minutes of that day, so the split moved all of it into validation and left the Wednesday fold with no DoS in training.

LODO folds that train on later days than the test day measure cross-day generalisation, not deployable forecasting.

**Reference rules.** These are not models:
- *Host identity:* score = 1 if the host is 172.16.0.1.
- *Last window under attack:* reads the label-derived `attack_flow_ratio`.

## Answer: Wednesday↔Friday DoS, controlling for host

Every DoS positive on both days is host 172.16.0.1 attacking 192.168.10.50. Windows are grouped by source host, so all DoS positives sit on host 172.16.0.1.

| Fold (train → test) | Model | DoS detected | Test FPR | DoS ROC-AUC vs all negatives | DoS ROC-AUC vs 172.16.0.1's own negatives | 172.16.0.1 negatives flagged |
|---|---|---:|---:|---:|---:|---:|
| **Fri DDoS → Wed DoS** | Logistic regression | 0/129 | 0.036 | 0.043 | 0.297 (163 negatives) | 0/163 |
| | LSTM | 12/129 | 0.379 | 0.366 | 0.660 | 6/163 |
| | Host-identity rule | 129/129 | 0.013 | 0.994 | 0.500 | 163/163 |
| **Wed DoS → Fri DDoS** | Logistic regression | 42/44 | 0.003 | 0.959 | 0.801 (29 negatives) | 27/29 |
| | LSTM | 42/44 | 0.009 | 0.998 | 0.901 | **29/29** |
| | Host-identity rule | **42/44** | 0.003 | 0.976 | 0.500 | 29/29 |

**Friday → Wednesday collapses.** Logistic regression ranks Wednesday's DoS *below* benign traffic (ROC-AUC 0.043). The LSTM catches 9% at a 38% false-positive rate. Training on DDoS doesn't transfer to the Hulk/GoldenEye/slowloris/Slowhttptest tools, even from the same attacker host.

**Wednesday → Friday is host recognition when thresholded.**
- Both models catch exactly the 42 positives on 172.16.0.1 that the host-identity rule catches.
- They miss the 2 DoS positives on other hosts.
- The LSTM flags 29 of 29 benign sequences from 172.16.0.1.

The within-host ROC-AUC (0.80 and 0.90) is the only sign of genuine DoS signal, but its "benign" control is weak:
- It rests on 29 sequences from one host.
- Their targets all fall between 16:17 and 16:59, immediately after the DDoS ended at 16:16.
- 12 of the 29 have attack traffic in their inputs.

That compares an attack in progress with the attack's tail on the same host. It isn't evidence of a DoS detector.

## All folds: test-day results

| Fold | Validation day | Test positives | Logistic regression F1 / FPR | LSTM F1 / FPR | Host-identity rule F1 | Last-window rule F1 |
|---|---|---|---|---|---|---|
| Mon | Tue | 0 | – / 0.040 | – / 0.084 | – | – |
| Tue | Wed | 238 Brute Force | 0.034 / 0.995 | 0.034 / 0.997 | 0.964 | 0.994 |
| Wed | Tue | 129 DoS | 0.000 / 0.036 | 0.005 / 0.379 | 0.613 | 0.855 |
| Thu | Wed | 118 Web Attack, 11 Infiltration | 0.021 / 0.996 | 0.040 / 0.509 | 0.814 | 0.923 |
| Fri | Thu | 409 Botnet, 44 DoS, 34 Recon | 0.244 / 0.003 | 0.247 / 0.009 | 0.257 | 0.550 |

- **Monday** is attack-free, and the false-positive rates are 4.0% (logistic regression) and 8.4% (LSTM).
- **Thresholds don't transfer across days or classes.** Thresholds tuned on another day's attack collapse to about 0 for logistic regression on Tuesday and Thursday, so it flags nearly everything.
- **The LSTM overfits immediately.** Its best checkpoint is epoch 1 in 4 of 5 folds, and epoch 3 in the Friday fold.

## Zero-shot classes (never seen in training)

| Fold | Class | Positives (onset) | Hosts | Campaigns / episodes | Logistic regression detected, ROC-AUC | LSTM detected, ROC-AUC | Test FPR (LR / LSTM) | Where detected (LR / LSTM) | ROC-AUC vs 172.16.0.1 negatives (LR / LSTM) |
|---|---|---|---:|---|---|---|---|---|---|
| Tue | Brute Force | 238 (0) | 1 | 1 / 2 | 238/238, 0.791 | 237/238, 0.617 | 0.995 / 0.997 | all on 172.16.0.1 | 0.977 / 0.969 (18 negatives) |
| Thu | Web Attack | 118 (0) | 1 | 1 / 1 | 118/118, 0.982 | 118/118, 0.974 | 0.996 / 0.509 | all on 172.16.0.1 | 0.868 / 0.915 (43 negatives) |
| Thu | Infiltration | 11 (6) | 1 | 1 / 1 | 11/11, 0.953 | 11/11, 0.915 | 0.996 / 0.509 | other hosts | – |
| Fri | Botnet | 409 (67) | 7 | 1 / 7 | **0/409**, 0.381 | **6/409**, 0.497 | 0.003 / 0.009 | other hosts | – |
| Fri | Reconnaissance | 34 (0) | 1 | 1 / 2 | 30/34, 0.989 | 34/34, 0.998 | 0.003 / 0.009 | all on 172.16.0.1 | 0.534 / 0.603 (29 negatives) |

- **Brute Force, Web Attack and Infiltration** are "detected" only at false-positive rates of 51–99.7%. That is not detection.
- **Reconnaissance** is detected at a low false-positive rate, but within host 172.16.0.1 its ranking is close to chance. That's host recognition.
- **Botnet**, the one clean zero-shot test (7 internal victim hosts, no attacker-host confound), gets 0 of 409 and 6 of 409, with ROC-AUC ≤ 0.50. There's no transfer.
- **Infiltration** has 11 positives, so it is just above the 10-positive "insufficient data" cut-off.

## Implication for the full 2.83M run

The full dataset adds rows. It doesn't add days, campaigns, attacker hosts or victim hosts. DoS stays a single attacker→victim pair on two days.
- The Friday → Wednesday failure is a failure to rank, not a lack of samples.
- The Wednesday → Friday result is limited by the host confound, not by sample size.

More data would only enlarge the 29-sequence post-attack "benign" set on 172.16.0.1, which isn't a clean control either. The next useful step is a host-controlled representation, such as the deferred per-host normalisation or windows keyed by victim host. Scaling up is not the next step.
