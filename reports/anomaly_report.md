# Benign-only anomaly baseline

Question: can a model that never sees an attack label detect attacks on this data? The supervised models can learn "which host" instead of "what attack", because nearly every attack class comes from one attacker host (172.16.0.1). These detectors are fit only on benign traffic, so labels cannot teach them that shortcut.

- **Isolation forest**: 300 trees on the flattened, preprocessed input sequence (10 windows by 28 features).
- **Autoencoder**: dense layers 280, 64, 16, 64, 280. The score is reconstruction MSE.

**Fit set:** 50,645 train-split sequences whose target is benign and whose 10 input windows contain no attack flow. 66 sequences with a benign target but attack traffic in their inputs were dropped. Inputs, preprocessing and split are the same as for the committed model, and label-derived features are excluded. The committed LSTM is read from `audit_predictions.csv` and is not retrained, so the demo model and its numbers are unchanged.

To reproduce: `python scripts/evaluation/anomaly_eval.py`. Raw numbers are in `reports/anomaly_eval.json`.

## Test split (487 positive targets, 10,892 negative)

| Model | Labels used | ROC-AUC | PR-AUC | Precision / recall at 2.5% FPR |
|---|---|---:|---:|---|
| LSTM (committed) | yes | 0.553 | 0.177 | 0.226 / 0.162 |
| Isolation forest | none | **0.640** | 0.116 | 0.196 / 0.136 |
| Autoencoder | none | 0.617 | 0.056 | 0.015 / 0.008 |

**How the 2.5% FPR is set:** the LSTM figure is the committed model at its stored threshold, and its test FPR comes out at 2.5%. For the anomaly models, each threshold is set so the test-split benign FPR equals that same 2.5%. That makes recall comparable at equal false alarms, but it is not a threshold you could deploy. The label-free operating point (99th percentile of the fit-set scores) is in the JSON. At that point the isolation forest has precision 0.45, recall 0.12 and FPR 0.7%.

**Seed stability (test split):**
- Isolation forest, 5 seeds: ROC-AUC 0.634 to 0.651, PR-AUC 0.107 to 0.116.
- Autoencoder, 3 seeds: ROC-AUC 0.615 to 0.626, PR-AUC 0.055 to 0.058.

## Host check (test split, 2.5% FPR)

| | LSTM | Isolation forest |
|---|---|---|
| Benign sequences flagged on 172.16.0.1 | 29/29 | 10/29 |
| Attacks detected on 172.16.0.1 | 75/76 | 59/76 |
| Within-host ROC-AUC on 172.16.0.1 | 0.772 | 0.780 |
| ROC-AUC on other hosts | 0.472 | **0.585** |
| Botnet detected (the only class with several attacker hosts) | 4/409 | 5/409 |
| DoS detected | 42/44 | 44/44 |
| Reconnaissance detected | 33/34 | 17/34 |

## Classes never seen in training (all splits, isolation forest at the 2.5% test-FPR threshold)

No attack label was used to fit the model, so attacks from every split can be scored. Each class is compared against held-out benign traffic (val and test).

| Class | Detected | ROC-AUC vs held-out benign |
|---|---:|---:|
| Denial of Service | 176/185 | 0.994 |
| Infiltration | 7/11 | 0.898 |
| Reconnaissance | 17/39 | 0.873 |
| Brute Force | 12/238 | 0.870 |
| Web Attack | 0/130 | 0.854 |
| Botnet | 5/409 | 0.690 |

**Caveat:** these per-class AUCs compare attacks from all five days against benign traffic from Thursday and Friday only, so day-to-day differences can inflate them. The test-split tables above do not have this problem.

## Reading

- **The ranking is better and less host-driven.** Test ROC-AUC rises from 0.553 to 0.64, and on hosts other than 172.16.0.1 it moves from below chance (0.47) to 0.585. The isolation forest flags 10 of the attacker host's 29 benign sequences, against all 29 for the LSTM. It never saw a label, so this gain cannot come from memorising the host.
- **Operating-point detection is not better.** PR-AUC is lower (0.116 against 0.177), and at equal false alarms recall is 13.6% against 16.2%. Botnet, the one class with several attacker hosts, is still almost entirely missed (5/409).
- **What it finds is volume.** DoS is caught almost completely without labels. Low-volume attacks (web attacks, brute force, botnet beacons) mostly sit inside the benign tail.
- **The attacker host is still unusual on its own.** The median benign sequence on 172.16.0.1 already scores above 88% of held-out benign traffic. Part of the remaining 172.16.0.1 result is therefore still "this host looks odd", though now measured from behaviour rather than learned from labels.
- **The autoencoder is weaker** than the isolation forest on every measure except Reconnaissance and Web Attack AUC. At usable thresholds it detects almost nothing.

**Conclusion:** a label-free detector is an honest second number for the deck. It ranks attacks better than the supervised LSTM (ROC-AUC 0.64 against 0.55) and does not depend on the attacker-host shortcut learned from labels. It does not raise detection at a usable alert rate, and it does not solve Botnet. The data limit from `hostrel_report.md` still applies: more attacker hosts and campaigns are needed (B13). Nothing here is wired into training, the backend or the dashboard.
