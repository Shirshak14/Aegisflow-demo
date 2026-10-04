# B1 experiment: host-relative inputs

Question: does expressing each window relative to the same host's own past (causal rolling median, last 240 windows, log space) remove the attacker-host shortcut? Same hyperparameters, seed and split as the committed model. Trained into `artifacts/experiments/cic_ids2017_hostrel`; the demo model is untouched. Reproduce: `.venv/Scripts/python.exe scripts/hostrel_eval.py` (raw numbers in `reports/hostrel_eval.json`).

## Test split (487 positive targets, 10,892 negative)

| Model | Inputs | ROC-AUC | PR-AUC | Precision / recall | FPR |
|---|---|---:|---:|---|---:|
| LSTM | absolute (committed) | 0.553 | 0.177 | 0.226 / 0.162 | 0.025 |
| LSTM | host-relative | 0.610 | 0.112 | 0.169 / 0.109 | 0.024 |
| Logistic regression | absolute (committed) | 0.473 | 0.143 | 0.642 / 0.107 | 0.003 |
| Logistic regression | host-relative | 0.437 | 0.046 | 0.042 / 0.986 | 0.995 |

## Host check (LSTM)

| | absolute | host-relative |
|---|---|---|
| 172.16.0.1 benign flagged | 29/29 | 29/29 |
| 172.16.0.1 attacks detected | 75/76 | 50/76 |
| 172.16.0.1 ROC-AUC (within host) | 0.772 | 0.545 |
| Other hosts, ROC-AUC | 0.472 | 0.559 |
| Botnet detected | 4/409 | 3/409 |

## Reading

- The host shortcut is weaker: within-host ROC on 172.16.0.1 drops from 0.77 to 0.55, and the off-host ranking moves from below chance to slightly above (0.47 to 0.56). That is the intended effect.
- It did not become detection. Botnet is still 3/409, all 29 benign sequences on 172.16.0.1 are still flagged, and PR-AUC is lower (0.177 to 0.112). The ROC gain is small and rests on one fold and one split.
- Logistic regression degenerated: the validation-selected threshold flags 99.5% of test sequences. Validation has only 127 positives from 2 episodes, so the threshold is fragile (see B9/B13).
- Not tested: per-host scaling by MAD, other lookbacks, and a leave-one-day-out rerun of `lodo_pilot.py`.

Conclusion: host-relative inputs remove some of the shortcut but are not enough on this data (one attacker host, one campaign per class). This supports the earlier finding that more hosts and campaigns are needed (B13). The committed demo model and its numbers are unchanged.
