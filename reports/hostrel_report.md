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

## Data limitation: why the next step is not simply "more data" (B13 scoping, 2026-10-04)

The host-relative result above is negative, and the likely reason is the data, not the representation. Every candidate source below was checked for the fields this pipeline needs: a source IP, a timestamp, and enough distinct attacker hosts per class that a model cannot learn "which host" instead of "what attack".

**Method.** CIC-IDS2017: all 8 raw CSVs read in full (3,119,345 rows; raw rows, before cleaning). CSE-CIC-IDS2018: one daily CSV downloaded and inspected (2 March, the botnet day). CTU-13: the 13 scenario READMEs read, plus two scenario flow files downloaded (52 and 53, about 60 MB) and checked against them. UNSW-NB15: documentation only; the download is gated and I could not open it. Nothing was re-ingested and no pcap was touched.

| Source | Source IP + timestamp? | Attacker hosts per class | Verdict |
|---|---|---|---|
| CIC-IDS2017, full 3.1M rows (already local) | Yes | Every attack class has exactly 1 victim. DoS (4 subtypes), PortScan, FTP/SSH-Patator, Web Attack (3 subtypes) and Heartbleed all come from 172.16.0.1 alone. DDoS has 2 source hosts (172.16.0.1 sends 128,024 of 128,027 flows). Infiltration has 1 (192.168.10.8). Bot has 8. | Adds classes and rows, not hosts. Re-ingesting it would not remove the shortcut. |
| CSE-CIC-IDS2018 processed CSVs (checked 1 of 10 daily files) | **No source IP**: the 80 columns have only `Dst Port`, `Timestamp` and `Label`. Timestamps are 12-hour with no AM/PM marker (first row 08:47, last row 02:08), and the file stops at 1,048,575 rows (the spreadsheet row limit, so possibly truncated; not verified). | Cannot be counted. | Unusable for per-host windowing as downloaded. The raw pcaps would need flow re-extraction, a much larger job that has not been scoped. |
| CTU-13 (checked) | Yes (`SrcAddr`, `DstAddr`, `StartTime`, `LastTime`) | 13 scenarios, 7 malware families (Neris, Rbot, Virut, DonBot, Sogou, Murlo, NSIS.ay). Infected hosts per scenario: 1 in nine scenarios (42-49, 54), 3 in two (52, 53), 10 in two (50, 51). Across all scenarios only **10 distinct infected IPs**, all on 147.32.84.x, reused across scenarios. | See below: host identity equals the label. Botnet only, so it adds no new classes. |
| UNSW-NB15 | Not verified (gated download) | Not verified. The testbed documentation says 45 IPs over 3 networks and one of the three virtual servers generated all attacks, which suggests very few attacker hosts. | Unknown. Do not plan around it without opening a CSV. |

**CTU-13 is not a fix for the host shortcut; it is the same problem in a stronger form.** Labels are assigned by IP address. In both downloaded scenarios, every flow sent by an infected host is labelled `From-Botnet` and every flow sent by a normal host is `From-Normal` (0 exceptions in 8,164 and 2,168 infected-host flows). A model can reach perfect accuracy by memorising the IP. Only 3 hosts are reliably normal (147.32.84.134, .164, .170), plus 3 servers the dataset itself calls unreliable. About 60% of flows are `Background`, which is unlabelled traffic, not benign. Scenario 52 lists 3 infected hosts but one sent only 7 flows.

**Conclusions.**
1. The weakness is structural. In the data we can use today, an attack class is almost always one attacker host, and in CTU-13 a host is its own label.
2. A host-relative representation cannot create host diversity that is absent from the data, which fits the small effect measured above.
3. Of the sources checked, none currently provides many independent attacker hosts per class together with usable IPs. CSE-CIC-IDS2018 is the only one with many attacker machines (50, per the dataset page), but only via raw pcaps.
4. Any future claim that this model detects attacks, rather than recognises hosts, needs a host-disjoint evaluation: train and test on different attacker hosts. This data cannot provide one for most classes.

**Not done.** No full CIC-IDS2017 re-ingest, no pcap extraction, no UNSW-NB15 inspection, and only one CSE-CIC-IDS2018 day and two CTU-13 scenarios were opened.
