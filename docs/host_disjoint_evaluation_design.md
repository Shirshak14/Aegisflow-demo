# Host-disjoint evaluation: design (scoping only, nothing implemented)

Status: proposal, bars agreed (see section 6). Follows from `reports/hostrel_report.md` (B1 negative result, B13 data limitation), which is in draft PR #1 and not yet on `main`. No pcap re-extraction, no full CIC-IDS2017 re-ingest and no new dataset is assumed. All counts below come from the committed `sequences.parquet` (85,077 sequences, 1,390 hosts) and the raw-CSV count in the B1 report.

## 1. What question the evaluation must answer

The current test split cannot separate "the model recognises an attack" from "the model recognises the host that sends attacks". A valid evaluation must show that the score ranks attack above benign **for hosts the model did not learn from, or within a single host that has both labels**, and must show that it beats a baseline that only knows the host's identity.

## 2. What the data can and cannot support

Positive sequences (target is an attack window) by class and host, from the current table:

| Class | Attacker hosts | Sequences (train / val / test) | Host-disjoint test possible? |
|---|---|---|---|
| Botnet | 7 (6 internal 192.168.10.x + 205.174.165.73) | 0 / 0 / 409 | **Yes, only case.** All in the test split (Friday). |
| Denial of Service | 172.16.0.1 (+192.168.10.50, 2 seqs) | 141 / 0 / 44 | No. One real attacker host. |
| Brute Force | 172.16.0.1 | 238 / 0 / 0 | No. |
| Web Attack | 172.16.0.1 | 3 / 116 / 0 | No. |
| Reconnaissance | 172.16.0.1 | 0 / 0 / 34 | No. |
| Infiltration | 192.168.10.8 | 0 / 11 / 0 | No (11 sequences, one host). |

Hosts that have **both** benign and attack targets, which is what a within-host contrast needs:

| Host | Benign seqs | Attack seqs | Note |
|---|---:|---:|---|
| 192.168.10.15 | 3,160 | 80 | Botnet |
| 192.168.10.8 | 2,981 | 73 | Botnet and Infiltration |
| 192.168.10.5 | 3,124 | 69 | Botnet |
| 192.168.10.9 | 3,069 | 67 | Botnet |
| 192.168.10.14 | 3,129 | 56 | Botnet |
| 192.168.10.17 | 3,556 | 2 | Botnet, too few |
| 172.16.0.1 | 259 | 590 | attack-heavy; benign control is small (29 in test) |
| 205.174.165.73 | 2 | 73 | no usable benign control |

Honest limit: **Botnet is the only class where a host-controlled test is possible**, and it exists only in the test day. Onset positives are small (2 to 20 per host; 67 in total across the 6 internal hosts). Every other class stays "not evaluable for generalisation" and must be reported that way, not as detection.

## 3. Proposed design

### D1. Headline metric: host-stratified ROC-AUC
For each host with at least 20 attack and 20 benign targets, compute ROC-AUC **within that host**, then report the per-host values and their median. Never headline a pooled AUC. This is already computed in `scripts/phase3_baselines.py` (`per_host`); the change is to make it the primary number and add the minimum-count rule.

### D2. Identity baseline and shortcut gap
Score every sequence 1 if its host is a known attacker host, else 0 (the `ref_host_identity` rule used in `lodo_pilot.py`). Report **shortcut gap = model metric minus identity-baseline metric** on the same evaluation set. A model "passes" only where it beats this baseline within hosts, where the identity rule scores exactly 0.5 by construction.

### D3. Leave-one-host-out for Botnet
Five folds, one per internal Botnet host with at least 20 attack sequences (192.168.10.5, .8, .9, .14, .15). Host .17 (2 attack sequences) and the external 205.174.165.73 (2 benign sequences) stay in training only. In each fold, hold out all of that host's sequences from training and validation; train on the other hosts' Botnet positives plus benign from all non-held-out hosts; threshold chosen on a validation set that also excludes the held-out host. Report recall at a 1% false-positive rate (a reported number, not a pass bar) and per-host AUC on the held-out host.
- Caveat to state in the report: this is host-disjoint, **not campaign-disjoint**. The hosts are infected in the same Friday campaign with the same command-and-control server, so campaign-level patterns can still leak across folds. Do not describe it as zero-shot.
- Caveat: Botnet exists only on one day, so folds share the same time period. Time-ordered training (train earlier days, test Friday) remains the existing split and is complementary, not replaced.

### D4. Within-host temporal control for 172.16.0.1
Where the data allow it, compare attack windows to that host's own benign windows (259 sequences: 181 train, 43 val, 29 test). With 29 test negatives the result is a sanity check, not a headline. Report it with the count beside it.

### D5. Uncertainty
Resample by **episode** (consecutive attack windows on one host separated by under 30 minutes, as defined in `lodo_pilot.py`), not by sequence, because neighbouring sequences overlap. Report 95% intervals. With 67 onset positives in total, expect wide intervals; say so rather than hide them.

### D6. Controls that must fail
- **Host-permutation control:** shuffle host labels across sequences before training. Performance on held-out hosts should fall to chance. If it does not, the evaluation leaks.
- **Identity-only model:** a model given only a host-ID feature should score near 0.5 within-host. If it does not, the within-host metric is contaminated.

### D7. Acceptance criteria, fixed before running
1. Within-host AUC lower 95% bound above 0.5 on at least 4 of the 5 evaluable Botnet hosts.
2. Shortcut gap positive and its interval excluding 0.
3. Host-permutation control at chance (interval contains 0.5).
4. Every class that is not host-disjoint-testable is listed in the report as "not evaluable", with its host count.

The 20-sequence minimum and the 95% intervals are fixed before any run and must not be adjusted after seeing results. Recall at 1% false-positive rate is reported alongside, but is not a pass criterion.

With so few positives, a pass will be weak evidence and a fail is just as informative. Both are reported plainly. A failure means the model does not demonstrate detection beyond host recognition; that is an acceptable outcome.

## 4. What this design does not give you
- No evidence for Brute Force, Web Attack, Reconnaissance, DoS or Infiltration generalisation. One attacker host each; no design can fix that without new data.
- No campaign-level generalisation for Botnet.
- No statement about other networks or datasets.

## 5. Effort and risk (estimates from reading the code, not measured)
- D1, D2 and D6: about 1 day. They extend `phase3_baselines.py` and need no change to the pipeline or the Parquet files.
- D3 and D5: 1 to 2 days. Needs a fold builder for host exclusion, which does not exist; `lodo_pilot.py` splits by day, not by host. Training the five LSTMs is the slow part.
- Demo risk: none, provided it stays in `scripts/` and writes only under `reports/` and `artifacts/experiments/`. It should not touch `backend/`, `index.html`, `aegisflow/ml/temporal/*` or the committed model.

## 6. Decisions

Agreed:
- Keep the 20-sequence minimum and the 95% episode-level intervals as fixed pass bars.
- The 1% false-positive rate is a reported number, not a pass bar.
- Report a pass or a fail plainly.
- The full CIC-IDS2017 re-ingest is not wanted now. It would only add Botnet volume, not hosts.

Not agreed: Botnet-only host-disjoint evidence as the only headline. The team wants other classes to become evaluable as well, which this data cannot do (section 2).

## 7. Deferred: CSE-CIC-IDS2018 raw captures

The only public set found with many attacker machines (about 50, per the dataset page) is CSE-CIC-IDS2018, but only through its raw packet captures. Its processed CSVs have no source IP, so flows would have to be re-extracted with CICFlowMeter. That is a much larger job than anything in this design.

Status: **deferred, not started, not sized.** Unknown today: capture size and download time, extraction time, whether attacker IPs map cleanly to classes, and how many distinct attacker hosts each class really has. None of this has been measured. It is the route to making more classes host-disjoint-testable, and the decision to pursue it should come after the Botnet-only result above is in.
