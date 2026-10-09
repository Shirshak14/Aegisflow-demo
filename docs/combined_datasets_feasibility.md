# Combining CIC-IDS2017 with CTU-13: feasibility (2026-10-09)

Question: can CTU-13 be added to CIC-IDS2017 so that AegisFlow can be trained and tested for attack **forecasting** across datasets? This is a feasibility study. No model was trained; the demo model, its threshold, the default dataset and `backend/` are unchanged.

Evidence: `scripts/ctu13_onset_census.py` → `reports/ctu13_onset_census.json` (all 13 scenarios, 19,976,700 flows), the opt-in adapter `aegisflow/ml/datasets/ctu_13.py` with `tests/test_ctu_13_adapter.py`, and a training-free per-feature comparison of host windows (section 4).

## Answer

**No-go for onset forecasting. Conditional go for two narrower uses.**

- CTU-13 adds **zero** usable attack onsets for the question the onset study asked (a host with its own benign traffic that then starts attacking). In all 35 infected-host captures, every flow the infected host sends is labelled Botnet; the host has no non-botnet traffic of its own before onset. Combined with CIC-IDS2017 the count stays at 8, and the bar in `onset_forecasting_report.md` §5 was at least 30 starts across at least 3 campaigns.
- **Go (about 3 to 4 days):** cross-dataset *next-window risk* for the Botnet class (CIC "Bot" with CTU "Botnet"), evaluated leave-one-scenario-out and leave-one-dataset-out with the controls in section 6. That tests whether the demo model's actual task transfers, which nothing has tested so far.
- **Go only as a separate pre-registered study (about 4 to 5 days):** *escalation* forecasting inside CTU-13, i.e. predicting when an infected host moves from C2/DNS traffic to spam, click fraud or ICMP flooding. There are 46 such events on 26 host captures, but they come from 8 independent scenarios and the 10-host scenarios are commanded in lockstep, so the effective sample is about 8.

## 1. Source, size, licence, what is on disk

| Item | Value |
|---|---|
| Official page | https://www.stratosphereips.org/datasets-ctu13 |
| Files used | 13 × `detailed-bidirectional-flow-labels/capture2011081x*.binetflow` from `mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42..54` |
| Size | 2.73 GB uncompressed (HTTP Content-Length, summed); stored gzipped as 460 MB in `data/raw/ctu_13/` (gitignored). All-in-one archive: `CTU-13-Dataset.tar.bz2`, 1.9 GB, also contains pcaps. |
| Format | CSV, 15 columns: `StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label`. Bidirectional Argus flows. |
| Licence | Not stated in the page text. The page metadata names CC BY 2.0, which may refer to the page rather than the data (inferred, unverified). The page asks for the citation below. Raw files stay out of git either way. |
| Citation | Garcia, Grill, Stiborek, Zunino (2014). An empirical comparison of botnet detection methods. Computers & Security 45, 100-123. |

The per-scenario READMEs list the infected hosts; all 13 used 147.32.84.165, and scenarios 9 to 12 added 147.32.84.191-193 and .204-.209.

## 2. Field mapping and which model features survive

Adapter: `aegisflow/ml/datasets/ctu_13.py`, selected only with `--dataset ctu_13` (`configs/datasets.yaml`, `status: opt-in prototype`). `active_dataset` stays `cic_ids2017`.

| Canonical column | CTU-13 source | Note |
|---|---|---|
| timestamp | StartTime | `%Y/%m/%d %H:%M:%S.%f`, Prague local time, kept naive like CIC |
| source_ip / destination_ip | SrcAddr / DstAddr | |
| source_port / destination_port | Sport / Dport | decimal; hex for ICMP type/code (`0x0303`); empty for ARP etc. → NA |
| protocol | Proto | upper-cased (`TCP`, `UDP`, `ICMP`, `ARP`, ...) |
| flow_duration | Dur | seconds, same unit as CIC after its µs → s conversion |
| packet_count | TotPkts | both directions, like CIC fwd+bwd |
| byte_count | TotBytes | **includes headers**; CICFlowMeter counts payload |
| fwd_byte_count / bwd_byte_count | SrcBytes / TotBytes − SrcBytes | |
| fwd/bwd_packet_count | none | NA |
| packet_size_mean | TotBytes / TotPkts | not CIC's payload "Packet Length Mean" |
| syn/ack/rst/fin/psh/urg_count | letters in Argus `State` (`FSPA_FSPA`) | 1 if seen in either direction, else 0; 0 for non-TCP. Presence, not counts |
| packet_size_std/min/max, inter_arrival_*, tcp_window_size, ttl_*, retransmission_count | none | NA, never filled |
| dataset_label | family of `Label`: Botnet / Normal / Background | raw label kept in `ctu_label`, scenario in `ctu_scenario` |

Of the 28 window features the model uses (`MODEL_FEATURES`):

| Status | Features |
|---|---|
| **Clean (10)** | flow_count, packet_count_sum, packet_count_mean, duration_mean, packets_per_second_mean, unique_destination_ips, unique_destination_ports, unique_source_ports, destination_port_entropy, connection_burst_max |
| **Degraded (15)**, different meaning across datasets | byte_count_sum, byte_count_mean, bytes_per_second_mean, bytes_per_packet_mean, packet_size_mean (header vs payload bytes); syn/ack/rst/fin/psh_count_sum and the five flag ratios (presence vs per-packet counts) |
| **Lost (3)** | packet_size_std_mean, inter_arrival_mean, tcp_window_size_mean |

Two structural differences matter beyond single columns: Argus and CICFlowMeter cut flows with different timeouts (CTU flows reach thousands of seconds), and CTU is a live university network with millions of Background flows from outside hosts, where CIC is a closed lab LAN.

## 3. Onsets per scenario and per infected host

Source: `reports/ctu13_onset_census.json`. "Own history" means non-botnet flows **sent by** the host before its first botnet flow, which is what AegisFlow's `source_ip` windows see. "Inbound" means flows **to** the host before onset. An episode is a run of botnet flows with no gap over 30 minutes (same rule as the onset study). "Escalation" is the first spam, click-fraud or ICMP flow after at least 10 minutes of other botnet traffic from that host.

| Scen. | Bot | Capture (h) | Infected hosts | Onset, min after capture start | Own history | Inbound before onset | Episodes | Escalations |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | Neris | 6.1 | 1 | 77.5 | 0 | 1 | 1 | 1 |
| 2 | Neris | 4.2 | 1 | 37.7 | 0 | 0 | 1 | 1 |
| 3 | Rbot | 66.8 | 1 | 29.3 | 0 | 1 | 9 | 0 |
| 4 | Rbot | 4.5 | 1 | 17.3 | 0 | 1 | 2 | 2 |
| 5 | Virut | 0.5 | 1 | 9.5 | 0 | 1 | 1 | 0 |
| 6 | DonBot | 2.2 | 1 | 0.4 | 0 | 0 | 1 | 0 |
| 7 | Sogou | 0.4 | 1 | 1.9 | 0 | 1 | 1 | 0 |
| 8 | Murlo | 19.5 | 1 | 0.1 | 0 | 0 | 2 | 1 |
| 9 | Neris | 5.6 | 10 | 53.9 to 67.9 | 0 | 10 | 35 | 25 |
| 10 | Rbot | 5.1 | 10 | 41.5 to 65.2 | 0 | 2 | 26 | 14 |
| 11 | Rbot | 0.3 | 3 | 8.4 to 8.5 | 0 | 3 | 3 | 0 |
| 12 | NSIS.ay | 1.7 | 3 | 39.7 to 47.0 | 0 | 1 | 3 | 1 |
| 13 | Virut | 16.4 | 1 | 3.1 | 0 | 1 | 1 | 1 |
| **Total** | | | **35** (10 IPs) | | **0** | **22** | **86** | **46** (26 hosts) |

What this means for forecasting:

- **No onset has own benign history.** The labelling marks every flow from an infected IP as Botnet, so the first window a source-grouped host ever has is an attack window. This is the same situation as 172.16.0.1 in CIC-IDS2017, which the onset study had to keep in training only.
- **Inbound history is thin.** 22 of 35 hosts receive something before onset, but only 1 receives 10 or more flows; mostly it is a few broadcast or scan packets.
- **The 51 restarts are not onsets either.** None has any own traffic in the 10 minutes before it, because the host is silent between episodes.
- **Onset time is scripted.** Onsets fall 0.1 to 78 minutes after capture start, and in scenarios 9 and 10 the ten hosts start one after another within 14 and 24 minutes. "Minutes since capture start" plays the role that "Friday 10 a.m." played in CIC-IDS2017.
- **Hosts are not independent across scenarios.** 147.32.84.165 is the infected host in all 13 captures, and ten IPs cover all 35 host captures. The 40 internal hosts with Normal labels also recur. Any split by host instead of by scenario leaks.
- **Escalations are the only within-host forecasting events**, and they come from 8 scenarios (1, 2, 4, 8, 9, 10, 12, 13). In scenarios 9 and 10 the C2 server commands all ten bots together, so their 39 escalations are close to two events.

## 4. Dataset identity is trivially visible

Training-free check: for each of the 28 model features, the ROC-AUC of that feature alone at separating CIC-IDS2017 host windows (132,460 windows from `data/processed/cic_ids2017/host_windows.parquet`) from CTU-13 windows (scenarios 5 and 7, internal source hosts, 9,850 windows built with the normal pipeline). 0.5 means indistinguishable.

| Feature | AUC | CIC median | CTU median |
|---|---:|---:|---:|
| inter_arrival_mean (lost in CTU) | 0.999 | 86 | 0 |
| tcp_window_size_mean (lost) | 0.915 | 349 | 0 |
| packet_size_mean (degraded) | 0.904 | 6 | 145 |
| bytes_per_packet_mean (degraded) | 0.900 | 6 | 145 |
| byte_count_mean (degraded) | 0.851 | 14 | 896 |
| ack_ratio (degraded) | 0.827 | 0.5 | 0.008 |
| duration_mean (clean) | 0.783 | 0.0001 s | 3.5 s |
| packets_per_second_mean (clean) | 0.752 | 29,412 | 2,216 |
| ...lowest: ack_count_sum, unique_destination_ports, destination_port_entropy | 0.53 to 0.61 | | |

One missing feature alone identifies the dataset almost perfectly, and even the clean features separate at 0.6 to 0.78 because the flow meters cut flows differently. A combined model would learn "which dataset" first, and because the Botnet share differs between the two datasets, dataset identity is also correlated with the label.

## 5. Label harmonisation and time handling

Labels (`configs/stages.yaml: ctu_13`):

| CTU family | normalized_attack_class | attack_stage | confidence |
|---|---|---|---|
| Botnet (`From-Botnet-*`, includes C2, DNS, spam, click fraud, ICMP) | Botnet | Command and Control | inferred |
| Normal (`*-Normal-*`) | Benign | Benign | ground_truth |
| Background (everything else) | Benign | Benign | inferred: unverified traffic, not confirmed benign |

- CIC "Bot" and CTU "Botnet" are the only shared attack class. CIC's DoS, PortScan, brute force, web attacks and infiltration have no CTU counterpart, and CTU's spam and click fraud have none in CIC. Cross-dataset work should use **Botnet vs non-Botnet** only; other CIC classes must be dropped or masked, not relabelled.
- For training, Background windows are safer excluded or down-weighted than treated as benign: they include unlabelled infections elsewhere on the campus.
- If an escalation study is run, the fine activity can be read from `ctu_label` (`-CC`, `SPAM`/`SMTP`, `HTTP-Ad`, `ICMP`) without changing the adapter.

Time:

- Both datasets use naive local time (CIC: New Brunswick, July 2017, working hours; CTU: Prague, August 2011, any hour). Wall-clock features must not be shared across datasets; they are a dataset fingerprint.
- Windows stay 60 s with a 30 s stride, built **per scenario**: CTU captures 4, 5 and 13 overlap in calendar time but are separate captures, so their windows must never be merged.
- Use **minutes since capture start** as CTU's time control, alongside hour of day and day of week.
- Scale: about 20 M flows in all. Load per scenario and keep internal (147.32.0.0/16) source hosts for windowing, which drops most Background rows. The adapter reads one file at a time, but scenario 3 (4.7 M rows) still needs a few GB of RAM.

## 6. Evaluation plan (for the "go" items)

Botnet next-window risk across datasets, Botnet vs non-Botnet only, features restricted to the 25 present in both and normalised **per capture** (rank or quantile transform within each CIC day / CTU scenario), so that absolute byte and duration scales cannot carry dataset identity:

1. **Leave-one-scenario-out (CTU only).** 13 folds; validation on a different scenario than test. No host-level split, because the same IPs recur.
2. **Leave-one-dataset-out.** Train on CIC, test on all of CTU; train on CTU, test on CIC Friday morning (the only CIC Botnet day). Report per scenario as well as pooled.
3. **Dataset-identity control.** Train the same model to predict *dataset* from the same features. If it scores well above 0.5 after normalisation, the risk model's cross-dataset numbers are suspect. As an extra check, train with dataset as the label and see whether its score alone explains the risk model's ranking (correlation between the two scores on test).
4. **Clock-only control.** Gradient boosting on hour, weekday and (CTU) minutes since capture start, exactly as in `onset_forecasting_report.md`. Any traffic model has to beat it by a pre-stated margin.
5. **Shifted-label control.** Labels moved to random times within the same capture; should sit at chance.
6. **Continuation split.** Report separately the targets where the input windows already contain Botnet traffic and the first-window targets, so detection of ongoing activity is not reported as forecasting.

Pre-state the bars before any training, as the onset study did. Escalation forecasting would reuse `scripts/onset_forecast_study.py` with anchors on botnet windows before the escalation, folds by scenario (8 folds), and "minutes since infection" added to the time control.

## 7. Effort

| Step | Estimate |
|---|---|
| Adapter, fixtures, tests (done, opt-in) | done |
| Per-scenario ingestion run, internal-host filter, windows for all 13 scenarios | 1 day |
| Per-capture feature normalisation and shared feature list (opt-in, demo untouched) | 0.5 day |
| Pre-registered design doc + LOSO/LODO script with the five controls | 1.5 to 2 days |
| Escalation study (optional, separate) | 4 to 5 days |

## 8. Recommendation

Do not build a combined dataset for onset forecasting: CTU-13 cannot supply a single benign-to-attack onset, so the onset study's conclusion stands. If cross-dataset work is still wanted, run the Botnet next-window-risk transfer test (about 3 to 4 days) with the dataset-identity and clock controls. For real onset forecasting, look at data where hosts have long benign histories before compromise (LANL 2015, DARPA OpTC), as `onset_forecasting_report.md` §5 already suggests.
