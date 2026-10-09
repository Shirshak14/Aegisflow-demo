# Attack-onset forecasting data: LANL 2015 and DARPA OpTC (2026-10-09)

Question: CIC-IDS2017 and CTU-13 together give 8 usable attack onsets (`onset_forecasting_report.md`, `combined_datasets_feasibility.md`). Can LANL 2015 or DARPA OpTC give enough **independent attack starts on hosts with their own benign traffic beforehand** to test real onset forecasting? The bar from `onset_forecasting_report.md` §5 is at least 30 starts across at least 3 campaigns, with no single start time shared by most of them.

This is a data census. No model was trained; the demo model, its threshold, the default dataset (`cic_ids2017`) and `backend/` are unchanged. Both adapters are opt-in prototypes.

Evidence: `scripts/lanl_onset_census.py` → `reports/lanl_onset_census.json` (all 749 red-team events, all 129 M flows scanned); `scripts/optc_onset_census.py` → `reports/optc_onset_census.json` (all 31 red-team host onsets; flow history for the victims whose attack-day traffic has been pulled, see §3.4); `reports/optc_tar_index.json` (every host file in all ten OpTC archives). Adapters: `aegisflow/ml/datasets/lanl_2015.py`, `aegisflow/ml/datasets/optc.py`, tests in `tests/test_lanl_2015_adapter.py`, `tests/test_optc_adapter.py`.

## Answer

**LANL 2015: go, with conditions. OpTC: no-go as a forecasting benchmark, useful as an external check.**

- **LANL** clears the bar on count: 102 victims had their own flows in the 10 minutes before their first red-team logon, forming 98 distinct onset moments and 23 bursts (onsets chained within 30 minutes) over 9 different days. Victims have a median of 8.5 days of flow history before onset. But it is one red team (701 of 749 events come from one machine), 55 of those 102 onsets fall on one day, and only 35 victims show any attacker traffic before onset. It is the only dataset of the four that can run the onset study with enough events to be significant.
- **OpTC** has real benign history (every victim workstation has 1 to 9 days of its own telemetry before onset, 27 of 29 have 5 or more), which CIC and CTU never had. But 31 host onsets collapse to **13 independent onset moments in 3 campaigns on 3 consecutive days**: the red team hit 14 hosts in one second on day 1 and 6 hosts in one second on day 2. 13 is under the bar of 30, and every onset is a weekday between 10:27 and 15:41. Use it as an out-of-distribution check of whatever the LANL study finds, not as the main benchmark.

## 1. What each dataset is, and what is on this laptop

| | LANL 2015 | DARPA OpTC (corrected) |
|---|---|---|
| Official page | https://csr.lanl.gov/data/cyber1/ | https://github.com/FiveDirections/OpTC-data; corrected release doi:10.57745/UXCWOC |
| Licence | CC0 | Distribution A (original); CC BY 4.0 (corrected) |
| Content | 58 days of a real enterprise network: auth (7.2 GB gz), process (2.2 GB), router flows (1.1 GB), DNS (177 MB), red team (4.8 KB) | 2019 exercise on ~1,000 Windows hosts (telemetry from up to 595): eCAR endpoint events (process, file, registry, flow) per host and day, plus Zeek on Google Drive |
| Full size | 12 GB compressed | about 940 GB (10 tar archives, 13 to 125 GB each) |
| Access | Short form on the official page (email + intended use). The Imperial College mirror (lanl.ma.ic.ac.uk) serves the same files directly; that is where the copy here came from. **Please fill in the official form once** so LANL knows the data is being used. | Public. Archives are uncompressed tars on an S3 store that honours HTTP Range, so single host-days can be pulled without the rest (`scripts/optc_remote.py`). |
| On disk (gitignored) | `data/raw/lanl/flows.txt.gz` (1.08 GB), `redteam.txt.gz` | `data/raw/optc/labels/` (ground truth CSVs, Zeek labels, red-team PDF, 4.5 MB); `data/raw/optc/flows/` (FLOW events of victim attack days up to onset + 30 min, about 70 MB per host) |
| Labels | 749 red-team **authentication** events (time, user, source, destination). No flow labels. | No official labels. Corrected host labels (Majorczyk et al. 2025): red-team process ids and lifetimes per host, children included; Zeek conn logs of malicious traffic per host |
| Time | Seconds from an undisclosed start; real hour/weekday unknown | Real local time (US Eastern, -04:00), 2019-09-16 19:37 to 2019-09-25 |

Download speed here was 0.5 to 0.9 MB/s. The LANL flows took about 30 minutes; pulling one OpTC host-day takes 3 to 5 minutes.

## 2. LANL 2015 onset census

Source: `reports/lanl_onset_census.json`. Onset = a computer's first red-team authentication. "Own flows" = flows with the victim as **source**, which is what AegisFlow's source-grouped windows see. Router flows cover seconds 1 to 3,126,928 (36 days), so every red-team onset (day 1.75 to 29.5) falls inside them.

| Item | Number |
|---|---:|
| Red-team events / victims / users | 749 / 301 / 104 |
| Red-team source computers | 4 (C17693 sends 701 of 749) |
| Distinct onset moments (chained within 60 s) / bursts (within 30 min) | 242 / 35 |
| Largest batch within 60 s | 5 |
| Victims that appear in router flows at all | 209 |
| ...with own flows before onset | 196 |
| ...with own flows in the 60 min before onset | 122 |
| **...with own flows in the 10 min before onset** | **102** (98 distinct moments, 23 bursts) |
| Median days of flow history before onset | 8.5 |
| Victims with attacker → victim flows before onset | 35 (all within the hour before; 1 to 1,122 flows each) |

Onsets per day (all 301 / the 102 usable):

| Day | 2 | 3 | 6 | 7 | 8 | 9 | 13 | 14 | 15 | 16 | 22 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All victims | 10 | 4 | 7 | 2 | 1 | 170 | 63 | 27 | 7 | 7 | 2 | 1 |
| Own flows 10 min before | 4 | 1 | 5 | 1 | 1 | 55 | 20 | 10 | 5 | 0 | 0 | 0 |

What this means:

- **Count is enough; independence is the weak point.** 98 moments in 23 bursts over 9 days is well above the bar, and no 60-second batch is larger than 5. But one red team drives everything, and day 9 alone holds 55 of the 102. Folds must be by burst or by day, not by host, and results reported with and without day 9.
- **Victims are chosen by the attacker, so the victim's own traffic should not predict its compromise.** The plausible precursors are network-level: the attacker's contact with the victim (visible for only 35 victims, and only within the hour before), DNS lookups of the victim, and the victim's neighbours being compromised earlier. A study on LANL should include those (own+net features, as in the onset study) and expect own-traffic-only models to sit at chance. That is a valid result, not a failure.
- **Labels are authentications; features are flows.** The onset time comes from auth.txt, the windows from flows.txt. The adapter labels a flow `RedTeam` when it goes from a red-team source to a computer it compromised, from one hour before that compromise (`contact_slack_s`). The 92 victims not in router flows can only be studied with auth.txt features (7.2 GB, not downloaded).
- **Clock control still applies.** Real hour and weekday are hidden, but relative time keeps daily periodicity, and red-team activity clusters in working bursts. The time-only control should use "seconds since start mod 86,400" and day index.

## 3. DARPA OpTC onset census

Source: `reports/optc_onset_census.json`, `reports/optc_tar_index.json`.

### 3.1 Onsets

Onset = start of the host's first red-team process in the corrected ground truth. The two DC rows (`dc1`) have no telemetry in the corrected release.

| Scenario (day) | Host onsets | Distinct onset moments | Largest batch | Onset clock times |
|---|---:|---:|---:|---|
| 1, PowerShell Empire (Mon 23 Sep) | 18 (17 workstations + DC) | 5 | **14 hosts in 3 s** at 14:44:53 | 11:23 to 14:44 |
| 2, DeathStar / custom malware (Tue 24 Sep) | 11 (10 + DC) | 6 | **6 hosts in 2 s** at 15:41:17 | 10:35 to 15:41 |
| 3, malicious software upgrade (Wed 25 Sep) | 2 | 2 | 1 | 10:27 to 11:17 |
| **Total** | **31** (29 workstations) | **13** | | |

### 3.2 Benign history (from the archive index, no download needed)

Telemetry was rolled out progressively: 350 hosts on 16 and 17 Sep, 595 on 18 Sep, 475 on 19 to 22 Sep, 500 on 23 to 25 Sep. Of the 29 victim workstations, 27 have telemetry on 5 or more days before their onset day; sysclient0005 and 0010 have 1. A host-day file is 140 to 280 MB of gzip.

### 3.3 Flow pipeline

eCAR records every flow the host sensor sees, inbound and outbound, with process id and image path. The adapter keeps one row per flow object (first `FLOW START`), joins byte totals from `FLOW MESSAGE`, drops the sensor's own Kafka uplink (port 9092), and labels a flow `Malicious` when its (host, pid) is a red-team process alive at that moment. Of the 28 model features, the count, port, and destination features are computable; packet counts, flags, packet sizes, durations, inter-arrival and window are NA (eCAR has none of them). About 95% of rows are inbound LAN broadcast and multicast (NetBIOS, LLMNR, mDNS); outbound flows are a few thousand per host per day.

### 3.4 Flow history before onset

Pulled so far: the 14 scenario-1 victims whose attack day had finished streaming when this was written (the remaining 15 are still downloading; rerun `scripts/optc_onset_census.py` to refresh). "Own outbound" excludes red-team processes and the Kafka uplink.

| Host | Onset | Batch | Own outbound 10 / 60 min before | Inbound 10 min before | Minutes since last own outbound | First red-team flow after onset |
|---|---|---:|---|---:|---:|---:|
| 0201 | 11:23:54 | 1 | 38 / 337 | 3,109 | 0.03 | 9.6 s |
| 0402 | 13:23:30 | 1 | 99 / 909 | 3,301 | 0.03 | 7.4 s |
| 0660 | 13:34:17 | 1 | 20 / 113 | 3,464 | 0.01 | 5.7 s |
| 0104, 0355, 0419, 0462, 0503, 0559, 0609, 0771, 0874, 0955, 0170 | 14:44:53 to :54 | 14 | 4 to 907 / 86 to 3,115 | 3,481 to 4,065 | 0.01 to 4.9 | 7.9 to 20.7 s |

- **Every victim pulled so far has its own outbound traffic in the 10 minutes before onset** (14 of 14), and the gap since its last own flow is under 5 minutes. So the CIC/CTU problem (no history at onset) does not exist here.
- The limit is independence, not history: these 14 onsets are 4 onset moments, and 11 of them are one command fanned out in 3 seconds. The full 29 workstations can give at most 11 moments (13 minus the two DC onsets, which have no telemetry).
- Red-team flows start 6 to 21 seconds after the process onset, so the first window that contains attack traffic is the onset window itself. Any forecasting signal must come from before the red-team process exists.

## 4. Comparison with what the onset study had

| | CIC-IDS2017 | CTU-13 | OpTC | LANL 2015 |
|---|---:|---:|---:|---:|
| Usable onsets (own traffic 10 min before) | 8 | 0 | 14 of 14 pulled (29 expected) | 102 |
| Independent onset moments among them | about 3 | 0 | 4 so far; at most 11 | 98 |
| Campaigns / bursts | 2 | 0 | 3 | 23 bursts, 1 red team |
| Days onsets span | 1 (Friday) | n/a | 3 | 9 |
| Real clock time | yes | yes | yes | no (relative only) |
| Meets the bar (≥30 starts, ≥3 campaigns) | no | no | no | **yes on count**; campaign independence arguable |

## 5. Recommendation and next steps

1. **Run the onset study on LANL** with `scripts/onset_forecast_study.py`'s method, pre-registered first as before. Folds by burst (23), not by host. Report with and without day 9. Variants: own flows only, own + attacker/neighbour contact (inbound from previously compromised computers), and the time-only control on relative time. Expect own-only to be at chance; the interesting question is whether network context beats the clock. About 3 to 4 days: per-victim windows from the streamed flows (the adapter's `computers` filter keeps memory small), then the study.
2. **Use OpTC as the external check**, not a benchmark: 13 moments can confirm or contradict a LANL result, not establish one. Pulling every victim's attack day plus one benign day each is about 60 host-days, about 6 to 8 hours at this link speed. The tooling for it is done.
3. **Fill in the LANL form** at https://csr.lanl.gov/data/cyber1/ (email and intended use). The copy here came from the mirror without it.
4. auth.txt.gz (7.2 GB) is needed only if the 92 victims missing from router flows matter, or for auth-based features. Not downloaded.

## 6. Not done

No model trained. OpTC benign-day flows not pulled (only the archive index shows they exist). LANL DNS, process and auth files not downloaded. The OpTC labels are a third party's reconstruction (no official labels exist); the LANL red-team list is known to be incomplete as ground truth for "all bad behaviour" (it lists compromise authentications only).
