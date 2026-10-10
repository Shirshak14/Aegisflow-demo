# More public datasets for attack-onset forecasting: feasibility (2026-10-09)

Question: CIC-IDS2017, CTU-13, OpTC and LANL 2015 all failed or were ruled out (`onset_forecasting_report.md`, `combined_datasets_feasibility.md`, and on section-c `optc_lanl_feasibility.md`, `lanl_onset_report.md`). Does any other public dataset offer what a real onset study needs?

Requirements, from the earlier studies:

1. At least 30 independent attack starts (not one burst split into many rows).
2. Clean benign history of the same host before each start.
3. Labels with exact onset times.
4. Flows or similar, with time of day not hidden.
5. The attacker chooses when to start, so that a clock cannot trivially predict it.
6. Downloadable here.

This is a desk review from each dataset's own page or paper. No data was downloaded and no model was trained. The reachability test in this container returned no response (proxy denial or no route) for archive.ll.mit.edu, datasets.uwf.edu, nesg.ugr.es, zenodo.org and cptc.rit.edu, and the project's laptop workspace was unavailable. So criterion 6 fails for every candidate below from this environment; each "go" would need a download on the laptop. Facts I could not confirm from the sources are marked "not confirmed".

## Answer

**No clear go.** Nothing found meets criteria 1 to 5 together. The two with the most onsets (UWF-ZeekData24, UGR'16) get them because the authors launched attacks on a script or schedule. A model that "forecasts" those is forecasting a cron job, and criterion 5 fails by construction. The rest have too few campaigns, no usable benign history, or no public data.

| Dataset | Independent starts | Benign history | Onset labels | Attacker timing | Fetchable here | Verdict |
|---|---|---|---|---|---|---|
| DARPA 1999 | 201 instances, but few victim hosts | Weeks 1 and 3 attack-free | Truth file with times | Scripted by the evaluators | No (blocked) | No-go |
| UGR'16 | Not confirmed | Real ISP traffic | Attack timestamps in minutes | Scheduled by the authors | No (blocked) | No-go |
| UWF-ZeekData24 | About 560 scheduled runs, 15 groups (paper inconsistent: 3 subnets) | Not documented | Mission logs, UTC start and end | Cron, 4 runs per script per day at random times in each hour | No (blocked) | No-go as forecasting; possible detection-in-progress check |
| UWF-ZeekData22 | Not stated; 99.97% reconnaissance | Benign collected in a separate period with no attacks | MITRE tactic per record, from student logs | Students in a course | No (blocked) | No-go |
| AIT-LDSv2 / Alert set | 8 scenarios, one multi-step attack each | Simulated employees | Step start and end times | State machine | No (blocked) | No-go (8 onsets) |
| CPTC | Many teams, but not confirmed | Not confirmed | Not confirmed as released | Competition teams | Page names only Suricata alerts; no public download found | No-go (no access) |
| DARPA TC (Engagement) | A handful of scripted campaigns | Host provenance, not flows | Ground-truth reports | Red team | Multi-terabyte | No-go |
| OpTC, LANL 2015 | See `optc_lanl_feasibility.md` | | | | | Already studied |

## Notes per candidate

**1999 DARPA evaluation** ([data page](https://archive.ll.mit.edu/ideval/data/1999data.html)). Weeks 1 and 3 are attack-free, week 2 is labelled, weeks 4 and 5 hold "201 instances of about 56 types of attacks". Data is tcpdump from inside and outside sensors plus BSM and NT audit logs; an attack truth file lists times. Problems: traffic is synthetic, a few named target hosts (pascal, marx, zeno, hume are the audit sources), and the evaluators scripted every attack, so there is no adversary choice to forecast. No licence is stated.

**UGR'16** ([page](https://nesg.ugr.es/nesg-ugr16/june_week1.php)). Real Spanish ISP NetFlow v9 with labelled CSV, and `attack_ts` files with the minute each attack ran. The injected attacks (DoS, scans) were run by the authors on a schedule. The page does not name victim hosts or sizes (not confirmed). The dataset is explicitly built around daily and weekly periodicity, which is the thing the time-only control exploits.

**UWF-ZeekData24** ([paper](https://www.mdpi.com/2306-5729/10/5/59)). CC BY. Zeek conn logs (about 47 M records), 29,550 mission log entries, five scripted attacks (nmap, PsExec, GlassFish brute force, ProFTPD exploit, SMB exploit) each run four times a day at random times within an hour. That gives hundreds of exact onsets, but each is a script firing, attacker and victim are the lab's own VMs, and the paper does not describe benign traffic. A model could at best learn the schedule. It could be useful later to test "intrusion already under way" detection, which is the only signal found so far.

**UWF-ZeekData22** ([COMIDDS entry](https://fkie-cad.github.io/COMIDDS/content/datasets/uwf_zeekdata22/), [paper](https://www.mdpi.com/2306-5729/8/1/18)). 209 GB, 81 student groups on a cyber range. The paper says the non-malicious traffic was collected when no attack was possible, so there is no benign history of a host that is later attacked. Reconnaissance is 99.97% of attacks.

**AIT-LDSv2 and AIT Alert Data Set** ([paper](https://www.skopik.at/ait/2024_cset.pdf), Zenodo 5789064). Eight scenarios of 4 to 6 days with a modelled multi-step attack and simulated employees, with exact step times and netflows. Only eight campaigns, so criterion 1 fails by a wide margin. It could support an escalation-style test (step N to step N+1 inside a scenario), but CTU-13 escalation already failed twice.

**CPTC** ([research page](https://cp.tc/research)). The page names Suricata alerts only and asks users to cite and contact the organisers. No public dataset repository was found (the globalcptc Hugging Face organisation lists none). Access would need a request to RIT; whether per-team onset labels are released is not confirmed.

**DARPA Transparent Computing** and **OpTC**. TC data is host provenance graphs with few campaigns per engagement and very large archives. OpTC was already shown to have 13 independent onset moments.

## Recommendation

Do not start a new onset study on any of these. The only dataset-side path still open is **access-dependent**: ask RIT whether CPTC per-team data with timestamps can be shared (criteria 1 to 5 would plausibly hold, since competing teams choose when to act), and ask only if the project wants to keep pursuing onset forecasting. Otherwise, the evidence across CIC, CTU-13, OpTC, LANL and this review is that no public dataset supports "predicts attacks before they start", and the defensible claim is early warning of attacks in progress.
