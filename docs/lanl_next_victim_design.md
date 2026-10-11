# LANL next-victim forecasting: design and pre-stated bars

Status: written and committed **before** any ranker in this study is scored. Script: `scripts/lanl_next_victim_study.py`. Results go in `docs/lanl_next_victim_report.md`. Nothing here changes the demo model, its threshold, the default dataset or the backend.

## 0. Why this question

Every onset study so far (CIC-IDS2017, CTU-13, LANL 5-minute, LANL short-lead, synthetic) failed: nothing predicts *when* an intrusion starts better than a clock-only control. The next-stage question on CTU-13 (spam, click fraud or flooding about to start on an infected bot) was already tested in `docs/escalation_forecasting_report.md` and failed its bars; CIC-IDS2017 stages are a fixed calendar from a single attacker host and cannot be forecast meaningfully. This study changes the target: **once an intrusion is under way, which host will the attacker compromise next?**

LANL 2015 is the only data we hold with enough victims (301) for this. The short-lead study found that for the 35 victims the red-team machine visibly contacts, its first flow arrives at least 19 minutes before the compromising logon. That is the obvious candidate signal, and the bars below are built to test whether it (or anything else) ranks the next victim better than simple host-level controls.

## 1. Question

At the moment of each compromise, with the compromised hosts so far and the attacker's source machines known to the defender, can a ranker put the **next** victim(s) near the top of all live hosts, better than rankers that only know how busy or how central a host is?

Premise, stated plainly: the defender already knows the attack is under way (it has the compromised set so far) and knows the attacker's source machines (the 4 red-team computers). This is the "attack in progress" setting, not onset forecasting.

## 2. Definitions (fixed here)

- **Victims, onsets, sources:** as in the earlier LANL studies. Onset of v = first red-team authentication to v (`redteam.txt.gz`); 301 victims; S = the 4 red-team source computers.
- **Decision point τ:** every 5-minute boundary while an attack is under way, meaning at least one onset happened in [τ − 60 min, τ]. At τ the compromised set is K(τ) = {v : onset_v ≤ τ}. (A grid rather than only onset times, so that a contact seen minutes before a logon can count.)
- **Candidates C(τ):** every computer other than S and K(τ) with at least one flow (as source or destination) in the 24 hours before τ, at hour resolution (hours h with (h+1)·3600 ≤ τ and h ≥ ⌊τ/3600⌋ − 24). Causal: a defender at τ knows this.
- **Positives at horizon H:** candidates with onset in (τ, τ + H]. H = 10 min and 60 min. A decision with no positive candidate at H is not scored at H. Victims that are never a candidate (no flows) are counted as coverage misses.
- **Bursts:** onsets chained when consecutive onsets are under 30 minutes apart (as in the earlier studies); a decision belongs to the burst of the latest onset at or before it. Metrics are averaged per burst first, so a batch of near-simultaneous compromises counts once.

## 3. Features (all computed from data strictly before τ)

| Feature | Meaning |
|---|---|
| `contact_60` | flows from S to the candidate in the 60 minutes before τ (minute bins ending ≤ τ) |
| `contact_7d` | same over 7 days |
| `contact_age` | minutes since the first flow from S to the candidate (large value if none) |
| `act24` | the candidate's flows (in + out) in the last 24 h |
| `nbr` | number of already-compromised hosts the candidate has exchanged at least one flow with |
| `deg` | number of distinct peers the candidate has exchanged flows with |

## 4. Rankers

- **Controls** (know nothing about the attacker): `C-ACT` (act24), `C-NBR` (nbr), `C-DEG` (deg), `M-CTRL` (gradient boosting on act24, nbr, deg). Within one decision every candidate shares the same clock time, so the time-of-day control that beat every earlier model cannot rank candidates; these host-level controls take its place.
- **Candidates:** `R-CONTACT` (rank by contact_60, then contact_7d; ties broken at random) and `M-GB` (gradient boosting on all six features).
- Learned rankers (`M-CTRL`, `M-GB`): `HistGradientBoostingClassifier` (max_iter 100, seed 42), pointwise on (decision, candidate) rows, **leave one burst out**; training decisions within 6 hours of the test burst are dropped; training negatives subsampled to 5%, positives all kept.
- Random ties: one fixed random tie-break per decision (seed 42) for top-k metrics; AUC uses averaged ties.

## 5. Metrics

Per decision: AUC of positives against the other candidates; hit@10 (a positive in the top 10); rank of the best positive. Aggregation: mean per burst, then the median (and mean) over bursts. **Lead:** for each victim, the earliest decision τ in [onset − H, onset) where it sits in the top 10; lead = onset − τ. **Coverage:** victims alerted that way, out of all 301.

Null for hit@10: per decision, the same number of pseudo-victims drawn at random from C(τ), 1000 repeats; p = (1 + #repeats with mean-over-bursts hit@10 ≥ real) / 1001.

## 6. Bars (pre-registered)

A next-victim claim needs one candidate ranker (`R-CONTACT` or `M-GB`) at one horizon to pass all of:

| # | Bar |
|---|---|
| N1 | Median burst AUC ≥ 0.80 (at least 10 scored bursts) |
| N2 | Median burst AUC ≥ best control (C-ACT, C-NBR, C-DEG, M-CTRL) + 0.10 |
| N3 | Mean burst hit@10 ≥ 2 × best control's, and pseudo-victim p < 0.05 |
| N4 | Median lead of alerted victims ≥ 2 minutes |
| N5 | Without day-9 bursts: N1 and N2 still hold |

`M-GB` passing must repeat on seed 7. Descriptive, with no bar: of all hosts S contacts in the flows, how many are later compromised, and how long after contact (the false-alarm side of a "contacted by the attacker" alert).

## 7. How the result is worded

- Pass: "once an intrusion is under way, AegisFlow ranks the likely next victim, with a lead of about X minutes, for hosts visible in the network flows" (plus the coverage number). Nothing about onset.
- Fail: no next-victim claim; the descriptive contact numbers can still be quoted as observations about the data.

## 8. Expectations stated in advance

`R-CONTACT` should rank the 35 contacted victims at or near the top, but most victims (153 of the 188 visible in flows) have no visible contact, so their decisions sit at chance and N1 is likely to fail on the median. Host-level controls may carry real signal (the red team may favour busy or central hosts), which would make N2 hard. The precision of "contacted" is unknown until the extraction runs.

## 9. Data and where it runs

The cloud cannot hold `flows.txt.gz` (1.1 GB). `extract` runs once on the laptop and writes four small tables under `data/processed/lanl/next_victim/` (host vocabulary, hourly host activity, first-seen host pairs, attacker flows per minute), all cut at the last onset. `study` runs anywhere from those tables plus `redteam.txt.gz`.
