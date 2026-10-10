# LANL next-victim study: results (2026-10-10)

Implements `docs/lanl_next_victim_design.md`. The design, the bars N1 to N5 and the script were committed and pushed (8016853) before anything was scored, and none of them changed for this run. Reproduce with:

    python scripts/lanl_next_victim_study.py extract   # one pass over flows.txt.gz, about 8 min
    python scripts/lanl_next_victim_study.py study     # about 6.5 min

Summary output: `reports/lanl_next_victim_study.json` (transcribed from the full run report, which stays on the laptop with its per-burst AUCs). The demo model, its threshold, the default dataset and `backend/` are unchanged.

## Answer

**No next-victim claim.** Neither candidate passes the bars at either horizon. The attacker-contact ranker does put a real next victim in the top 10 far more often than any control at 10 minutes (25% of decisions per burst against 12% for the best control), with a median lead of 7.5 minutes. But its median burst AUC is 0.50, because in most bursts the next victim has no visible attacker contact. A ranker that knows nothing about the attacker, "most peers first" (C-DEG), scores 0.74. The learned model is worse than the plain rankers.

**What the data shows, descriptively:** the red-team machines send flows to **514** hosts in the period studied, and only **35** of them are later compromised (about 1 in 15). A "contacted by the attacker" alert would therefore be wrong about 14 times out of 15, though when it is right it arrives about 19 minutes before the logon.

## 1. What was run

| Item | Value |
|---|---|
| Hosts in flows (up to the last onset) | 12,027 |
| Victims | 301, of which 209 appear in the flows |
| Decision points | 683 (every 5 min while an onset happened in the last hour), 35 bursts |
| Candidates per decision | median 8,185 live hosts |
| Rows | 5.2 million (decision, candidate) pairs |
| Scored decisions (a victim among the candidates within H) | 173 in 19 bursts at H = 10 min; 394 in 21 bursts at H = 60 min |
| Seed | 42. Nothing passed, so the seed-7 repeat was not triggered. |

## 2. Results

Median burst AUC / mean burst hit@10 / victims ever in the top 10 inside the horizon (of 301) / median lead. A random ranker's hit@10 is 0.002 (H = 10) and 0.003 (H = 60).

| Ranker | H = 10 min | H = 60 min |
|---|---|---|
| C-ACT (busiest in 24 h) | 0.715 / 0.07 / 5 / 9.1 min | 0.660 / 0.15 / 5 / 58.7 min |
| C-NBR (talked to past victims) | 0.499 / 0.02 / 1 / 9.2 min | 0.499 / 0.06 / 3 / 50.0 min |
| **C-DEG (most peers)** | **0.740** / 0.12 / 5 / 9.1 min | **0.691** / 0.19 / 5 / 58.7 min |
| M-CTRL (learned, controls only) | 0.442 / 0.01 / 1 / 6.3 min | 0.448 / 0.01 / 2 / 30.2 min |
| **R-CONTACT** | 0.499 / **0.25** / **32** / 7.5 min | 0.499 / 0.18 / 32 / 17.1 min |
| M-GB (learned, all features) | 0.465 / 0.17 / 13 / 7.2 min | 0.584 / 0.16 / 26 / 17.1 min |

| Bar | R-CONTACT H10 | M-GB H10 | R-CONTACT H60 | M-GB H60 |
|---|:-:|:-:|:-:|:-:|
| N1 median burst AUC ≥ 0.80 | – | – | – | – |
| N2 ≥ best control + 0.10 | – | – | – | – |
| N3 hit@10 ≥ 2 × best control, p < 0.05 | **pass** | – | – | – |
| N4 median lead ≥ 2 min | pass | pass | pass | pass |
| N5 without day 9 | – | – | – | – |
| All | no | no | no | no |

## 3. What the numbers mean

- **The contact signal is real but narrow.** R-CONTACT brings 32 victims into the top 10 before their logon (inferred to be almost all contacted ones, since uncontacted candidates tie at zero among about 8,000), which is why its hit@10 is double the best control's at 10 minutes. In the other bursts no victim has visible contact, every candidate ties at zero and its AUC is 0.50. The per-burst AUCs show it: 1.00 and 1.00 in the two bursts dominated by contacted victims, 0.49 to 0.50 in 11 of 19.
- **Busy, well-connected hosts get hit more.** C-DEG and C-ACT reach 0.66 to 0.74 with no knowledge of the attacker. That is the bar any attacker-aware model has to clear, and none does. It is also not a forecast: it ranks the same hosts high no matter what the attacker does.
- **Learning made it worse.** M-CTRL sits below chance (0.44) and M-GB swings from 0.00 to 1.00 across bursts. Trained on some bursts, the model's preferences (for example, which activity level means "next") flip on others, so a learned ranker does not transfer between bursts any better than the onset models did.
- **Contact is a weak alarm on its own.** 479 of the 514 hosts the attacker contacts are never compromised in the data. A ranker built on contact alone therefore wastes most of its top slots, and the 19-minute lead only holds for the hosts that are in fact hit.
- **Day 9 did not matter here.** Removing day 9 changed no AUC and no scored-decision count: no decision on day 9 had a victim among the live candidates within the horizon. Inferred, not checked: most day-9 victims are either absent from the flows or were silent in the 24 hours before.

## 4. What the deck can honestly say

- **Do not say** AegisFlow predicts the next victim. It fails the pre-stated bars, and a plain "most-connected hosts first" list beats it on AUC.
- **Can say, as an observation about the data, not a model result:** "In LANL 2015, the attacker's machines touched 514 hosts; 35 of them were later compromised, each about 19 minutes after first contact. Flagging hosts a known attacker is talking to gives early warning for those 35, at the cost of about 14 false flags per true one."
- This adds to the earlier conclusion: the honest framing stays **early warning of attacks in progress**.

## 5. Not done

No seed-7 repeat (nothing passed). No hyperparameter search. Victims with no flows (92 of 301) are never candidates. Candidates are limited to hosts active in the last 24 hours. Next-stage forecasting was not rerun: the CTU-13 escalation study (`docs/escalation_forecasting_report.md`) already tested it and failed, and CIC-IDS2017 stages are a fixed schedule from one attacker host.
