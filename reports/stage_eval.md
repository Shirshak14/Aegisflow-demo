# Stage labels on alerts: can we beat 34/79?

Script: `scripts/evaluation/stage_eval.py`. Numbers: `reports/stage_eval.json`. The demo model, the
multi-task model directory and `configs/config.yaml` are unchanged (`replay.stage_model_dir` stays null).

**Evaluation set.** The demo LSTM raises 349 test alerts at its threshold (0.3308). 79 of them are real
attacks: 42 Impact (DDoS), 33 Reconnaissance (PortScan) and 4 Command and Control (Botnet). Every
row below scores the stage label on those same 79 alerts. Macro-F1 averages the three stages.

## Results

| Variant | Exact stage | Macro-F1 | Impact | Recon | C2 |
|---|---|---|---|---|---|
| Multi-task model, 0.5 confidence gate (the "43%") | 34/79 | 0.30 | 34/42 | 0/33 | 0/4 |
| Multi-task model, no gate | 39/79 | 0.30 | 39/42 | 0/33 | 0/4 |
| (1) Stage loss weighted 5x, retrained | 39/79 | 0.32 | 39/42 | 0/33 | 0/4 |
| (2) Two-step, committed split | 28/79 | 0.26 | 28/42 | 0/33 | 0/4 |
| Two-step, grouped cross-validation incl. Friday | 35/79 | 0.46 | 31/42 | 0/33 | 4/4 |
| Two-step + the alerted window's own traffic, same CV | **53/79** | **0.59** | 39/42 | 10/33 | 4/4 |

Ceiling for anything trained on the committed split: **42/79**.

## Why weighting and two-step can't move it much

Under the chronological split, training and validation contain only Initial Access, Impact and
Lateral Movement attacks. Reconnaissance and Command and Control appear **only on the test day**
(Friday). No loss weighting or architecture can predict a class it has never seen, so the best any
committed-split model can do is get all 42 Impact alerts right. The current model already gets 39.

- **(1) Class weighting** was already in the training code (inverse-frequency, about 30x for Initial
  Access and 52x for Impact). Weighting the stage loss a further 5x did not change exact matches.
  It also labelled more false-positive alerts "Benign" (247/270 vs 217/270).
- **(2) Two-step** removes "Benign" as an answer, but a classifier trained on 509 Monday to Thursday
  attack sequences confuses DDoS with brute force and infiltration. It scored lower (28/79). It is also
  just as confident on false alerts (mean 0.85) as on real ones (0.80), so it cannot flag a wrong alert.

## What does help

**Train the stage classifier on every attack type.** The only way to have Recon and C2 examples is to
train on Friday. The CV rows hold out one 20-minute block of one stage at a time (Botnet: one whole
host at a time) and drop same-host training rows within 15 minutes of it. C2 then becomes learnable
(4/4 here, 372/409 over all test Botnet sequences, host-disjoint). Reconnaissance stays at 0/33 even
with no purge at all, because the 10 windows *before* a PortScan window look like the attacker's other
traffic. Stage cannot be forecast from them.

**Label the attack in progress, not the next one.** Adding the alerted window's own traffic to the
stage classifier lifts the result to 53/79 (67%) and Recon to 10/33. This names the stage once that
window has been observed, which matches the project's safe claim (early warning of attacks in progress).

## Caveats (say these in the deck)

- Every attack stage except C2 comes from the single attacker host 172.16.0.1. Within that host the
  model has to tell stages apart by traffic, but the CV rows still train on the same attack episodes
  of the test day. Treat 53/79 as an **upper bound on this dataset**, not a measured generalisation.
- Only C2 (Botnet, 7 hosts) supports a host-disjoint check, and there are only 4 C2 alerts in the 79.
- Stage labels are the label-to-stage proxy in `configs/stages.yaml`, not ground truth.

Safe slide line: "With a stage classifier trained on all attack types, AegisFlow names the stage of
67% of true alerts on CIC-IDS2017 (up from 43%), measured with grouped, purged cross-validation on
the same dataset."

## Reproduce

```
python scripts/evaluation/stage_eval.py            # retrains the 5x variant into artifacts/experiments/
python scripts/evaluation/stage_eval.py --no-retrain
```
