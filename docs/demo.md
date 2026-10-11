# Demo workflow

Prerequisites: raw CIC-IDS2017 CSVs in `data/raw/cic_ids2017/`, then `preprocess` and `train` already run (see README). These outputs are gitignored, so the demo runs from the machine that holds them.

## Run

```
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Wait until the progress line stops saying "loading model and scoring test split" (the model scores the whole test split once at startup).

## Click path

The sidebar has seven pages; each is a link (for example `/#/alerts/12`) you can open directly.

1. **Overview**: choose speed (1000x finishes quickly), press **Start**. Each Start opens a new session; earlier sessions stay in the ledger, and the dashboard shows the current one. Watch hosts, alerts, progress and the sim clock; the replay covers the held-out test period (from 2017-07-07 09:15). The bottom card shows the honest test numbers and the attacker-host limitation.
2. **Alerts**: filter by risk or status, open one. Left: risk score and components (max reachable 75), LSTM probability vs threshold, logistic-regression comparison, dataset label (evaluation only), MITRE lookup for that label, ledger record. Right: feature attribution, attention (if trained), stage model (off by default) and analyst review.
3. **Hosts**: open 172.16.0.1 for its risk timeline and alerts.
4. **Audit ledger**: press **Verify chains**: expect `VERIFIED`, record count and head hash.
5. **Models and evaluation**: LSTM test numbers, the label-free anomaly baseline next to it, then the evaluation table (switch any/onset target and val/test) with the host-check and leave-one-day-out lines.
6. **How it works**: the pipeline, what the demo claims, and its known limits.

## Expected numbers (full replay)

11,379 test sequences replayed, 349 alerts, 79 of them matching an attack label. Test LSTM F1 0.189, ROC-AUC 0.553.

## Optional tamper demo

Use a throwaway copy of the database (point the server at it with the `AEGISFLOW_DB` environment variable), then run `UPDATE alerts SET risk_score = 0 WHERE id = 3;` with any SQLite client and press **Verify chain**: expect `CORRUPTED`, record 3.

## Fallback

If the model fails to load, `/replay/status` shows the error. Keep a screen recording of a full run as backup. Do not run `preprocess`, `rebuild-temporal` or `train` right before presenting; they overwrite reports and model files.
