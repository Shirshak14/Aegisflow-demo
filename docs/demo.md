# Demo workflow

Prerequisites: raw CIC-IDS2017 CSVs in `data/raw/cic_ids2017/`, then `preprocess` and `train` already run (see README). These outputs are gitignored, so the demo runs from the machine that holds them.

## Run

```
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Wait until the progress line stops saying "loading model and scoring test split" (the model scores the whole test split once at startup).

## Click path

1. Read the "Honest finding" banner: detections are largely attacker-host recognition; stage is not predicted.
2. Choose speed (1000x finishes quickly), press **Start**. Note: Start clears the alert ledger.
3. Watch Hosts monitored, Alerts, progress and the sim clock. Replay covers the held-out test period (from 2017-07-07 09:15).
4. Click an alert: risk score and components (max reachable 75), LSTM probability vs threshold, logistic-regression comparison, dataset label (revealed for evaluation only), MITRE lookup for that label.
5. Click a host for its risk timeline.
6. Press **Verify chain**: expect `VERIFIED`, record count and head hash.
7. Scroll to Model evaluation: switch any/onset target and val/test; read the host-check and leave-one-day-out lines.

## Expected numbers (full replay)

11,379 test sequences replayed, 349 alerts, 79 of them matching an attack label. Test LSTM F1 0.189, ROC-AUC 0.553.

## Optional tamper demo

Use a throwaway copy of the database (point the server at it with the `AEGISFLOW_DB` environment variable), then run `UPDATE alerts SET risk_score = 0 WHERE id = 3;` with any SQLite client and press **Verify chain**: expect `CORRUPTED`, record 3.

## Fallback

If the model fails to load, `/replay/status` shows the error. Keep a screen recording of a full run as backup. Do not run `preprocess`, `rebuild-temporal` or `train` right before presenting; they overwrite reports and model files.
