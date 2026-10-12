# Reproducible release

What a fresh checkout needs, how to check it, and what is and is not distributed. Every result below was produced
by running the command shown, on a clean copy of this branch with no `data/`, `artifacts/` or `reports/figures`
content (Linux, Python 3.13, CPU torch). Anything that was not run is marked **not run**.

## 1. Clean-clone walkthrough

```
pip install -r requirements.txt        # add: pip install torch (CPU build is enough)
python -m aegisflow doctor             # lists what is missing and the command that creates each file
python -m aegisflow.smoke              # synthetic end-to-end run, no dataset needed (section 2)
pytest                                 # full suite
```

Actual results on the clean copy:

| Step | Result |
|---|---|
| `python -m aegisflow doctor` | Reports the CIC-IDS2017 CSVs, processed sequences, split metadata and model files as `[MISSING]`, each with its create command. This is expected before the data is prepared |
| `python -m aegisflow.smoke` | Exit 0, all 9 checks pass, 24 s wall time (table in section 2) |
| `pytest` | 318 passed, 7 skipped, 102 s. The 7 skips are the tests that need the real processed data and trained model (for example the 349-alert replay check) |

The real demo path then needs the dataset and a local training run. **Not run on the clean copy** (the container has
no CIC-IDS2017 download), so no result is claimed here:

```
python scripts/setup/download_data.py --dataset cic_ids2017    # prints where to download; unpack under data/raw/cic_ids2017/
python -m aegisflow preprocess --dataset cic_ids2017
python -m aegisflow train --dataset cic_ids2017
python -m aegisflow calibrate --dataset cic_ids2017            # optional: calibrated confidence
python -m aegisflow manifest --dataset cic_ids2017             # record what you built (section 3)
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The reference outcome on the maintainer's machine is 349 alerts, 79 of them on attack labels, at alert threshold
0.3308 (`tests/api/test_replay.py` checks the alert count against the model's own confusion matrix). A rebuilt model
should land on the same numbers when seeds, library versions and data match, but weight files are not bit-for-bit
reproducible across machines, so compare the metrics, not the hashes.

## 2. Synthetic smoke test

`python -m aegisflow.smoke` (also `make smoke`, and `tests/test_smoke_e2e.py` in `pytest`) generates seeded
synthetic flows (8 benign hosts plus one attacker host with three DoS episodes) in a temporary directory and runs
every stage: flows, windows, sequences with a chronological split, training, manifest, replay of the test split, the
hash-chained ledger and an external checkpoint. It does not touch `data/`, `artifacts/` or `reports/`.

| Check | Result on the clean copy |
|---|---|
| Synthetic flows | 7,661 flows, 1,721 attack |
| Windows and sequences, all splits non-empty and each holding attack targets | 3,140 windows, 3,032 sequences |
| Model trained, artifacts written | `best.pt`, `last.pt`, `logistic.joblib`, `preprocessor.joblib`, `metadata.json`, `metrics.json` |
| Manifest written and verified; a changed file is detected | pass |
| Replay of the test split | 490 sequences, 25 alerts |
| Ledger chain verifies | `VERIFIED`, 25 records |
| External checkpoint matches | `MATCHES` |

These numbers show that the pipeline runs and hands valid data from stage to stage. They say nothing about detection
quality: the data is synthetic.

## 3. Model / artifact manifest

`python -m aegisflow manifest [--dataset cic_ids2017] [--model-dir DIR]` writes `manifest.json` next to the model:

- SHA-256 and size of each artifact file (`best.pt`, `last.pt`, `logistic.joblib`, `preprocessor.joblib`, `metadata.json`, `metrics.json`, `calibration.json` when present)
- feature schema: ordered feature names, input shape, and a hash over them
- alert thresholds (LSTM and logistic regression) and how they were selected (validation only)
- training configuration (hyperparameters, epochs run, seed) and the windowing configuration
- data split sizes and cut-off times, library versions, git commit

`python -m aegisflow manifest --verify` recomputes the hashes and exits 1 if any file changed, vanished or is
unlisted. The manifest identifies what you are running; it is not evidence that the model is good. See
`/evaluation/provenance` and the Models and evaluation page for that.

## 4. Release bundle: what is and is not distributed

`python -m aegisflow manifest --bundle demo_model.zip` zips the model files, `manifest.json` and a `NOTICE.txt`. It
never includes flows, sequences or any dataset file (tested). It was **not built or published for the real demo
model**, because the trained model is not available in the cloud environment.

Licence position, checked 2026-10-12 against the dataset page (unb.ca/cic/datasets/ids-2017.html): CIC-IDS2017 is
described as publicly available to researchers, with a request to cite Sharafaldin, Lashkari and Ghorbani (ICISSP
2018). No licence is named, and the page does not address redistribution or derived works. Therefore:

- **The dataset itself (raw, interim or processed) is not redistributed** and must not be added to a release. Users download it from the source.
- **Model weights trained on it** contain no flow records, but the page does not say whether sharing derived models is allowed. That is a decision for the maintainer (or a short email to the dataset authors). Until then, share the bundle only with people who can download the dataset themselves, and keep the citation.

## 5. Tamper evidence for the audit ledger

See "Audit Ledger" in the README: `python -m backend.app.audit checkpoint | verify | recover`. A checkpoint is useful
only if it is stored away from the host that holds the ledger.
