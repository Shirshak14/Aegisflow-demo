# AegisFlow

**AI-Based Network Attack Forecasting from Network Traffic Data**
Smart India Hackathon 2026 · Problem Statement **SIH26153** · Team **CyberVanguard** (Team ID 40)
Theme: Blockchain & Cybersecurity

AegisFlow analyzes network traffic over time to move intrusion detection from
**reactive detection** to **predictive defence**: forecasting future network
behaviour, predicting the next likely attack stage, explaining why, mapping
it to MITRE ATT&CK, and logging every alert in a tamper-evident hash chain.

> **Status: Phases 1–3 complete, plus a minimal working demo.**
> - Implemented, tested and audited: ingestion, cleaning and labelling (Phase 1); host-level temporal windows and sequences with strictly-future targets (Phase 2); and majority, logistic-regression and LSTM baselines (Phase 3).
> - Demo: a FastAPI backend, a replay engine that runs the saved model over the held-out test day, a SHA-256 hash-chained audit ledger, and a single-page dashboard.
> - Not built: SHAP/attention explanations, a stage-forecasting model, and the React frontend. The later phases are listed in [`docs/architecture.md`](docs/architecture.md).

## Run the live demo

The demo replays real held-out CIC-IDS2017 traffic through the trained model. The dataset and the trained model are **not** in this repo: the dataset is licensed, and the model is regenerated in step 4. From a fresh clone (Python 3.10+; tested with 3.14 on Windows):

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # ~15 min (torch, shap, xgboost, jupyter). Windows: clone to a short
                                  # path such as C:\Users\<you>\Aegisflow-demo; very long paths hit the
                                  # 260-character limit inside lxml and pip fails (or enable long paths)

# 2. Dataset: download GeneratedLabelledFlows.zip from the official CIC page and unzip
#    the 8 day CSVs anywhere under data/raw/cic_ids2017/ (instructions: docs/data_pipeline.md)
python scripts/download_data.py --dataset cic_ids2017      # prints the exact instructions
python -m aegisflow validate-dataset --dataset cic_ids2017

# 3. Phase 1+2: 500k-row spread-out sample -> flows, host windows, sequences, split (~10 min)
python -m aegisflow preprocess --dataset cic_ids2017 --reingest --sample-size 500000

# 4. Phase 3: train the baselines the demo uses (~4 min on CPU)
python -m aegisflow train --dataset cic_ids2017 --epochs 8 --batch-size 256 --learning-rate 0.001 --hidden-size 16 --dropout 0.2 --patience 3

# 5. Start the demo, then open http://localhost:8000/
uvicorn backend.app.main:app --port 8000
```

In the dashboard, choose a speed (1000× replays the whole test day in about 40 s) and click **Start**. Click a host or an alert for detail, and click **Verify chain** to check the audit ledger. Every score shown is a real output of the saved model on the held-out test split.

The **Model evaluation** panel reads the committed result files in `reports/`, which come from the run documented there. A retrained model may differ slightly from those numbers.

## Honest finding: the model mostly recognises the attacker host, not attacks

On CIC-IDS2017, the model's apparent detections are **largely recognition of one attacker host (`172.16.0.1`), not attack detection**.
- On the test day, the LSTM flags all 29 benign sequences from `172.16.0.1` and 75 of its 76 attack sequences, but only 4 of 411 attack sequences on every other host.
- It detects none of the Botnet attacks, a class it never saw in training.
- A leave-one-day-out test on the only attack class that occurs on two days (DoS, Wednesday and Friday) does not survive a host control:
  - Friday → Wednesday: the LSTM detects 12 of 129 DoS sequences at a 38% false-positive rate.
  - Wednesday → Friday: it catches exactly the 42 of 44 that a rule flagging `172.16.0.1` catches.
- None of the models beats a trivial "the last input window was already under attack" rule, and none predicts attack *onsets* above chance.

The demo therefore does **not** show reliable forecasting of unseen attacks. Evidence:
- [`reports/phase3_onset_baseline_audit.md`](reports/phase3_onset_baseline_audit.md): baselines on any-attack and onset-only targets, and the per-host check.
- [`reports/lodo_pilot_results.json`](reports/lodo_pilot_results.json), with a written summary in [`reports/lodo_pilot_report.md`](reports/lodo_pilot_report.md): leave-one-day-out results per fold.
- [`reports/phase3_training_report.md`](reports/phase3_training_report.md): the Phase 3 training run.

The model predicts attack versus benign only, so the dashboard always shows the predicted stage as **UNCERTAIN**. Its MITRE ATT&CK mapping is looked up from the dataset's label and marked as such; it is not a model output.

## What's real right now

- A documented, versioned **canonical network-flow schema**
  (`aegisflow/schema.py`).
- A working **CIC-IDS2017 dataset adapter** that reads the real,
  official `GeneratedLabelledFlows` CSVs — no synthetic data anywhere in
  this path.
- **Cleaning** that drops (and counts) unusable rows instead of silently
  keeping bad data.
- A **documented attack-stage proxy mapping** (`configs/stages.yaml`) that
  clearly distinguishes the dataset's own label from AegisFlow's inferred
  stage — see [Honesty requirement](#honesty-requirement) below.
- A **feature registry** describing every canonical column: source,
  calculation, and whether CIC-IDS2017 actually supplies it.
- **EDA** producing real summary stats and figures from the ingested data.
- A **CLI** (`python -m aegisflow ...`) and **58 passing tests** (`pytest tests backend/tests`).

## Phase 1 pipeline quickstart

```bash
# 1. Set up (Windows: run inside WSL2, or a native venv)
python -m venv .venv && source .venv/bin/activate      # or .venv\Scripts\activate on native Windows
pip install -r requirements.txt

# 2. See what dataset to download and where it goes
python scripts/download_data.py --dataset cic_ids2017

# 3. After downloading + unzipping GeneratedLabelledFlows.zip into
#    data/raw/cic_ids2017/, confirm it's set up correctly
python -m aegisflow validate-dataset --dataset cic_ids2017

# 4. Fast pipeline check on a sample (seconds, not minutes)
python -m aegisflow ingest --dataset cic_ids2017 --sample-size 200000
python -m aegisflow eda --dataset cic_ids2017

# 5. Full dataset (2.8M rows -- a few minutes on a laptop)
python -m aegisflow ingest --dataset cic_ids2017

# 6. Run the tests
pytest tests backend/tests
```

No dataset yet? `pytest tests/` still passes fully (the demo replay test in `backend/tests` is skipped until the data and model exist) — the test suite ships a
small CSV fixture (`tests/fixtures/cic_ids2017/`) so the pipeline logic is
verified without the real 2.8M-row download.

## Repository structure

```
aegisflow/
├── aegisflow/                  # installable package
│   ├── cli.py                  # python -m aegisflow ...
│   ├── config.py                # loads configs/*.yaml, supports --set overrides
│   ├── schema.py                 # canonical flow schema
│   ├── errors.py / logging_setup.py
│   └── ml/
│       ├── datasets/            # adapter interface + cic_ids2017.py (implemented)
│       ├── preprocessing/        # cleaning.py, labels.py
│       ├── features/registry.py  # documents every canonical column
│       ├── ingestion/pipeline.py
│       └── eda.py
├── backend/app/                 # demo: FastAPI (main.py), replay.py, audit.py (hash chain), static/index.html
├── configs/                     # config.yaml, datasets.yaml, stages.yaml, mitre_mapping.yaml
├── data/                        # raw/ (gitignored) interim/ processed/
├── scripts/                     # download_data.py, prepare_data.py, validate_dataset.py
├── tests/                       # pytest suite + tests/fixtures/ (small, checked-in CSV)
├── notebooks/01_eda.ipynb
├── docs/                        # architecture.md, data_pipeline.md, +stubs for later phases
├── reports/                     # committed Phase 2/3 audit reports and result JSON (figures/logs gitignored)
└── requirements.txt / pyproject.toml / Makefile
```

Phase 2/3 code lives in `aegisflow/ml/temporal/` (windowing, sequences, split, reporting) and
`aegisflow/ml/modeling.py`; audit/evaluation scripts are in `scripts/`. Explainability and a
stage-forecasting model are not built — see [`docs/architecture.md`](docs/architecture.md).

## CLI reference

```
python -m aegisflow list-datasets
python -m aegisflow validate-dataset --dataset cic_ids2017
python -m aegisflow ingest --dataset cic_ids2017 [--sample-size N]
python -m aegisflow eda --dataset cic_ids2017
python -m aegisflow feature-registry
python -m aegisflow preprocess --dataset cic_ids2017 [--reingest] [--sample-size N]
python -m aegisflow train --dataset cic_ids2017 [--epochs ...]
python -m aegisflow predict --input <sequences.parquet>
```

`evaluate`, `replay` and `serve` are registered but raise a clear "not
implemented" error; the demo's replay and API run through
`uvicorn backend.app.main:app` instead.

Every parameter (window size, split fractions, model hyperparameters,
confidence thresholds, replay speed, ...) lives in `configs/config.yaml`,
never hardcoded; override any of them ad hoc with `--set key.path=value`.

## Honesty requirement

This project does not fabricate results. Concretely:

- `configs/stages.yaml` marks every attack-stage mapping as `"inferred"`
  (proxy) except `Benign`, which is `"ground_truth"`. The ingested data
  carries a `label_confidence` column so this distinction survives into the
  dashboard, not just the docs.
- Canonical columns a dataset genuinely can't supply (e.g. `ttl_mean` for
  CIC-IDS2017) are left `NA`, never zero-filled — check with
  `python -m aegisflow feature-registry`.
- An unmapped dataset label is a hard error (`LabelMappingError`), not a
  silent `"Unknown"`.
- A missing dataset is a hard error with the exact official download
  instructions, never a silent fallback to fake data.
- CLI commands for unimplemented phases fail loudly
  (`NotImplementedPhaseError`) instead of doing nothing.

## Limitations

- Only CIC-IDS2017 is implemented; UNSW-NB15 / CTU-13 / CIC-IDS2018 adapters
  are planned (Phase 9) but not built.
- Attack-stage labels are a documented proxy, not ground truth — see
  `configs/stages.yaml`.
- The model's apparent detection is largely attacker-host recognition, not attack
  detection. See [Honest finding](#honest-finding-the-model-mostly-recognises-the-attacker-host-not-attacks).
- The demo uses a 500k-row sample of CIC-IDS2017, not all 2.83M rows.
- No stage-forecasting model and no explanations (SHAP/attention). The demo's
  frontend is one static HTML page, not the planned React app.
- The audit ledger detects edited, reordered or deleted records. Deleting the newest
  records leaves a valid shorter chain, so compare the head hash reported by
  `/audit/verify` with one recorded elsewhere.

## Team

CyberVanguard — Team ID 40. SIH26153 / Theme: Blockchain & Cybersecurity.
