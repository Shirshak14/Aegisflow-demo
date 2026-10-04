# Model documentation

Current state. Numbers are from `artifacts/models/cic_ids2017/metrics.json` and `reports/phase3_training_report.md`. Full evaluation context is in `reports/`.

## Task

Binary next-window prediction per host: given the last 10 observed 60 s windows (30 s stride) of a source host, predict whether the host's next strictly-future window contains attack traffic (`target_attack_present`). Windows are built per `source_ip`.

## Input

Tensor `[batch, 10, 28]`: 28 numeric traffic features per window (counts, byte/packet sums, flag counts and ratios, unique destination IPs/ports, port entropy, burst). The two label-derived columns (`attack_flow_ratio`, `benign_flow_ratio`) are excluded. Preprocessing (train-only median imputation, signed log1p on count features, RobustScaler) is fitted on the training split only.

## Models

| Model | Description |
|---|---|
| Majority | Predicts the training attack prior; never alerts. |
| Logistic regression | Flattened normalised history, `class_weight="balanced"`. |
| LSTM (demo model) | 1-layer LSTM (hidden 16), dropout 0.2 on the last hidden state, linear output; `BCEWithLogitsLoss` with `pos_weight` 132.75; Adam 1e-3, batch 256; early stopping on validation loss. About 3k parameters. Best checkpoint was epoch 1 of 4. |

Hyperparameters are the defaults in `configs/config.yaml` under `model:` (epochs 8, batch 256, lr 0.001, hidden 16, dropout 0.2, patience 3 = the committed model). CLI flags or `--set model.hidden_size=8` override them; `model.type` must be `lstm`.
`python -m aegisflow train --dataset cic_ids2017`

There is no stage-prediction head or attention. Predicted stage is always `UNCERTAIN`.

## Explanations (`aegisflow/ml/explain.py`)

Every prediction can be attributed to its 10 × 28 inputs (time step × feature):

| Model | Method | Property |
|---|---|---|
| LSTM | `shap`: SHAP values via `shap.GradientExplainer` (expected gradients), background = 100 training sequences | attributions sum to output − mean background output, approximately (sampled) |
| LSTM | `integrated_gradients`: Integrated Gradients from the median training sequence, 128 steps, pure PyTorch | attributions sum to output − baseline output (gap reported as `additivity_gap`) |
| Logistic regression | exact linear SHAP, `w · (x − E[x])` | exact |

All three explain the **log-odds** of an attack in the target window. Output per sequence: top features with sign (towards attack / towards benign), the time step where each mattered most, the raw last-window value, and attribution per time step.

`python -m aegisflow explain --dataset cic_ids2017 --limit 5` (or `--sequence-id <id>`, `--model logistic_regression`, `--method integrated_gradients`), and `GET /explain/{sequence_id}` in the API. The dashboard does not call it. Explanations are faithful to the model; they do not make the model more accurate, and given the host shortcut below, features that identify host `172.16.0.1`'s traffic are expected to dominate.

## Split

Chronological 60/20/20 on sequence target-end time; sequences straddling a cutoff are `boundary_excluded`. Cutoffs: 2017-07-06 09:20:22 and 2017-07-07 09:15:22. Thresholds are chosen on validation (max F1), never on test.

## Test results (11,379 sequences, 487 positives, prevalence 4.28 %)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.5 | 0 | 0 | 0 | 0.500 | 0.043 | 0 |
| Logistic regression | 0.99868 | 0.642 | 0.107 | 0.183 | 0.473 | 0.143 | 0.0027 |
| LSTM | 0.33075 | 0.226 | 0.162 | 0.189 | 0.553 | 0.177 | 0.0248 |

## Known limitations

- All training positives come from one host (`172.16.0.1`); the LSTM's apparent skill is largely recognition of that host (`reports/phase3_onset_baseline_audit.md` §5, `reports/lodo_pilot_report.md`).
- Botnet (84 % of test positives) never appears in training and is not detected.
- Most positives are continuations of an attack already visible in the inputs; onset prediction is at or below chance.
- Validation is thin (127 positives, 2 episodes), so thresholds do not transfer.
