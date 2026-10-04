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

### Temporal GNN (`aegisflow/ml/graph.py`, opt-in)

For each window, flows define a host-communication graph: an edge joins two hosts that exchanged a flow in that window. For each step of a host's 10-window sequence, the model combines three inputs:
- the host's own 28 features
- the mean features of its neighbours in that window
- log(1 + neighbour count)

It combines them with a GraphSAGE-style mean-aggregation layer, then runs a GRU over time and a linear attack head. Target, split, class weighting, early stopping and threshold selection are the same as the LSTM's.

`python -m aegisflow train-gnn` reads `sequences.parquet`, `host_windows.parquet` and `flows.parquet` and writes to `artifacts/models/cic_ids2017_tgnn/`. A synthetic test checks that the graph carries information per-host models cannot see: there, the label depends only on which peer a host talked to, and the GNN reaches ROC-AUC ~0.88 vs ~0.48 for the LSTM. That is a property of the test data, not a CIC-IDS2017 result. On CIC-IDS2017 the graph is dominated by the attacker host 172.16.0.1, so expect the same shortcut. No CIC-IDS2017 metrics are reported here.

Hyperparameters are the defaults in `configs/config.yaml` under `model:` (epochs 8, batch 256, lr 0.001, hidden 16, dropout 0.2, patience 3 = the committed model). CLI flags or `--set model.hidden_size=8` override them; `model.type` must be `lstm`.
`python -m aegisflow train --dataset cic_ids2017`

There is no stage-prediction head, attention, or explainability module. Predicted stage is always `UNCERTAIN`.

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
