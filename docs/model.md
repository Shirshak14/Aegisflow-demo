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

The demo LSTM has no stage head, so in the default replay the predicted stage is always `UNCERTAIN`.

## Multi-task model: stage prediction and K-step future state (`aegisflow/ml/multitask.py`)

An opt-in second model with one LSTM encoder and three heads:

| Head | Target | Loss | Reported in `metrics.json` |
|---|---|---|---|
| attack | attack in the horizon-1 target window (same as the LSTM) | BCE with `pos_weight` | precision / recall / F1 / ROC-AUC / PR-AUC / FPR at a validation-selected threshold |
| stage | dominant stage of the horizon-1 target window, over `stages.yaml: stages_order` | cross-entropy, inverse-frequency class weights | accuracy, macro-F1, **accuracy on attack-positive targets**, confusion matrix, all next to a majority-class baseline; stages with no training examples are listed |
| future state | the 28 traffic features of each of the next K windows (`--horizons`, default 3) | masked MSE in preprocessed units | MAE / RMSE per horizon next to a persistence baseline (last input window repeated) |

Horizon k means the k-th same-host window that starts after the input ends, the same rule `build_host_sequences` uses; targets for k > 1 come from `host_windows.parquet`, and a horizon with no window is masked.

```
python -m aegisflow train-multitask --dataset cic_ids2017 --horizons 3   # -> artifacts/models/cic_ids2017_multitask/
python -m aegisflow forecast --limit 5                                   # attack p, stage + MITRE tactic, future windows in raw units
```

`forecast` keeps a stage only if its probability is at least `confidence.stage_prediction_threshold`, otherwise it prints `confidence.uncertain_label`. Setting `replay.stage_model_dir: artifacts/models/cic_ids2017_multitask` in `config.yaml` makes the replay label each alert with that stage, so the stage-severity risk term and the MITRE lookup apply to live alerts. Default `null` keeps the demo exactly as it was.

Limits: stages are the label-to-stage proxy in `stages.yaml`, not ground truth. CIC-IDS2017 has no Exfiltration traffic and very few attack windows per stage in the test split, so stage numbers will be noisy. No metrics are quoted here until the model is trained on the full processed dataset.

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
