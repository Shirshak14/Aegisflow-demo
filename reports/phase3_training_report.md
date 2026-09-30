# Phase 3 training on the corrected 500k CIC-IDS2017 sample

**Status:** Training completed and the evaluation protocol checks out. The models do **not** show useful attack forecasting. Every model's test F1 is 0.21 or lower. The LSTM's apparent skill comes mostly from recognising one host, and no model detects Botnet, which is 84% of test positives.

Earlier Phase 3 reports (`phase3_evaluation_audit.md`, `phase3_temporal_alignment_audit.md`) describe models trained on the mis-clocked data and are superseded. Those checkpoints are archived in `artifacts/archive/stale_pre_clockfix_models/`.

## Command and setup

```
python -m aegisflow train --dataset cic_ids2017 --epochs 8 --batch-size 256 --learning-rate 0.001 --hidden-size 16 --dropout 0.2 --patience 3
```

- **Data:** `data/processed/cic_ids2017/sequences.parquet` from the corrected 500k spread-out ingest (446,137 flows, 85,077 sequences).
- **Input:** `[batch, 10, 28]`. `attack_flow_ratio` and `benign_flow_ratio` are excluded. Feature order is deterministic and identical across splits. Changing every target and both label-derived columns leaves the model input unchanged.
- **Preprocessing:** median imputation, signed log1p and RobustScaler, all fitted on train only. A fresh refit on train matches the saved `preprocessor.joblib`.
- **Class imbalance:** LSTM `pos_weight = 50,711 / 382 = 132.75`, computed from train only. Logistic regression uses `class_weight="balanced"`.
- **LSTM training:** 4 of the 8 requested epochs ran. Early stopping triggered after 3 epochs without improvement. The best checkpoint is **epoch 1** by minimum validation BCE.
- **Threshold selection:** each model's threshold maximises F1 on the **validation** split. Candidates are {0.01, 0.05, 0.1, 0.2, 0.3, 0.5} plus every validation score. Recomputing from validation alone reproduces the stored thresholds exactly. The majority baseline stays at 0.5.

### Epoch history

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.8906 | **1.6828** (best) |
| 2 | 0.1725 | 2.3014 |
| 3 | 0.0847 | 2.0977 |
| 4 | 0.0547 | 2.7515 |

Training loss falls while validation loss rises from epoch 2 on: the model overfits the training attack classes immediately.

## Split class composition (the main limitation)

| Split | Total | Positive | Negative | Positive classes |
|---|---:|---:|---:|---|
| Train | 51,093 | 382 | 50,711 | Brute Force 238, DoS 141, Web Attack 3 (all on host 172.16.0.1) |
| Validation | 11,970 | 127 | 11,843 | Web Attack 116 (172.16.0.1), Infiltration 11 |
| Test | 11,379 | 487 | 10,892 | **Botnet 409**, DoS 44, Reconnaissance 34 |

Botnet has **no** training positives, and Infiltration and Reconnaissance have none either. Test prevalence (4.28%) is about 6× train prevalence (0.75%).

## Validation metrics (at each model's selected threshold)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Confusion matrix [[TN, FP], [FN, TP]] | Predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Majority | 0.5 | 0 | 0 | 0 | 0.500 | 0.0106 | 0 | [[11843, 0], [127, 0]] | 0 |
| Logistic regression | 0.99868 | 0.793 | 0.362 | 0.497 | 0.552 | 0.355 | 0.0010 | [[11831, 12], [81, 46]] | 58 |
| LSTM | 0.33075 | 0.194 | 0.709 | 0.304 | 0.942 | 0.151 | 0.0317 | [[11468, 375], [37, 90]] | 465 |

Actual validation split: 127 positives, 11,843 negatives.

## Test metrics (final chronological evaluation, never used for selection)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Confusion matrix [[TN, FP], [FN, TP]] | Predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Majority | 0.5 | 0 | 0 | 0 | 0.500 | 0.0428 | 0 | [[10892, 0], [487, 0]] | 0 |
| Logistic regression | 0.99868 | 0.642 | 0.107 | 0.183 | 0.473 | 0.143 | 0.0027 | [[10863, 29], [435, 52]] | 81 |
| LSTM | 0.33075 | 0.226 | 0.162 | 0.189 | 0.553 | 0.177 | 0.0248 | [[10622, 270], [408, 79]] | 349 |

Actual test split: 487 positives, 10,892 negatives. The PR-AUC prevalence baseline is 0.0428.

## Per-class test recall at the selected threshold

| Class (n) | Logistic regression | LSTM | LSTM class-vs-negatives ROC-AUC |
|---|---:|---:|---:|
| Botnet (409) | 0 / 409 | 4 / 409 | 0.473 |
| DoS (44) | 38 / 44 | 42 / 44 | 0.958 |
| Reconnaissance (34) | 14 / 34 | 33 / 34 | 0.995 |

## Diagnostics that limit interpretation

1. **Host shortcut.** Every training positive is on `172.16.0.1`.
   - On test, the LSTM flags **29 of 29** benign sequences from that host, plus 75 of 76 positives from it.
   - On all other hosts its ROC-AUC is 0.472 and it detects 4 of 411 positives.
   - On validation, its within-host ROC-AUC on 172.16.0.1 is 0.531.

   The high validation ROC-AUC (0.942) and the DoS and Reconnaissance "detections" mostly reflect recognising that host, not predicting attacks.
2. **Continuation, not forecasting.** For 118 of 127 validation positives and 270 of 487 test positives, the final input window already contains attack flows. A reference rule, "predict attack if the last input window has attack flows", scores F1 0.92 on validation and 0.55 on test. It uses label-derived information, so it isn't a deployable model, but it outperforms all three models. Most of the measurable signal is "an attack in progress continues".
3. **Logistic regression is badly calibrated.** Its scores collapse to about 0 or 1, and the selected threshold is 0.99868. Its ROC-AUC is at or below chance on both splits.
4. **Validation is thin for threshold selection.** It has 127 positives from only 2 episodes: one Web Attack burst on 172.16.0.1 that continues a training episode, and 11 Infiltration sequences built from 5 flows. The selected thresholds don't transfer: LSTM F1 drops from 0.304 to 0.189, and logistic regression from 0.497 to 0.183.
5. **Stale text in metrics.json.** Its `stage_proxy.status` string still says "four attack-positive test targets". This comes from hard-coded text in `modeling.py`, not from this run.

## Artifacts (`artifacts/models/cic_ids2017/`)

- `best.pt`: LSTM, epoch 1. `last.pt`: LSTM, epoch 4.
- `logistic.joblib`, `preprocessor.joblib`, `metadata.json`, `metrics.json` (test metrics only).
- `validation_and_test_metrics.json`: validation and test metrics plus per-class results, recomputed from the saved artifacts with no refitting.
- `evaluation_audit.json`, `audit_predictions.csv`: output of `scripts/audit_phase3.py`.
