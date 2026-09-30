"""Audit saved Phase 3 checkpoint and emit every validation/test prediction."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import (ForecastDataset, LSTMForecaster, MODEL_FEATURES,
                                   SequencePreprocessor, metrics_at, positive_class_weight)

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
ARTIFACT = ROOT / "artifacts/models/cic_ids2017"


def summarize(values: np.ndarray) -> dict:
    return {"count": int(len(values)), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)}}


def run() -> dict:
    data = ForecastDataset.from_parquet(DATA)
    metadata = json.loads((ARTIFACT / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    prep = joblib.load(ARTIFACT / "preprocessor.joblib")
    indices = {s: data.indices(s) for s in ("train", "val", "test")}
    counts = {s: {"total": int(len(ix)), "positive": int(data.attack[ix].sum()),
                  "negative": int(np.sum(data.attack[ix] == 0))} for s, ix in indices.items()}

    refit = SequencePreprocessor(data.feature_names).fit(data.X[indices["train"]])
    preprocessing_match = bool(np.allclose(prep.medians, refit.medians) and
        np.allclose(prep.scaler.center_, refit.scaler.center_) and
        np.allclose(prep.scaler.scale_, refit.scaler.scale_))
    tensors = {s: prep.transform(data.X[ix]) for s, ix in indices.items()}
    feature_order_ok = metadata["feature_names"] == MODEL_FEATURES == data.feature_names
    dimensions_ok = all(x.shape[-1] == len(metadata["feature_names"]) for x in tensors.values())
    weight, train_counts = positive_class_weight(data.attack[indices["train"]])
    weight_ok = bool(np.isclose(weight, metrics["positive_class_weight"]) and
                     train_counts == metrics["train_class_counts"])

    chronology = {}
    for split, ix in indices.items():
        chronology[split] = bool(pd.to_datetime(data.frame.iloc[ix]["seq_start_time"]).is_monotonic_increasing)

    altered = data.frame.copy()
    altered["target_attack_present"] = 1 - altered["target_attack_present"]
    altered["target_dominant_class"] = "AUDIT_MUTATED"
    altered["target_dominant_stage"] = "AUDIT_MUTATED"
    altered["target_features"] = [np.zeros(30, dtype=np.float32) for _ in range(len(altered))]
    no_target_leakage = np.array_equal(data.X, ForecastDataset.from_frame(altered).X)

    cfg = metadata["training_config"]
    model = LSTMForecaster(len(metadata["feature_names"]), cfg["hidden_size"], cfg["dropout"]).net
    model.load_state_dict(torch.load(ARTIFACT / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    best_epoch = int(np.argmin([row["val_loss"] for row in metrics["history"]])) + 1
    checkpoint_ok = cfg.get("best_checkpoint_epoch") == best_epoch == metrics["training"].get("best_checkpoint_epoch")
    probabilities = {}
    with torch.no_grad():
        for split in indices:
            probabilities[split] = torch.sigmoid(model(torch.tensor(tensors[split]))).numpy()

    yv, pv = data.attack[indices["val"]], probabilities["val"]
    yt, pt = data.attack[indices["test"]], probabilities["test"]
    validation = {"roc_auc": float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else None,
        "pr_auc": float(average_precision_score(yv, pv)) if yv.sum() else None,
        "pr_auc_status": "not estimable: validation has zero positives" if yv.sum() == 0 else "computed",
        "at_0.5": metrics_at(yv, pv, .5)}

    # Thresholds derive exclusively from validation score quantiles. With no positive
    # validation targets these control alert volume; they do not optimize recall/F1.
    thresholds = {"fixed_0.5": .5}
    for q in (.90, .95, .99, .999, 1.0):
        thresholds[f"val_upper_tail_{1-q:.3%}"] = float(np.nextafter(np.quantile(pv, q), np.inf))
    threshold_table = {name: {"threshold": t, "validation": metrics_at(yv, pv, t),
                              "test": metrics_at(yt, pt, t)} for name, t in thresholds.items()}

    pos, neg = pt[yt == 1], pt[yt == 0]
    test_roc = float(roc_auc_score(yt, pt))
    score_summary = {s: {"all": summarize(probabilities[s]),
        "positive": summarize(probabilities[s][data.attack[ix] == 1]) if np.any(data.attack[ix] == 1) else None,
        "negative": summarize(probabilities[s][data.attack[ix] == 0])} for s, ix in indices.items()}
    predictions = []
    for split, ix in indices.items():
        for j, row_idx in enumerate(ix):
            row = data.frame.iloc[row_idx]
            predictions.append({"row_index": int(row_idx), "split": split,
                "sequence_id": row.get("sequence_id", ""), "seq_start_time": row.get("seq_start_time"),
                "seq_end_time": row.get("seq_end_time"), "target_window_start": row.get("target_window_start"),
                "target_attack_present": int(data.attack[row_idx]),
                "attack_probability": float(probabilities[split][j])})
    pd.DataFrame(predictions).to_csv(ARTIFACT / "audit_predictions.csv", index=False)

    report = {
        "split_counts": counts,
        "input_contract": {"per_sample_shape": list(tensors["test"].shape[1:]),
            "test_batch_shape": list(tensors["test"].shape), "features": data.feature_names,
            "identical_order_all_splits": bool(feature_order_ok and dimensions_ok),
            "excluded_label_derived_columns": ["attack_flow_ratio", "benign_flow_ratio"],
            "X_built_from_sequence_features_only": True,
            "mutating_all_targets_keeps_X_identical": bool(no_target_leakage)},
        "chronology": {"rows_monotonic_by_seq_start_per_split": chronology,
            "within_sequence_order": "Phase 2 sorts each host by window_start before stacking input rows"},
        "preprocessing": {"independent_training_only_refit_matches_saved_parameters": preprocessing_match,
            "fit_split": "train", "fit_sequence_count": counts["train"]["total"],
            "transformed_splits": ["train", "val", "test"]},
        "class_weight": {"positive_weight": weight, "counts": train_counts,
            "verified_training_only": weight_ok},
        "checkpoint": {"file": "best.pt", "best_epoch_by_min_validation_loss": best_epoch,
            "declared_best_epoch": cfg.get("best_checkpoint_epoch"),
            "selection_metric": "validation BCEWithLogitsLoss", "verified": checkpoint_ok},
        "label_orientation": {"metadata": metadata["label_mapping"],
            "stored_labels": sorted(np.unique(data.attack).tolist()),
            "test_roc_auc_attack_as_1": test_roc,
            "test_roc_auc_reversed_orientation": float(roc_auc_score(1-yt, pt)),
            "positive_probability_mean": float(pos.mean()), "negative_probability_mean": float(neg.mean()),
            "positive_probability_values": pos.tolist(),
            "positive_sample_probability_ranks_ascending": [int(np.sum(pt <= value)) for value in pos]},
        "validation": {"metrics": validation, "probabilities": score_summary["val"]},
        "test": {"roc_auc": test_roc, "pr_auc": float(average_precision_score(yt, pt)),
            "probabilities": score_summary["test"]},
        "validation_derived_threshold_analysis": threshold_table,
        "all_split_sample_predictions_csv": "artifacts/models/cic_ids2017/audit_predictions.csv",
    }
    (ARTIFACT / "evaluation_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
