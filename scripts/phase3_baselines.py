"""Re-run the existing Phase 3 baselines on any-positive and onset-only targets.

No new model code: reuses SequencePreprocessor (train-only fit), the same
LogisticRegression configuration as ``train_experiment``, ``select_threshold``
(validation only) and ``metrics_at``. No deep model is trained; the existing
LSTM checkpoint is only *scored* for the per-host shortcut check.

Target variants:
  any    -- target_attack_present (all sequences).
  onset  -- target_is_onset; continuation positives (attack target whose inputs
            already contain attack traffic) are dropped from every split, since
            they are neither onsets nor clean negatives.

The "last input window was under attack" rule reads the label-derived
attack_flow_ratio of the final input window. It is a reference score only and
is never a model input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import (ForecastDataset, LSTMForecaster, SequencePreprocessor,  # noqa: E402
                                   metrics_at, select_threshold)
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES  # noqa: E402

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
ARTIFACT = ROOT / "artifacts/models/cic_ids2017"
OUT = ROOT / "reports/phase3_baselines_onset.json"
SEED = 42
SHORTCUT_HOST = "172.16.0.1"


def per_host(y, p, hosts, classes, t):
    """Per-host ROC-AUC (hosts with both labels), shortcut-host vs rest, per-class recall."""
    rows = {}
    for h in pd.unique(hosts):
        m = hosts == h
        if 0 < y[m].sum() < m.sum():
            rows[str(h)] = {"n": int(m.sum()), "pos": int(y[m].sum()), "roc_auc": float(roc_auc_score(y[m], p[m]))}
    groups = {}
    for name, m in ((SHORTCUT_HOST, hosts == SHORTCUT_HOST), ("other_hosts", hosts != SHORTCUT_HOST)):
        neg = m & (y == 0); pos = m & (y == 1)
        groups[name] = {"n": int(m.sum()), "pos": int(pos.sum()),
                        "neg_flagged": int((p[neg] >= t).sum()), "neg": int(neg.sum()),
                        "pos_detected": int((p[pos] >= t).sum()),
                        "roc_auc": float(roc_auc_score(y[m], p[m])) if 0 < y[m].sum() < m.sum() else None}
    per_class = {str(c): {"n": int(((classes == c) & (y == 1)).sum()),
                          "detected": int((p[(classes == c) & (y == 1)] >= t).sum())}
                 for c in sorted(set(classes[y == 1]))}
    return {"per_host_roc_auc": rows, "shortcut_host_vs_rest": groups, "per_class_detection": per_class}


def run() -> dict:
    data = ForecastDataset.from_parquet(DATA)
    f = data.frame
    std = list(STANDARD_NUMERIC_WINDOW_FEATURES)
    raw_last = np.stack([np.stack(v)[-1] for v in f.sequence_features])
    last_attack = (raw_last[:, std.index("attack_flow_ratio")] > 0).astype(float)
    onset = f.target_is_onset.to_numpy(bool)
    any_pos = data.attack.astype(bool)
    hosts = f.host_id.to_numpy(str)

    lstm_scores = None
    if (ARTIFACT / "best.pt").exists():
        import torch
        meta = json.loads((ARTIFACT / "metadata.json").read_text(encoding="utf-8"))
        cfg = meta["training_config"]
        net = LSTMForecaster(len(meta["feature_names"]), cfg["hidden_size"], cfg["dropout"]).net
        net.load_state_dict(torch.load(ARTIFACT / "best.pt", map_location="cpu", weights_only=True)); net.eval()
        prep_saved = joblib.load(ARTIFACT / "preprocessor.joblib")
        with torch.no_grad():
            lstm_scores = torch.sigmoid(net(torch.tensor(prep_saved.transform(data.X)))).numpy()
        lstm_threshold = float(meta["threshold"])

    out = {"notes": __doc__.strip(), "variants": {}}
    for variant in ("any", "onset"):
        keep = np.ones(len(f), bool) if variant == "any" else ~(any_pos & ~onset)
        y_all = (any_pos if variant == "any" else onset).astype(np.int64)
        ix = {s: np.flatnonzero((f.split.to_numpy() == s) & keep) for s in ("train", "val", "test")}
        y = {s: y_all[i] for s, i in ix.items()}
        prep = SequencePreprocessor(data.feature_names).fit(data.X[ix["train"]])
        flat = {s: prep.transform(data.X[i]).reshape(len(i), -1) for s, i in ix.items()}
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(flat["train"], y["train"])
        prior = float(y["train"].mean())
        scores = {"majority": {s: np.full(len(i), prior) for s, i in ix.items()},
                  "logistic_regression": {s: lr.predict_proba(flat[s])[:, 1] for s in ix},
                  "last_window_under_attack_rule": {s: last_attack[i] for s, i in ix.items()}}
        thresholds = {"majority": (0.5, "fixed 0.5"),
                      "logistic_regression": select_threshold(y["val"], scores["logistic_regression"]["val"]),
                      "last_window_under_attack_rule": (0.5, "binary rule")}
        if lstm_scores is not None:
            scores["lstm_existing_checkpoint"] = {s: lstm_scores[i] for s, i in ix.items()}
            thresholds["lstm_existing_checkpoint"] = (lstm_threshold, "stored validation-selected threshold (not re-tuned)")
        v = {"split_counts": {s: {"total": int(len(i)), "positive": int(y[s].sum()), "negative": int((y[s] == 0).sum()),
                                  "positive_classes": pd.Series(data.classes[i][y[s] == 1]).value_counts().to_dict()}
                              for s, i in ix.items()},
             "train_prior": prior, "models": {}}
        for m, sc in scores.items():
            t, why = thresholds[m]
            v["models"][m] = {"threshold": float(t), "threshold_reason": why,
                              "val": metrics_at(y["val"], sc["val"], t), "test": metrics_at(y["test"], sc["test"], t)}
            if m in ("logistic_regression", "lstm_existing_checkpoint"):
                v["models"][m]["host_check"] = {s: per_host(y[s], sc[s], hosts[ix[s]], data.classes[ix[s]], t)
                                                for s in ("val", "test")}
        if variant == "any" and (ARTIFACT / "logistic.joblib").exists():
            saved = joblib.load(ARTIFACT / "logistic.joblib")
            v["logistic_matches_saved_artifact"] = bool(np.allclose(
                saved.predict_proba(flat["test"])[:, 1], scores["logistic_regression"]["test"]))
        out["variants"][variant] = v
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run()
    for variant, v in res["variants"].items():
        print(f"\n=== {variant} targets ===")
        for s, c in v["split_counts"].items():
            print(f"  {s:5s} total={c['total']:6d} pos={c['positive']:4d} neg={c['negative']:6d} {c['positive_classes']}")
        for m, r in v["models"].items():
            for s in ("val", "test"):
                x = r[s]
                print(f"  {m:32s} {s:4s} t={r['threshold']:.5f} P={x['precision']:.3f} R={x['recall']:.3f} "
                      f"F1={x['f1']:.3f} ROC={x['roc_auc'] if x['roc_auc'] is None else round(x['roc_auc'], 3)} "
                      f"PR={x['pr_auc'] if x['pr_auc'] is None else round(x['pr_auc'], 3)} "
                      f"FPR={x['false_positive_rate']:.4f} CM={x['confusion_matrix']} pred+={x['positive_predictions']}")
    print(f"\nwritten: {OUT}")
