"""Leave-one-day-out (LODO) pilot on the existing 500k sequence table.

Current code paths only: each fold calls the existing ``train_experiment``
(train-only preprocessing, existing class weighting, validation-only checkpoint
and threshold selection) with the Phase 3 hyperparameters. The Phase 3
artifacts are not touched; each fold writes to its own directory.

Fold construction for held-out test day d (calendar date of the capture):
  test   -- sequences whose inputs AND target lie entirely on day d
  val    -- sequences entirely on the validation day v: the capture day before d,
            except that Monday (attack-free per the dataset description) is never
            used; the Monday and Tuesday folds use the following day instead
  purged -- sequences touching d or v only partially (inputs or target cross in)
  train  -- everything else (all remaining days; may span two training days)
Train/val/test are day-disjoint. A within-day time split was rejected: attacks
sit at fixed times of day (Friday's DDoS is its last 20 minutes), so a per-day
80/20 cut moved all Friday DDoS into validation and left the Wednesday fold
with no DoS in training.

LODO folds that train on later days than the test day measure cross-day
generalisation, not deployable forward-in-time forecasting.

Classes absent from a fold's TRAIN positives are reported as ZERO-SHOT.
The "last window under attack" rule (label-derived attack_flow_ratio) and the
"host identity" rule (score 1 for host 172.16.0.1) are reference scores only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import ForecastDataset, LSTMForecaster, metrics_at, train_experiment  # noqa: E402
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES  # noqa: E402

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
OUT = ROOT / "artifacts/models/cic_ids2017_lodo_pilot"
REPORT = ROOT / "reports/lodo_pilot_results.json"
HOST = "172.16.0.1"
HP = dict(seed=42, epochs=8, batch_size=256, learning_rate=0.001, hidden_size=16, dropout=0.2, patience=3)
MIN_POS = 10


def validation_day(days: list, test_day):
    i = days.index(test_day)
    return days[i - 1] if i >= 2 else days[i + 1]


def assign_fold(frame: pd.DataFrame, test_day, val_day) -> pd.Series:
    start_day = frame.seq_start_time.dt.date
    end_day = frame.target_window_end.dt.date
    inside = lambda d: (start_day == d) & (end_day == d)  # noqa: E731
    touching = lambda d: (start_day <= d) & (end_day >= d)  # noqa: E731
    split = pd.Series("purged", index=frame.index, dtype=object)
    split[inside(test_day)] = "test"
    split[inside(val_day)] = "val"
    split[~touching(test_day) & ~touching(val_day)] = "train"
    return split


def rate(n, d):
    return float(n / d) if d else None


def auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < int(np.sum(y)) < len(y) else None


def host_breakdown(y, p, t, hosts):
    out = {}
    for name, m in ((HOST, hosts == HOST), ("other_hosts", hosts != HOST)):
        neg, pos = m & (y == 0), m & (y == 1)
        out[name] = {"n": int(m.sum()), "pos": int(pos.sum()), "neg": int(neg.sum()),
                     "neg_flagged": int((p[neg] >= t).sum()), "neg_flag_rate": rate((p[neg] >= t).sum(), neg.sum()),
                     "pos_detected": int((p[pos] >= t).sum()), "pos_detect_rate": rate((p[pos] >= t).sum(), pos.sum()),
                     "within_group_roc_auc": auc(y[m], p[m])}
    return out


def episodes(sub: pd.DataFrame) -> int:
    s = sub.sort_values(["host_id", "target_window_start"])
    new = (s.host_id != s.host_id.shift()) | (s.target_window_start.diff() > pd.Timedelta("30min"))
    return int(new.sum())


def evaluate_fold(frame, split, fold_dir, train_classes, last_attack):
    data = ForecastDataset.from_frame(frame.assign(split=split.values))
    metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    prep = joblib.load(fold_dir / "preprocessor.joblib")
    lr = joblib.load(fold_dir / "logistic.joblib")
    import torch
    net = LSTMForecaster(len(meta["feature_names"]), HP["hidden_size"], HP["dropout"]).net
    net.load_state_dict(torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=True)); net.eval()
    ix = data.indices("test")
    y, cls, hosts = data.attack[ix], data.classes[ix], data.frame.host_id.to_numpy(str)[ix]
    onset = data.frame.target_is_onset.to_numpy(bool)[ix]
    xt = prep.transform(data.X[ix])
    with torch.no_grad():
        lstm_p = torch.sigmoid(net(torch.tensor(xt))).numpy()
    scores = {"logistic_regression": (lr.predict_proba(xt.reshape(len(xt), -1))[:, 1],
                                      metrics["models"]["logistic_regression"]["threshold"]),
              "lstm": (lstm_p, metrics["threshold"]),
              "ref_last_window_under_attack": (last_attack[ix], 0.5),
              "ref_host_identity": ((hosts == HOST).astype(float), 0.5)}
    res = {"test_metrics": {"majority": metrics["models"]["majority"]}, "host_breakdown": {}, "classes": {},
           "thresholds": {k: float(v[1]) for k, v in scores.items()},
           "lstm_training": {k: metrics["training"][k] for k in ("epochs_run", "best_checkpoint_epoch")},
           "lstm_history": metrics["history"]}
    for m, (p, t) in scores.items():
        res["test_metrics"][m] = metrics_at(y, p, t)
        res["host_breakdown"][m] = host_breakdown(y, p, t, hosts)
    neg = y == 0
    for c in sorted(set(cls[y == 1])):
        pc = (y == 1) & (cls == c)
        sub = data.frame.iloc[ix[pc]]
        entry = {"status": "SEEN in training" if c in train_classes else "ZERO-SHOT: class never seen in training",
                 "positives": int(pc.sum()), "onset_positives": int((pc & onset).sum()),
                 "distinct_hosts": int(sub.host_id.nunique()), "hosts": sorted(sub.host_id.unique().tolist()),
                 "campaigns": 1, "episodes_30min_gap": episodes(sub),
                 "insufficient_data": bool(pc.sum() < MIN_POS), "models": {}}
        for m in ("logistic_regression", "lstm", "ref_host_identity"):
            p, t = scores[m]
            on_host = pc & (hosts == HOST); off_host = pc & (hosts != HOST)
            host_neg = neg & (hosts == HOST)
            entry["models"][m] = {
                "detected": int((p[pc] >= t).sum()), "detection_rate": rate((p[pc] >= t).sum(), pc.sum()),
                "onset_detected": int((p[pc & onset] >= t).sum()),
                "roc_auc_vs_all_negatives": auc(np.r_[np.zeros(neg.sum()), np.ones(pc.sum())], np.r_[p[neg], p[pc]]),
                "same_day_fpr": rate((p[neg] >= t).sum(), neg.sum()),
                "detected_on_172.16.0.1": f"{int((p[on_host] >= t).sum())}/{int(on_host.sum())}",
                "detected_on_other_hosts": f"{int((p[off_host] >= t).sum())}/{int(off_host.sum())}",
                # host control: this class's positives on 172.16.0.1 vs that host's OWN negatives
                "roc_auc_vs_172.16.0.1_negatives": auc(np.r_[np.zeros(host_neg.sum()), np.ones(on_host.sum())],
                                                       np.r_[p[host_neg], p[on_host]]) if on_host.any() and host_neg.any() else None,
                "host_172.16.0.1_negatives": int(host_neg.sum()),
            }
        res["classes"][str(c)] = entry
    return res


def run() -> dict:
    frame = pd.read_parquet(DATA)
    for c in ("seq_start_time", "seq_end_time", "target_window_start", "target_window_end"):
        frame[c] = pd.to_datetime(frame[c])
    std = list(STANDARD_NUMERIC_WINDOW_FEATURES)
    last_attack = np.array([np.stack(v)[-1][std.index("attack_flow_ratio")] > 0 for v in frame.sequence_features], float)
    days = sorted(frame.seq_start_time.dt.date.unique())
    out = {"notes": __doc__.strip(), "hyperparameters": HP, "folds": {}}
    for day in days:
        t0 = time.time()
        name = f"{pd.Timestamp(day).strftime('%a')}_{day}"
        vday = validation_day(days, day)
        split = assign_fold(frame, day, vday)
        fold_dir = OUT / name
        counts = {}
        for s in ("train", "val", "test", "purged"):
            m = (split == s).to_numpy()
            counts[s] = {"total": int(m.sum()), "positive": int(frame.target_attack_present.to_numpy()[m].sum()),
                         "onset": int(frame.target_is_onset.to_numpy(bool)[m].sum()),
                         "positive_classes": frame.loc[m & (frame.target_attack_present == 1).to_numpy(),
                                                       "target_dominant_class"].value_counts().to_dict()}
        print(f"[{name}] counts: " + ", ".join(f"{s}={c['total']}/{c['positive']}" for s, c in counts.items()), flush=True)
        train_classes = set(counts["train"]["positive_classes"])
        if counts["test"]["total"] == 0:
            out["folds"][name] = {"counts": counts, "skipped": "no test sequences"}
            continue
        train_experiment(ForecastDataset.from_frame(frame.assign(split=split.values)), fold_dir, **HP)
        fold = {"test_day": str(day), "val_day": str(vday), "counts": counts,
                "train_positive_classes": sorted(train_classes),
                "val_has_positives": counts["val"]["positive"] > 0}
        fold.update(evaluate_fold(frame, split, fold_dir, train_classes, last_attack))
        fold["runtime_seconds"] = round(time.time() - t0, 1)
        out["folds"][name] = fold
        REPORT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"[{name}] done in {fold['runtime_seconds']} s", flush=True)
    REPORT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
    print(f"written: {REPORT}")
