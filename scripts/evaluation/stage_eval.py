"""Stage-label experiment: can the alert's attack stage be named more often than 34/79?

Scores stage labels on the demo LSTM's true-positive test alerts (attack probability >= the demo
threshold AND an attack really is in the target window), for:

  baseline       the multi-task model in artifacts/models/cic_ids2017_multitask (its stage head)
  weighted       the same multi-task model retrained with the stage loss weighted 5x (on top of the
                 inverse-frequency class weights it already uses), into a separate directory
  two_step       binary LSTM decides the alert; a separate stage classifier trained ONLY on attack
                 sequences of the committed train+val splits names the stage (never "Benign")
  two_step_cv    the same two-step classifier, but trained on attack sequences from every day with
                 grouped, purged cross-validation so each stage has training examples. Each held-out
                 group is a 20-minute block of one stage (Botnet / C2: one host), and training rows of
                 the same host within 15 minutes of the held-out block are dropped.
  two_step_cv_window  as two_step_cv, but the stage classifier also sees the alerted (target) window's
                 own traffic, i.e. it names the stage of an attack already in progress once that window
                 has been observed, instead of forecasting it from the 10 earlier windows.

Why two_step_cv exists: under the committed chronological split the test day's attack stages
(Reconnaissance, Command and Control) never occur in train or val, so no model trained on that split
can name them. two_step_cv shows what the design does once it has seen each stage, but it trains on
the test day and every attack stage except C2 comes from the one attacker host 172.16.0.1, so its
number is an upper bound on this dataset, not evidence of generalisation.

Nothing here changes the demo model, the multi-task model directory or configs/config.yaml.
Output: reports/stage_eval.json and a console table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # HistGradientBoosting thrashes on many small fits otherwise

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import MODEL_FEATURES, ForecastDataset, SequencePreprocessor  # noqa: E402
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES  # noqa: E402
from aegisflow.ml.multitask import MultiTaskData, MultiTaskPredictor, train_multitask  # noqa: E402

SEED = 42
BLOCK = pd.Timedelta(minutes=20)
PURGE = pd.Timedelta(minutes=15)


def sequence_summary(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Per-sequence flow-behaviour summary: last window, mean, max and std over the 10 windows."""
    x = np.asarray(X, dtype=np.float64).copy()
    for i, n in enumerate(names):
        if n in SequencePreprocessor.LOG_FEATURES:
            x[..., i] = np.sign(x[..., i]) * np.log1p(np.abs(x[..., i]))
    x = np.nan_to_num(x)
    return np.concatenate([x[:, -1], x.mean(1), x.max(1), x.std(1)], axis=1)


def stage_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                          random_state=SEED)


def grouped_purged_oof(frame: pd.DataFrame, feats: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Out-of-fold stage predictions for ``rows`` (indices into frame) with time-block / host groups."""
    sub = frame.iloc[rows]
    t = pd.to_datetime(sub["target_window_start"]).to_numpy()
    s0 = pd.to_datetime(sub["seq_start_time"]).to_numpy()
    host = sub["host_id"].astype(str).to_numpy()
    stage = y[rows]
    block = ((t - t.min()) // BLOCK.to_timedelta64()).astype(np.int64)
    group = np.array([f"c2|{h}" if st == "Command and Control" else f"{st}|{b}"
                      for st, h, b in zip(stage, host, block)])
    pred = np.empty(len(rows), dtype=object)
    for g in np.unique(group):
        test = group == g
        lo, hi = s0[test].min() - PURGE.to_timedelta64(), t[test].max() + PURGE.to_timedelta64()
        near = (np.isin(host, np.unique(host[test]))) & (t >= lo) & (s0 <= hi)
        train = ~test & ~near
        clf = stage_classifier().fit(feats[rows[train]], stage[train])
        pred[test] = clf.predict(feats[rows[test]])
    return pred


def score(truth: np.ndarray, pred: np.ndarray) -> dict:
    labels = sorted(set(truth))
    per = f1_score(truth, pred, labels=labels, average=None, zero_division=0)
    return {"exact": int((truth == pred).sum()), "of": int(len(truth)),
            "exact_rate": float((truth == pred).mean()),
            "macro_f1": float(per.mean()),
            "per_stage_f1": {s: float(v) for s, v in zip(labels, per)},
            "per_stage_correct": {s: f"{int(((truth == s) & (pred == s)).sum())}/{int((truth == s).sum())}"
                                  for s in labels},
            "predicted": pd.Series(pred).value_counts().to_dict()}


def run(data_path: Path, demo_dir: Path, mt_dir: Path, weighted_dir: Path, retrain: bool) -> dict:
    d = ForecastDataset.from_parquet(data_path)
    f = d.frame
    demo_meta = json.loads((demo_dir / "metadata.json").read_text(encoding="utf-8"))
    threshold = float(demo_meta["threshold"])
    audit = pd.read_csv(demo_dir / "audit_predictions.csv")
    test = audit[audit.split == "test"]
    alerts = test[test.attack_probability >= threshold]
    pos = pd.Series(np.arange(len(f)), index=f.sequence_id.astype(str))
    a_ix = pos[alerts.sequence_id.astype(str)].to_numpy()
    tp = a_ix[d.attack[a_ix] == 1]
    fp = a_ix[d.attack[a_ix] == 0]
    truth = d.stages[tp]
    out: dict = {"demo_threshold": threshold, "alerts": int(len(a_ix)), "true_positive_alerts": int(len(tp)),
                 "true_positive_stages": pd.Series(truth).value_counts().to_dict(),
                 "attack_stages_by_split": {s: pd.Series(d.stages[d.indices(s)][d.attack[d.indices(s)] == 1])
                                            .value_counts().to_dict() for s in ("train", "val", "test")}}
    trainable = set(out["attack_stages_by_split"]["train"]) | set(out["attack_stages_by_split"]["val"])
    out["committed_split_ceiling"] = {"exact": int(np.isin(truth, list(trainable)).sum()), "of": int(len(tp)),
                                      "why": "stages absent from train+val cannot be predicted"}

    res = {}
    mt = MultiTaskPredictor(mt_dir).predict(d.X[a_ix])
    k = np.isin(a_ix, tp)
    gated = np.where(mt["stage_probability"] >= 0.5, mt["stage"], "UNCERTAIN")
    res["baseline_gated"] = score(truth, gated[k].astype(str))
    res["baseline"] = score(truth, mt["stage"][k].astype(str))
    res["baseline"]["false_positive_alerts_labeled_benign"] = int((mt["stage"][~k] == "Benign").sum())

    if retrain:
        mtd = MultiTaskData.from_dir(data_path.parent, horizons=3)
        hp = json.loads((mt_dir / "metadata.json").read_text(encoding="utf-8"))["training_config"]
        train_multitask(mtd, weighted_dir, seed=SEED, epochs=hp["epochs_requested"], batch_size=hp["batch_size"],
                        learning_rate=hp["learning_rate"], hidden_size=hp["hidden_size"], dropout=hp["dropout"],
                        patience=hp["patience"], stage_weight=5.0, state_weight=hp["state_weight"])
    if (weighted_dir / "best.pt").exists():
        w = MultiTaskPredictor(weighted_dir).predict(d.X[a_ix])
        res["weighted"] = score(truth, w["stage"][k].astype(str))
        res["weighted"]["false_positive_alerts_labeled_benign"] = int((w["stage"][~k] == "Benign").sum())

    feats = sequence_summary(d.X, d.feature_names)
    split = f.split.to_numpy()
    fit = np.flatnonzero((d.attack == 1) & np.isin(split, ["train", "val"]))
    clf = stage_classifier().fit(feats[fit], d.stages[fit])
    res["two_step"] = score(truth, clf.predict(feats[tp]))
    res["two_step"]["trained_on"] = pd.Series(d.stages[fit]).value_counts().to_dict()
    fp_conf = clf.predict_proba(feats[fp]).max(1) if len(fp) else np.array([])
    tp_conf = clf.predict_proba(feats[tp]).max(1)
    res["two_step"]["mean_confidence"] = {"true_positive_alerts": float(tp_conf.mean()),
                                          "false_positive_alerts": float(fp_conf.mean()) if len(fp) else None}

    rows = np.flatnonzero(d.attack == 1)
    oof = pd.Series(grouped_purged_oof(f, feats, d.stages, rows), index=rows)
    res["two_step_cv"] = score(truth, oof[tp].to_numpy().astype(str))
    all_test = rows[split[rows] == "test"]
    res["two_step_cv"]["all_test_attack_sequences"] = score(d.stages[all_test], oof[all_test].to_numpy().astype(str))

    raw = list(STANDARD_NUMERIC_WINDOW_FEATURES)
    target = d.future_state[:, [raw.index(n) for n in MODEL_FEATURES]][:, None, :]
    feats_w = sequence_summary(np.concatenate([d.X, target], axis=1), d.feature_names)
    oof_w = pd.Series(grouped_purged_oof(f, feats_w, d.stages, rows), index=rows)
    res["two_step_cv_window"] = score(truth, oof_w[tp].to_numpy().astype(str))
    res["two_step_cv_window"]["all_test_attack_sequences"] = score(d.stages[all_test],
                                                                   oof_w[all_test].to_numpy().astype(str))
    out["results"] = res
    out["caveats"] = [
        "Stage labels are the label-to-stage proxy in configs/stages.yaml, not ground truth.",
        "Every attack stage except Command and Control (Botnet) comes from the single attacker host 172.16.0.1.",
        "two_step_cv trains on the test day (grouped and purged, but the same attack episodes); treat it as an "
        "upper bound on this dataset.",
        "two_step_cv_window uses the alerted window's own traffic, so it labels an attack in progress; it is "
        "not a forecast of the stage.",
    ]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=str(ROOT / "data/processed/cic_ids2017/sequences.parquet"))
    p.add_argument("--demo-dir", default=str(ROOT / "artifacts/models/cic_ids2017"))
    p.add_argument("--multitask-dir", default=str(ROOT / "artifacts/models/cic_ids2017_multitask"))
    p.add_argument("--weighted-dir", default=str(ROOT / "artifacts/experiments/cic_ids2017_multitask_stage5"))
    p.add_argument("--no-retrain", action="store_true", help="reuse --weighted-dir instead of retraining")
    p.add_argument("--output", default=str(ROOT / "reports/stage_eval.json"))
    a = p.parse_args()
    out = run(Path(a.data), Path(a.demo_dir), Path(a.multitask_dir), Path(a.weighted_dir), not a.no_retrain)
    Path(a.output).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"{out['true_positive_alerts']} true-positive alerts: {out['true_positive_stages']}")
    print(f"committed-split ceiling: {out['committed_split_ceiling']['exact']}/{out['committed_split_ceiling']['of']}")
    for name, r in out["results"].items():
        print(f"{name:16s} exact {r['exact']}/{r['of']}  macro-F1 {r['macro_f1']:.3f}  {r['per_stage_correct']}")
    print(f"Wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
