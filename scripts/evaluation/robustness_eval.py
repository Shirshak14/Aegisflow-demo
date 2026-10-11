"""Robustness of the committed demo LSTM: threshold sweep, results by capture day, noisy and missing inputs.

Nothing is retrained and no config or threshold changes. Three parts:

  thresholds   precision / recall / F1 / FPR / alert count of the committed model over a grid of alert
               thresholds, on the validation split (where 0.3308 was chosen) and the test split, plus a
               bootstrap interval for test F1 at 0.3308. Read from ``audit_predictions.csv`` only.
  by_day       the same metrics per capture day (date of the target window) and per test hour, at 0.3308.
               Train-split days are in-sample (the model was fit on them) and are marked as such. The
               chronological split puts ALL test data on one day (Fri 2017-07-07) and all validation data
               on Thu 2017-07-06, so this is not a multi-day held-out test. Read from ``audit_predictions.csv``.
  perturbation the test split re-scored after corrupting the raw input features: multiplicative log-normal
               noise, cells missing at random, every value of one feature missing, and whole windows
               missing. Missing values go through the model's own train-median imputation
               (``SequencePreprocessor``), as unseen gaps would in deployment. Needs the processed
               ``sequences.parquet``; when it is absent the previous perturbation section is kept as is.

Output: reports/robustness_eval.json (read by GET /evaluation/robustness). Nothing here changes the replay.

    python scripts/evaluation/robustness_eval.py [--data PATH] [--model-dir PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import metrics_at

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
ARTIFACT = ROOT / "artifacts/models/cic_ids2017"
OUT = ROOT / "reports/robustness_eval.json"
SEED = 42
SHORTCUT_HOST = "172.16.0.1"
GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
BOOTSTRAP = 1000
SEEDS = 5
NOISE_SIGMAS = [0.05, 0.1, 0.25, 0.5, 1.0]
MISSING_RATES = [0.05, 0.1, 0.25, 0.5]
WINDOWS_MISSING = [1, 3, 5]
FEATURE_GROUPS = {
    "volume": ["flow_count", "packet_count_sum", "packet_count_mean", "byte_count_sum", "byte_count_mean"],
    "rates_and_timing": ["duration_mean", "packets_per_second_mean", "bytes_per_second_mean", "inter_arrival_mean",
                         "connection_burst_max"],
    "packet_size": ["bytes_per_packet_mean", "packet_size_mean", "packet_size_std_mean", "tcp_window_size_mean"],
    "tcp_flags": ["syn_count_sum", "ack_count_sum", "rst_count_sum", "fin_count_sum", "psh_count_sum",
                  "syn_ratio", "ack_ratio", "rst_ratio", "fin_ratio", "psh_ratio"],
    "destination_diversity": ["unique_destination_ips", "unique_destination_ports", "unique_source_ports",
                              "destination_port_entropy"],
}


def summary(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    m = metrics_at(y, p, t)
    tn, fp = m["confusion_matrix"][0]
    fn, tp = m["confusion_matrix"][1]
    return {"n": int(len(y)), "positives": int(y.sum()), "alerts": m["positive_predictions"], "tp": tp, "fp": fp,
            "fn": fn, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "false_positive_rate": m["false_positive_rate"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]}


def host_of(sequence_id: pd.Series) -> np.ndarray:
    return sequence_id.str.rsplit("_seq_", n=1).str[0].to_numpy(str)


def threshold_part(audit: pd.DataFrame, t0: float) -> dict:
    out = {"committed_threshold": t0, "grid": sorted(set(GRID) | {t0}), "splits": {}}
    for split in ("val", "test"):
        a = audit[audit.split == split]
        y, p = a.target_attack_present.to_numpy(int), a.attack_probability.to_numpy(float)
        rows = [summary(y, p, t) | {"threshold": t} for t in out["grid"]]
        best = max(rows, key=lambda r: r["f1"])
        out["splits"][split] = {"rows": rows, "best_f1_threshold": best["threshold"], "best_f1": best["f1"]}

    # Bootstrap interval for test F1 / recall / precision at the committed threshold (sequences resampled).
    a = audit[audit.split == "test"]
    y, p = a.target_attack_present.to_numpy(int), a.attack_probability.to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = {"f1": [], "precision": [], "recall": [], "roc_auc": []}
    for _ in range(BOOTSTRAP):
        ix = rng.integers(0, len(y), len(y))
        m = metrics_at(y[ix], p[ix], t0)
        for k in draws:
            draws[k].append(m[k])
    out["test_bootstrap_95ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                  for k, v in draws.items()}
    out["bootstrap_note"] = (f"{BOOTSTRAP} resamples of test sequences. Sequences from the same host overlap in time, "
                             "so these intervals are narrower than the real uncertainty.")
    return out


def by_day_part(audit: pd.DataFrame, t0: float) -> dict:
    a = audit.copy()
    ts = pd.to_datetime(a.target_window_start)
    a["day"], a["hour"] = ts.dt.strftime("%Y-%m-%d"), ts.dt.hour
    a["host"] = host_of(a.sequence_id)
    days = []
    for (day, split), g in a.groupby(["day", "split"], sort=True):
        y, p = g.target_attack_present.to_numpy(int), g.attack_probability.to_numpy(float)
        h = g.host.to_numpy() == SHORTCUT_HOST
        days.append({"day": day, "weekday": pd.Timestamp(day).day_name()[:3], "split": split,
                     "in_sample": split == "train", **summary(y, p, t0),
                     "positives_from_172_16_0_1": int(y[h].sum()), "alerts_on_172_16_0_1": int((p[h] >= t0).sum()),
                     "tp_on_172_16_0_1": int(((p >= t0) & h & (y == 1)).sum())})
    hours = []
    test = a[a.split == "test"]
    for hour, g in test.groupby("hour", sort=True):
        y, p = g.target_attack_present.to_numpy(int), g.attack_probability.to_numpy(float)
        h = g.host.to_numpy() == SHORTCUT_HOST
        hours.append({"hour": int(hour), **summary(y, p, t0),
                      "positives_from_172_16_0_1": int(y[h].sum()), "alerts_on_172_16_0_1": int((p[h] >= t0).sum()),
                      "tp_on_172_16_0_1": int(((p >= t0) & h & (y == 1)).sum())})
    return {"days": days, "test_hours": hours,
            "note": ("Chronological split: train = Mon-Thu morning (in-sample), validation = Thu 2017-07-06, test = "
                     "Fri 2017-07-07 only. Each held-out split is a single day, so this shows how the score moves "
                     "between days, not how it generalises to unseen days. Monday has no attacks.")}


def perturbation_part(t0: float) -> dict | None:
    if not DATA.exists():
        return None
    import joblib
    import torch

    from aegisflow.ml.modeling import ForecastDataset, load_model, score_batched

    data = ForecastDataset.from_parquet(DATA)
    test = data.indices("test")
    X, y = data.X[test].astype(np.float64), data.attack[test]
    classes, hosts = data.classes[test], data.frame.host_id.to_numpy(str)[test]
    names = data.feature_names
    prep = joblib.load(ARTIFACT / "preprocessor.joblib")
    net, meta = load_model(ARTIFACT)
    if list(meta["feature_names"]) != list(names):
        raise SystemExit("feature names in metadata.json differ from the sequence table")

    def score(x: np.ndarray) -> np.ndarray:
        return score_batched(lambda b: torch.sigmoid(net(b)), prep.transform(x))

    clean = score(X)
    audit = pd.read_csv(ARTIFACT / "audit_predictions.csv", usecols=["sequence_id", "attack_probability"])
    ref = data.frame.iloc[test][["sequence_id"]].merge(audit, on="sequence_id", how="left").attack_probability
    max_diff = float(np.nanmax(np.abs(ref.to_numpy() - clean)))
    clean_alert = clean >= t0
    hmask = hosts == SHORTCUT_HOST

    def evaluate(p: np.ndarray) -> dict:
        m = summary(y, p, t0)
        alert = p >= t0
        m["alerts_changed_vs_clean"] = int((alert != clean_alert).sum())
        m["roc_auc_hosts_other_than_172_16_0_1"] = (
            float(roc_auc_score(y[~hmask], p[~hmask])) if 0 < y[~hmask].sum() < (~hmask).sum() else None)
        m["detected_by_class"] = {str(c): int((alert & (classes == c) & (y == 1)).sum())
                                  for c in sorted(set(classes[y == 1]))}
        return m

    def repeated(make) -> dict:
        runs = [evaluate(score(make(np.random.default_rng(SEED + s)))) for s in range(SEEDS)]
        keys = ("precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc", "alerts",
                "alerts_changed_vs_clean", "roc_auc_hosts_other_than_172_16_0_1")
        agg = {k: {"mean": float(np.mean([r[k] for r in runs])), "min": float(np.min([r[k] for r in runs])),
                   "max": float(np.max([r[k] for r in runs]))} for k in keys}
        agg["detected_by_class_mean"] = {c: float(np.mean([r["detected_by_class"][c] for r in runs]))
                                         for c in runs[0]["detected_by_class"]}
        return agg

    out = {"clean": evaluate(clean), "max_abs_diff_vs_audit_predictions": max_diff, "seeds": SEEDS,
           "test_attacks_by_class": {str(c): int(((classes == c) & (y == 1)).sum()) for c in sorted(set(classes[y == 1]))}}

    out["noise"] = [{"sigma": s, **repeated(lambda r, s=s: X * np.exp(r.normal(0.0, s, X.shape)))}
                    for s in NOISE_SIGMAS]

    def drop_cells(r, rate):
        x = X.copy()
        x[r.random(X.shape) < rate] = np.nan
        return x

    out["missing_cells"] = [{"rate": q, **repeated(lambda r, q=q: drop_cells(r, q))} for q in MISSING_RATES]

    def drop_windows(r, k):
        x = X.copy()
        for i in range(len(x)):
            x[i, r.choice(x.shape[1], k, replace=False), :] = np.nan
        return x

    out["missing_windows"] = [{"windows": k, **repeated(lambda r, k=k: drop_windows(r, k))} for k in WINDOWS_MISSING]
    x = X.copy()
    x[:, -1, :] = np.nan
    out["missing_last_window"] = evaluate(score(x))

    def without(cols: list[str]) -> np.ndarray:
        x = X.copy()
        x[:, :, [names.index(c) for c in cols]] = np.nan
        return x

    out["missing_feature"] = sorted(({"feature": c, **evaluate(score(without([c])))} for c in names),
                                    key=lambda r: r["roc_auc"])
    out["missing_group"] = [{"group": g, "features": cols, **evaluate(score(without(cols)))}
                            for g, cols in FEATURE_GROUPS.items()]
    out["note"] = ("Test split re-scored by the committed model with corrupted raw inputs; alert threshold fixed at "
                   f"{t0:.4f}. Noise multiplies every raw value by exp(N(0, sigma)); missing values are filled with "
                   "the model's train-split medians. Ranges are over seeds.")
    return out


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    global DATA, ARTIFACT, OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA, help="processed sequences.parquet")
    ap.add_argument("--model-dir", type=Path, default=ARTIFACT, help="committed demo model directory")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    DATA, ARTIFACT, OUT = args.data.resolve(), args.model_dir.resolve(), args.out.resolve()
    meta = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    t0 = float(meta["threshold"])
    audit = pd.read_csv(ARTIFACT / "audit_predictions.csv")
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    report = {
        "notes": __doc__.strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": _rel(ARTIFACT),
        "thresholds": threshold_part(audit, t0),
        "by_day": by_day_part(audit, t0),
    }
    pert = perturbation_part(t0)
    if pert is not None:
        report["perturbation"] = pert
        report["perturbation_generated_at"] = report["generated_at"]
    elif "perturbation" in previous:
        report["perturbation"] = previous["perturbation"]
        report["perturbation_generated_at"] = previous.get("perturbation_generated_at")
        print(f"{DATA} not found: kept the previous perturbation results")
    else:
        print(f"{DATA} not found: perturbation tests skipped")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    th = report["thresholds"]
    print(f"threshold {t0:.4f}; best test F1 {th['splits']['test']['best_f1']:.3f} at "
          f"{th['splits']['test']['best_f1_threshold']}; test F1 95% CI {th['test_bootstrap_95ci']['f1']}")
    for d in report["by_day"]["days"]:
        print(f"{d['day']} {d['split']:5s} n={d['n']:6d} pos={d['positives']:4d} alerts={d['alerts']:5d} "
              f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} FPR={d['false_positive_rate']:.4f}")
    if "perturbation" in report:
        p = report["perturbation"]
        print(f"clean test F1 {p['clean']['f1']:.3f} ROC-AUC {p['clean']['roc_auc']:.3f} "
              f"(max diff vs audit {p['max_abs_diff_vs_audit_predictions']:.2e})")
        for r in p["noise"]:
            print(f"noise sigma={r['sigma']}: F1 {r['f1']['mean']:.3f} ROC-AUC {r['roc_auc']['mean']:.3f} "
                  f"alerts changed {r['alerts_changed_vs_clean']['mean']:.0f}")
        for r in p["missing_cells"]:
            print(f"missing {r['rate']:.0%} cells: F1 {r['f1']['mean']:.3f} ROC-AUC {r['roc_auc']['mean']:.3f}")
    print(f"wrote {_rel(OUT)}")


if __name__ == "__main__":
    main()
