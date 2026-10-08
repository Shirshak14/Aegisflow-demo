"""B1 experiment: does a host-relative input representation remove the host shortcut?

Trains the existing LSTM + logistic regression on host-relative inputs
(``ForecastDataset.from_parquet(..., host_relative=True)``) into a SEPARATE artifact
directory so the committed demo model is untouched, then reports the same host check as
``phase3_baselines.py`` for both representations. Nothing here is read by the backend.

Output: reports/hostrel_eval.json and a console table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from aegisflow.ml.modeling import ForecastDataset, LSTMForecaster, metrics_at, train_experiment  # noqa: E402
from phase3_baselines import SHORTCUT_HOST, per_host  # noqa: E402

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
BASE_DIR = ROOT / "artifacts/models/cic_ids2017"
REL_DIR = ROOT / "artifacts/experiments/cic_ids2017_hostrel"
OUT = ROOT / "reports/hostrel_eval.json"


def score(model_dir: Path, data: ForecastDataset):
    import joblib, torch
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    prep = joblib.load(model_dir / "preprocessor.joblib")
    lr = joblib.load(model_dir / "logistic.joblib")
    tc = meta["training_config"]
    net = LSTMForecaster(len(meta["feature_names"]), tc["hidden_size"], tc["dropout"]).net
    net.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)); net.eval()
    ix = data.indices("test"); x = prep.transform(data.X[ix])
    with torch.no_grad():
        lstm = torch.sigmoid(net(torch.tensor(x))).numpy()
    return ix, {"lstm": lstm, "logistic_regression": lr.predict_proba(x.reshape(len(x), -1))[:, 1]}, float(meta["threshold"])


def run() -> dict:
    hp = {"epochs": 20, "batch_size": 128, "learning_rate": 0.001, "hidden_size": 32, "dropout": 0.2, "patience": 4}
    try:  # same hyperparameters the committed model was trained with
        hp = json.loads((BASE_DIR / "metadata.json").read_text(encoding="utf-8"))["training_config"]
        hp = {k: hp[k] for k in ("epochs_requested", "batch_size", "learning_rate", "hidden_size", "dropout", "patience")}
        hp["epochs"] = hp.pop("epochs_requested")
    except Exception:
        pass
    rel = ForecastDataset.from_parquet(DATA, host_relative=True)
    train_experiment(rel, REL_DIR, seed=42, **hp)
    base = ForecastDataset.from_parquet(DATA)
    out = {"hyperparameters": hp, "representations": {}}
    for name, model_dir, data in (("absolute (committed)", BASE_DIR, base), ("host-relative (B1)", REL_DIR, rel)):
        ix, scores, thr = score(model_dir, data)
        y = data.attack[ix]; hosts = data.frame.host_id.to_numpy(str)[ix]; cls = data.classes[ix]
        entry = {"threshold": thr}
        for m, p in scores.items():
            t = thr if m == "lstm" else json.loads((model_dir / "metrics.json").read_text())["models"][m]["threshold"]
            entry[m] = {"test": metrics_at(y, p, t), "host_check": per_host(y, p, hosts, cls, t)}
        out["representations"][name] = entry
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run()
    for name, e in res["representations"].items():
        print(f"\n=== {name} ===")
        for m in ("logistic_regression", "lstm"):
            t = e[m]["test"]; h = e[m]["host_check"]["shortcut_host_vs_rest"]
            s, o = h[SHORTCUT_HOST], h["other_hosts"]
            print(f"  {m:20s} ROC={t['roc_auc']:.3f} PR={t['pr_auc']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} FPR={t['false_positive_rate']:.4f}")
            print(f"      {SHORTCUT_HOST}: benign flagged {s['neg_flagged']}/{s['neg']}, attacks detected {s['pos_detected']}/{s['pos']}, ROC {s['roc_auc']}")
            print(f"      other hosts: benign flagged {o['neg_flagged']}/{o['neg']}, attacks detected {o['pos_detected']}/{o['pos']}, ROC {o['roc_auc']}")
            print(f"      per class: {e[m]['host_check']['per_class_detection']}")
    print(f"\nwritten: {OUT}")
