"""Benign-only anomaly baseline: can a model that never sees an attack label detect attacks?

The supervised models (LSTM, logistic regression) can learn "which host" instead of "what
attack", because nearly every attack class comes from one attacker host (172.16.0.1). These
detectors are fit on benign traffic only, so they cannot learn that shortcut from labels:

  isolation_forest  sklearn IsolationForest on the flattened, preprocessed input sequence
  autoencoder       dense autoencoder (280 -> 64 -> 16 -> 64 -> 280); score = reconstruction MSE

Fit set: train-split sequences whose target is benign AND whose 10 input windows contain no
attack flow (``attack_flow_ratio == 0`` in every window). Same inputs, preprocessing and
split as the committed model; the label-derived features never enter the model.

Thresholds (none uses an attack label):
  benign_p99     99th percentile of the scores on the benign fit set (nominal 1% FPR)
  lstm_fpr       the score at which the test-split benign FPR equals the committed LSTM's test FPR
                 (2.5%). It uses which test sequences are benign, so it is a comparison point for
                 recall at equal false alarms, not a deployable threshold

Because no attack label is used to fit, attacks from every split can be scored, so the
per-class table covers all six classes (train-split attacks were never in the fit set).
The committed LSTM is read from ``audit_predictions.csv`` and is NOT retrained.

Output: reports/anomaly_eval.json and a console table. Nothing here is read by the backend.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import ForecastDataset, SequencePreprocessor, metrics_at
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
ARTIFACT = ROOT / "artifacts/models/cic_ids2017"
OUT = ROOT / "reports/anomaly_eval.json"
SEED = 42
SHORTCUT_HOST = "172.16.0.1"


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def input_attack_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where any of the sequence's input windows carries an attack flow (labels only, for the fit filter)."""
    col = list(STANDARD_NUMERIC_WINDOW_FEATURES).index("attack_flow_ratio")
    return np.array([np.nan_to_num(np.stack(v)[:, col]).max() > 0 for v in frame.sequence_features])


def fit_autoencoder(x: np.ndarray, epochs: int = 30, batch_size: int = 256, lr: float = 1e-3, seed: int = SEED):
    import torch
    from torch import nn

    d = x.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d)
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    # Hold out 10% of the benign fit set for early stopping (still benign only).
    order = rng.permutation(len(x))
    hold, fit = order[: len(x) // 10], order[len(x) // 10 :]
    xt = torch.tensor(x)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(fit), max(1, len(fit) // batch_size)):
            opt.zero_grad()
            loss = ((net(xt[b]) - xt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            h = float(((net(xt[hold]) - xt[hold]) ** 2).mean())
        if h < best - 1e-4:
            best, best_state, bad = h, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()

    def score(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = []
            for i in range(0, len(z), 4096):
                zt = torch.tensor(z[i : i + 4096])
                out.append(((net(zt) - zt) ** 2).mean(dim=1).numpy())
            return np.concatenate(out)

    return score


def host_and_class(y, p, hosts, classes, t) -> dict:
    groups = {}
    for name, m in ((SHORTCUT_HOST, hosts == SHORTCUT_HOST), ("other_hosts", hosts != SHORTCUT_HOST)):
        neg, pos = m & (y == 0), m & (y == 1)
        groups[name] = {
            "neg": int(neg.sum()),
            "neg_flagged": int((p[neg] >= t).sum()),
            "pos": int(pos.sum()),
            "pos_detected": int((p[pos] >= t).sum()),
            "roc_auc": float(roc_auc_score(y[m], p[m])) if 0 < y[m].sum() < m.sum() else None,
        }
    per_class = {
        str(c): {
            "n": int(((classes == c) & (y == 1)).sum()),
            "detected": int((p[(classes == c) & (y == 1)] >= t).sum()),
        }
        for c in sorted(set(classes[y == 1]))
    }
    return {"shortcut_host_vs_rest": groups, "per_class_detection": per_class}


def threshold_at_fpr(benign_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(benign_scores, 1.0 - fpr))


def run() -> dict:
    seed_all()
    data = ForecastDataset.from_parquet(DATA)
    f = data.frame
    split = f.split.to_numpy()
    y = data.attack
    hosts = f.host_id.to_numpy(str)
    contaminated = input_attack_mask(f)
    fit_ix = np.flatnonzero((split == "train") & (y == 0) & ~contaminated)

    prep = SequencePreprocessor(data.feature_names).fit(data.X[fit_ix])
    flat = prep.transform(data.X).reshape(len(f), -1)

    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(flat[fit_ix])
    ae_score = fit_autoencoder(flat[fit_ix])
    scores = {"isolation_forest": -iso.score_samples(flat), "autoencoder": ae_score(flat)}

    # Committed LSTM, as stored by the audit (test split only for the head-to-head table).
    audit = pd.read_csv(ARTIFACT / "audit_predictions.csv", usecols=["sequence_id", "attack_probability"])
    lstm_meta = json.loads((ARTIFACT / "metrics.json").read_text(encoding="utf-8"))
    lstm_t = float(lstm_meta["threshold"])
    lstm = f[["sequence_id"]].merge(audit, on="sequence_id", how="left").attack_probability.to_numpy()
    test = split == "test"
    lstm_test_fpr = float(lstm_meta["models"]["lstm"]["false_positive_rate"])

    heldout_benign = (split != "train") & (y == 0)
    out = {
        "notes": __doc__.strip(),
        "fit_set": {
            "sequences": len(fit_ix),
            "excluded_train_benign_target_but_attack_in_inputs": int(
                ((split == "train") & (y == 0) & contaminated).sum()
            ),
        },
        "test_counts": {"positive": int(y[test].sum()), "negative": int((y[test] == 0).sum())},
        "lstm_committed": {
            "threshold": lstm_t,
            "test": metrics_at(y[test], lstm[test], lstm_t),
            "test_host_and_class": host_and_class(y[test], lstm[test], hosts[test], data.classes[test], lstm_t),
        },
        "models": {},
    }

    for name, s in scores.items():
        th = {
            "benign_p99": float(np.quantile(s[fit_ix], 0.99)),
            "lstm_fpr": threshold_at_fpr(s[test & (y == 0)], lstm_test_fpr),
        }
        r = {
            "thresholds": th,
            "test_roc_auc": float(roc_auc_score(y[test], s[test])),
            "test_pr_auc": float(average_precision_score(y[test], s[test])),
            "heldout_benign_fpr": {k: float((s[heldout_benign] >= t).mean()) for k, t in th.items()},
        }
        for k, t in th.items():
            m = metrics_at(y[test], s[test], t)
            m.pop("roc_auc", None), m.pop("pr_auc", None)  # metrics_at assumes probabilities; AUCs reported above
            r[f"test_at_{k}"] = m
            r[f"test_host_and_class_at_{k}"] = host_and_class(y[test], s[test], hosts[test], data.classes[test], t)
            # Every attack in every split (none was in the fit set) vs held-out benign.
            allm = (y == 1) | heldout_benign
            r[f"all_splits_host_and_class_at_{k}"] = host_and_class(
                y[allm], s[allm], hosts[allm], data.classes[allm], t
            )
        r["all_splits_roc_auc"] = float(roc_auc_score(y[(y == 1) | heldout_benign], s[(y == 1) | heldout_benign]))
        per_class_auc = {}
        for c in sorted(set(data.classes[y == 1])):
            m = ((data.classes == c) & (y == 1)) | heldout_benign
            per_class_auc[str(c)] = float(roc_auc_score(y[m], s[m]))
        r["all_splits_per_class_roc_auc_vs_heldout_benign"] = per_class_auc
        # The host question without labels: is 172.16.0.1's own benign traffic already anomalous?
        sh_b = (hosts == SHORTCUT_HOST) & heldout_benign
        r["shortcut_host_benign_median_percentile"] = (
            float((s[heldout_benign] < np.median(s[sh_b])).mean()) if sh_b.any() else None
        )
        out["models"][name] = r

    # Seed stability of the threshold-free test numbers.
    stab = {"isolation_forest": [], "autoencoder": []}
    for sd in range(5):
        seed_all(sd)
        s = -IsolationForest(n_estimators=300, random_state=sd, n_jobs=-1).fit(flat[fit_ix]).score_samples(flat)
        stab["isolation_forest"].append([roc_auc_score(y[test], s[test]), average_precision_score(y[test], s[test])])
    for sd in range(3):
        seed_all(sd)
        s = fit_autoencoder(flat[fit_ix], seed=sd)(flat)
        stab["autoencoder"].append([roc_auc_score(y[test], s[test]), average_precision_score(y[test], s[test])])
    out["seed_stability_test_roc_pr"] = {
        k: {
            "roc_auc": [min(r[0] for r in v), max(r[0] for r in v)],
            "pr_auc": [min(r[1] for r in v), max(r[1] for r in v)],
            "seeds": len(v),
        }
        for k, v in stab.items()
    }

    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run()
    print(f"fit set: {res['fit_set']}")
    lt = res["lstm_committed"]["test"]
    print(
        f"LSTM (committed) test: ROC={lt['roc_auc']:.3f} PR={lt['pr_auc']:.3f} P={lt['precision']:.3f} "
        f"R={lt['recall']:.3f} FPR={lt['false_positive_rate']:.4f}"
    )
    print(f"  {res['lstm_committed']['test_host_and_class']}")
    for name, r in res["models"].items():
        print(
            f"\n== {name}: test ROC={r['test_roc_auc']:.3f} PR={r['test_pr_auc']:.3f}  all-splits ROC={r['all_splits_roc_auc']:.3f}"
        )
        print(
            f"   per-class ROC (all splits vs held-out benign): {json.dumps({k: round(v, 3) for k, v in r['all_splits_per_class_roc_auc_vs_heldout_benign'].items()})}"
        )
        print(f"   172.16.0.1 benign median sits at percentile {r['shortcut_host_benign_median_percentile']}")
        for k in r["thresholds"]:
            m = r[f"test_at_{k}"]
            print(
                f"   [{k}] test P={m['precision']:.3f} R={m['recall']:.3f} FPR={m['false_positive_rate']:.4f} "
                f"held-out benign FPR={r['heldout_benign_fpr'][k]:.4f}"
            )
            print(f"      test: {r[f'test_host_and_class_at_{k}']}")
            print(f"      all : {r[f'all_splits_host_and_class_at_{k}']['per_class_detection']}")
    print(f"\nseed stability (min, max): {res['seed_stability_test_roc_pr']}")
    print(f"\nwritten: {OUT}")
