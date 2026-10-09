"""Host-disjoint evaluation (docs/host_disjoint_evaluation_design.md, D1-D7).

Answers one question: does the score rank attack above benign *within a host the model
never saw*, better than a rule that only knows host identity? Only Botnet has enough
attacker hosts for this; every other class is reported as not evaluable.

Parts:
  committed -- D1/D2/D4 on the existing time split, scoring the committed demo model
               (no retraining): per-host ROC-AUC with the 20/20 minimum, the identity
               baseline on the pooled test set, and the 172.16.0.1 within-host control.
  folds     -- D3: leave-one-host-out over the 5 internal Botnet hosts with >= 20 attack
               targets. Each fold trains the existing LSTM + logistic regression with
               ``train_experiment`` (committed hyperparameters) into its own directory.
  permuted  -- D6: the same folds with training and validation labels shuffled. The design
               says "shuffle host labels"; literally shuffling host IDs before building
               folds would put every real host in both train and test, so this implements
               the label-shuffle null instead: held-out performance must fall to chance.
  time_control -- added after the first results, not pre-registered: re-scores the real fold
               models against the same host's benign windows inside the Botnet hours (see
               ``time_control_part``). Needs the ``folds`` models on disk.

Fold for held-out Botnet host H (hosts listed in BOTNET_HOSTS, V = the next one in the list):
  test  -- every sequence of H
  val   -- every sequence of V (selects the checkpoint and the 1%-FPR threshold)
  train -- every sequence of every other host
Sequences whose target is a non-Botnet attack are dropped everywhere (positives are
Botnet only, per D3). Hosts are disjoint between train, val and test, so no purge gap is
needed. Folds are host-disjoint, NOT campaign-disjoint: all Botnet hosts belong to one
Friday campaign with one command-and-control server.

Intervals (D5): 2,000 bootstrap resamples, stratified by host. Positives are resampled by
attack episode (consecutive attack targets on one host less than 30 minutes apart, as in
lodo_pilot.py). Benign targets cannot use that rule (a host's benign windows form one
continuous run), so they are resampled in 30-minute clock blocks per host. On this data every
Botnet host's positives form a single episode, so the intervals carry no positive-side
uncertainty; the report says so.

Writes artifacts/experiments/cic_ids2017_host_disjoint/<fold>/ and
reports/host_disjoint_eval.json. Nothing here is read by the backend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import ForecastDataset, LSTMForecaster, train_experiment  # noqa: E402

DATA = ROOT / "data/processed/cic_ids2017/sequences.parquet"
COMMITTED = ROOT / "artifacts/models/cic_ids2017"
OUT_DIR = ROOT / "artifacts/experiments/cic_ids2017_host_disjoint"
REPORT = ROOT / "reports/host_disjoint_eval.json"

CLASS = "Botnet"
BOTNET_HOSTS = ["192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14", "192.168.10.15"]
MIN_PER_LABEL = 20            # D1, fixed before running
BOOTSTRAP = 2000              # D5
SEED = 42
FPR_TARGET = 0.01
SHORTCUT_HOST = "172.16.0.1"
HP = dict(seed=SEED, epochs=8, batch_size=256, learning_rate=0.001, hidden_size=16, dropout=0.2, patience=3)


# ---------------------------------------------------------------- fold construction
def assign_host_fold(frame: pd.DataFrame, test_host: str, val_host: str, cls: str = CLASS) -> pd.Series:
    """Split label per row: 'test' (test_host), 'val' (val_host), 'train', or 'dropped'.

    Rows whose target is an attack of another class are 'dropped' so the only positives
    anywhere are ``cls``.
    """
    if test_host == val_host:
        raise ValueError("test and validation host must differ")
    hosts = frame.host_id.astype(str)
    split = pd.Series("train", index=frame.index, dtype=object)
    split[hosts == val_host] = "val"
    split[hosts == test_host] = "test"
    other_attack = (frame.target_attack_present == 1) & (frame.target_dominant_class.astype(str) != cls)
    split[other_attack] = "dropped"
    return split


# ---------------------------------------------------------------- resampling units
def resampling_units(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Cluster id per row: attack episodes (30-min gap) for positives, 30-min clock blocks for negatives."""
    t = pd.to_datetime(frame.target_window_start).reset_index(drop=True)
    hosts = frame.host_id.astype(str).reset_index(drop=True)
    unit = np.empty(len(frame), dtype=object)
    neg = np.flatnonzero(y == 0)
    unit[neg] = [f"{h}|neg|{b}" for h, b in zip(hosts.iloc[neg], t.iloc[neg].dt.floor("30min"))]
    pos = np.flatnonzero(y == 1)
    if len(pos):
        order = pos[np.lexsort((t.iloc[pos].to_numpy(), hosts.iloc[pos].to_numpy()))]
        hs, ts = hosts.iloc[order].to_numpy(), t.iloc[order]
        new = (hs != np.r_[None, hs[:-1]]) | (ts.diff() > pd.Timedelta("30min")).to_numpy()
        ep = np.cumsum(new)
        unit[order] = [f"{h}|pos|{e}" for h, e in zip(hs, ep)]
    return unit


def auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < int(np.sum(y)) < len(y) else None


def _cluster_index(units: np.ndarray) -> dict:
    groups: dict = {}
    for i, u in enumerate(units):
        groups.setdefault(u, []).append(i)
    return {k: np.asarray(v) for k, v in groups.items()}


def bootstrap_host_aucs(y, scores: dict, hosts, units, rng, n=BOOTSTRAP):
    """Resample units within each host and label; return {score_name: array (n, n_hosts)} of AUCs."""
    host_list = list(dict.fromkeys(hosts))
    plan = []
    for h in host_list:
        groups = []
        for lab in (1, 0):
            idx = np.flatnonzero((hosts == h) & (y == lab))
            groups.append([idx[local] for local in _cluster_index(units[idx].astype(str)).values()])
        plan.append(groups)
    out = {k: np.full((n, len(host_list)), np.nan) for k in scores}
    for b in range(n):
        for j, (pc, nc) in enumerate(plan):
            pi = np.concatenate([pc[i] for i in rng.integers(0, len(pc), len(pc))])
            ni = np.concatenate([nc[i] for i in rng.integers(0, len(nc), len(nc))])
            yy = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
            for k, p in scores.items():
                out[k][b, j] = roc_auc_score(yy, np.r_[p[pi], p[ni]])
    return host_list, out


def ci(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


# ---------------------------------------------------------------- scoring
def score_model_dir(model_dir: Path, data: ForecastDataset, ix: np.ndarray) -> dict:
    import joblib
    import torch
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    tc = meta["training_config"]
    prep = joblib.load(model_dir / "preprocessor.joblib")
    lr = joblib.load(model_dir / "logistic.joblib")
    net = LSTMForecaster(len(meta["feature_names"]), tc["hidden_size"], tc["dropout"]).net
    net.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)); net.eval()
    x = prep.transform(data.X[ix])
    with torch.no_grad():
        lstm = torch.sigmoid(net(torch.tensor(x))).numpy()
    return {"lstm": lstm, "logistic_regression": lr.predict_proba(x.reshape(len(x), -1))[:, 1]}


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float = FPR_TARGET) -> float:
    """Smallest threshold whose FPR on these negatives is <= fpr (predict positive when p >= t)."""
    s = np.sort(np.asarray(neg_scores, float))[::-1]
    k = int(np.floor(fpr * len(s)))          # at most k negatives may score strictly above s[k]
    return float(np.nextafter(s[min(k, len(s) - 1)], np.inf))


# ---------------------------------------------------------------- part 1: committed model
def committed_part(base: ForecastDataset, rng) -> dict:
    f = base.frame
    ix = base.indices("test")
    y = base.attack[ix]; hosts = f.host_id.to_numpy(str)[ix]
    scores = score_model_dir(COMMITTED, base, ix)
    train_attackers = sorted(set(f.host_id.to_numpy(str)[base.indices("train")][base.attack[base.indices("train")] == 1]))
    scores["ref_host_identity"] = np.isin(hosts, train_attackers).astype(float)
    units = resampling_units(f.iloc[ix], y)
    per_host = {}
    for h in pd.unique(hosts):
        m = hosts == h
        pos, neg = int(y[m].sum()), int((y[m] == 0).sum())
        if pos and neg:
            per_host[str(h)] = {"pos": pos, "neg": neg, "evaluable": pos >= MIN_PER_LABEL and neg >= MIN_PER_LABEL,
                                **{k: auc(y[m], p[m]) for k, p in scores.items()}}
    ev = [h for h, r in per_host.items() if r["evaluable"]]
    m = np.isin(hosts, ev)
    host_list, boot = bootstrap_host_aucs(y[m], {k: scores[k][m] for k in ("lstm", "logistic_regression")},
                                          hosts[m], units[m], rng)
    for k in ("lstm", "logistic_regression"):
        for j, h in enumerate(host_list):
            per_host[h][f"{k}_ci95"] = ci(boot[k][:, j])
    # pooled set: identity baseline vs model on the same sequences (D2)
    pooled = {k: auc(y, p) for k, p in scores.items()}
    pooled_gap = {}
    # one bootstrap over the whole test set, positives by episode and negatives by block
    clusters = list(_cluster_index(units.astype(str)).values())
    pos_cl = [c for c in clusters if y[c[0]] == 1]; neg_cl = [c for c in clusters if y[c[0]] == 0]
    gaps = {k: [] for k in ("lstm", "logistic_regression")}
    for _ in range(BOOTSTRAP):
        sel = np.concatenate([pos_cl[i] for i in rng.integers(0, len(pos_cl), len(pos_cl))] +
                             [neg_cl[i] for i in rng.integers(0, len(neg_cl), len(neg_cl))])
        ident = roc_auc_score(y[sel], scores["ref_host_identity"][sel])
        for k in gaps:
            gaps[k].append(roc_auc_score(y[sel], scores[k][sel]) - ident)
    for k in gaps:
        pooled_gap[k] = {"gap": pooled[k] - pooled["ref_host_identity"], "ci95": ci(gaps[k])}
    return {"split": "existing time split, test (Friday) only", "n": int(len(ix)), "pos": int(y.sum()),
            "train_attacker_hosts": train_attackers, "per_host": per_host,
            "evaluable_hosts": ev, "pooled_roc_auc": pooled, "pooled_shortcut_gap_vs_identity": pooled_gap,
            "positive_episodes": int(len(pos_cl)), "negative_blocks": int(len(neg_cl))}


# ---------------------------------------------------------------- part 2/3: host folds
def run_fold(frame: pd.DataFrame, test_host: str, val_host: str, fold_dir: Path, permute: bool, rng) -> dict:
    split = assign_host_fold(frame, test_host, val_host)
    keep = (split != "dropped").to_numpy()
    sub = frame.loc[keep].assign(split=split[keep].values).reset_index(drop=True)
    if permute:
        y = sub.target_attack_present.to_numpy().copy()
        for s in ("train", "val"):
            i = np.flatnonzero(sub.split.to_numpy() == s)
            y[i] = y[rng.permutation(i)]
        sub["target_attack_present"] = y
    data = ForecastDataset.from_frame(sub)
    t0 = time.time()
    metrics = train_experiment(data, fold_dir, **HP)
    counts = {s: {"total": int((sub.split == s).sum()),
                  "pos": int(sub.target_attack_present[sub.split == s].sum()),
                  "hosts": int(sub.host_id[sub.split == s].nunique())} for s in ("train", "val", "test")}
    vi, ti = data.indices("val"), data.indices("test")
    sv, st = score_model_dir(fold_dir, data, vi), score_model_dir(fold_dir, data, ti)
    # evaluate against the TRUE labels of the held-out host (never permuted)
    true = frame.loc[keep].reset_index(drop=True).target_attack_present.to_numpy()
    y_t = true[ti]; y_v = data.attack[vi]
    fri = pd.to_datetime(sub.target_window_start.iloc[ti]).dt.date.to_numpy() == max(pd.to_datetime(sub.target_window_start).dt.date)
    res = {"test_host": test_host, "val_host": val_host, "counts": counts,
           "lstm_epochs_run": metrics["training"]["epochs_run"],
           "lstm_best_epoch": metrics["training"]["best_checkpoint_epoch"], "models": {}}
    for k in ("lstm", "logistic_regression"):
        t = threshold_at_fpr(sv[k][y_v == 0])
        p = st[k]
        res["models"][k] = {
            "roc_auc_within_host": auc(y_t, p),
            "roc_auc_within_host_friday_only": auc(y_t[fri], p[fri]),
            "threshold_1pct_val_fpr": t,
            "recall_at_threshold": float((p[y_t == 1] >= t).mean()),
            "test_fpr_at_threshold": float((p[y_t == 0] >= t).mean()),
            "val_roc_auc": auc(y_v, sv[k]),
        }
    res["test_pos"], res["test_neg"] = int(y_t.sum()), int((y_t == 0).sum())
    res["test_pos_onset"] = int(sub.target_is_onset.to_numpy(bool)[ti][y_t == 1].sum())
    res["runtime_seconds"] = round(time.time() - t0, 1)
    res["_scores"] = {k: st[k] for k in st}
    res["_rows"] = (frame.loc[keep].reset_index(drop=True).iloc[ti], y_t)
    # identity-only model (D6): one-hot host ID trained on the fold's train split
    tr = data.indices("train")
    host_codes = {h: i for i, h in enumerate(sorted(sub.host_id.iloc[tr].unique()))}
    def onehot(idx):
        z = np.zeros((len(idx), len(host_codes) + 1), np.float32)
        for r, h in enumerate(sub.host_id.iloc[idx]):
            z[r, host_codes.get(h, len(host_codes))] = 1
        return z
    ident = LogisticRegression(max_iter=1000, class_weight="balanced").fit(onehot(tr), data.attack[tr])
    res["models"]["identity_only"] = {"roc_auc_within_host": auc(y_t, ident.predict_proba(onehot(ti))[:, 1])}
    return res


def summarise_folds(folds: list[dict], rng) -> dict:
    rows = pd.concat([f["_rows"][0] for f in folds], ignore_index=True)
    y = np.concatenate([f["_rows"][1] for f in folds])
    hosts = rows.host_id.to_numpy(str)
    units = resampling_units(rows, y)
    scores = {k: np.concatenate([f["_scores"][k] for f in folds]) for k in ("lstm", "logistic_regression")}
    host_list, boot = bootstrap_host_aucs(y, scores, hosts, units, rng)
    out = {"per_host": {}, "aggregate": {}}
    for k in scores:
        point = np.array([auc(y[hosts == h], scores[k][hosts == h]) for h in host_list])
        for j, h in enumerate(host_list):
            out["per_host"].setdefault(h, {})[k] = {
                "roc_auc": float(point[j]), "ci95": ci(boot[k][:, j]),
                "pos_units": int(len({u for u in units[(hosts == h) & (y == 1)]})),
                "neg_units": int(len({u for u in units[(hosts == h) & (y == 0)]}))}
        out["aggregate"][k] = {
            "median_host_auc": float(np.median(point)), "median_ci95": ci(np.median(boot[k], axis=1)),
            "mean_host_auc": float(np.mean(point)),
            "shortcut_gap_mean": float(np.mean(point) - 0.5), "shortcut_gap_ci95": ci(np.mean(boot[k], axis=1) - 0.5),
            "hosts_lower_bound_above_0.5": int(sum(ci(boot[k][:, j])[0] > 0.5 for j in range(len(host_list)))),
        }
    return out


def public(fold: dict) -> dict:
    return {k: v for k, v in fold.items() if not k.startswith("_")}


def time_control_part(frame: pd.DataFrame) -> dict:
    """Added after the first results (NOT pre-registered): is the held-out signal the Botnet hours?

    Each host's Botnet positives form one 30-minute episode inside one shared Friday span, so a
    model could score "Friday late morning" rather than Botnet behaviour. Re-scores the saved real
    fold models (no retraining) on the held-out host only, Friday only:
      time_matched_auc  -- Botnet positives vs the same host's benign targets inside the span
      span_benign_auc   -- the host's benign targets inside the span vs outside it, using only
                           benign targets whose 10 input windows contain no attack traffic.
                           ~0.5 means the model does not simply flag the span.
      attack_in_inputs_auc -- benign targets inside the span whose inputs DO contain attack
                           traffic vs clean benign targets outside the span: does the score follow
                           Botnet traffic that is already present in the inputs?
    """
    from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES
    t = pd.to_datetime(frame.target_window_start)
    bot = (frame.target_dominant_class.astype(str) == CLASS).to_numpy()
    lo, hi = t[bot & frame.host_id.isin(BOTNET_HOSTS).to_numpy()].min(), t[bot & frame.host_id.isin(BOTNET_HOSTS).to_numpy()].max()
    k = list(STANDARD_NUMERIC_WINDOW_FEATURES).index("attack_flow_ratio")
    out = {"span": [str(lo), str(hi)], "hosts": {}}
    for i, h in enumerate(BOTNET_HOSTS):
        v = BOTNET_HOSTS[(i + 1) % len(BOTNET_HOSTS)]
        split = assign_host_fold(frame, h, v)
        keep = (split != "dropped").to_numpy()
        sub = frame.loc[keep].assign(split=split[keep].values).reset_index(drop=True)
        data = ForecastDataset.from_frame(sub)
        ti = data.indices("test")
        tt = pd.to_datetime(sub.target_window_start.iloc[ti]).reset_index(drop=True)
        y = data.attack[ti]
        clean_inputs = np.array([not (np.stack(x)[:, k] > 0).any() for x in sub.sequence_features.iloc[ti]])
        friday = (tt.dt.date == lo.date()).to_numpy()
        in_span = ((tt >= lo) & (tt <= hi)).to_numpy()
        scores = score_model_dir(OUT_DIR / "folds" / h.replace(".", "_"), data, ti)
        neg_in, neg_out = (y == 0) & friday & in_span, (y == 0) & friday & ~in_span
        row = {"pos": int((y == 1).sum()), "benign_in_span": int(neg_in.sum()),
               "benign_in_span_clean_inputs": int((neg_in & clean_inputs).sum()),
               "benign_friday_outside_span_clean_inputs": int((neg_out & clean_inputs).sum())}
        for m, p in scores.items():
            tm = (y == 1) | neg_in
            sb = (neg_in | neg_out) & clean_inputs
            dirty = (neg_in & ~clean_inputs) | (neg_out & clean_inputs)
            row[m] = {"time_matched_auc": auc(y[tm], p[tm]),
                      "span_benign_auc": auc(in_span[sb].astype(int), p[sb]),
                      "attack_in_inputs_auc": auc((neg_in & ~clean_inputs)[dirty].astype(int), p[dirty])}
        out["hosts"][h] = row
    return out


def run(parts=("committed", "folds", "permuted", "time_control")) -> dict:
    rng = np.random.default_rng(SEED)
    frame = pd.read_parquet(DATA)
    out = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    out.update({"notes": __doc__.strip(), "hyperparameters": HP, "min_per_label": MIN_PER_LABEL,
                "bootstrap_resamples": BOOTSTRAP, "botnet_hosts": BOTNET_HOSTS})
    if "committed" in parts:
        out["committed"] = committed_part(ForecastDataset.from_frame(frame), rng)
        REPORT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print("[committed] done", flush=True)
    for part, permute in (("folds", False), ("permuted", True)):
        if part not in parts:
            continue
        folds = []
        for i, h in enumerate(BOTNET_HOSTS):
            v = BOTNET_HOSTS[(i + 1) % len(BOTNET_HOSTS)]
            fold = run_fold(frame, h, v, OUT_DIR / part / h.replace(".", "_"), permute, np.random.default_rng(SEED + i))
            folds.append(fold)
            print(f"[{part}] held out {h}: LSTM AUC {fold['models']['lstm']['roc_auc_within_host']:.3f} "
                  f"LR AUC {fold['models']['logistic_regression']['roc_auc_within_host']:.3f} "
                  f"({fold['runtime_seconds']} s)", flush=True)
        out[part] = {"folds": [public(f) for f in folds], "summary": summarise_folds(folds, rng)}
        REPORT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if "time_control" in parts:
        out["time_control"] = time_control_part(frame)
        REPORT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    parts = tuple(sys.argv[1:]) or ("committed", "folds", "permuted", "time_control")
    run(parts)
    print(f"written: {REPORT}")
