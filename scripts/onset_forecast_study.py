"""Onset forecasting study (docs/onset_forecasting_design.md).

Asks whether any model can forecast that a host is about to start attack traffic while its own
recent traffic is still attack-free. Anchors are host windows; see the design doc for the target,
folds, controls and the bars P1-P5, which were committed before this script was run.

Parts:
  diagnose -- episode and onset counts, lead-time spread of the existing sequence target.
  classical -- logistic regression and gradient boosting, every feature set and horizon,
               with the shifted-label retraining control (3 shifts).
  neural   -- LSTM, Transformer, multi-task LSTM and temporal GNN at H = 5 and 15 minutes,
               with one shifted-label LSTM per horizon.

Writes reports/onset_forecast_study.json. Nothing here is read by the backend.
Run: .venv/Scripts/python.exe scripts/onset_forecast_study.py [part ...]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import MODEL_FEATURES, SequencePreprocessor, build_model  # noqa: E402

PROCESSED = ROOT / "data/processed/cic_ids2017"
REPORT = ROOT / "reports/onset_forecast_study.json"

L = 10                                # input windows, as in the demo model
EPISODE_GAP = pd.Timedelta("30min")   # episode rule and attack-free lookback
HORIZONS_MIN = (1, 5, 15, 30)
NEURAL_HORIZONS_MIN = (5, 15)
FOLD_HOSTS = ["192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14",
              "192.168.10.15", "192.168.10.17", "192.168.10.50"]
N_PSEUDO = 200
CLOCK_TOL_MIN = 60
NEG_SUBSAMPLE = 0.3                   # neural training negatives only
SEED = 42
HP = dict(epochs=8, batch_size=256, lr=1e-3, hidden=16, dropout=0.2, patience=3)
BARS = dict(p1_median_auc=0.70, p2_min_folds=5, p2_alpha=0.05, p3_clock_auc=0.65,
            p4_vs_time=0.10, p4_vs_shift=0.15, p5_val_fpr=0.01, p5_max_test_fpr=0.02)
NET_FEATURES = ["net_active_hosts", "net_flow_count", "net_byte_count", "net_dst_ports_mean"]


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


# ------------------------------------------------------------------------------------------- data
def load_windows() -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED / "host_windows.parquet")
    w["host_id"] = w.host_id.astype(str)
    w["window_start"] = pd.to_datetime(w.window_start)
    w["window_end"] = pd.to_datetime(w.window_end)
    w[MODEL_FEATURES] = w[MODEL_FEATURES].fillna(0.0)
    return w.sort_values(["host_id", "window_start"]).reset_index(drop=True)


def episodes(w: pd.DataFrame) -> pd.DataFrame:
    a = w[w.attack_present == 1].sort_values(["host_id", "window_start"])
    new = (a.host_id != a.host_id.shift()) | (a.window_start.diff() > EPISODE_GAP)
    a = a.assign(ep=new.cumsum())
    ep = a.groupby("ep").agg(host=("host_id", "first"), cls=("dominant_class", lambda x: x.mode()[0]),
                             start=("window_start", "min"), end=("window_start", "max"),
                             attack_windows=("window_start", "size"))
    prior = []
    for e in ep.itertuples():
        hw = w[(w.host_id == e.host) & (w.window_start < e.start)]
        prior.append((len(hw), int((hw.window_start >= e.start - pd.Timedelta("10min")).sum())))
    ep["prior_windows"], ep["prior_windows_10min"] = zip(*prior)
    return ep.reset_index(drop=True)


def neighbour_features(w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean model features of each host window's graph neighbours in the same window, and their count."""
    from aegisflow.ml.graph import window_neighbours
    flows = pd.read_parquet(PROCESSED / "flows.parquet", columns=["timestamp", "source_ip", "destination_ip"])
    nb = window_neighbours(flows, w, 60.0)
    feats = {(h, s): v for h, s, v in zip(w.host_id, w.window_start, w[MODEL_FEATURES].to_numpy(np.float32))}
    NX = np.zeros((len(w), len(MODEL_FEATURES)), np.float32); D = np.zeros(len(w), np.float32)
    for i, (h, s) in enumerate(zip(w.host_id, w.window_start)):
        peers = nb.get((h, s), ())
        vecs = [feats[(p, s)] for p in peers if (p, s) in feats]
        if vecs:
            NX[i] = np.mean(vecs, axis=0)
        D[i] = len(peers)
    return NX, D


def build_anchors(w: pd.DataFrame, with_graph: bool = False) -> dict:
    """One row per host window with a full 10-window history. Labels are only meaningful where eligible."""
    gap = w.groupby("host_id").window_start.diff().dt.total_seconds().fillna(86400.0).to_numpy()
    net = w.groupby("window_start").agg(net_active_hosts=("host_id", "size"), net_flow_count=("flow_count", "sum"),
                                        net_byte_count=("byte_count_sum", "sum"),
                                        net_dst_ports_mean=("unique_destination_ports", "mean"))
    netv = np.log1p(net.reindex(w.window_start).to_numpy(np.float32))
    own = np.concatenate([w[MODEL_FEATURES].to_numpy(np.float32), np.log1p(gap)[:, None].astype(np.float32)], 1)
    attack = w.attack_present.to_numpy(np.int8)
    starts = w.window_start.to_numpy(); ends = w.window_end.to_numpy()
    idx_rows, host_of, t_of = [], [], []
    elig, labels, next_onset = [], {h: [] for h in HORIZONS_MIN}, []
    for host, g in w.groupby("host_id", sort=False):
        rows = g.index.to_numpy()
        if len(rows) < L:
            continue
        ast = starts[rows][attack[rows] == 1]
        a = np.arange(L - 1, len(rows))
        win = rows[a[:, None] - (L - 1) + np.arange(L)]
        t = ends[rows[a]]
        recent = np.searchsorted(ast, t, "left") - np.searchsorted(ast, t - EPISODE_GAP.to_timedelta64(), "left")
        e = (recent == 0) & (attack[win].max(1) == 0)
        for h in HORIZONS_MIN:
            n_future = np.searchsorted(ast, t + np.timedelta64(h, "m"), "left") - np.searchsorted(ast, t, "left")
            labels[h].append((n_future > 0).astype(np.int8))
        k = np.searchsorted(ast, t, "left")
        nxt = np.full(len(t), np.datetime64("NaT", "ns"), dtype=starts.dtype)
        if len(ast):
            nxt[k < len(ast)] = ast[k[k < len(ast)]]
        idx_rows.append(win); host_of.append(np.full(len(a), host, object)); t_of.append(t)
        elig.append(e); next_onset.append(nxt)
    A = dict(win=np.concatenate(idx_rows), host=np.concatenate(host_of), t=np.concatenate(t_of),
             eligible=np.concatenate(elig), next_attack=np.concatenate(next_onset),
             y={h: np.concatenate(v) for h, v in labels.items()}, own=own, net=netv)
    if with_graph:
        A["nbr"], A["deg"] = neighbour_features(w)
    keep = A["eligible"]
    for k in ("win", "host", "t", "next_attack"):
        A[k] = A[k][keep]
    A["y"] = {h: v[keep] for h, v in A["y"].items()}
    tt = pd.DatetimeIndex(A["t"])
    A["clock"] = (tt.hour + tt.minute / 60).to_numpy(np.float32)
    A["dow"] = tt.dayofweek.to_numpy(np.float32)
    A["date"] = tt.normalize().to_numpy()
    return A


def tensor(A: dict, rows: np.ndarray, features: str) -> tuple[np.ndarray, list[str]]:
    win = A["win"][rows]
    names = list(MODEL_FEATURES) + ["log_gap_s"]
    X = A["own"][win]
    if features == "own+net":
        X = np.concatenate([X, A["net"][win]], -1); names += NET_FEATURES
    return X, names


# ------------------------------------------------------------------------------------------- folds
def folds(A: dict) -> list[dict]:
    out = []
    for i, test in enumerate(FOLD_HOSTS):
        val = FOLD_HOSTS[(i + 1) % len(FOLD_HOSTS)]
        hosts = A["host"]
        out.append(dict(test=test, val=val, test_i=np.flatnonzero(hosts == test), val_i=np.flatnonzero(hosts == val),
                        train_i=np.flatnonzero((hosts != test) & (hosts != val))))
    return out


def shifted_labels(A: dict, y: np.ndarray, rows: np.ndarray, rng) -> np.ndarray:
    """Roll each host's label sequence by a random offset: same episode shape, onset at a random time."""
    y2 = y.copy()
    hosts = A["host"][rows]
    for h in np.unique(hosts):
        m = np.flatnonzero(hosts == h)
        if y[rows[m]].any() and len(m) > 10:
            y2[rows[m]] = np.roll(y[rows[m]], int(rng.integers(len(m) // 10, len(m) - len(m) // 10)))
    return y2


# ------------------------------------------------------------------------------------------- scoring
def evaluate_fold(A, f, h, score_test, score_val, rng) -> dict:
    ti = f["test_i"]; y = A["y"][h][ti]; t = A["t"][ti]
    res = dict(host=f["test"], positives=int(y.sum()), negatives=int((y == 0).sum()), auc=auc(y, score_test))
    if res["auc"] is None:
        return res
    # Pseudo-onset null: random onset times on the same host, same horizon rule.
    cand = np.flatnonzero(y == 0)  # real positives are left out of the null
    tn, sn = t[cand], score_test[cand]
    null = []
    width = np.timedelta64(h, "m")
    for t0 in tn[rng.choice(len(cand), size=min(N_PSEUDO, len(cand)), replace=False)] + np.timedelta64(1, "s"):
        yp = (tn >= t0 - width) & (tn < t0)
        a = auc(yp.astype(int), sn)
        if a is not None:
            null.append(a)
    null = np.asarray(null)
    res["pseudo_null_median"] = float(np.median(null))
    res["pseudo_null_p95"] = float(np.quantile(null, 0.95))
    res["pseudo_p"] = float((1 + (null >= res["auc"]).sum()) / (1 + len(null)))
    # Clock-matched: same host, other days, within CLOCK_TOL_MIN of a positive's clock time.
    clock = A["clock"][ti]; date = A["date"][ti]
    pos = np.flatnonzero(y == 1)
    near = np.zeros(len(ti), bool)
    for c, d in zip(clock[pos], date[pos]):
        near |= (np.abs(clock - c) <= CLOCK_TOL_MIN / 60) & (date != d)
    m = (y == 1) | near
    res["clock_matched_auc"] = auc(y[m], score_test[m])
    res["clock_matched_negatives"] = int(near.sum())
    # P5: threshold at 1% FPR on the validation host's negatives.
    yv = A["y"][h][f["val_i"]]
    thr = float(np.quantile(score_val[yv == 0], 1 - BARS["p5_val_fpr"]))
    res["test_fpr"] = float((score_test[y == 0] >= thr).mean())
    onsets = A["next_attack"][ti][pos]
    hit = pd.Series(score_test[pos] >= thr).groupby(onsets).any()
    res["episodes"] = int(len(hit)); res["episodes_alerted"] = int(hit.sum())
    return res


def summarise(per_fold: list[dict]) -> dict:
    scored = [r for r in per_fold if r.get("auc") is not None]
    if not scored:
        return dict(folds_scored=0)
    med = lambda k: float(np.median([r[k] for r in scored if r.get(k) is not None])) if any(r.get(k) is not None for r in scored) else None
    return dict(folds_scored=len(scored), median_auc=med("auc"), median_clock_auc=med("clock_matched_auc"),
                folds_p_lt_05=int(sum(r["pseudo_p"] < BARS["p2_alpha"] for r in scored)),
                episodes=int(sum(r["episodes"] for r in scored)),
                episodes_alerted=int(sum(r["episodes_alerted"] for r in scored)),
                max_test_fpr=float(max(r["test_fpr"] for r in scored)))


def bars(s: dict, time_auc: float | None, shift_auc: float | None) -> dict:
    if not s.get("folds_scored"):
        return dict(all=False)
    b = dict(P1=s["median_auc"] >= BARS["p1_median_auc"],
             P2=s["folds_p_lt_05"] >= BARS["p2_min_folds"],
             P3=(s["median_clock_auc"] or 0) >= BARS["p3_clock_auc"],
             P4=(time_auc is not None and s["median_auc"] - time_auc >= BARS["p4_vs_time"])
                and (shift_auc is None or s["median_auc"] - shift_auc >= BARS["p4_vs_shift"]),
             P5=s["episodes_alerted"] * 2 >= s["episodes"] and s["max_test_fpr"] <= BARS["p5_max_test_fpr"])
    b["all"] = all(b.values())
    return b


# ------------------------------------------------------------------------------------------- classical
def summary_features(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X[:, -1], X.mean(1), X.max(1), X.min(1), X[:, -1] - X[:, 0]], 1)


def fit_classical(kind, A, f, h, features, y_train, seed):
    if features == "time":
        mk = lambda rows: np.stack([A["clock"][rows], A["dow"][rows]], 1)
        Xtr, Xv, Xte = mk(f["train_i"]), mk(f["val_i"]), mk(f["test_i"])
    else:
        Xs = []
        names = None
        for rows in (f["train_i"], f["val_i"], f["test_i"]):
            X, names = tensor(A, rows, features); Xs.append(X)
        prep = SequencePreprocessor(names).fit(Xs[0])
        Xs = [prep.transform(x) for x in Xs]
        if kind == "logistic":
            Xtr, Xv, Xte = (x.reshape(len(x), -1) for x in Xs)
        else:
            Xtr, Xv, Xte = (summary_features(x) for x in Xs)
    if kind == "logistic":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=seed)
        m.fit(sc.transform(Xtr), y_train)
        return m.predict_proba(sc.transform(Xte))[:, 1], m.predict_proba(sc.transform(Xv))[:, 1]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                       class_weight="balanced", random_state=seed)
    m.fit(Xtr, y_train)
    return m.predict_proba(Xte)[:, 1], m.predict_proba(Xv)[:, 1]


def classical_part(A, rng) -> dict:
    out = {}
    fl = folds(A)
    variants = [("logistic", "own"), ("logistic", "own+net"), ("gboost", "own"), ("gboost", "own+net"),
                ("gboost", "time")]
    for h in HORIZONS_MIN:
        for kind, features in variants:
            key = f"{kind}|{features}|H{h}"
            t0 = time.time()
            per = []
            shift_aucs = []
            for f in fl:
                y = A["y"][h]
                if y[f["train_i"]].sum() == 0:
                    continue
                pt, pv = fit_classical(kind, A, f, h, features, y[f["train_i"]], SEED)
                per.append(evaluate_fold(A, f, h, pt, pv, rng))
                if features != "time":
                    for s in range(3):
                        ys = shifted_labels(A, y, f["train_i"], np.random.default_rng(1000 + s))
                        ps, _ = fit_classical(kind, A, f, h, features, ys[f["train_i"]], SEED + s)
                        a = auc(y[f["test_i"]], ps)
                        if a is not None:
                            shift_aucs.append(a)
            out[key] = dict(per_fold=per, summary=summarise(per),
                            shifted_label_median_auc=float(np.median(shift_aucs)) if shift_aucs else None,
                            seconds=round(time.time() - t0, 1))
            print(key, out[key]["summary"], "shift", out[key]["shifted_label_median_auc"], flush=True)
    for h in HORIZONS_MIN:
        time_auc = out[f"gboost|time|H{h}"]["summary"].get("median_auc")
        for kind, features in variants:
            key = f"{kind}|{features}|H{h}"
            if features != "time":
                out[key]["bars"] = bars(out[key]["summary"], time_auc, out[key]["shifted_label_median_auc"])
    return out


# ------------------------------------------------------------------------------------------- neural
def neural_net(kind: str, F: int):
    import torch
    from torch import nn
    if kind in ("lstm", "transformer"):
        net = build_model(kind, F, HP["hidden"], HP["dropout"])
        return net, lambda net, b: net(b[0])
    if kind == "multitask":
        from aegisflow.ml.multitask import MultiTaskForecaster
        net = MultiTaskForecaster(F, 1, 3, HP["hidden"], HP["dropout"]).net
        return net, lambda net, b: net(b[0])[0]
    if kind == "tgnn":
        from aegisflow.ml.graph import TemporalGNNForecaster
        net = TemporalGNNForecaster(F, HP["hidden"], HP["dropout"]).net
        return net, lambda net, b: net(b[0], b[1], b[2])
    raise ValueError(kind)


def neural_inputs(A, rows, kind, prep, nprep=None):
    X, _ = tensor(A, rows, "own+net" if kind != "tgnn" else "own")
    out = [prep.transform(X)]
    if kind == "tgnn":
        win = A["win"][rows]
        NX = np.concatenate([A["nbr"][win], np.zeros((*win.shape, 1), np.float32)], -1)
        out += [nprep.transform(NX), A["deg"][win][..., None].astype(np.float32)]
    return out


def future_state(A, rows, prep):
    """Own+net features of the anchor host's next 3 windows (preprocessed); mask where the host has none."""
    last = A["win"][rows][:, -1]
    nxt = last[:, None] + np.arange(1, 4)
    ok = nxt < len(A["own"])
    nxt = np.minimum(nxt, len(A["own"]) - 1)
    ok &= A["own_host"][nxt] == A["own_host"][last][:, None]
    return prep.transform(np.concatenate([A["own"][nxt], A["net"][nxt]], -1)), ok


def train_neural(kind, A, f, h, y, seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    r = np.random.default_rng(seed)
    tr = f["train_i"]
    ytr = y[tr]
    keep = (ytr == 1) | (r.random(len(tr)) < NEG_SUBSAMPLE)
    tr, ytr = tr[keep], ytr[keep]
    Xtr_raw, names = tensor(A, tr, "own+net" if kind != "tgnn" else "own")
    prep = SequencePreprocessor(names).fit(Xtr_raw)
    nprep = None
    if kind == "tgnn":
        win = A["win"][tr]
        nprep = SequencePreprocessor(names).fit(np.concatenate([A["nbr"][win], np.zeros((*win.shape, 1), np.float32)], -1))
    T = lambda rows: [torch.tensor(x) for x in neural_inputs(A, rows, kind, prep, nprep)]
    btr = T(tr); bv = T(f["val_i"]); bte = T(f["test_i"])
    yv = torch.tensor(y[f["val_i"]], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    extra = []
    if kind == "multitask":
        fs, fm = future_state(A, tr, prep)
        extra = [torch.tensor(fs), torch.tensor(fm)]
    net, fwd = neural_net(kind, btr[0].shape[-1])
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    n = len(tr); best, stale = float("inf"), 0
    state = None
    g = torch.Generator().manual_seed(seed)
    for epoch in range(HP["epochs"]):
        net.train()
        for b in torch.randperm(n, generator=g).split(HP["batch_size"]):
            opt.zero_grad()
            batch = [x[b] for x in btr]
            loss = bce(fwd(net, batch), yt[b])
            if kind == "multitask":
                pred = net(batch[0])[2]
                m = extra[1][b].unsqueeze(-1).float()
                loss = loss + 0.5 * (((pred - extra[0][b]) ** 2) * m).sum() / (m.sum() * pred.shape[-1] + 1e-9)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(bce(fwd(net, bv), yv))
        if vl < best:
            best, stale = vl, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= HP["patience"]:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return torch.sigmoid(fwd(net, bte)).numpy(), torch.sigmoid(fwd(net, bv)).numpy()


def neural_part(A, rng, classical: dict | None) -> dict:
    out = {}
    fl = folds(A)
    for h in NEURAL_HORIZONS_MIN:
        y = A["y"][h]
        shift_aucs = []
        for f in fl:
            ys = shifted_labels(A, y, f["train_i"], np.random.default_rng(2000))
            ps, _ = train_neural("lstm", A, f, h, ys, SEED)
            a = auc(y[f["test_i"]], ps)
            if a is not None:
                shift_aucs.append(a)
        shift_med = float(np.median(shift_aucs)) if shift_aucs else None
        time_auc = (classical or {}).get(f"gboost|time|H{h}", {}).get("summary", {}).get("median_auc")
        for kind in ("lstm", "transformer", "multitask", "tgnn"):
            key = f"{kind}|{'own' if kind == 'tgnn' else 'own+net'}|H{h}"
            t0 = time.time()
            per = []
            for f in fl:
                pt, pv = train_neural(kind, A, f, h, y, SEED)
                per.append(evaluate_fold(A, f, h, pt, pv, rng))
            s = summarise(per)
            out[key] = dict(per_fold=per, summary=s, shifted_label_lstm_median_auc=shift_med,
                            bars=bars(s, time_auc, shift_med), seconds=round(time.time() - t0, 1))
            print(key, s, out[key]["bars"], flush=True)
    return out


# ------------------------------------------------------------------------------------------- diagnose
def diagnose(w: pd.DataFrame, A: dict) -> dict:
    seq = pd.read_parquet(PROCESSED / "sequences.parquet",
                          columns=["host_id", "seq_end_time", "target_window_start", "target_attack_present",
                                   "target_is_onset", "target_dominant_class"])
    lead = (pd.to_datetime(seq.target_window_start) - pd.to_datetime(seq.seq_end_time)).dt.total_seconds() / 60
    ep = episodes(w)
    on = seq[seq.target_is_onset]
    # a sequence onset is an episode start when its target window is the first window of an episode
    starts = set(zip(ep.host, ep.start))
    on_starts = sum((h, pd.Timestamp(t)) in starts for h, t in zip(on.host_id, on.target_window_start))
    pos = int(seq.target_attack_present.sum())
    per_h = {}
    for h in HORIZONS_MIN:
        y = A["y"][h]
        per_h[f"H{h}"] = dict(positives=int(y.sum()),
                              hosts_with_positives=sorted(set(A["host"][y == 1].tolist())),
                              episodes_with_positives=int(len(set(A["next_attack"][y == 1].tolist()))))
    return dict(
        sequence_target=dict(positives=pos, onset=int(len(on)), continuation=pos - int(len(on)),
                             onset_that_start_an_episode=int(on_starts),
                             lead_minutes_quantiles={str(q): float(lead.quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)},
                             onset_lead_minutes_max=float(lead[seq.target_is_onset].max())),
        episodes=json.loads(ep.to_json(orient="records", date_format="iso")),
        anchors=dict(eligible=int(len(A["t"])), hosts=int(len(set(A["host"].tolist()))), by_horizon=per_h))


def run(parts) -> dict:
    t0 = time.time()
    w = load_windows()
    A = build_anchors(w, with_graph="neural" in parts)
    A["own_host"] = w.host_id.to_numpy()
    rng = np.random.default_rng(SEED)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report.update(design="docs/onset_forecasting_design.md", bars=BARS, horizons_min=list(HORIZONS_MIN),
                  fold_hosts=FOLD_HOSTS, hyperparameters=HP)
    if "diagnose" in parts:
        report["diagnose"] = diagnose(w, A)
        print(json.dumps(report["diagnose"]["sequence_target"], indent=1), report["diagnose"]["anchors"], flush=True)
    if "classical" in parts:
        report["classical"] = classical_part(A, rng)
    if "neural" in parts:
        report["neural"] = neural_part(A, rng, report.get("classical"))
    report["runtime_seconds"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    run(sys.argv[1:] or ["diagnose", "classical", "neural"])
