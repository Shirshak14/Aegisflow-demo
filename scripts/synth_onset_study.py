"""Onset forecasting study on the synthetic lab (docs/synthetic_onset_design.md).

    python scripts/synth_onset_study.py CONDITION [--seed 11] [--confirm-seed 12]

Writes reports/synthetic_onset_<CONDITION>_s<seed>.json. Bars S1-S5 are evaluated per variant; S6 is
the same variant on the second generator seed (evaluated by --seed 12 runs). Nothing here is read by
the backend.
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts"))
import synth_lab as lab  # noqa: E402

L = 10
HORIZONS_MIN = (1, 5, 15, 30)
STRIDE = lab.STRIDE_S
GAP_WIN = 30 * 60 // STRIDE            # 30 min episode / eligibility rule, in windows
N_FOLDS = 10
N_PSEUDO = 200
NEG_KEEP = 0.3                         # logistic training negatives kept
NEG_KEEP_GB = 0.1                      # gradient boosting training negatives kept (deviation 1)
CLOCK_TOL_MIN = 60
BARS = dict(s1=0.70, s2_folds=7, s2_alpha=0.05, s3=0.65, s4_time=0.10, s4_shift=0.15, s5_val_fpr=0.01, s5_test_fpr=0.02)


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def summarise(w: np.ndarray) -> np.ndarray:
    """w: (n, L, d) -> last, mean, max, slope per feature."""
    x = np.arange(L) - (L - 1) / 2
    slope = (w * x[None, :, None]).sum(1) / (x ** 2).sum()
    return np.concatenate([w[:, -1], w.mean(1), w.max(1), slope], 1)


def build(df: pd.DataFrame):
    hosts = sorted(df.host_id.unique())
    nh = len(hosts)
    df = df.sort_values(["host_id", "window_start"])
    nw = len(df) // nh
    X = np.log1p(df[lab.F].to_numpy(np.float32)).reshape(nh, nw, -1)
    att = df.attack_present.to_numpy(np.int8).reshape(nh, nw)
    runid = df.run_id.to_numpy(np.int32).reshape(nh, nw)
    # network context per window (labels never used)
    flows = np.expm1(X[:, :, lab.F.index("flow_count")]); byt = np.expm1(X[:, :, lab.F.index("byte_count_sum")])
    ports = X[:, :, lab.F.index("unique_destination_ports")]; ips = np.expm1(X[:, :, lab.F.index("unique_destination_ips")])
    net = np.log1p(np.stack([flows.sum(0), byt.sum(0), ports.mean(0), ips.sum(0)], 1)).astype(np.float32)   # (nw, 4)
    cs = np.concatenate([np.zeros((nh, 1), np.int32), np.cumsum(att, 1)], 1)

    rows_h, rows_a = [], []
    for h in range(nh):
        a = np.arange(L - 1, nw - 2 - 2 * max(HORIZONS_MIN))
        recent = cs[h, a + 1] - cs[h, np.maximum(a + 1 - GAP_WIN, 0)]
        ok = recent == 0
        rows_h.append(np.full(ok.sum(), h)); rows_a.append(a[ok])
    H = np.concatenate(rows_h); A = np.concatenate(rows_a)
    win = X[H[:, None], A[:, None] - (L - 1) + np.arange(L)[None, :]]            # (n, L, d)
    own = summarise(win)
    netw = net[A[:, None] - (L - 1) + np.arange(L)[None, :]]
    netf = np.concatenate([netw[:, -1], netw.mean(1)], 1)
    t_end = A * STRIDE + 60
    clock = (t_end / 3600.0) % 24
    day = (t_end // 86400).astype(int)
    tfe = np.stack([np.sin(2 * np.pi * clock / 24), np.cos(2 * np.pi * clock / 24), day % 7], 1).astype(np.float32)
    y = {}
    for m in HORIZONS_MIN:
        k = 2 * m
        y[m] = np.array([att[h, a + 2: a + 2 + k].any() for h, a in zip(H, A)], np.int8) if False else \
            (cs[H, A + 2 + k] - cs[H, A + 2] > 0).astype(np.int8)
    onset_run = np.full(len(H), -1, np.int32)
    pos = y[HORIZONS_MIN[-1]] == 1
    for i in np.flatnonzero(pos):
        j = A[i] + 2 + np.argmax(att[H[i], A[i] + 2: A[i] + 2 + 2 * HORIZONS_MIN[-1]])
        onset_run[i] = runid[H[i], j]
    onset_win = np.full(len(H), -1)
    for r in set(onset_run[pos].tolist()):
        pass
    return dict(h=H, a=A, own=own, net=netf, time=tfe, y=y, clock=clock, day=day, n_hosts=nh, att=att, runid=runid)


def make_folds(D, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(D["n_hosts"])
    fold_of_host = np.empty(D["n_hosts"], int)
    fold_of_host[order] = np.arange(D["n_hosts"]) % N_FOLDS
    f = fold_of_host[D["h"]]
    out = []
    for k in range(N_FOLDS):
        v = (k + 1) % N_FOLDS
        out.append(dict(test=np.flatnonzero(f == k), val=np.flatnonzero(f == v), train=np.flatnonzero((f != k) & (f != v))))
    return out


def feats(D, name):
    if name == "own":
        return D["own"]
    if name == "own+net":
        return np.concatenate([D["own"], D["net"]], 1)
    return D["time"]


def fit_score(model, Xtr, ytr, Xs, rng):
    keep_p = NEG_KEEP if model == "logistic" else NEG_KEEP_GB
    keep = (ytr == 1) | (rng.random(len(ytr)) < keep_p)
    Xtr, ytr = Xtr[keep], ytr[keep]
    if ytr.sum() < 5:
        return [np.zeros(len(x)) for x in Xs]
    if model == "logistic":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=300, C=0.5).fit(sc.transform(Xtr), ytr)
        return [m.decision_function(sc.transform(x)) for x in Xs]
    m = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_depth=4, random_state=0).fit(Xtr, ytr)
    return [m.decision_function(x) for x in Xs]


def shifted(D, y, rows, rng):
    y2 = y.copy()
    hh = D["h"][rows]
    for h in np.unique(hh):
        m = np.flatnonzero(hh == h)
        if y[rows[m]].any() and len(m) > 10:
            y2[rows[m]] = np.roll(y[rows[m]], int(rng.integers(len(m) // 10, len(m) - len(m) // 10)))
    return y2


def fold_metrics(D, f, m, s_test, s_val, rng):
    ti, vi = f["test"], f["val"]
    y = D["y"][m][ti]
    res = dict(positives=int(y.sum()), auc=auc(y, s_test))
    if res["auc"] is None:
        return res
    t = D["a"][ti] * STRIDE + 60
    cand = np.flatnonzero(y == 0)
    tn, sn = t[cand], s_test[cand]
    hn = D["h"][ti][cand]
    width = m * 60
    null = []
    for t0 in tn[rng.choice(len(cand), size=min(N_PSEUDO, len(cand)), replace=False)] + 1:
        yp = ((hn == hn[np.searchsorted(tn, t0 - 1) - 1 if False else 0]) | True) & (tn >= t0 - width) & (tn < t0)
        a = auc(yp.astype(int), sn)
        if a is not None:
            null.append(a)
    null = np.asarray(null)
    res["pseudo_p"] = float((1 + (null >= res["auc"]).sum()) / (1 + len(null))) if len(null) else None
    # clock-matched: positives vs same-host negatives within CLOCK_TOL_MIN of a positive clock time, other day
    hh, clk, day = D["h"][ti], D["clock"][ti], D["day"][ti]
    keep = y == 1
    for h in np.unique(hh[y == 1]):
        pc = clk[(hh == h) & (y == 1)]; pd_ = day[(hh == h) & (y == 1)]
        near = (hh == h) & (y == 0) & (np.abs((clk[:, None] - pc[None, :] + 12) % 24 - 12) <= CLOCK_TOL_MIN / 60).any(1) \
            & ~np.isin(day, pd_)
        keep |= near
    res["clock_auc"] = auc(y[keep], s_test[keep])
    # S5
    yv = D["y"][m][vi]
    thr = np.quantile(s_val[yv == 0], 1 - BARS["s5_val_fpr"]) if (yv == 0).any() else np.inf
    res["test_fpr"] = float((s_test[y == 0] >= thr).mean())
    ra = D["att"]; rid = D["runid"]
    hit = tot = 0
    for h in np.unique(hh[y == 1]):
        sel = np.flatnonzero((hh == h) & (y == 1))
        runs = {}
        for i in sel:
            gi = ti[i]
            j = D["a"][gi] + 2 + int(np.argmax(ra[D["h"][gi], D["a"][gi] + 2: D["a"][gi] + 2 + 2 * m]))
            runs.setdefault(int(rid[D["h"][gi], j]), []).append(i)
        for r, idx in runs.items():
            tot += 1; hit += int((s_test[idx] >= thr).any())
    res["episodes"] = tot
    res["episode_alert_rate"] = hit / tot if tot else None
    return res


def run(condition, seed, out):
    t0 = time.time()
    df, runs = lab.generate(condition, seed)
    D = build(df)
    print(f"[{condition} s{seed}] anchors {len(D['h']):,}, runs {len(runs)}, built in {time.time()-t0:.0f}s", flush=True)
    F = make_folds(D)
    rng = np.random.default_rng(42)
    results = {}
    variants = [("own", "logistic"), ("own", "gboost"), ("own+net", "logistic"), ("own+net", "gboost"), ("time", "gboost")]
    for m in HORIZONS_MIN:
        y = D["y"][m]
        for fs, mod in variants:
            X = feats(D, fs)
            folds = []
            for f in F:
                s_te, s_va = fit_score(mod, X[f["train"]], y[f["train"]], [X[f["test"]], X[f["val"]]], rng)
                met = fold_metrics(D, f, m, s_te, s_va, rng)
                if fs != "time":
                    y_sh = shifted(D, y, f["train"], rng)
                    s_sh, = fit_score(mod, X[f["train"]], y_sh[f["train"]], [X[f["test"]]], rng)
                    met["shift_auc"] = auc(y[f["test"]], s_sh)
                folds.append(met)
            results[f"{fs}|{mod}|H{m}"] = folds
            a = [r["auc"] for r in folds if r.get("auc") is not None]
            print(f"  H{m:>2} {fs:8s} {mod:8s} median AUC {np.median(a):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    out.write_text(json.dumps(dict(condition=condition, seed=seed, n_runs=len(runs), n_anchors=int(len(D["h"])),
                                   bars=BARS, results=results), default=float))
    print("wrote", out)


def verdicts(res: dict) -> dict:
    """Bars S1-S5 per variant from a result file; S6 is read off the confirmation seed by the report."""
    out = {}
    for key, folds in res["results"].items():
        fs, mod, h = key.split("|")
        if fs == "time":
            continue
        a = [r["auc"] for r in folds if r.get("auc") is not None]
        tkey = f"time|gboost|{h}"
        ta = [r["auc"] for r in res["results"][tkey] if r.get("auc") is not None]
        sh = [r["shift_auc"] for r in folds if r.get("shift_auc") is not None]
        ca = [r["clock_auc"] for r in folds if r.get("clock_auc") is not None]
        p = [r["pseudo_p"] for r in folds if r.get("pseudo_p") is not None]
        ep = [r for r in folds if r.get("episodes")]
        hit = sum(r["episode_alert_rate"] * r["episodes"] for r in ep); tot = sum(r["episodes"] for r in ep)
        fpr = [r["test_fpr"] for r in folds if "test_fpr" in r]
        v = dict(median_auc=float(np.median(a)), time_auc=float(np.median(ta)), shift_auc=float(np.median(sh)),
                 clock_auc=float(np.median(ca)) if ca else None, folds_p05=int(sum(x < 0.05 for x in p)),
                 episode_alert_rate=hit / tot if tot else None, median_test_fpr=float(np.median(fpr)))
        v["S1"] = v["median_auc"] >= BARS["s1"]
        v["S2"] = v["folds_p05"] >= BARS["s2_folds"]
        v["S3"] = v["clock_auc"] is not None and v["clock_auc"] >= BARS["s3"]
        v["S4"] = v["median_auc"] - v["time_auc"] >= BARS["s4_time"] and v["median_auc"] - v["shift_auc"] >= BARS["s4_shift"]
        v["S5"] = v["episode_alert_rate"] is not None and v["episode_alert_rate"] >= 0.5 and v["median_test_fpr"] <= BARS["s5_test_fpr"]
        v["S1-S5"] = all(v[k] for k in ("S1", "S2", "S3", "S4", "S5"))
        out[key] = v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=sorted(lab.CONDITIONS))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = ROOT / f"reports/synthetic_onset_{a.condition}_s{a.seed}.json"
    out.parent.mkdir(exist_ok=True)
    run(a.condition, a.seed, out)
