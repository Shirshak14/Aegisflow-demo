"""Escalation forecasting study on CTU-13 (docs/escalation_forecasting_design.md).

Asks whether a model can forecast that an infected host is about to escalate from C2/DNS-style
botnet traffic to spam, click fraud or ICMP flooding, from that host's own recent traffic while it
has not escalated yet. Anchors, horizons, folds, controls and bars are in the design doc, committed
before any model here was trained.

Parts:
  build     -- infected-host windows for all 13 scenarios (60 s, 30 s stride, existing pipeline);
               writes data/processed/ctu_13/escalation_windows.parquet.
  diagnose  -- escalation episodes and how many have eligible anchors before them.
  classical -- logistic regression and gradient boosting, plus the time-only control,
               the pseudo-onset null and the shifted-label retraining control.

Writes reports/escalation_forecast_study.json. Nothing here is read by the backend.
Run: .venv/Scripts/python.exe scripts/escalation_forecast_study.py [part ...]
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT))
from aegisflow.config import load_config  # noqa: E402
from aegisflow.ml.datasets.ctu_13 import Ctu13Adapter, scenario_of  # noqa: E402
from aegisflow.ml.features.flow_features import compute_flow_features  # noqa: E402
from aegisflow.ml.modeling import MODEL_FEATURES, SequencePreprocessor  # noqa: E402
from aegisflow.ml.temporal.windowing import aggregate_host_windows  # noqa: E402
from aegisflow.schema import coerce_canonical_frame  # noqa: E402

RAW = ROOT / "data/raw/ctu_13"
CENSUS = ROOT / "reports/ctu13_onset_census.json"
WINDOWS = ROOT / "data/processed/ctu_13/escalation_windows.parquet"
REPORT = ROOT / "reports/escalation_forecast_study.json"

L = 10                                     # input windows, as in the demo model
EPISODE_GAP = pd.Timedelta("30min")        # escalation episode rule and escalation-free lookback
HORIZONS_MIN = (1, 5, 15, 30)
N_PSEUDO = 200
SHIFTS = {"logistic": 3, "gboost": 1}
SEED = 42
ESCALATION = r"SPAM|SMTP|HTTP-Ad|ICMP"     # spam, click fraud, ICMP flooding (fine CTU label)
LOST = {"packet_size_std_mean", "inter_arrival_mean", "tcp_window_size_mean"}  # absent in CTU-13
FEATURES = [f for f in MODEL_FEATURES if f not in LOST]
BARS = dict(e1_median_auc=0.70, e2_min_folds=5, e2_alpha=0.05, e3_vs_time=0.10, e3_vs_shift=0.15,
            e4_val_fpr=0.01, e4_max_test_fpr=0.02)


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


# ------------------------------------------------------------------------------------------- build
def build() -> pd.DataFrame:
    """Windows of every infected host's own (source) traffic, one host per (scenario, ip)."""
    cfg = load_config()
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    frames = []
    for f in sorted(RAW.glob("*.binetflow.gz"), key=lambda p: scenario_of(p) or 99):
        s = scenario_of(f)
        infected = {h["ip"] for h in census[str(s)]["infected_hosts"]}
        start = pd.Timestamp(census[str(s)]["capture_start"])
        parts = [c[c["SrcAddr"].str.strip().isin(infected)]
                 for c in pd.read_csv(f, dtype=str, keep_default_na=False, chunksize=500_000)]
        raw = pd.concat(parts, ignore_index=True)
        raw.columns = [c.strip() for c in raw.columns]
        raw["ctu_scenario"] = s
        flows = coerce_canonical_frame(Ctu13Adapter._to_canonical(raw))
        flows = flows[flows["dataset_label"] == "Botnet"].copy()
        esc = flows["ctu_label"].str.contains(ESCALATION, regex=True, na=False)
        # windowing counts any non-"Benign" class as attack: here "attack" means "escalation activity"
        flows["normalized_attack_class"] = np.where(esc, "Escalation", "Benign")
        flows["attack_stage"] = np.where(esc, "Escalation", "Benign")
        flows["host_ip"] = flows["source_ip"]
        flows["source_ip"] = f"s{s:02d}:" + flows["source_ip"].astype(str)
        w = aggregate_host_windows(compute_flow_features(flows), cfg=cfg)
        w["scenario"] = s
        w["capture_start"] = start
        first = flows.groupby("source_ip")["timestamp"].min()
        w["infection_start"] = w["host_id"].map(first)
        frames.append(w)
        print(f"scenario {s}: {len(flows)} botnet flows, {int(esc.sum())} escalation flows, {len(w)} windows", flush=True)
    w = pd.concat(frames, ignore_index=True)
    WINDOWS.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(WINDOWS, index=False)
    return w


def load_windows() -> pd.DataFrame:
    w = pd.read_parquet(WINDOWS)
    for c in ("window_start", "window_end", "capture_start", "infection_start"):
        w[c] = pd.to_datetime(w[c])
    w[FEATURES] = w[FEATURES].fillna(0.0)
    return w.sort_values(["host_id", "window_start"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------- anchors
def build_anchors(w: pd.DataFrame) -> dict:
    """Anchor = host window with 10 windows of history, no escalation in the inputs or the 30 min before."""
    gap = w.groupby("host_id").window_start.diff().dt.total_seconds().fillna(86400.0).to_numpy()
    own = np.concatenate([w[FEATURES].to_numpy(np.float32), np.log1p(gap)[:, None].astype(np.float32)], 1)
    esc = w.attack_present.to_numpy(np.int8)
    starts, ends = w.window_start.to_numpy(), w.window_end.to_numpy()
    keep_rows, host_of, t_of, labels, nxt_of = [], [], [], {h: [] for h in HORIZONS_MIN}, []
    for host, g in w.groupby("host_id", sort=False):
        rows = g.index.to_numpy()
        if len(rows) < L:
            continue
        est = starts[rows][esc[rows] == 1]
        a = np.arange(L - 1, len(rows))
        win = rows[a[:, None] - (L - 1) + np.arange(L)]
        t = ends[rows[a]]
        recent = np.searchsorted(est, t, "left") - np.searchsorted(est, t - EPISODE_GAP.to_timedelta64(), "left")
        ok = (recent == 0) & (esc[win].max(1) == 0)
        for h in HORIZONS_MIN:
            n = np.searchsorted(est, t + np.timedelta64(h, "m"), "left") - np.searchsorted(est, t, "left")
            labels[h].append((n > 0).astype(np.int8)[ok])
        k = np.searchsorted(est, t, "left")
        nxt = np.full(len(t), np.datetime64("NaT", "ns"), dtype=starts.dtype)
        nxt[k < len(est)] = est[k[k < len(est)]]
        keep_rows.append(win[ok]); host_of.append(np.full(ok.sum(), host, object)); t_of.append(t[ok]); nxt_of.append(nxt[ok])
    A = dict(win=np.concatenate(keep_rows), host=np.concatenate(host_of), t=np.concatenate(t_of),
             next_esc=np.concatenate(nxt_of), y={h: np.concatenate(v) for h, v in labels.items()}, own=own)
    last = A["win"][:, -1]
    A["scenario"] = w.scenario.to_numpy()[last]
    tt = pd.DatetimeIndex(A["t"])
    A["time"] = np.stack([
        (tt.hour + tt.minute / 60).to_numpy(np.float32),
        ((tt - pd.DatetimeIndex(w.capture_start.to_numpy()[last])).total_seconds() / 60).to_numpy(np.float32),
        ((tt - pd.DatetimeIndex(w.infection_start.to_numpy()[last])).total_seconds() / 60).to_numpy(np.float32),
    ], 1)
    return A


def escalation_episodes(w: pd.DataFrame) -> pd.DataFrame:
    a = w[w.attack_present == 1].sort_values(["host_id", "window_start"])
    new = (a.host_id != a.host_id.shift()) | (a.window_start.diff() > EPISODE_GAP)
    a = a.assign(ep=new.cumsum())
    return a.groupby("ep").agg(host=("host_id", "first"), scenario=("scenario", "first"),
                               start=("window_start", "min"), windows=("window_start", "size")).reset_index(drop=True)


def diagnose(w: pd.DataFrame, A: dict) -> dict:
    ep = escalation_episodes(w)
    out = dict(windows=int(len(w)), hosts=int(w.host_id.nunique()), escalation_windows=int(w.attack_present.sum()),
               escalation_episodes=int(len(ep)), anchors=int(len(A["t"])), per_scenario={})
    for s in sorted(w.scenario.unique()):
        m = A["scenario"] == s
        onsets_with_anchor = {h: int(len(set(A["next_esc"][m][A["y"][h][m] == 1]))) for h in HORIZONS_MIN}
        out["per_scenario"][str(int(s))] = dict(
            hosts=int(w[w.scenario == s].host_id.nunique()), episodes=int((ep.scenario == s).sum()),
            anchors=int(m.sum()), positives={h: int(A["y"][h][m].sum()) for h in HORIZONS_MIN},
            episodes_with_positive_anchor=onsets_with_anchor)
    return out


# ------------------------------------------------------------------------------------------- folds
def folds(A: dict, h: int) -> list[dict]:
    scen = [s for s in sorted(set(A["scenario"].tolist())) if A["y"][h][A["scenario"] == s].sum() > 0]
    out = []
    for i, s in enumerate(scen):
        v = scen[(i + 1) % len(scen)]
        out.append(dict(test=int(s), val=int(v), test_i=np.flatnonzero(A["scenario"] == s),
                        val_i=np.flatnonzero(A["scenario"] == v),
                        train_i=np.flatnonzero((A["scenario"] != s) & (A["scenario"] != v))))
    return out


def shifted_labels(A: dict, y: np.ndarray, rows: np.ndarray, rng) -> np.ndarray:
    y2 = y.copy()
    hosts = A["host"][rows]
    for hst in np.unique(hosts):
        m = np.flatnonzero(hosts == hst)
        if y[rows[m]].any() and len(m) > 10:
            y2[rows[m]] = np.roll(y[rows[m]], int(rng.integers(len(m) // 10, len(m) - len(m) // 10)))
    return y2


def evaluate_fold(A, f, h, score_test, score_val, rng) -> dict:
    ti = f["test_i"]; y = A["y"][h][ti]; t = A["t"][ti]; hosts = A["host"][ti]
    res = dict(scenario=f["test"], positives=int(y.sum()), negatives=int((y == 0).sum()), auc=auc(y, score_test))
    if res["auc"] is None:
        return res
    # Pseudo-onset null: for every host with a real escalation in this scenario, one random
    # pseudo-onset time on that host, labelled with the same horizon rule.
    width = np.timedelta64(h, "m")
    esc_hosts = np.unique(hosts[y == 1])
    neg = y == 0
    null = []
    for _ in range(N_PSEUDO):
        yp = np.zeros(len(ti), bool)
        for hst in esc_hosts:
            cand = np.flatnonzero(neg & (hosts == hst))
            if len(cand):
                t0 = t[rng.choice(cand)] + np.timedelta64(1, "s")
                yp |= (hosts == hst) & neg & (t >= t0 - width) & (t < t0)
        a = auc(yp[neg].astype(int), score_test[neg])
        if a is not None:
            null.append(a)
    null = np.asarray(null)
    res["pseudo_null_median"] = float(np.median(null))
    res["pseudo_null_p95"] = float(np.quantile(null, 0.95))
    res["pseudo_p"] = float((1 + (null >= res["auc"]).sum()) / (1 + len(null)))
    yv = A["y"][h][f["val_i"]]
    thr = float(np.quantile(score_val[yv == 0], 1 - BARS["e4_val_fpr"]))
    res["test_fpr"] = float((score_test[y == 0] >= thr).mean())
    pos = np.flatnonzero(y == 1)
    key = pd.Series(list(zip(hosts[pos], A["next_esc"][ti][pos])))
    hit = pd.Series(score_test[pos] >= thr).groupby(key).any()
    res["episodes"] = int(len(hit)); res["episodes_alerted"] = int(hit.sum())
    return res


def summarise(per_fold: list[dict]) -> dict:
    scored = [r for r in per_fold if r.get("auc") is not None]
    if not scored:
        return dict(folds_scored=0)
    return dict(folds_scored=len(scored), median_auc=float(np.median([r["auc"] for r in scored])),
                folds_p_lt_05=int(sum(r["pseudo_p"] < BARS["e2_alpha"] for r in scored)),
                episodes=int(sum(r["episodes"] for r in scored)),
                episodes_alerted=int(sum(r["episodes_alerted"] for r in scored)),
                max_test_fpr=float(max(r["test_fpr"] for r in scored)))


def bars(s: dict, time_auc: float | None, shift_auc: float | None) -> dict:
    if not s.get("folds_scored"):
        return dict(all=False)
    b = dict(E1=s["median_auc"] >= BARS["e1_median_auc"],
             E2=s["folds_p_lt_05"] >= BARS["e2_min_folds"],
             E3=(time_auc is not None and s["median_auc"] - time_auc >= BARS["e3_vs_time"])
                and (shift_auc is None or s["median_auc"] - shift_auc >= BARS["e3_vs_shift"]),
             E4=s["episodes_alerted"] * 2 >= s["episodes"] and s["max_test_fpr"] <= BARS["e4_max_test_fpr"])
    b["all"] = all(b.values())
    return b


# ------------------------------------------------------------------------------------------- classical
def summary_features(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X[:, -1], X.mean(1), X.max(1), X.min(1), X[:, -1] - X[:, 0]], 1)


def fit_classical(kind, A, f, features, y_train, seed):
    if features == "time":
        Xtr, Xv, Xte = (A["time"][f[k]] for k in ("train_i", "val_i", "test_i"))
    else:
        names = list(FEATURES) + ["log_gap_s"]
        Xs = [A["own"][A["win"][f[k]]] for k in ("train_i", "val_i", "test_i")]
        prep = SequencePreprocessor(names).fit(Xs[0])
        Xs = [prep.transform(x) for x in Xs]
        Xtr, Xv, Xte = ((x.reshape(len(x), -1) for x in Xs) if kind == "logistic" else (summary_features(x) for x in Xs))
    if kind == "logistic":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, random_state=seed)
        m.fit(sc.transform(Xtr), y_train)
        return m.predict_proba(sc.transform(Xte))[:, 1], m.predict_proba(sc.transform(Xv))[:, 1]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                       class_weight="balanced", random_state=seed)
    m.fit(Xtr, y_train)
    return m.predict_proba(Xte)[:, 1], m.predict_proba(Xv)[:, 1]


def classical_part(A, rng) -> dict:
    out = {}
    variants = [("logistic", "own"), ("gboost", "own"), ("gboost", "time")]
    for h in HORIZONS_MIN:
        fl = folds(A, h)
        for kind, features in variants:
            key = f"{kind}|{features}|H{h}"
            t0 = time.time()
            per, shift_aucs = [], []
            y = A["y"][h]
            for f in fl:
                if y[f["train_i"]].sum() == 0 or y[f["val_i"]].sum() == 0:
                    continue
                pt, pv = fit_classical(kind, A, f, features, y[f["train_i"]], SEED)
                per.append(evaluate_fold(A, f, h, pt, pv, rng))
                if features != "time":
                    for s in range(SHIFTS[kind]):
                        ys = shifted_labels(A, y, f["train_i"], np.random.default_rng(1000 + s))
                        ps, _ = fit_classical(kind, A, f, features, ys[f["train_i"]], SEED + s)
                        a = auc(y[f["test_i"]], ps)
                        if a is not None:
                            shift_aucs.append(a)
            out[key] = dict(per_fold=per, summary=summarise(per),
                            shifted_label_median_auc=float(np.median(shift_aucs)) if shift_aucs else None,
                            seconds=round(time.time() - t0, 1))
            print(key, out[key]["summary"], "shift", out[key]["shifted_label_median_auc"], flush=True)
        time_auc = out[f"gboost|time|H{h}"]["summary"].get("median_auc")
        for kind, features in variants:
            if features != "time":
                key = f"{kind}|{features}|H{h}"
                out[key]["bars"] = bars(out[key]["summary"], time_auc, out[key]["shifted_label_median_auc"])
    return out


def run(parts) -> dict:
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    w = build() if "build" in parts or not WINDOWS.exists() else load_windows()
    if "build" in parts:
        w = load_windows()
    A = build_anchors(w)
    rng = np.random.default_rng(SEED)
    if "diagnose" in parts:
        report["diagnose"] = diagnose(w, A)
        print(json.dumps(report["diagnose"], indent=1))
    if "classical" in parts:
        report["classical"] = classical_part(A, rng)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    run(sys.argv[1:] or ["build", "diagnose", "classical"])
