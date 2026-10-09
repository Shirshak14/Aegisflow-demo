"""
LANL 2015 onset forecasting study. Design and bars: docs/lanl_onset_design.md (committed first).

    python scripts/lanl_onset_study.py bins    # stream flows.txt.gz -> data/processed/lanl/victim_bins.parquet
    python scripts/lanl_onset_study.py study   # -> reports/lanl_onset_study.json
    python scripts/lanl_onset_study.py study --seed 7   # second-seed confirmation

Nothing here touches the demo model, its threshold or the default dataset.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegisflow.ml.datasets.lanl_2015 import iter_flows, load_redteam, victim_onsets  # noqa: E402

RAW = ROOT / "data/raw/lanl"
BINS = ROOT / "data/processed/lanl/victim_bins.parquet"
REPORT = ROOT / "reports/lanl_onset_study.json"

BIN = 300                      # 5-minute bins
L = 6                          # input bins (30 minutes)
LOOKBACK = 7 * 86400           # anchors in the 7 days before onset
HORIZONS_MIN = (5, 15, 30, 60)
BURST_GAP = 1800               # onsets chained when < 30 min apart
EXCLUDE_NEAR_TEST = 6 * 3600   # training anchors this close to a test onset are dropped
NEG_SUBSAMPLE = 0.25           # training negatives kept (deviation recorded in the design doc)
N_PSEUDO = 200
DAY9 = 9
BARS = dict(l1_auc=0.70, l2_alpha=0.05, l2_frac=0.5, l3_time=0.10, l3_shift=0.15,
            l4_auc=0.70, l4_time=0.10, l5_alert_frac=0.5, l5_fpr=0.02)

OWN = ["own_flows", "own_packets", "own_bytes", "own_dsts", "own_dports", "own_dur_mean"]
INB = ["in_flows", "in_srcs", "in_bytes"]
CTX = ["ctx_flows"]
FEATURE_SETS = {"own": OWN, "own+inbound": OWN + INB, "own+inbound+context": OWN + INB + CTX}


# --------------------------------------------------------------------------- bins
def build_bins(raw: Path = RAW, out: Path = BINS, chunksize: int = 4_000_000) -> pd.DataFrame:
    rt = load_redteam(raw / "redteam.txt.gz")
    v = victim_onsets(rt)
    onset = v.set_index("computer")["onset_s"]
    victims = set(onset.index)
    sources = set(rt["src"])
    t_stop = int(onset.max())
    parts = []
    t0 = time.time()
    for chunk in iter_flows(raw / "flows.txt.gz", victims, chunksize=chunksize):
        t = pd.to_numeric(chunk["time"]).astype("int64")
        if t.iloc[0] > t_stop:
            break
        num = lambda c: pd.to_numeric(chunk[c], errors="coerce")  # noqa: E731
        base = pd.DataFrame({"t": t, "src": chunk["src"], "dst": chunk["dst"], "dport": chunk["dst_port"],
                             "pk": num("packets"), "by": num("bytes"), "du": num("duration")})
        for role in ("src", "dst"):
            sub = base[base[role].isin(victims)]
            on = sub[role].map(onset)
            sub = sub[(sub["t"] < on) & (sub["t"] >= on - LOOKBACK - L * BIN)]
            if sub.empty:
                continue
            sub = sub.assign(victim=sub[role], bin=sub["t"] // BIN)
            g = sub.groupby(["victim", "bin"])
            if role == "src":
                agg = pd.DataFrame({"own_flows": g.size(), "own_packets": g["pk"].sum(), "own_bytes": g["by"].sum(),
                                    "own_dsts": g["dst"].nunique(), "own_dports": g["dport"].nunique(),
                                    "own_dur_sum": g["du"].sum()})
            else:
                src_on = sub["src"].map(onset)
                known = sub["src"].isin(sources) | (src_on.notna() & (src_on <= sub["t"]))
                sub = sub.assign(ctx=known.astype("int64"))
                g = sub.groupby(["victim", "bin"])
                agg = pd.DataFrame({"in_flows": g.size(), "in_srcs": g["src"].nunique(), "in_bytes": g["by"].sum(),
                                    "ctx_flows": g["ctx"].sum()})
            parts.append(agg)
        print(f"t={int(t.iloc[-1]):,} parts={len(parts)} {time.time() - t0:.0f}s", flush=True)
    bins = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    bins["own_dur_mean"] = bins["own_dur_sum"] / bins["own_flows"].where(bins["own_flows"] > 0)
    bins = bins.drop(columns="own_dur_sum").fillna(0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    bins.to_parquet(out, index=False)
    print("bins", bins.shape, "victims", bins["victim"].nunique())
    return bins


# --------------------------------------------------------------------------- anchors
def anchors(bins: pd.DataFrame, onsets: pd.Series) -> dict:
    """One anchor per victim per bin end in the 7 days before onset; inputs are the last L bins."""
    feats = OWN + INB + CTX
    rows_t, rows_v, X = [], [], []
    for victim, on in onsets.items():
        b_last = (on - 1) // BIN                       # last bin starting before onset
        b_first = (on - LOOKBACK) // BIN
        grid = np.arange(b_first - L + 1, b_last + 1)
        vb = bins[bins["victim"] == victim].set_index("bin")[feats].reindex(grid, fill_value=0.0).to_numpy()
        win = np.lib.stride_tricks.sliding_window_view(vb, (L, len(feats)))[:, 0]   # (n, L, F)
        ends = (grid[L - 1:] + 1) * BIN                # anchor time = end of last input bin
        keep = ends <= on
        X.append(win[keep])
        rows_t.append(ends[keep])
        rows_v.append(np.full(keep.sum(), victim, dtype=object))
    X = np.concatenate(X).astype("float32")
    t = np.concatenate(rows_t)
    v = np.concatenate(rows_v)
    on = onsets.reindex(v).to_numpy()
    return {"X": X, "t": t, "victim": v, "onset": on, "features": feats}


def summarise(X: np.ndarray, idx: list[int]) -> np.ndarray:
    x = np.log1p(np.clip(X[:, :, idx], 0, None))
    slope = x[:, -1] - x[:, 0]
    return np.concatenate([x[:, -1], x.mean(1), x.max(1), slope], axis=1)


def time_features(t: np.ndarray) -> np.ndarray:
    return np.stack([(t % 86400) / 3600.0, (t // 86400) % 7], axis=1).astype("float32")


def labels(t: np.ndarray, onset: np.ndarray, h: int) -> np.ndarray:
    return ((onset >= t) & (onset < t + h * 60)).astype(int)


# --------------------------------------------------------------------------- folds
def bursts(onsets: pd.Series) -> pd.Series:
    s = onsets.sort_values()
    ids, last, cur = [], None, -1
    for x in s:
        if last is None or x - last >= BURST_GAP:
            cur += 1
        ids.append(cur)
        last = x
    return pd.Series(ids, index=s.index)


def make_folds(A: dict, burst_of: pd.Series, onsets: pd.Series) -> list[dict]:
    b = burst_of.reindex(A["victim"]).to_numpy()
    n_b = int(burst_of.max()) + 1
    folds = []
    for k in range(n_b):
        test = b == k
        val_k = k + 1 if k + 1 < n_b else k - 1
        val = b == val_k
        test_onsets = onsets[burst_of == k].to_numpy()
        near = np.zeros(len(b), bool)
        for o in test_onsets:
            near |= np.abs(A["t"] - o) < EXCLUDE_NEAR_TEST
        train = ~test & ~val & ~near
        day = int(test_onsets.min() // 86400 + 1)
        folds.append({"burst": k, "test": np.where(test)[0], "val": np.where(val)[0], "train": np.where(train)[0],
                      "day": day, "victims": sorted(burst_of[burst_of == k].index)})
    return folds


# --------------------------------------------------------------------------- models
def fit(kind: str, X: np.ndarray, y: np.ndarray, seed: int):
    if kind == "logistic":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        m = HistGradientBoostingClassifier(random_state=seed)
    return m.fit(X, y)


def subsample(rows: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    neg = rows[y[rows] == 0]
    keep = rng.random(len(neg)) < NEG_SUBSAMPLE
    return np.sort(np.concatenate([rows[y[rows] == 1], neg[keep]]))


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def pseudo_p(A, rows, p, h, real, rng) -> float | None:
    if real is None:
        return None
    t, v = A["t"][rows], A["victim"][rows]
    lo = {vic: t[v == vic].min() for vic in np.unique(v)}
    hi = {vic: t[v == vic].max() for vic in np.unique(v)}
    count = 0
    for _ in range(N_PSEUDO):
        fake = {vic: rng.uniform(lo[vic], hi[vic]) for vic in lo}
        on = np.array([fake[x] for x in v])
        a = safe_auc(labels(t, on, h), p)
        count += a is not None and a >= real
    return (1 + count) / (N_PSEUDO + 1)


def shifted_onsets(A, rows, rng) -> np.ndarray:
    v = A["victim"][rows]
    t = A["t"][rows]
    fake = {}
    for vic in np.unique(v):
        tv = t[v == vic]
        fake[vic] = rng.uniform(tv.min(), tv.max())
    return np.array([fake[x] for x in v])


def run_fold(A, f, h, design, kind, seed, rng, with_controls: bool) -> dict:
    y = labels(A["t"], A["onset"], h)
    tr = subsample(f["train"], y, rng)
    m = fit(kind, design[tr], y[tr], seed)
    p_test = m.predict_proba(design[f["test"]])[:, 1]
    p_val = m.predict_proba(design[f["val"]])[:, 1]
    yt, yv = y[f["test"]], y[f["val"]]
    real = safe_auc(yt, p_test)
    out = {"burst": f["burst"], "day": f["day"], "auc": real, "n_test": int(len(yt)), "pos": int(yt.sum())}
    # L5: threshold at 1% FPR on validation negatives
    neg_v = p_val[yv == 0]
    thr = float(np.quantile(neg_v, 0.99)) if len(neg_v) else np.inf
    out["test_fpr"] = float((p_test[yt == 0] >= thr).mean()) if (yt == 0).any() else None
    tv, tt = A["victim"][f["test"]], A["t"][f["test"]]
    alerted = []
    for vic in np.unique(tv):
        on = A["onset"][f["test"]][tv == vic][0]
        w = (tv == vic) & (tt >= on - h * 60) & (tt <= on)
        alerted.append(bool((p_test[w] >= thr).any()))
    out["alerted"], out["onsets"] = int(sum(alerted)), len(alerted)
    # per-victim AUC (for the contact breakdown)
    out["victim_auc"] = {str(vic): safe_auc(yt[tv == vic], p_test[tv == vic]) for vic in np.unique(tv)}
    if with_controls:
        out["pseudo_p"] = pseudo_p(A, f["test"], p_test, h, real, rng)
        ys = labels(A["t"][f["train"]], shifted_onsets(A, f["train"], rng), h)
        ys_full = np.zeros(len(y), int)
        ys_full[f["train"]] = ys
        trs = subsample(f["train"], ys_full, rng)
        ms = fit(kind, design[trs], ys_full[trs], seed)
        out["shift_auc"] = safe_auc(yt, ms.predict_proba(design[f["test"]])[:, 1])
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def judge(folds_out: list[dict], time_folds: list[dict]) -> dict:
    auc = med(f["auc"] for f in folds_out)
    t_auc = med(f["auc"] for f in time_folds)
    s_auc = med(f.get("shift_auc") for f in folds_out)
    ps = [f.get("pseudo_p") for f in folds_out if f.get("pseudo_p") is not None]
    n9 = [f for f in folds_out if f["day"] != DAY9]
    t9 = [f for f in time_folds if f["day"] != DAY9]
    auc9, t_auc9 = med(f["auc"] for f in n9), med(f["auc"] for f in t9)
    alerted = sum(f["alerted"] for f in folds_out)
    onsets = sum(f["onsets"] for f in folds_out)
    fpr = med(f["test_fpr"] for f in folds_out)
    b = BARS
    res = {
        "median_auc": auc, "time_auc": t_auc, "shift_auc": s_auc,
        "pseudo_sig_folds": sum(p < b["l2_alpha"] for p in ps), "pseudo_folds": len(ps),
        "median_auc_no_day9": auc9, "time_auc_no_day9": t_auc9,
        "alerted": alerted, "onsets": onsets, "median_test_fpr": fpr,
    }
    res["L1"] = auc is not None and auc >= b["l1_auc"]
    res["L2"] = bool(ps) and res["pseudo_sig_folds"] >= b["l2_frac"] * len(ps)
    res["L3"] = None not in (auc, t_auc, s_auc) and auc - t_auc >= b["l3_time"] and auc - s_auc >= b["l3_shift"]
    res["L4"] = None not in (auc9, t_auc9) and auc9 >= b["l4_auc"] and auc9 - t_auc9 >= b["l4_time"]
    res["L5"] = onsets > 0 and alerted >= b["l5_alert_frac"] * onsets and fpr is not None and fpr <= b["l5_fpr"]
    res["all"] = all(res[k] for k in ("L1", "L2", "L3", "L4", "L5"))
    return res


def study(seed: int = 42, report: Path = REPORT, raw: Path = RAW, bins_path: Path = BINS) -> dict:
    rng = np.random.default_rng(seed)
    rt = load_redteam(raw / "redteam.txt.gz")
    bins = pd.read_parquet(bins_path)
    on_all = victim_onsets(rt).set_index("computer")["onset_s"]
    onsets = on_all[on_all.index.isin(set(bins["victim"]))]
    A = anchors(bins, onsets)
    burst_of = bursts(onsets)
    folds = make_folds(A, burst_of, onsets)
    print(f"victims {len(onsets)} anchors {len(A['t']):,} bursts {len(folds)}", flush=True)
    feats = A["features"]
    designs = {name: summarise(A["X"], [feats.index(c) for c in cols]) for name, cols in FEATURE_SETS.items()}
    designs["time"] = time_features(A["t"])
    census = json.loads((ROOT / "reports/lanl_onset_census.json").read_text())
    contacted = {r["computer"] for r in census["victims"] if (r.get("attacker_before") or 0) > 0}

    results = {"seed": seed, "victims": int(len(onsets)), "anchors": int(len(A["t"])), "bursts": len(folds),
               "fold_days": [f["day"] for f in folds], "variants": {}}
    t0 = time.time()
    for h in HORIZONS_MIN:
        time_folds = [run_fold(A, f, h, designs["time"], "gboost", seed, rng, False) for f in folds]
        results["variants"][f"time-only|H{h}"] = {"folds": time_folds, "median_auc": med(f["auc"] for f in time_folds)}
        for fs in FEATURE_SETS:
            for kind in ("logistic", "gboost"):
                fo = [run_fold(A, f, h, designs[fs], kind, seed, rng, True) for f in folds]
                verdict = judge(fo, time_folds)
                vauc = {}
                for f in fo:
                    vauc.update(f["victim_auc"])
                verdict["victim_auc_contacted"] = med(a for k, a in vauc.items() if k in contacted)
                verdict["victim_auc_not_contacted"] = med(a for k, a in vauc.items() if k not in contacted)
                results["variants"][f"{kind}|{fs}|H{h}"] = {"verdict": verdict, "folds": fo}
                print(f"H{h} {kind:8s} {fs:22s} auc={verdict['median_auc']} time={verdict['time_auc']} "
                      f"shift={verdict['shift_auc']} sig={verdict['pseudo_sig_folds']}/{verdict['pseudo_folds']} "
                      f"no9={verdict['median_auc_no_day9']} alert={verdict['alerted']}/{verdict['onsets']} "
                      f"fpr={verdict['median_test_fpr']} pass={verdict['all']} ({time.time() - t0:.0f}s)", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=1, default=str))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["bins", "study"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", type=Path, default=RAW, help="folder with flows.txt.gz and redteam.txt.gz")
    ap.add_argument("--bins", type=Path, default=BINS)
    a = ap.parse_args(argv)
    if a.part == "bins":
        build_bins(a.raw, a.bins)
    else:
        out = REPORT if a.seed == 42 else REPORT.with_name(f"lanl_onset_study_seed{a.seed}.json")
        study(a.seed, out, a.raw, a.bins)


if __name__ == "__main__":
    main()
