"""
LANL 2015 short-lead login study. Design and bars: docs/lanl_short_lead_design.md (committed first).

    python scripts/lanl_short_lead_study.py bins     # stream flows.txt.gz -> data/processed/lanl/victim_bins_1m.parquet (+ contacts)
    python scripts/lanl_short_lead_study.py study    # -> reports/lanl_short_lead_study.json
    python scripts/lanl_short_lead_study.py study --seed 7

Nothing here touches the demo model, its threshold or the default dataset.
"""
from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location("lanl_onset_study", ROOT / "scripts" / "lanl_onset_study.py")
P = importlib.util.module_from_spec(_spec)   # the 5-minute study: folds, bursts, time features, labels
_spec.loader.exec_module(P)

RAW = ROOT / "data/raw/lanl"
BINS = ROOT / "data/processed/lanl/victim_bins_1m.parquet"
CONTACTS = ROOT / "data/processed/lanl/victim_contacts.parquet"
REPORT = ROOT / "reports/lanl_short_lead_study.json"

BIN = 60
L = 5
LOOKBACK = 7 * 86400
HORIZONS_MIN = (2, 5, 10, 15)
NEG_SUBSAMPLE = 0.05
N_PSEUDO = 200
DAY9 = 9
MIN_FOLDS, MIN_POS = 10, 30
BARS = dict(f1_auc=0.70, f2_time=0.10, f3_alpha=0.05, f3_frac=0.5, f4_shift=0.15, f5_auc=0.70, f5_time=0.10,
            d1_auc=0.85, d2_time=0.10, d2_shift=0.15, d3_alert_frac=0.5, d3_fpr=0.02, d3_lead_s=120)

OWN = ["own_flows", "own_packets", "own_bytes", "own_dsts", "own_dports", "own_dur_mean"]
INB = ["in_flows", "in_srcs", "in_bytes"]
CTX = ["ctx_flows"]
ATT = ["att_flows", "att_bytes"]
ALL = OWN + INB + CTX + ATT
FEATURE_SETS = {"own": OWN, "own+inbound": OWN + INB, "own+inbound+attacker": OWN + INB + ATT}


# --------------------------------------------------------------------------- bins
def build_bins(raw: Path = RAW, out: Path = BINS, contacts_out: Path = CONTACTS, chunksize: int = 4_000_000):
    rt = load_redteam(raw / "redteam.txt.gz")
    onset = victim_onsets(rt).set_index("computer")["onset_s"]
    victims, sources = set(onset.index), set(rt["src"])
    t_stop = int(onset.max())
    parts, first_contact = [], {}
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
                att = sub["src"].isin(sources)
                known = att | (src_on.notna() & (src_on <= sub["t"]))
                sub = sub.assign(ctx=known.astype("int64"), att=att.astype("int64"), att_by=sub["by"].where(att, 0.0))
                g = sub.groupby(["victim", "bin"])
                agg = pd.DataFrame({"in_flows": g.size(), "in_srcs": g["src"].nunique(), "in_bytes": g["by"].sum(),
                                    "ctx_flows": g["ctx"].sum(), "att_flows": g["att"].sum(),
                                    "att_bytes": g["att_by"].sum()})
                for vic, tt in sub[att].groupby("victim")["t"].min().items():
                    first_contact[vic] = min(first_contact.get(vic, tt), int(tt))
            parts.append(agg)
        print(f"t={int(t.iloc[-1]):,} parts={len(parts)} {time.time() - t0:.0f}s", flush=True)
    bins = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    bins["own_dur_mean"] = bins["own_dur_sum"] / bins["own_flows"].where(bins["own_flows"] > 0)
    bins = bins.drop(columns="own_dur_sum").fillna(0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    bins.to_parquet(out, index=False)
    pd.DataFrame({"victim": list(onset.index), "onset": onset.to_numpy(),
                  "contact": [first_contact.get(v, -1) for v in onset.index]}).to_parquet(contacts_out, index=False)
    print("bins", bins.shape, "victims", bins["victim"].nunique(), "contacted", len(first_contact))


# --------------------------------------------------------------------------- anchors
def anchors(bins: pd.DataFrame, onsets: pd.Series, contact: pd.Series) -> dict:
    """Anchor = bin end t in the 7 days before onset; input = last L bins. Adds the visible-contact flag."""
    feats = ALL
    groups = {v: g.set_index("bin")[feats] for v, g in bins.groupby("victim")}
    t_all, v_all, X = [], [], []
    for victim, on in onsets.items():
        b_last, b_first = (on - 1) // BIN, (on - LOOKBACK) // BIN
        grid = np.arange(b_first - L + 1, b_last + 1)
        vb = groups.get(victim, pd.DataFrame(columns=feats)).reindex(grid, fill_value=0.0).to_numpy(dtype="float32")
        win = np.lib.stride_tricks.sliding_window_view(vb, (L, len(feats)))[:, 0]
        ends = (grid[L - 1:] + 1) * BIN
        keep = ends <= on
        X.append(win[keep])
        t_all.append(ends[keep])
        v_all.append(np.full(keep.sum(), victim, dtype=object))
    t, v = np.concatenate(t_all), np.concatenate(v_all)
    on = onsets.reindex(v).to_numpy()
    c = contact.reindex(v).to_numpy()
    visible_at = np.where(c >= 0, (c // BIN + 1) * BIN, np.iinfo(np.int64).max)
    return {"X": np.concatenate(X), "t": t, "victim": v, "onset": on, "features": feats,
            "pre": t < visible_at,                 # contact not yet visible at t
            "contacted": c >= 0}


def summarise(X: np.ndarray, idx: list[int]) -> np.ndarray:
    x = np.log1p(np.clip(X[:, :, idx], 0, None))
    return np.concatenate([x[:, -1], x.mean(1), x.max(1), x[:, -1] - x[:, 0]], axis=1)


# --------------------------------------------------------------------------- models
def fit(kind: str, X, y, seed: int):
    if kind == "logistic":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        m = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    return m.fit(X, y)


def subsample(rows, y, rng):
    neg = rows[y[rows] == 0]
    keep = rng.random(len(neg)) < NEG_SUBSAMPLE
    return np.sort(np.concatenate([rows[y[rows] == 1], neg[keep]]))


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def pseudo_p(A, rows, p, h, real, rng):
    if real is None:
        return None
    t, v = A["t"][rows], A["victim"][rows]
    uv = np.unique(v)
    lo = {x: t[v == x].min() for x in uv}
    hi = {x: t[v == x].max() for x in uv}
    count = 0
    for _ in range(N_PSEUDO):
        fake = {x: rng.uniform(lo[x], hi[x]) for x in uv}
        a = safe_auc(P.labels(t, np.array([fake[x] for x in v]), h), p)
        count += a is not None and a >= real
    return (1 + count) / (N_PSEUDO + 1)


def shifted_onsets(A, rows, rng):
    t, v = A["t"][rows], A["victim"][rows]
    fake = {x: rng.uniform(t[v == x].min(), t[v == x].max()) for x in np.unique(v)}
    return np.array([fake[x] for x in v])


def run_fold(A, f, h, design, kind, seed, rng, mask, controls: bool) -> dict:
    """mask: boolean over anchors; the model is trained and evaluated only on rows where mask is True."""
    y = P.labels(A["t"], A["onset"], h)
    tr = f["train"][mask[f["train"]]]
    te = f["test"][mask[f["test"]]]
    va = f["val"][mask[f["val"]]]
    out = {"burst": f["burst"], "day": f["day"], "n_test": int(len(te)), "pos": int(y[te].sum())}
    if len(tr) == 0 or y[tr].sum() == 0 or len(te) == 0:
        return out | {"auc": None}
    trs = subsample(tr, y, rng)
    m = fit(kind, design[trs], y[trs], seed)
    p_te, p_va = m.predict_proba(design[te])[:, 1], m.predict_proba(design[va])[:, 1] if len(va) else np.array([])
    yt, yv = y[te], y[va]
    real = safe_auc(yt, p_te)
    out["auc"] = real
    neg_v = p_va[yv == 0]
    thr = float(np.quantile(neg_v, 0.99)) if len(neg_v) else np.inf
    out["test_fpr"] = float((p_te[yt == 0] >= thr).mean()) if (yt == 0).any() else None
    tv, tt, to = A["victim"][te], A["t"][te], A["onset"][te]
    contacted = A["contacted"][te]
    alerted, leads, cont_alerted, cont_n = 0, [], 0, 0
    for vic in np.unique(tv):
        w = tv == vic
        on = to[w][0]
        near = w & (tt >= on - h * 60) & (tt <= on)
        hit = bool((p_te[near] >= thr).any())
        alerted += hit
        win60 = w & (tt >= on - 3600) & (tt <= on) & (p_te >= thr)
        lead = float(on - tt[win60].min()) if win60.any() else None
        if contacted[w][0]:
            cont_n += 1
            cont_alerted += hit
            if lead is not None:
                leads.append(lead)
    out |= {"alerted": int(alerted), "onsets": int(len(np.unique(tv))), "cont_alerted": int(cont_alerted),
            "cont_onsets": int(cont_n), "leads_s": leads}
    if controls:
        out["pseudo_p"] = pseudo_p(A, te, p_te, h, real, rng)
        ys = np.zeros(len(y), int)
        ys[tr] = P.labels(A["t"][tr], shifted_onsets(A, tr, rng), h)
        trs_s = subsample(tr, ys, rng)
        ms = fit(kind, design[trs_s], ys[trs_s], seed)
        out["shift_auc"] = safe_auc(yt, ms.predict_proba(design[te])[:, 1])
    return out


def valid(folds):
    return [f for f in folds if f.get("auc") is not None]


def judge_forecast(fo, tf):
    b = BARS
    v, tv = valid(fo), valid(tf)
    pos = sum(f["pos"] for f in v)
    auc, tauc, sauc = med(f["auc"] for f in v), med(f["auc"] for f in tv), med(f.get("shift_auc") for f in v)
    ps = [f["pseudo_p"] for f in v if f.get("pseudo_p") is not None]
    a9, t9 = med(f["auc"] for f in v if f["day"] != DAY9), med(f["auc"] for f in tv if f["day"] != DAY9)
    r = {"valid_folds": len(v), "pooled_pos": pos, "median_auc": auc, "time_auc": tauc, "shift_auc": sauc,
         "pseudo_sig_folds": sum(p < b["f3_alpha"] for p in ps), "pseudo_folds": len(ps),
         "median_auc_no_day9": a9, "time_auc_no_day9": t9}
    r["testable"] = len(v) >= MIN_FOLDS and pos >= MIN_POS
    r["F1"] = r["testable"] and auc >= b["f1_auc"]
    r["F2"] = r["testable"] and None not in (auc, tauc) and auc - tauc >= b["f2_time"]
    r["F3"] = r["testable"] and bool(ps) and r["pseudo_sig_folds"] >= b["f3_frac"] * len(ps)
    r["F4"] = r["testable"] and None not in (auc, sauc) and auc - sauc >= b["f4_shift"]
    r["F5"] = r["testable"] and None not in (a9, t9) and a9 >= b["f5_auc"] and a9 - t9 >= b["f5_time"]
    r["all"] = all(r[k] for k in ("F1", "F2", "F3", "F4", "F5"))
    return r


def judge_detect(fo, tf):
    b = BARS
    v, tv = valid(fo), valid(tf)
    auc, tauc, sauc = med(f["auc"] for f in v), med(f["auc"] for f in tv), med(f.get("shift_auc") for f in v)
    ca, cn = sum(f["cont_alerted"] for f in v), sum(f["cont_onsets"] for f in v)
    leads = [x for f in v for x in f["leads_s"]]
    fpr = med(f["test_fpr"] for f in v)
    r = {"valid_folds": len(v), "median_auc": auc, "time_auc": tauc, "shift_auc": sauc,
         "contacted_alerted": ca, "contacted_onsets": cn, "alerted": sum(f["alerted"] for f in v),
         "onsets": sum(f["onsets"] for f in v), "median_test_fpr": fpr,
         "median_alert_lead_s": float(np.median(leads)) if leads else None, "n_leads": len(leads)}
    r["D1"] = auc is not None and auc >= b["d1_auc"]
    r["D2"] = None not in (auc, tauc, sauc) and auc - tauc >= b["d2_time"] and auc - sauc >= b["d2_shift"]
    r["D3"] = (cn > 0 and ca >= b["d3_alert_frac"] * cn and fpr is not None and fpr <= b["d3_fpr"]
               and r["median_alert_lead_s"] is not None and r["median_alert_lead_s"] >= b["d3_lead_s"])
    r["all"] = all(r[k] for k in ("D1", "D2", "D3"))
    return r


def contact_lead_summary(onsets: pd.Series, contact: pd.Series) -> dict:
    lead = (onsets - contact)[contact >= 0] / 60.0
    return {"victims": int(len(onsets)), "contacted": int(len(lead)),
            "lead_min_quantiles": {str(q): float(lead.quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)} if len(lead) else {},
            "lead_ge_min": {str(m): int((lead >= m).sum()) for m in (1, 2, 5, 10, 15, 30)}}


def study(seed=42, report=REPORT, raw=RAW, bins_path=BINS, contacts_path=CONTACTS, horizons=HORIZONS_MIN) -> dict:
    rng = np.random.default_rng(seed)
    bins = pd.read_parquet(bins_path)
    ct = pd.read_parquet(contacts_path)
    onsets = ct.set_index("victim")["onset"]
    onsets = onsets[onsets.index.isin(set(bins["victim"]))]
    contact = ct.set_index("victim")["contact"].reindex(onsets.index)
    A = anchors(bins, onsets, contact)
    burst_of = P.bursts(onsets)
    folds = P.make_folds(A, burst_of, onsets)
    feats = A["features"]
    designs = {n: summarise(A["X"], [feats.index(c) for c in cols]) for n, cols in FEATURE_SETS.items()}
    designs["time"] = P.time_features(A["t"])
    allrows = np.ones(len(A["t"]), bool)
    res = {"seed": seed, "victims": int(len(onsets)), "anchors": int(len(A["t"])), "pre_contact_anchors": int(A["pre"].sum()),
           "bursts": len(folds), "contact_lead": contact_lead_summary(onsets, contact), "variants": {}}
    print(json.dumps({k: res[k] for k in ("victims", "anchors", "pre_contact_anchors", "bursts")}),
          json.dumps(res["contact_lead"]), flush=True)
    t0 = time.time()
    for h in horizons:
        tf_pre = [run_fold(A, f, h, designs["time"], "gboost", seed, rng, A["pre"], False) for f in folds]
        tf_all = [run_fold(A, f, h, designs["time"], "gboost", seed, rng, allrows, False) for f in folds]
        res["variants"][f"time-only|pre|H{h}"] = {"median_auc": med(f["auc"] for f in valid(tf_pre))}
        res["variants"][f"time-only|all|H{h}"] = {"median_auc": med(f["auc"] for f in valid(tf_all))}
        for fs in ("own", "own+inbound"):
            for kind in ("logistic", "gboost"):
                fo = [run_fold(A, f, h, designs[fs], kind, seed, rng, A["pre"], True) for f in folds]
                v = judge_forecast(fo, tf_pre)
                res["variants"][f"forecast|{kind}|{fs}|H{h}"] = {"verdict": v, "folds": fo}
                print(f"FORECAST H{h} {kind:8s} {fs:12s} auc={v['median_auc']} time={v['time_auc']} shift={v['shift_auc']} "
                      f"sig={v['pseudo_sig_folds']}/{v['pseudo_folds']} folds={v['valid_folds']} pos={v['pooled_pos']} "
                      f"pass={v['all']} ({time.time() - t0:.0f}s)", flush=True)
        for kind in ("logistic", "gboost"):
            fo = [run_fold(A, f, h, designs["own+inbound+attacker"], kind, seed, rng, allrows, True) for f in folds]
            v = judge_detect(fo, tf_all)
            res["variants"][f"detect|{kind}|own+inbound+attacker|H{h}"] = {"verdict": v, "folds": fo}
            print(f"DETECT H{h} {kind:8s} auc={v['median_auc']} time={v['time_auc']} shift={v['shift_auc']} "
                  f"contacted_alerted={v['contacted_alerted']}/{v['contacted_onsets']} fpr={v['median_test_fpr']} "
                  f"lead_s={v['median_alert_lead_s']} pass={v['all']} ({time.time() - t0:.0f}s)", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(res, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["bins", "study"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--bins", type=Path, default=BINS)
    ap.add_argument("--contacts", type=Path, default=CONTACTS)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS_MIN))
    a = ap.parse_args(argv)
    if a.part == "bins":
        build_bins(a.raw, a.bins, a.contacts)
    else:
        out = REPORT if a.seed == 42 else REPORT.with_name(f"lanl_short_lead_study_seed{a.seed}.json")
        study(a.seed, out, a.raw, a.bins, a.contacts, tuple(a.horizons))


if __name__ == "__main__":
    main()
