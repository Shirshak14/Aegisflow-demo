"""
LANL 2015 next-victim study. Design and bars: docs/lanl_next_victim_design.md (committed first).

    python scripts/lanl_next_victim_study.py extract   # one pass over flows.txt.gz -> data/processed/lanl/next_victim/
    python scripts/lanl_next_victim_study.py study     # -> reports/lanl_next_victim_study.json
    python scripts/lanl_next_victim_study.py study --seed 7

Standalone (pandas, numpy, scikit-learn only). Nothing here touches the demo model,
its threshold or the default dataset.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/lanl"
OUT = ROOT / "data/processed/lanl/next_victim"
REPORT = ROOT / "reports/lanl_next_victim_study.json"

FLOW_COLUMNS = ["time", "duration", "src", "src_port", "dst", "dst_port", "protocol", "packets", "bytes"]
HORIZONS_MIN = (10, 60)
BURST_GAP = 1800
GRID = 300                     # decision every 5 minutes
ACTIVE = 3600                  # ... while an onset happened in the last hour
EXCLUDE_NEAR_TEST = 6 * 3600
NEG_SUBSAMPLE = 0.05
N_NULL = 1000
TOPK = 10
DAY9 = 9
MIN_BURSTS = 10
NO_CONTACT_AGE = 1e6
BARS = dict(n1_auc=0.80, n2_margin=0.10, n3_ratio=2.0, n3_alpha=0.05, n4_lead_s=120)

FEATURES = ["contact_60", "contact_7d", "contact_age", "act24", "nbr", "deg"]
CONTROL_FEATURES = ["act24", "nbr", "deg"]


def load_redteam(path: Path) -> pd.DataFrame:
    rt = pd.read_csv(path, header=None, names=["time", "user", "src", "dst"], dtype={"time": "int64"})
    return rt.sort_values("time").reset_index(drop=True)


def onsets_of(rt: pd.DataFrame) -> pd.Series:
    return rt.groupby("dst")["time"].min().sort_values()


# --------------------------------------------------------------------------- extract
class Vocab:
    """Computer name -> int id, grown chunk by chunk."""

    def __init__(self):
        self.names: list[str] = []
        self.index = pd.Index([], dtype=object)

    def ids(self, s: pd.Series) -> np.ndarray:
        idx = self.index.get_indexer(s)
        if (idx < 0).any():
            new = pd.unique(s[idx < 0])
            self.names.extend(new)
            self.index = pd.Index(self.names, dtype=object)
            idx = self.index.get_indexer(s)
        return idx.astype("int64")


def _reduce_edges(parts: list[pd.DataFrame]) -> pd.DataFrame:
    e = pd.concat(parts)
    return e.groupby(level=0).agg(first_t=("first_t", "min"), n=("n", "sum"))


def extract(raw: Path = RAW, out: Path = OUT, chunksize: int = 4_000_000) -> None:
    rt = load_redteam(raw / "redteam.txt.gz")
    onset = onsets_of(rt)
    sources = set(rt["src"])
    t_stop = int(onset.max())
    vocab = Vocab()
    att_parts, hour_parts, edge_pending = [], [], []
    edges = None
    t0 = time.time()
    reader = pd.read_csv(raw / "flows.txt.gz", header=None, names=FLOW_COLUMNS, dtype=str, chunksize=chunksize,
                         keep_default_na=False, usecols=["time", "src", "dst", "dst_port", "bytes"])
    for chunk in reader:
        t = pd.to_numeric(chunk["time"]).astype("int64")
        if t.iloc[0] > t_stop:
            break
        keep = (t <= t_stop).to_numpy()
        chunk, t = chunk[keep], t[keep].to_numpy()
        src, dst = chunk["src"], chunk["dst"]
        a = src.isin(sources).to_numpy()
        if a.any():
            att = pd.DataFrame({"src": src[a].to_numpy(), "dst": dst[a].to_numpy(), "minute": t[a] // 60,
                                "bytes": pd.to_numeric(chunk["bytes"][a], errors="coerce").fillna(0).to_numpy(),
                                "dport": chunk["dst_port"][a].to_numpy()})
            att_parts.append(att.groupby(["src", "dst", "minute"]).agg(
                n=("bytes", "size"), bytes=("bytes", "sum"), ports=("dport", "nunique")).reset_index())
        si, di = vocab.ids(src), vocab.ids(dst)
        hour = t // 3600
        hout = pd.DataFrame({"c": si, "h": hour}).value_counts().rename("out")
        hin = pd.DataFrame({"c": di, "h": hour}).value_counts().rename("in")
        hour_parts.append(pd.concat([hout, hin], axis=1).fillna(0))
        lo, hi = np.minimum(si, di), np.maximum(si, di)
        ok = lo != hi
        key = (lo[ok] << 32) | hi[ok]
        e = pd.DataFrame({"k": key, "t": t[ok]}).groupby("k")["t"].agg(first_t="min", n="size")
        edge_pending.append(e)
        if len(edge_pending) >= 8:
            edges = _reduce_edges(([edges] if edges is not None else []) + edge_pending)
            edge_pending = []
        print(f"t={int(t[-1]):,}/{t_stop:,} hosts={len(vocab.names):,} "
              f"edges~{(0 if edges is None else len(edges)):,} {time.time() - t0:.0f}s", flush=True)
    if edge_pending:
        edges = _reduce_edges(([edges] if edges is not None else []) + edge_pending)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": np.arange(len(vocab.names)), "name": vocab.names}).to_parquet(out / "hosts.parquet", index=False)
    hourly = pd.concat(hour_parts).groupby(level=[0, 1]).sum().astype("int64").reset_index()
    hourly.columns = ["c", "h", "out", "in"]
    hourly.to_parquet(out / "hourly.parquet", index=False)
    k = edges.index.to_numpy()
    pd.DataFrame({"a": k >> 32, "b": k & 0xFFFFFFFF, "first_t": edges["first_t"].to_numpy(),
                  "n": edges["n"].to_numpy()}).to_parquet(out / "edges.parquet", index=False)
    att = (pd.concat(att_parts).groupby(["src", "dst", "minute"]).agg(n=("n", "sum"), bytes=("bytes", "sum"),
                                                                      ports=("ports", "max")).reset_index()
           if att_parts else pd.DataFrame(columns=["src", "dst", "minute", "n", "bytes", "ports"]))
    att.to_parquet(out / "attacker_flows.parquet", index=False)
    print(f"done hosts={len(vocab.names):,} hourly={len(hourly):,} edges={len(k):,} attacker_rows={len(att):,} "
          f"{time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- study data
def load_tables(raw: Path = RAW, out: Path = OUT) -> dict:
    rt = load_redteam(raw / "redteam.txt.gz")
    return dict(rt=rt, hosts=pd.read_parquet(out / "hosts.parquet"), hourly=pd.read_parquet(out / "hourly.parquet"),
                edges=pd.read_parquet(out / "edges.parquet"), att=pd.read_parquet(out / "attacker_flows.parquet"))


def bursts(times: np.ndarray, gap: int = BURST_GAP) -> np.ndarray:
    """Burst id per (sorted) decision time: chained while consecutive gaps are < gap."""
    b = np.zeros(len(times), dtype=int)
    for i in range(1, len(times)):
        b[i] = b[i - 1] + (times[i] - times[i - 1] >= gap)
    return b


def decision_grid(onset_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decision points: every GRID-second boundary tau with at least one onset in [tau - ACTIVE, tau]
    (an attack is under way). Burst of tau = burst of the latest onset at or before tau."""
    ob = bursts(onset_times)
    grid = np.unique(np.concatenate([np.arange(-(-o // GRID) * GRID, o + ACTIVE + 1, GRID) for o in onset_times]))
    last = np.searchsorted(onset_times, grid, side="right") - 1
    keep = (last >= 0) & (grid - onset_times[np.maximum(last, 0)] <= ACTIVE)
    return grid[keep], ob[last[keep]]


def build_rows(T: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One row per (decision, candidate) with the six features and the onset of the candidate (inf if never)."""
    rt, hosts = T["rt"], T["hosts"]
    name_to_id = pd.Series(hosts["id"].to_numpy(), index=hosts["name"].to_numpy())
    n_hosts = len(hosts)
    onset_s = onsets_of(rt)
    onset_id = np.full(n_hosts, np.inf)
    vis = onset_s[onset_s.index.isin(name_to_id.index)]
    onset_id[name_to_id[vis.index].to_numpy()] = vis.to_numpy()
    src_ids = name_to_id[[s for s in set(rt["src"]) if s in name_to_id.index]].to_numpy()

    hourly = T["hourly"]
    n_hours = int(hourly["h"].max()) + 1
    act = np.zeros((n_hours + 1, n_hosts), dtype="float64")
    np.add.at(act, (hourly["h"].to_numpy() + 1, hourly["c"].to_numpy()), (hourly["out"] + hourly["in"]).to_numpy())
    act_cum = act.cumsum(axis=0)        # act_cum[h] = flows in hours < h

    att = T["att"]
    att = att[att["dst"].isin(name_to_id.index)]
    att_dst = name_to_id[att["dst"]].to_numpy()
    att_end = (att["minute"].to_numpy() + 1) * 60
    att_n = att["n"].to_numpy().astype(float)
    order = np.argsort(att_end, kind="stable")
    att_dst, att_end, att_n = att_dst[order], att_end[order], att_n[order]

    edges = T["edges"].sort_values("first_t")
    ea, eb, et = edges["a"].to_numpy(), edges["b"].to_numpy(), edges["first_t"].to_numpy()

    times, burst_of = decision_grid(np.unique(onset_s.to_numpy()))
    deg = np.zeros(n_hosts)
    e_ptr = 0
    rows = []
    for d, tau in enumerate(times):
        e_new = np.searchsorted(et, tau, side="left")         # edges first seen strictly before tau
        np.add.at(deg, ea[e_ptr:e_new], 1)
        np.add.at(deg, eb[e_ptr:e_new], 1)
        e_ptr = e_new
        h0 = int(tau // 3600)
        hi = min(h0, n_hours)
        lo = max(h0 - 24, 0)
        act24 = act_cum[hi] - act_cum[lo]
        compromised = onset_id <= tau
        cand = act24 > 0
        cand &= ~compromised
        cand[src_ids] = False
        cid = np.flatnonzero(cand)
        # attacker contact
        a_hi = np.searchsorted(att_end, tau, side="right")
        a60 = np.searchsorted(att_end, tau - 3600, side="right")
        a7d = np.searchsorted(att_end, tau - 7 * 86400, side="right")
        c60 = np.bincount(att_dst[a60:a_hi], weights=att_n[a60:a_hi], minlength=n_hosts)
        c7d = np.bincount(att_dst[a7d:a_hi], weights=att_n[a7d:a_hi], minlength=n_hosts)
        first = np.full(n_hosts, np.inf)
        np.minimum.at(first, att_dst[:a_hi], att_end[:a_hi] - 60)
        age = np.where(np.isfinite(first), (tau - first) / 60.0, NO_CONTACT_AGE)
        # neighbours already compromised (pairs first seen before tau)
        comp_ids = np.flatnonzero(compromised)
        m = np.isin(ea[:e_new], comp_ids)
        m2 = np.isin(eb[:e_new], comp_ids)
        nbr = (np.bincount(eb[:e_new][m], minlength=n_hosts) + np.bincount(ea[:e_new][m2], minlength=n_hosts))
        rows.append(pd.DataFrame({
            "decision": d, "tau": tau, "burst": burst_of[d], "cand": cid,
            "contact_60": c60[cid], "contact_7d": c7d[cid], "contact_age": age[cid],
            "act24": act24[cid], "nbr": nbr[cid].astype(float), "deg": deg[cid], "onset": onset_id[cid]}))
    R = pd.concat(rows, ignore_index=True)
    dec = pd.DataFrame({"decision": np.arange(len(times)), "tau": times, "burst": burst_of})
    meta = dict(n_hosts=n_hosts, victims=int(len(onset_s)), victims_in_flows=int(np.isfinite(onset_id).sum()),
                decisions=int(len(times)), bursts=int(len(np.unique(burst_of))), rows=int(len(R)),
                median_candidates=float(R.groupby("decision").size().median()))
    return R, dec, meta


# --------------------------------------------------------------------------- scoring
def rank_scores(R: pd.DataFrame, name: str, preds: dict | None = None) -> np.ndarray:
    if name == "R-CONTACT":
        return R["contact_60"].to_numpy() * 1e9 + R["contact_7d"].to_numpy()
    if name == "C-ACT":
        return R["act24"].to_numpy()
    if name == "C-NBR":
        return R["nbr"].to_numpy()
    if name == "C-DEG":
        return R["deg"].to_numpy()
    return preds[name]


def fit_loo(R: pd.DataFrame, y: np.ndarray, feats: list[str], seed: int) -> np.ndarray:
    """Leave one burst out; training decisions within 6 h of the test burst dropped; 5% negatives."""
    rng = np.random.default_rng(seed)
    X = R[feats].to_numpy(dtype="float64")   # trees: no scaling needed
    pred = np.full(len(R), np.nan)
    burst, tau = R["burst"].to_numpy(), R["tau"].to_numpy()
    for b in np.unique(burst):
        test = burst == b
        if y[test].sum() == 0:
            continue
        t_lo, t_hi = tau[test].min() - EXCLUDE_NEAR_TEST, tau[test].max() + EXCLUDE_NEAR_TEST
        train = ~test & ~((tau >= t_lo) & (tau <= t_hi))
        keep = train & ((y == 1) | (rng.random(len(R)) < NEG_SUBSAMPLE))
        if y[keep].sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(X[keep], y[keep])
        pred[test] = clf.predict_proba(X[test])[:, 1]
    return pred


def evaluate(R: pd.DataFrame, y: np.ndarray, score: np.ndarray, seed: int, h_s: int) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"decision": R["decision"].to_numpy(), "burst": R["burst"].to_numpy(),
                       "tau": R["tau"].to_numpy(), "onset": R["onset"].to_numpy(), "y": y, "s": score,
                       "tie": rng.random(len(R))})
    per_dec, alerts, null_inputs = [], [], []
    for d, g in df.groupby("decision", sort=True):
        if g["y"].sum() == 0 or g["s"].isna().any():
            continue
        n, m = len(g), int(g["y"].sum())
        auc = roc_auc_score(g["y"], g["s"]) if m < n else np.nan
        order = np.lexsort((-g["tie"].to_numpy(), -g["s"].to_numpy()))
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        yy = g["y"].to_numpy()
        best = int(ranks[yy == 1].min())
        per_dec.append(dict(decision=d, burst=int(g["burst"].iloc[0]), n=n, m=m, auc=auc,
                            hit=float(best <= TOPK), best_rank=best))
        null_inputs.append((n, m, int(g["burst"].iloc[0])))
        top = ranks <= TOPK
        for on, tau in zip(g["onset"].to_numpy()[top & (yy == 1)], g["tau"].to_numpy()[top & (yy == 1)]):
            alerts.append((on, on - tau))
    P = pd.DataFrame(per_dec)
    if P.empty:
        return dict(scored_decisions=0, scored_bursts=0)
    B = P.groupby("burst").agg(auc=("auc", "mean"), hit=("hit", "mean"), best_rank=("best_rank", "median"))
    # pseudo-victim null: same n and m per decision, random positives
    sims = np.zeros((N_NULL, len(null_inputs)))
    for j, (n, m, _) in enumerate(null_inputs):
        sims[:, j] = rng.hypergeometric(min(TOPK, n), max(n - TOPK, 0), m, size=N_NULL) > 0
    nb = np.array([b for _, _, b in null_inputs])
    ub = np.unique(nb)
    null_mean = np.stack([sims[:, nb == b].mean(axis=1) for b in ub], axis=1).mean(axis=1)
    real_mean = float(B["hit"].mean())
    lead = pd.DataFrame(alerts, columns=["onset", "lead"]).groupby("onset")["lead"].max() if alerts else pd.Series(dtype=float)
    return dict(scored_decisions=int(len(P)), scored_bursts=int(len(B)),
                median_burst_auc=float(B["auc"].median()), mean_burst_auc=float(B["auc"].mean()),
                mean_burst_hit10=real_mean, median_burst_best_rank=float(B["best_rank"].median()),
                null_mean_hit10=float(null_mean.mean()),
                null_p=float((1 + (null_mean >= real_mean).sum()) / (N_NULL + 1)),
                victims_alerted=int(len(lead)), median_lead_s=(float(lead.median()) if len(lead) else None),
                burst_auc={int(k): float(v) for k, v in B["auc"].items()})


def bars(cand: dict, controls: dict, cand_no9: dict, controls_no9: dict) -> dict:
    best_ctrl = max(c.get("median_burst_auc", 0.0) for c in controls.values())
    best_ctrl_hit = max(c.get("mean_burst_hit10", 0.0) for c in controls.values())
    best_ctrl9 = max(c.get("median_burst_auc", 0.0) for c in controls_no9.values())
    auc = cand.get("median_burst_auc", 0.0)
    auc9 = cand_no9.get("median_burst_auc", 0.0)
    out = dict(
        N1=bool(cand.get("scored_bursts", 0) >= MIN_BURSTS and auc >= BARS["n1_auc"]),
        N2=bool(auc >= best_ctrl + BARS["n2_margin"]),
        N3=bool(cand.get("mean_burst_hit10", 0.0) >= BARS["n3_ratio"] * best_ctrl_hit
                and cand.get("null_p", 1.0) < BARS["n3_alpha"]),
        N4=bool((cand.get("median_lead_s") or 0) >= BARS["n4_lead_s"]),
        N5=bool(auc9 >= BARS["n1_auc"] and auc9 >= best_ctrl9 + BARS["n2_margin"]),
        best_control_auc=best_ctrl, best_control_hit10=best_ctrl_hit, best_control_auc_no_day9=best_ctrl9)
    out["all"] = all(out[k] for k in ("N1", "N2", "N3", "N4", "N5"))
    return out


def contact_precision(T: dict) -> dict:
    """Descriptive: hosts S contacts in the flows, how many are later compromised, contact-to-onset delay."""
    att = T["att"]
    onset = onsets_of(T["rt"])
    first = att.groupby("dst")["minute"].min() * 60
    first = first[~first.index.isin(set(T["rt"]["src"]))]
    on = onset.reindex(first.index)
    later = on.notna() & (on >= first)
    delay = (on[later] - first[later]) / 60.0
    return dict(hosts_contacted=int(len(first)), later_compromised=int(later.sum()),
                compromised_before_contact=int((on.notna() & (on < first)).sum()),
                never_compromised=int(on.isna().sum()),
                delay_min_quantiles={q: float(delay.quantile(q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
                if len(delay) else {})


def study(seed: int = 42, report: Path = REPORT, raw: Path = RAW, out: Path = OUT) -> dict:
    T = load_tables(raw, out)
    R, dec, meta = build_rows(T)
    day = (R["tau"] // 86400).astype(int)
    no9 = (day != DAY9).to_numpy()
    res = dict(meta=meta, seed=seed, bars=BARS, contact=contact_precision(T), horizons={})
    for h in HORIZONS_MIN:
        hs = h * 60
        y = ((R["onset"] > R["tau"]) & (R["onset"] <= R["tau"] + hs)).to_numpy().astype(int)
        preds = {"M-CTRL": fit_loo(R, y, CONTROL_FEATURES, seed), "M-GB": fit_loo(R, y, FEATURES, seed)}
        names = ["C-ACT", "C-NBR", "C-DEG", "M-CTRL", "R-CONTACT", "M-GB"]
        full = {n: evaluate(R, y, rank_scores(R, n, preds), seed, hs) for n in names}
        sub = R[no9].reset_index(drop=True)
        part = {n: evaluate(sub, y[no9], rank_scores(R, n, preds)[no9], seed, hs) for n in names}
        ctrl = {n: full[n] for n in ("C-ACT", "C-NBR", "C-DEG", "M-CTRL")}
        ctrl9 = {n: part[n] for n in ("C-ACT", "C-NBR", "C-DEG", "M-CTRL")}
        res["horizons"][str(h)] = dict(
            positives=int(y.sum()), results=full, results_no_day9=part,
            bars={n: bars(full[n], ctrl, part[n], ctrl9) for n in ("R-CONTACT", "M-GB")})
        print(h, {n: round(full[n].get("median_burst_auc", float("nan")), 3) for n in names}, flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "study"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.raw, a.out)
    else:
        rep = a.report or (REPORT if a.seed == 42 else REPORT.with_name(f"lanl_next_victim_study_seed{a.seed}.json"))
        study(a.seed, rep, a.raw, a.out)


if __name__ == "__main__":
    main()
