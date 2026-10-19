"""Leave-one-scenario-out forecasting study on the v2 synthetic lab (research/synth_lab_v2/DESIGN.md).

    python research/synth_lab_v2/study.py CONDITION [--seed 1] [--leave-chain-out]

Writes research/synth_lab_v2/results/<CONDITION>_s<seed>.json. All data is synthetic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth_lab_v2 as lab  # noqa: E402

HORIZONS = {"30m": 3, "2h": 12, "12h": 72}         # in 10-minute windows
LOOKBACK = (1, 6, 36, 144)                          # 10 min, 1 h, 6 h, 24 h
BASELINE = 7 * lab.PER_DAY
COOLDOWN = lab.PER_DAY                              # rows within 24 h after an onset are not scored
NEG_KEEP = 0.05
BUDGET = 0.001                                      # flagged share of negative windows
MIN_LEAD = 1                                        # warning must come at least one window (10 min) early
ALERT_W = (1, 6, 36, 144)
BARS = dict(p1_ratio=1.5, p1_orgs=5, p2_auc=0.75, p3_recall=0.25, p3_ratio=1.5, p4_lead_min=30,
            p5_null_ratio=1.2, p5_z_mult=3.0, p5_z_margin=0.05)
CONTROLS = ["C-TIME", "C-CTX", "C-ALERT", "C-SCAN", "C-ALERT+CTX"]
MODELS = ["M-LR", "M-GB"]


def window_sums(X: np.ndarray, k: int) -> np.ndarray:
    """Sum over the last k windows including the current one (truncated at the start)."""
    cs = np.concatenate([np.zeros_like(X[:, :1]), np.cumsum(X, 1, dtype=np.float64)], 1)
    t = np.arange(X.shape[1])
    return (cs[:, t + 1] - cs[:, np.maximum(t + 1 - k, 0)]).astype(np.float32)


def build_org(org: dict) -> dict:
    X, roles, shift = org["X"], org["roles"], org["shift"]
    n, T, d = X.shape
    sums = {k: window_sums(X, k) for k in LOOKBACK}
    base = window_sums(X, BASELINE)
    t_idx = np.arange(T)
    span = np.minimum(t_idx + 1, BASELINE).astype(np.float32)[None, :, None]
    dev = np.log1p(sums[36]) - np.log1p(base * (36.0 / span))
    hour = lab.local_hour(t_idx, shift)
    weekend = (lab.weekday(t_idx, shift) >= 5).astype(np.float32)

    # next onset per host-window, last onset per host-window
    next_dt = np.full((n, T), 10 ** 6, np.int32)
    next_ct = np.full((n, T), -1, np.int8)
    last_on = np.full((n, T), -(10 ** 6), np.int32)
    for h, w, ct in org["onsets"]:
        tt = t_idx[:w]
        dt = w - tt
        upd = dt < next_dt[h, :w]
        next_dt[h, :w][upd] = dt[upd]
        next_ct[h, :w][upd] = lab.CHAINS.index(ct)
        last_on[h, w:] = np.maximum(last_on[h, w:], w)
    near_abort = np.zeros((n, T), bool)
    for h, w0, w1, _ct in org["aborted"]:
        near_abort[h, max(w0, 0):min(w1 + 2 * lab.PER_DAY, T)] = True

    maxH = max(HORIZONS.values())
    tt = np.arange(lab.BURN_IN, T - maxH)
    ok = (tt[None, :] - last_on[:, tt]) > COOLDOWN
    hh, ti = np.nonzero(ok)
    tw = tt[ti]
    feats = [np.log1p(sums[k][hh, tw]) for k in LOOKBACK] + [dev[hh, tw]]
    rad = 2 * np.pi * hour[tw] / 24
    feats.append(np.stack([np.sin(rad), np.cos(rad), weekend[tw]], 1).astype(np.float32))
    feats.append(np.eye(len(lab.ROLES), dtype=np.float32)[roles[hh]])
    F = np.concatenate(feats, 1).astype(np.float32)
    alert = np.stack([sums[k][hh, tw][:, lab.ALERT_CH].sum(1) for k in ALERT_W], 1)
    scan2 = window_sums(X[:, :, lab.SCAN_CH], 2)[hh, tw].sum(1)
    return dict(F=F, host=hh.astype(np.int32), t=tw.astype(np.int32), role=roles[hh],
                hour=np.floor(hour[tw]).astype(np.int8), weekend=weekend[tw],
                next_dt=next_dt[hh, tw], next_ct=next_ct[hh, tw], near_abort=near_abort[hh, tw],
                alert=alert.astype(np.float32), scan2=scan2.astype(np.float32),
                onsets=[(h, w, ct) for h, w, ct in org["onsets"] if lab.BURN_IN + maxH <= w < T],
                n_hosts=n)


def labels(D, Hw):
    return ((D["next_dt"] >= 1) & (D["next_dt"] <= Hw)).astype(np.int8)


def ctx_matrix(D, with_role):
    parts = [np.eye(24, dtype=np.float32)[D["hour"]], D["weekend"][:, None]]
    if with_role:
        parts.append(np.eye(len(lab.ROLES), dtype=np.float32)[D["role"]])
    return np.concatenate(parts, 1)


def design(D, name):
    if name in ("M-LR", "M-GB"):
        return D["F"]
    if name == "C-TIME":
        return ctx_matrix(D, False)
    if name == "C-CTX":
        return ctx_matrix(D, True)
    if name == "C-ALERT+CTX":
        return np.concatenate([np.log1p(D["alert"]), ctx_matrix(D, True)], 1)
    raise KeyError(name)


def make_model(name):
    if name == "M-GB":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=False, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))


def concat(Ds, idx_list):
    keys = ["F", "host", "t", "role", "hour", "weekend", "next_dt", "next_ct", "near_abort", "alert", "scan2"]
    return {k: np.concatenate([D[k][ix] for D, ix in zip(Ds, idx_list)]) for k in keys}


def jitter(rng, n):
    return rng.random(n) * 1e-6


def onset_eval(D, score, thr, Hw):
    """Per-onset: warned if any eligible row in [onset-H, onset-MIN_LEAD] is flagged; lead of first flag."""
    out = []
    flag = score >= thr
    for h, w, ct in D["onsets"]:
        m = (D["host"] == h) & (D["t"] >= w - Hw) & (D["t"] <= w - MIN_LEAD)
        f = m & flag
        lead = int(w - D["t"][f].min()) * lab.WIN_MIN if f.any() else None
        out.append(dict(chain=ct, warned=bool(f.any()), lead_min=lead, rows=int(m.sum())))
    return out


def run(condition, seed, leave_chain_out):
    t0 = time.time()
    rng = np.random.default_rng(seed + 1000)
    Ds = []
    for o in range(len(lab.ORGS)):
        org = lab.generate_org(o, condition, seed)
        Ds.append(build_org(org))
        del org
        print(f"org {o}: rows {len(Ds[-1]['t'])} onsets {len(Ds[-1]['onsets'])} ({time.time()-t0:.0f}s)", flush=True)

    res = {"condition": condition, "seed": seed, "orgs": [], "horizons": {}}
    for o, D in enumerate(Ds):
        res["orgs"].append(dict(org=o, hosts=D["n_hosts"], rows=int(len(D["t"])), onsets=len(D["onsets"]),
                                chains={c: sum(1 for x in D["onsets"] if x[2] == c) for c in lab.CHAINS}))

    for hname, Hw in HORIZONS.items():
        per = {m: [] for m in MODELS + CONTROLS}
        onsets_all = {m: [] for m in MODELS + CONTROLS}
        flag_rate = {m: [] for m in MODELS + CONTROLS}
        fp_abort = {m: [] for m in MODELS + CONTROLS}
        lco = {}
        for test in range(len(Ds)):
            tr = [i for i in range(len(Ds)) if i != test]
            fit_idx, cal_idx = [], []
            for i in tr:
                y = labels(Ds[i], Hw)
                r = rng.random(len(y))
                fit_idx.append(np.nonzero((y == 1) | (r < NEG_KEEP))[0])
                cal_idx.append(np.nonzero((y == 0) & (r >= NEG_KEEP) & (r < 2 * NEG_KEEP))[0])
            Tr = concat([Ds[i] for i in tr], fit_idx)
            Ca = concat([Ds[i] for i in tr], cal_idx)
            Te = Ds[test]
            ytr, yte = labels(Tr, Hw), labels(Te, Hw)
            scores, thrs = {}, {}
            for m in MODELS + ["C-TIME", "C-CTX", "C-ALERT+CTX"]:
                mdl = make_model(m)
                mdl.fit(design(Tr, m), ytr)
                scores[m] = mdl.predict_proba(design(Te, m))[:, 1] + jitter(rng, len(yte))
                cal = mdl.predict_proba(design(Ca, m))[:, 1] + jitter(rng, len(Ca["t"]))
                thrs[m] = float(np.quantile(cal, 1 - BUDGET))
                if m == "M-GB" and hname == "2h" and leave_chain_out:
                    lco[test] = {}
                    for ci, ct in enumerate(lab.CHAINS[:-1]):
                        keep = ~((ytr == 1) & (Tr["next_ct"] == ci))
                        m2 = make_model("M-GB").fit(Tr["F"][keep], ytr[keep])
                        s2 = m2.predict_proba(Te["F"])[:, 1] + jitter(rng, len(yte))
                        th2 = float(np.quantile(m2.predict_proba(Ca["F"])[:, 1] + jitter(rng, len(Ca["t"])), 1 - BUDGET))
                        ev = [e for e in onset_eval(Te, s2, th2, Hw) if e["chain"] == ct]
                        lco[test][ct] = dict(n=len(ev), warned=sum(e["warned"] for e in ev))
            # rule controls: W picked on training orgs by PR-AUC
            best_w, best_ap = 0, -1
            for j in range(len(ALERT_W)):
                ap = average_precision_score(ytr, Tr["alert"][:, j] + jitter(rng, len(ytr)))
                if ap > best_ap:
                    best_w, best_ap = j, ap
            scores["C-ALERT"] = Te["alert"][:, best_w] + jitter(rng, len(yte))
            thrs["C-ALERT"] = float(np.quantile(Ca["alert"][:, best_w] + jitter(rng, len(Ca["t"])), 1 - BUDGET))
            scores["C-SCAN"] = Te["scan2"] + jitter(rng, len(yte))
            thrs["C-SCAN"] = float(np.quantile(Ca["scan2"] + jitter(rng, len(Ca["t"])), 1 - BUDGET))

            for m in MODELS + CONTROLS:
                s = scores[m]
                per[m].append(dict(org=test, pr_auc=float(average_precision_score(yte, s)),
                                   roc_auc=float(roc_auc_score(yte, s)), base_rate=float(yte.mean()),
                                   alert_w=ALERT_W[best_w] if m == "C-ALERT" else None))
                neg = yte == 0
                fl = s >= thrs[m]
                flag_rate[m].append(float(fl[neg].mean()))
                fp_abort[m].append([int((fl & neg & Te["near_abort"]).sum()), int((fl & neg).sum())])
                onsets_all[m] += [dict(e, org=test) for e in onset_eval(Te, s, thrs[m], Hw)]
            print(f"{condition} s{seed} {hname} org{test} done ({time.time()-t0:.0f}s) "
                  f"GB pr {per['M-GB'][-1]['pr_auc']:.4f} ALERT pr {per['C-ALERT'][-1]['pr_auc']:.4f}", flush=True)

        summ = {}
        for m in MODELS + CONTROLS:
            ev = onsets_all[m]
            leads = [e["lead_min"] for e in ev if e["warned"]]
            by_chain = {}
            for ct in lab.CHAINS:
                e2 = [e for e in ev if e["chain"] == ct]
                by_chain[ct] = dict(n=len(e2), warned=sum(e["warned"] for e in e2))
            fa = np.array(fp_abort[m]).sum(0)
            summ[m] = dict(
                pr_auc_median=float(np.median([p["pr_auc"] for p in per[m]])),
                roc_auc_median=float(np.median([p["roc_auc"] for p in per[m]])),
                per_org=per[m],
                recall=float(np.mean([e["warned"] for e in ev])) if ev else None,
                n_onsets=len(ev), median_lead_min=float(np.median(leads)) if leads else None,
                by_chain=by_chain, test_flag_rate_median=float(np.median(flag_rate[m])),
                false_flags_near_aborted_chain=int(fa[0]), false_flags=int(fa[1]),
            )
        res["horizons"][hname] = dict(H_windows=Hw, results=summ, leave_chain_out=lco or None,
                                      random_flag_expectation=float(1 - (1 - BUDGET) ** Hw),
                                      verdict=verdict(summ, Hw))
    res["seconds"] = round(time.time() - t0)
    return res


def verdict(summ, Hw):
    best_m = max(MODELS, key=lambda m: summ[m]["pr_auc_median"])
    best_c = max(CONTROLS, key=lambda m: summ[m]["pr_auc_median"])
    best_c_rec = max(CONTROLS, key=lambda m: summ[m]["recall"] or 0)
    M, Cc = summ[best_m], summ[best_c]
    wins = sum(a["pr_auc"] > b["pr_auc"] for a, b in zip(M["per_org"], Cc["per_org"]))
    rand = 1 - (1 - BUDGET) ** Hw
    zc = M["by_chain"]["Z"]
    z_rec = zc["warned"] / zc["n"] if zc["n"] else 0.0
    zctl = max((summ[c]["by_chain"]["Z"]["warned"] / summ[c]["by_chain"]["Z"]["n"]) if summ[c]["by_chain"]["Z"]["n"] else 0
               for c in CONTROLS)
    v = dict(
        best_model=best_m, best_control=best_c, best_control_by_recall=best_c_rec,
        P1=bool(M["pr_auc_median"] >= BARS["p1_ratio"] * Cc["pr_auc_median"] and wins >= BARS["p1_orgs"]),
        P1_detail=dict(model=M["pr_auc_median"], control=Cc["pr_auc_median"], wins=wins),
        P2=bool(M["roc_auc_median"] >= BARS["p2_auc"]),
        P3=bool((M["recall"] or 0) >= BARS["p3_recall"]
                and (M["recall"] or 0) >= BARS["p3_ratio"] * (summ[best_c_rec]["recall"] or 0)),
        P3_detail=dict(model=M["recall"], control=summ[best_c_rec]["recall"]),
        P4=bool(Hw < 12 or (M["median_lead_min"] or 0) >= BARS["p4_lead_min"]),
        Z_recall=z_rec, Z_limit=max(BARS["p5_z_mult"] * rand, zctl + BARS["p5_z_margin"]),
        P5_z=bool(z_rec <= max(BARS["p5_z_mult"] * rand, zctl + BARS["p5_z_margin"])),
        null_ratio=M["pr_auc_median"] / Cc["pr_auc_median"] if Cc["pr_auc_median"] else None,
    )
    v["pass"] = bool(v["P1"] and v["P2"] and v["P3"] and v["P4"])
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=list(lab.CONDITIONS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--leave-chain-out", action="store_true")
    a = ap.parse_args()
    out = run(a.condition, a.seed, a.leave_chain_out)
    (HERE / "results").mkdir(exist_ok=True)
    p = HERE / "results" / f"{a.condition}_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    for h, r in out["horizons"].items():
        print(h, json.dumps(r["verdict"]))
