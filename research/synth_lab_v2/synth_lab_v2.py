"""Synthetic multi-stage attack lab, v2 (research/synth_lab_v2/DESIGN.md).

Everything generated here is synthetic. Each organisation ("scenario") is a set of hosts observed in 10-minute
windows over 24 days, with 19 channels per host-window (6 IDS alert channels, 13 telemetry channels), benign
daily rhythm, benign decoys, completed attack chains and aborted chains.

    generate_org(org_id, condition, seed) -> dict with
        X       float32 (n_hosts, T, n_channels)  raw channel values
        roles   int8 (n_hosts,)
        onsets  list of (host, window, chain_type)          completed attacks (impact start)
        aborted list of (host, first_window, last_window, chain_type)  chains stopped before impact
"""
from __future__ import annotations

import numpy as np

WIN_MIN = 10
PER_DAY = 24 * 60 // WIN_MIN          # 144
DAYS = 24
T = DAYS * PER_DAY
BURN_IN = 3 * PER_DAY

CH = [
    # IDS alert channels
    "ids_scan_in", "ids_scan_out", "ids_brute_in", "ids_exploit_in", "ids_misc", "ids_impact",
    # telemetry channels (never alert)
    "auth_ok_in", "auth_fail_in", "auth_newsrc_in", "auth_priv_in", "admin_tool_exec", "conns_in",
    "conns_out", "new_dst_out", "mb_out_int", "mb_out_ext", "file_read_burst", "file_mod_burst", "cpu_high",
]
C = {c: i for i, c in enumerate(CH)}
ALERT_CH = [C[c] for c in CH if c.startswith("ids_")]
SCAN_CH = [C["ids_scan_in"], C["ids_scan_out"]]

ROLES = ["ws", "app", "web", "fs", "dc", "dev"]
R = {r: i for i, r in enumerate(ROLES)}

# Benign base rate per window at full activity, per role (order of ROLES). Count channels are Poisson means;
# mb_* and file_* are log-normal means; cpu_high is a fraction.
BASE = {
    "ids_scan_in":     [0.004, 0.006, 0.40, 0.006, 0.006, 0.004],
    "ids_scan_out":    [0.0005, 0.0005, 0.0005, 0.0005, 0.001, 0.002],
    "ids_brute_in":    [0.001, 0.003, 0.05, 0.002, 0.004, 0.003],
    "ids_exploit_in":  [0.001, 0.002, 0.03, 0.001, 0.001, 0.002],
    "ids_misc":        [0.04, 0.03, 0.10, 0.03, 0.05, 0.06],
    "ids_impact":      [0.0002, 0.0002, 0.0005, 0.0002, 0.0002, 0.0002],
    "auth_ok_in":      [1.5, 4.0, 3.0, 8.0, 40.0, 2.5],
    "auth_fail_in":    [0.08, 0.25, 0.6, 0.4, 2.5, 0.15],
    "auth_newsrc_in":  [0.01, 0.03, 0.05, 0.04, 0.10, 0.03],
    "auth_priv_in":    [0.01, 0.12, 0.08, 0.10, 0.6, 0.05],
    "admin_tool_exec": [0.02, 0.10, 0.05, 0.08, 0.30, 0.15],
    "conns_in":        [5.0, 60.0, 120.0, 80.0, 200.0, 20.0],
    "conns_out":       [40.0, 30.0, 25.0, 15.0, 30.0, 60.0],
    "new_dst_out":     [0.3, 0.1, 0.1, 0.05, 0.2, 0.6],
    "mb_out_int":      [5.0, 20.0, 10.0, 60.0, 15.0, 15.0],
    "mb_out_ext":      [3.0, 2.0, 15.0, 1.0, 0.5, 8.0],
    "file_read_burst": [0.5, 0.8, 0.3, 3.0, 0.5, 1.5],
    "file_mod_burst":  [0.3, 0.4, 0.1, 1.5, 0.2, 1.0],
    "cpu_high":        [0.10, 0.20, 0.15, 0.10, 0.15, 0.30],
}
CONT = {"mb_out_int", "mb_out_ext", "file_read_burst", "file_mod_burst"}

# Six organisations: (n_hosts, role mix, work-hour offset, IDS detection prob, attacks, aborted per attack,
# chain mix over R X W I M Z, impact-at-night skew, pentest week?)
ORGS = [
    dict(n=220, mix=[.70, .08, .04, .08, .03, .07], shift=0, pdet=0.8, attacks=55, abort=3.0,
         chains=[.30, .20, .10, .10, .15, .15], night=0.6, pentest=False),
    dict(n=160, mix=[.60, .12, .08, .10, .03, .07], shift=2, pdet=0.5, attacks=45, abort=2.0,
         chains=[.15, .25, .20, .15, .10, .15], night=0.0, pentest=True),
    dict(n=260, mix=[.72, .07, .03, .08, .02, .08], shift=-1, pdet=0.9, attacks=65, abort=4.0,
         chains=[.25, .15, .10, .15, .15, .20], night=0.4, pentest=False),
    dict(n=120, mix=[.55, .12, .12, .08, .03, .10], shift=1, pdet=0.4, attacks=40, abort=2.5,
         chains=[.10, .15, .30, .10, .20, .15], night=0.2, pentest=False),
    dict(n=200, mix=[.65, .10, .05, .10, .03, .07], shift=3, pdet=0.7, attacks=50, abort=3.5,
         chains=[.25, .20, .15, .15, .10, .15], night=0.7, pentest=True),
    dict(n=180, mix=[.60, .10, .06, .08, .03, .13], shift=-2, pdet=0.6, attacks=50, abort=3.0,
         chains=[.20, .20, .15, .15, .15, .15], night=0.0, pentest=False),
]
CHAINS = ["R", "X", "W", "I", "M", "Z"]
TARGET_ROLES = {
    "R": {"fs": 4, "app": 2, "dc": 1, "ws": 1},
    "X": {"fs": 4, "dev": 2, "app": 1},
    "W": {"web": 1},
    "I": {"fs": 3, "ws": 2, "dev": 1},
    "M": {"app": 2, "web": 2, "dev": 2},
    "Z": {"ws": 2, "app": 1, "fs": 1, "web": 1, "dev": 1},
}

# Stage spec: (name, duration windows (lo, hi), {channel: rate per window}, gap to next stage: (median min, sigma))
# Counts are Poisson-added; CONT channels are added as rate x lognormal; cpu_high is set to max(cur, rate).
STAGES = {
    "R": [
        ("recon", (2, 6), {"ids_scan_in": 1.0, "conns_in": 40}, (360, 1.0)),
        ("cred", (1, 3), {"auth_fail_in": 4, "ids_brute_in": 0.6}, (120, 1.0)),
        ("lateral", (1, 2), {"auth_priv_in": 2, "auth_newsrc_in": 1, "admin_tool_exec": 3}, (60, 1.0)),
        ("staging", (2, 6), {"file_read_burst": 6, "new_dst_out": 2, "mb_out_int": 40}, (30, 0.8)),
        ("impact", (3, 12), {"file_mod_burst": 40, "ids_impact": 2}, None),
    ],
    "X": [
        ("login", (1, 2), {"auth_newsrc_in": 1, "auth_ok_in": 1}, (180, 1.0)),
        ("recon", (1, 3), {"ids_scan_in": 0.4, "conns_in": 15}, (120, 1.0)),
        ("staging", (2, 8), {"file_read_burst": 5, "mb_out_int": 30}, (60, 1.0)),
        ("impact", (2, 8), {"mb_out_ext": 500, "conns_out": 20}, None),
    ],
    "W": [
        ("scan", (3, 12), {"ids_scan_in": 2.5, "conns_in": 150}, (60, 1.2)),
        ("exploit", (1, 4), {"ids_exploit_in": 1.5, "ids_brute_in": 1.0, "auth_fail_in": 3}, (20, 1.0)),
        ("impact", (2, 10), {"mb_out_ext": 150, "ids_misc": 2, "conns_out": 30}, None),
    ],
    "I": [
        ("oddlogin", (1, 3), {"auth_ok_in": 2}, (1440, 0.7)),
        ("staging", (2, 6), {"file_read_burst": 2, "mb_out_int": 10}, (240, 1.0)),
        ("impact", (2, 8), {"mb_out_ext": 250}, None),
    ],
    "M": [
        ("brute", (3, 18), {"ids_brute_in": 2.0, "auth_fail_in": 8}, (120, 1.2)),
        ("login", (1, 2), {"auth_newsrc_in": 1, "auth_ok_in": 1}, (30, 1.0)),
        ("impact", (6, 24), {"cpu_high": 0.95, "conns_out": 60, "new_dst_out": 3}, None),
    ],
    "Z": [
        ("impact", (3, 12), {"file_mod_burst": 30, "conns_out": 80, "ids_impact": 1}, None),
    ],
}
NIGHT_STAGES = {("X", "login"), ("I", "oddlogin")}

CONDITIONS = {
    "NULL":   dict(precursors=False, abort_mult=1.0, p_skip=0.3, stealth_mult=1.0, alert_precursors=True),
    "MAIN":   dict(precursors=True, abort_mult=1.0, p_skip=0.3, stealth_mult=1.0, alert_precursors=True),
    "SUBTLE": dict(precursors=True, abort_mult=1.0, p_skip=0.3, stealth_mult=1.0, alert_precursors=False),
    "EASY":   dict(precursors=True, abort_mult=0.0, p_skip=0.0, stealth_mult=2.0, alert_precursors=True),
}


def local_hour(t: np.ndarray, shift: int) -> np.ndarray:
    return ((t % PER_DAY) * WIN_MIN / 60.0 + shift) % 24


def weekday(t: np.ndarray, shift: int) -> np.ndarray:
    # day index from local clock; day 0 is a Monday
    return ((t * WIN_MIN / 60.0 + shift) // 24).astype(int) % 7


def activity(t: np.ndarray, shift: int):
    h = local_hour(t, shift)
    wd = weekday(t, shift) < 5
    day = np.exp(-0.5 * ((h - 13.0) / 3.2) ** 2)                 # workday bump
    ws = np.where(wd, 0.04 + 0.96 * day, 0.04 + 0.12 * day)
    srv = 0.45 + 0.55 * np.where(wd, day, 0.3 * day)
    return ws.astype(np.float32), srv.astype(np.float32), h, wd


def is_offhours(t: int, shift: int) -> bool:
    h = local_hour(np.array([t]), shift)[0]
    wd = weekday(np.array([t]), shift)[0] < 5
    return (not wd) or h < 7 or h >= 20


def generate_org(org_id: int, condition: str = "MAIN", seed: int = 1) -> dict:
    cfg, cond = ORGS[org_id], CONDITIONS[condition]
    rng = np.random.default_rng([seed, org_id])
    n, shift = cfg["n"], cfg["shift"]
    roles = rng.choice(len(ROLES), size=n, p=cfg["mix"]).astype(np.int8)
    roles[0] = R["dc"]                                            # every org has at least one DC and web server
    roles[1] = R["web"]
    t = np.arange(T)
    act_ws, act_srv, hour, wd = activity(t, shift)
    X = np.zeros((n, T, len(CH)), np.float32)
    hostmult = rng.lognormal(0.0, 0.4, size=(n, len(CH))).astype(np.float32)

    # ---------------- benign background ----------------
    for c, rates in BASE.items():
        ci = C[c]
        lam = np.asarray(rates, np.float32)[roles][:, None] * hostmult[:, ci:ci + 1]
        act = np.where((roles == R["ws"])[:, None] | (roles == R["dev"])[:, None], act_ws[None], act_srv[None])
        if c.startswith("ids_") and c != "ids_misc":
            act = np.ones_like(act)                               # internet noise has no office hours
        if c == "cpu_high":
            X[:, :, ci] = np.clip(lam * act * rng.lognormal(0, 0.3, (n, T)), 0, 1)
        elif c in CONT:
            X[:, :, ci] = lam * act * rng.lognormal(-0.18, 0.6, (n, T))
        else:
            X[:, :, ci] = rng.poisson(lam * act)

    def add(h, w0, w1, rates, scale=1.0, alerts=True, pdet=1.0):
        w0, w1 = max(w0, 0), min(w1, T)
        if w1 <= w0:
            return
        k = w1 - w0
        for c, r in rates.items():
            ci = C[c]
            if c == "cpu_high":
                X[h, w0:w1, ci] = np.maximum(X[h, w0:w1, ci], min(r * max(scale, 0.6), 1.0))
            elif c in CONT:
                X[h, w0:w1, ci] += r * scale * rng.lognormal(-0.12, 0.5, k)
            else:
                cnt = rng.poisson(r * scale, k)
                if c.startswith("ids_"):
                    if not alerts:
                        continue
                    cnt = rng.binomial(cnt, pdet)
                X[h, w0:w1, ci] += cnt

    # weekly vulnerability scanner: every host, one 2-hour sweep a week
    scan_dow, scan_hour = rng.integers(0, 7), rng.integers(0, 24)
    for d in range(DAYS):
        if weekday(np.array([d * PER_DAY]), 0)[0] == scan_dow:
            start = d * PER_DAY + scan_hour * 6
            for h in range(n):
                w = start + rng.integers(0, 12)
                add(h, w, w + 1, {"ids_scan_in": 1.5, "conns_in": 60})
    # nightly backups on file/app servers (staging-like)
    for h in np.where(np.isin(roles, [R["fs"], R["app"], R["dc"]]))[0]:
        bh = rng.integers(0, 5)
        for d in range(DAYS):
            w = d * PER_DAY + ((bh - shift) % 24) * 6
            add(h, w, w + rng.integers(2, 6), {"mb_out_int": 300, "file_read_burst": 8})
    # monthly patch night: admin tools + privileged logins everywhere
    pd_ = rng.integers(5, DAYS - 2)
    for h in range(n):
        w = pd_ * PER_DAY + ((22 - shift) % 24) * 6 + rng.integers(0, 18)
        add(h, w, w + 2, {"admin_tool_exec": 4, "auth_priv_in": 2, "auth_newsrc_in": 0.3})
    # travelling users / new devices, lockout bursts, admins working late: scattered per host
    for h in range(n):
        for _ in range(rng.poisson(1.5)):
            w = rng.integers(0, T)
            add(h, w, w + 1, {"auth_newsrc_in": 1, "auth_ok_in": 1})
        for _ in range(rng.poisson(1.0)):
            w = rng.integers(0, T)
            add(h, w, w + rng.integers(1, 3), {"auth_fail_in": 6})
        if roles[h] in (R["app"], R["fs"], R["dc"], R["dev"]):
            for _ in range(rng.poisson(1.0)):
                w = rng.integers(0, T)
                add(h, w, w + rng.integers(1, 4), {"admin_tool_exec": 3, "auth_priv_in": 1})
    # pentest week: scans and brute force on 30% of hosts, never followed by impact
    if cfg["pentest"]:
        d0 = rng.integers(6, DAYS - 7)
        for h in rng.choice(n, size=int(0.3 * n), replace=False):
            w = d0 * PER_DAY + rng.integers(0, 5 * PER_DAY)
            add(h, w, w + rng.integers(2, 8), {"ids_scan_in": 2, "conns_in": 80}, pdet=cfg["pdet"])
            if rng.random() < 0.5:
                w2 = w + rng.integers(6, 60)
                add(h, w2, w2 + rng.integers(1, 4), {"ids_brute_in": 1.5, "auth_fail_in": 6}, pdet=cfg["pdet"])

    # ---------------- attack chains ----------------
    role_hosts = {r: np.where(roles == R[r])[0] for r in ROLES}
    busy = np.zeros((n, T), bool)                                 # impact windows +- margin, to avoid overlap

    def pick_target(ct):
        prefs = {r: wgt for r, wgt in TARGET_ROLES[ct].items() if len(role_hosts[r])}
        rs = list(prefs)
        p = np.array([prefs[r] for r in rs], float)
        r = rs[rng.choice(len(rs), p=p / p.sum())]
        return int(rng.choice(role_hosts[r]))

    def pick_impact_window():
        while True:
            w = int(rng.integers(BURN_IN + 2 * PER_DAY, T - PER_DAY // 2))
            if rng.random() < cfg["night"] and not is_offhours(w, shift):
                continue
            return w

    onsets, aborted = [], []
    n_abort = int(round(cfg["attacks"] * cfg["abort"] * cond["abort_mult"]))
    jobs = [("done", i) for i in range(cfg["attacks"])] + [("abort", i) for i in range(n_abort)]
    for kind, _ in jobs:
        ct = CHAINS[rng.choice(len(CHAINS), p=cfg["chains"])]
        if kind == "abort" and ct == "Z":
            continue                                              # an aborted no-precursor chain leaves nothing
        for _try in range(50):
            h = pick_target(ct)
            w_imp = pick_impact_window()
            if not busy[h, max(w_imp - 2 * PER_DAY, 0):w_imp + 2 * PER_DAY].any():
                break
        else:
            continue
        stealth = float(np.clip(rng.lognormal(0, 0.5), 0.2, 2.5)) * cond["stealth_mult"]
        stages = STAGES[ct]
        # schedule stages backwards from impact
        sched, end = [], w_imp
        dur_imp = int(rng.integers(stages[-1][1][0], stages[-1][1][1] + 1))
        for name, (lo, hi), rates, gap in reversed(stages[:-1]):
            g = int(round(rng.lognormal(np.log(gap[0]), gap[1]) / WIN_MIN))
            dur = int(rng.integers(lo, hi + 1))
            s = end - g - dur
            if (ct, name) in NIGHT_STAGES:                        # odd-hours activity: push into the night
                for back in range(0, 3 * PER_DAY):
                    if is_offhours(s - back, shift):
                        s -= back
                        break
            sched.append((name, s, s + dur, rates))
            end = s
        sched.reverse()
        if kind == "abort":
            # attacker stops after a random number of precursor stages (at least one)
            keep = int(rng.integers(1, len(sched) + 1)) if sched else 0
            sched = sched[:keep]
        if cond["precursors"]:
            for name, s, e, rates in sched:
                if rng.random() < cond["p_skip"]:
                    continue
                add(h, s, e, rates, scale=stealth, alerts=cond["alert_precursors"], pdet=cfg["pdet"])
        if kind == "done":
            add(h, w_imp, w_imp + dur_imp, stages[-1][2], scale=max(stealth, 1.0), pdet=0.9)
            busy[h, max(w_imp - PER_DAY, 0):w_imp + PER_DAY] = True
            onsets.append((h, w_imp, ct))
        elif sched:
            aborted.append((h, sched[0][1], sched[-1][2], ct))
    onsets.sort(key=lambda o: (o[0], o[1]))
    return dict(X=X, roles=roles, onsets=onsets, aborted=aborted, shift=shift, org=org_id)
