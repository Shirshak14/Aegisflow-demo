"""Synthetic host-window generator for the onset study (docs/synthetic_onset_design.md).

Produces the same 28 host-window features the demo model uses, for many hosts with many independent
scripted attack runs whose onset times and precursor strength are controlled. Nothing is sent over a
network; the data is entirely synthetic.

    python scripts/synth_lab.py --condition B2 --seed 11 --out data/synthetic/B2_s11.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.modeling import MODEL_FEATURES  # noqa: E402

F = list(MODEL_FEATURES)
NF = len(F)
N_HOSTS = 60
HOURS = 96
STRIDE_S = 30
N_WIN = HOURS * 3600 // STRIDE_S
N_RUNS = 400
MIN_GAP_H = 2.0
PRE_MIN = 10          # precursor length L
NET_PRE_MIN = 5
ATTACKS = ["scan", "brute", "flood", "exfil", "lateral"]

# condition -> (precursor strength s, clock-concentrated onsets, network-context precursor)
CONDITIONS = {
    "A": (0.0, False, False),
    "B1": (0.25, False, False),
    "B2": (0.5, False, False),
    "B3": (1.0, False, False),
    # exploratory, added after the pre-registered conditions had been run (docs/synthetic_onset_report.md)
    "B4": (2.0, False, False),
    "B5": (4.0, False, False),
    "C": (0.0, True, False),
    "D": (0.0, False, True),
}


def _signatures(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per attack type, a shift in benign-std units on a subset of features (fixed across conditions)."""
    def vec(**kw):
        v = np.zeros(NF)
        for k, x in kw.items():
            v[F.index(k)] = x
        return v
    return {
        "scan": vec(unique_destination_ports=5, destination_port_entropy=4, unique_destination_ips=3,
                    packet_size_mean=-3, syn_count_sum=3, syn_ratio=3, flow_count=3),
        "brute": vec(flow_count=4, rst_count_sum=3, rst_ratio=3, fin_count_sum=2, duration_mean=-3,
                     connection_burst_max=4, packet_count_mean=-2),
        "flood": vec(flow_count=6, syn_count_sum=6, syn_ratio=4, packet_count_sum=5, packets_per_second_mean=5,
                     connection_burst_max=5, unique_source_ports=4),
        "exfil": vec(byte_count_sum=6, byte_count_mean=5, bytes_per_second_mean=4, packet_size_mean=3,
                     duration_mean=3, psh_count_sum=3, unique_destination_ips=-1),
        "lateral": vec(unique_destination_ips=4, unique_destination_ports=3, ack_count_sum=2, flow_count=2,
                       destination_port_entropy=3, tcp_window_size_mean=-2),
    }


def _schedule(rng, n_runs, clock_conc):
    runs = []
    per_host: dict[int, list] = {h: [] for h in range(N_HOSTS)}
    tries = 0
    while len(runs) < n_runs and tries < n_runs * 200:
        tries += 1
        h = int(rng.integers(N_HOSTS))
        dur_min = float(rng.uniform(5, 90))
        if clock_conc:
            day = int(rng.integers(HOURS // 24))
            start_h = day * 24 + float(rng.uniform(7, 9))
        else:
            start_h = float(rng.uniform(0.7, HOURS - dur_min / 60 - 0.2))
        end_h = start_h + dur_min / 60
        if any(not (end_h + MIN_GAP_H < s or e + MIN_GAP_H < start_h) for s, e, _ in per_host[h]):
            continue
        per_host[h].append((start_h, end_h, len(runs)))
        runs.append(dict(run_id=len(runs), host=h, type=ATTACKS[int(rng.integers(len(ATTACKS)))],
                         start_h=start_h, end_h=end_h))
    return runs


def generate(condition: str, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    s_pre, clock_conc, net_pre = CONDITIONS[condition]
    rng = np.random.default_rng(seed)
    sigs = _signatures(np.random.default_rng(1))          # same attack shapes in every run/condition
    t = np.arange(N_WIN) * STRIDE_S / 3600.0               # hours
    hod = (t % 24)
    runs = _schedule(rng, N_RUNS, clock_conc)

    # benign latent per host: baseline + diurnal + AR(1) noise + decoy bursts, all in std units of z
    mu = rng.normal(3.0, 0.8, (N_HOSTS, NF))
    sd = rng.uniform(0.25, 0.6, (N_HOSTS, NF))
    load = rng.normal(0, 1, NF) * 0.8
    amp = rng.uniform(0.3, 1.2, N_HOSTS)
    phase = rng.uniform(0, 24, N_HOSTS)
    Z = np.empty((N_HOSTS, N_WIN, NF), np.float32)
    phi = 0.9
    eps = rng.normal(0, 1, (N_HOSTS, N_WIN, NF)).astype(np.float32) * np.sqrt(1 - phi ** 2)
    ar = np.zeros((N_HOSTS, NF), np.float32)
    for i in range(N_WIN):
        ar = phi * ar + eps[:, i]
        Z[:, i] = ar
    diurnal = np.cos(2 * np.pi * (hod[None, :] - phase[:, None]) / 24)            # host x win
    Z = Z * sd[:, None, :] + mu[:, None, :] + (amp[:, None, None] * diurnal[:, :, None] * load[None, None, :]
                                               * sd[:, None, :]).astype(np.float32)

    sig_unit = {k: v / np.abs(v).max() for k, v in sigs.items()}
    # decoy benign bursts that look like attack signatures but are not attacks
    for h in range(N_HOSTS):
        for _ in range(rng.poisson(HOURS / 3.0)):
            k = ATTACKS[int(rng.integers(len(ATTACKS)))]
            a = int(rng.integers(N_WIN)); n = int(rng.uniform(2, 20) * 2)
            Z[h, a:a + n] += (rng.uniform(1.5, 3.5) * sd[h] * sigs[k] / np.abs(sigs[k]).max())[None, :].astype(np.float32)

    attack = np.zeros((N_HOSTS, N_WIN), np.int8)
    run_of = np.full((N_HOSTS, N_WIN), -1, np.int32)
    for r in runs:
        h = r["host"]; a = int(r["start_h"] * 3600 / STRIDE_S); b = int(r["end_h"] * 3600 / STRIDE_S)
        sh = sigs[r["type"]]
        ramp = np.clip(np.arange(b - a) / 4.0, 0, 1)[:, None] * rng_amp(rng)
        Z[h, a:b] += (ramp * sd[h] * sh[None, :]).astype(np.float32)
        attack[h, a:b] = 1; run_of[h, a:b] = r["run_id"]
        r["onset_win"] = a
        if s_pre > 0:
            n = int(PRE_MIN * 60 / STRIDE_S)
            ramp = (np.arange(1, n + 1) / n)[:, None]
            Z[h, a - n:a] += (s_pre * ramp * sd[h] * sig_unit[r["type"]][None, :]).astype(np.float32)
        if net_pre:
            n = int(NET_PRE_MIN * 60 / STRIDE_S)
            others = rng.choice([x for x in range(N_HOSTS) if x != h], size=N_HOSTS // 3, replace=False)
            bump = np.zeros(NF); bump[F.index("flow_count")] = 1.5; bump[F.index("unique_destination_ips")] = 1.5
            Z[others, a - n:a] += (bump * sd[others][:, None, :]).astype(np.float32)

    X = np.exp(np.clip(Z, -5, 18)).astype(np.float32)
    hosts = np.repeat([f"10.0.0.{h + 1}" for h in range(N_HOSTS)], N_WIN)
    base = pd.Timestamp("2026-01-05 00:00:00")
    ws = base + pd.to_timedelta(np.tile(np.arange(N_WIN) * STRIDE_S, N_HOSTS), unit="s")
    df = pd.DataFrame(X.reshape(-1, NF), columns=F)
    df.insert(0, "host_id", hosts)
    df.insert(1, "window_start", ws)
    df.insert(2, "window_end", ws + pd.Timedelta(seconds=60))
    df["attack_present"] = attack.reshape(-1)
    df["run_id"] = run_of.reshape(-1)
    return df, runs


def rng_amp(rng):
    return float(rng.uniform(0.8, 1.2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df, runs = generate(a.condition, a.seed)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    Path(str(out) + ".runs.json").write_text(json.dumps(runs))
    print(f"{a.condition} seed {a.seed}: {len(df):,} windows, {len(runs)} runs, "
          f"{int(df.attack_present.sum()):,} attack windows -> {out}")


if __name__ == "__main__":
    main()
