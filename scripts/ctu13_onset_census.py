"""CTU-13 onset census for docs/combined_datasets_feasibility.md.

Counts, per scenario and per infected host, the attack onsets that a forecasting study could use:
when the host's botnet traffic starts, whether the host had any non-botnet traffic of its own
before that (benign history in source_ip windows, which is how AegisFlow groups hosts), how much
inbound-only traffic it had, how many episodes it has under the 30-minute gap rule, and when each
botnet activity (C2, spam, click fraud, ICMP, scan) first appears.

Reads data/raw/ctu_13/*.binetflow[.gz] with the CTU-13 adapter's helpers, one scenario at a time,
only the columns it needs. Writes reports/ctu13_onset_census.json. No model is trained.
Run: .venv/Scripts/python.exe scripts/ctu13_onset_census.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegisflow.ml.datasets.ctu_13 import label_family, scenario_of  # noqa: E402

RAW = ROOT / "data/raw/ctu_13"
REPORT = ROOT / "reports/ctu13_onset_census.json"
EPISODE_GAP = pd.Timedelta("30min")
HISTORY = pd.Timedelta("10min")      # "has history" = traffic of its own in the 10 min before onset
INTERNAL = "147.32."                 # CTU university network; infected VMs live in 147.32.84.0/24

# Botnet activity from the fine CTU label (first match wins). These words are in the labels.
ACTIVITIES = [("c2", r"-CC\d"), ("spam", r"SPAM|SMTP"), ("click_fraud", r"HTTP-Ad"),
              ("icmp", r"ICMP"), ("scan", r"Scan"), ("dns", r"DNS")]


def activity(label: pd.Series) -> pd.Series:
    out = pd.Series("other", index=label.index, dtype="object")
    for name, pattern in reversed(ACTIVITIES):
        out[label.str.contains(pattern, regex=True, na=False)] = name
    return out


def episodes(times: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    t = times.sort_values().reset_index(drop=True)
    if t.empty:
        return []
    cuts = [0, *(t.index[t.diff() > EPISODE_GAP].tolist()), len(t)]
    return [(t[a], t[b - 1]) for a, b in zip(cuts[:-1], cuts[1:])]


def census(path: Path) -> dict:
    df = pd.read_csv(path, usecols=["StartTime", "Proto", "SrcAddr", "DstAddr", "Dport", "Label"],
                     dtype=str, keep_default_na=False)
    df["t"] = pd.to_datetime(df["StartTime"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f")
    df["fam"] = label_family(df["Label"]).astype(str)
    start, end = df["t"].min(), df["t"].max()

    bot = df[df["fam"] == "Botnet"]
    infected = sorted(ip for ip in bot["SrcAddr"].unique() if ip.startswith(INTERNAL))
    external_bot_sources = int(bot.loc[~bot["SrcAddr"].str.startswith(INTERNAL), "SrcAddr"].nunique())
    normal_hosts = sorted(df.loc[(df["fam"] == "Normal") & df["SrcAddr"].str.startswith(INTERNAL), "SrcAddr"].unique())

    hosts = []
    for ip in infected:
        src = df[df["SrcAddr"] == ip]
        dst = df[df["DstAddr"] == ip]
        hb = src[src["fam"] == "Botnet"]
        onset = hb["t"].min()
        own_before = src[(src["fam"] != "Botnet") & (src["t"] < onset)]
        inbound_before = dst[dst["t"] < onset]
        eps = episodes(hb["t"])
        ep_rows = []
        for i, (a, b) in enumerate(eps):
            prev_end = eps[i - 1][1] if i else None
            own_hist = src[(src["t"] >= a - HISTORY) & (src["t"] < a)]
            ep_rows.append({
                "start": str(a), "end": str(b), "minutes": round((b - a).total_seconds() / 60, 1),
                "quiet_minutes_before": None if prev_end is None else round((a - prev_end).total_seconds() / 60, 1),
                "own_flows_in_10min_before": int(len(own_hist)),
                "non_botnet_own_flows_in_10min_before": int((own_hist["fam"] != "Botnet").sum()),
            })
        acts = activity(hb["Label"])
        first = {name: str(hb.loc[acts == name, "t"].min()) for name, _ in ACTIVITIES if (acts == name).any()}
        # escalation: first flow of a monetisation/attack activity after >= 10 min of other botnet traffic
        escalations = {}
        for name in ("spam", "click_fraud", "icmp", "scan"):
            if name in first:
                t1 = pd.Timestamp(first[name])
                lead = hb[(hb["t"] < t1) & (acts != name)]
                if not lead.empty and t1 - lead["t"].min() >= HISTORY:
                    escalations[name] = {"at": str(t1), "minutes_of_prior_botnet_traffic":
                                         round((t1 - lead["t"].min()).total_seconds() / 60, 1)}
        hosts.append({
            "ip": ip,
            "botnet_src_flows": int(len(hb)),
            "non_botnet_src_flows_total": int((src["fam"] != "Botnet").sum()),
            "onset": str(onset),
            "onset_clock": onset.strftime("%H:%M"),
            "onset_minutes_after_capture_start": round((onset - start).total_seconds() / 60, 1),
            "non_botnet_own_flows_before_onset": int(len(own_before)),
            "inbound_flows_before_onset": int(len(inbound_before)),
            "inbound_history_minutes": round((onset - inbound_before["t"].min()).total_seconds() / 60, 1)
            if not inbound_before.empty else 0.0,
            "episodes": ep_rows,
            "activity_first_seen": first,
            "activity_flow_counts": {k: int(v) for k, v in acts.value_counts().items()},
            "escalation_onsets": escalations,
        })
    return {
        "file": path.name,
        "capture_start": str(start), "capture_end": str(end),
        "capture_hours": round((end - start).total_seconds() / 3600, 2),
        "flows": int(len(df)),
        "label_families": {k: int(v) for k, v in df["fam"].value_counts().items()},
        "infected_hosts": hosts,
        "external_ips_with_botnet_label": external_bot_sources,
        "normal_hosts": normal_hosts,
    }


def main() -> None:
    files = sorted({p for pat in ("*.binetflow", "*.binetflow.gz") for p in RAW.glob(f"**/{pat}")},
                   key=lambda p: scenario_of(p) or 99)
    out = {}
    for f in files:
        s = scenario_of(f)
        if s is None:
            continue
        print(f"scenario {s}: {f.name}", flush=True)
        out[str(s)] = census(f)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nscen hosts episodes own-history inbound-history escalations onset-min-after-start")
    for s, r in out.items():
        hs = r["infected_hosts"]
        n_ep = sum(len(h["episodes"]) for h in hs)
        own = sum(1 for h in hs for e in h["episodes"] if e["non_botnet_own_flows_in_10min_before"] > 0)
        inb = sum(1 for h in hs if h["inbound_flows_before_onset"] > 0)
        esc = sum(len(h["escalation_onsets"]) for h in hs)
        mins = [h["onset_minutes_after_capture_start"] for h in hs]
        print(f"{s:>4} {len(hs):>5} {n_ep:>8} {own:>11} {inb:>15} {esc:>11} {mins}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
