"""
Onset census for LANL 2015 (flows + red-team authentications).

For every computer the red team authenticated to (a "victim"):
  * onset = its first red-team authentication (redteam.txt.gz);
  * lockstep: onsets chained within 60 s (one scripted sweep), and bursts within 30 min;
  * benign history in flows.txt.gz: does the victim appear in router flows at all, how many
    of its own flows (as source) and inbound flows exist before onset, in the 10 and 60
    minutes before onset, and how long the flow record covers it before onset;
  * attacker contact: flows from the red-team source computer to the victim before onset,
    which is the only plausible precursor (victims are picked by the attacker).

    python scripts/lanl_onset_census.py [--raw data/raw/lanl]

Streams the 1.1 GB flows file once (about 129 M rows). Writes reports/lanl_onset_census.json.
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

from aegisflow.ml.datasets.lanl_2015 import iter_flows, load_redteam, victim_onsets  # noqa: E402

DAY = 86400


def chain(times: np.ndarray, gap: int) -> np.ndarray:
    ids, last, cur = [], None, -1
    for x in times:
        if last is None or x - last > gap:
            cur += 1
        ids.append(cur)
        last = x
    return np.array(ids)


def census(raw: Path, chunksize: int = 4_000_000) -> tuple[dict, pd.DataFrame]:
    rt = load_redteam(raw / "redteam.txt.gz")
    v = victim_onsets(rt)
    v["onset_event_60s"] = chain(v["onset_s"].to_numpy(), 60)
    v["onset_burst_30m"] = chain(v["onset_s"].to_numpy(), 1800)
    v["batch_size_60s"] = v.groupby("onset_event_60s")["computer"].transform("size")
    onset = v.set_index("computer")["onset_s"]
    attacker = v.set_index("computer")["attacker"]
    victims = set(v["computer"])
    sources = set(rt["src"])

    stats = {c: dict(first_seen=None, own_before=0, own_10m=0, own_60m=0, in_before=0, in_10m=0, in_60m=0,
                     attacker_before=0, attacker_60m=0, attacker_24h=0, any_after=0) for c in victims}
    t_min = t_max = None
    rows = 0
    for chunk in iter_flows(raw / "flows.txt.gz", victims | sources, chunksize=chunksize):
        t = pd.to_numeric(chunk["time"]).to_numpy()
        t_min = t.min() if t_min is None else min(t_min, t.min())
        t_max = t.max() if t_max is None else max(t_max, t.max())
        rows += len(chunk)
        for role in ("src", "dst"):
            m = chunk[role].isin(victims).to_numpy()
            if not m.any():
                continue
            sub = chunk[m]
            comp = sub[role].to_numpy()
            tt = t[m]
            on = onset.reindex(comp).to_numpy()
            other = sub["dst" if role == "src" else "src"].to_numpy()
            from_attacker = (role == "dst") & (other == attacker.reindex(comp).to_numpy())
            df = pd.DataFrame({"c": comp, "dt": on - tt, "att": from_attacker})
            for c, g in df.groupby("c"):
                s = stats[c]
                first = (on[comp == c][0] - g["dt"].max())
                s["first_seen"] = first if s["first_seen"] is None else min(s["first_seen"], first)
                before = g["dt"] > 0
                s["any_after"] += int((~before).sum())
                key = "own" if role == "src" else "in"
                s[f"{key}_before"] += int(before.sum())
                s[f"{key}_10m"] += int((before & (g["dt"] <= 600)).sum())
                s[f"{key}_60m"] += int((before & (g["dt"] <= 3600)).sum())
                if role == "dst":
                    a = before & g["att"]
                    s["attacker_before"] += int(a.sum())
                    s["attacker_60m"] += int((a & (g["dt"] <= 3600)).sum())
                    s["attacker_24h"] += int((a & (g["dt"] <= DAY)).sum())
        print(f"rows kept {rows:,} up to t={t_max:,}", flush=True)

    st = pd.DataFrame.from_dict(stats, orient="index")
    v = v.join(st, on="computer")
    v["days_of_flows_before_onset"] = ((v["onset_s"] - v["first_seen"]) / DAY).round(2)
    v["onset_day"] = (v["onset_s"] // DAY + 1).astype(int)

    seen = v[v["first_seen"].notna()]
    own10 = seen[seen["own_10m"] > 0]
    summary = {
        "red_team_events": int(len(rt)),
        "victims": int(len(v)),
        "red_team_sources": sorted(sources),
        "onset_span_days": [round(v["onset_s"].min() / DAY, 2), round(v["onset_s"].max() / DAY, 2)],
        "onsets_per_day": {int(k): int(n) for k, n in v["onset_day"].value_counts().sort_index().items()},
        "independent_onset_events_60s": int(v["onset_event_60s"].nunique()),
        "onset_bursts_30m": int(v["onset_burst_30m"].nunique()),
        "largest_60s_batch": int(v["batch_size_60s"].max()),
        "flow_time_range_s": [None if t_min is None else int(t_min), None if t_max is None else int(t_max)],
        "victims_seen_in_flows": int(len(seen)),
        "victims_with_onset_inside_flow_range": int(((v["onset_s"] >= (t_min or 0)) & (v["onset_s"] <= (t_max or 0))).sum()),
        "victims_with_own_flows_before_onset": int((seen["own_before"] > 0).sum()),
        "victims_with_own_flows_10m_before": int(len(own10)),
        "victims_with_own_flows_60m_before": int((seen["own_60m"] > 0).sum()),
        "independent_events_with_own_flows_10m_before": int(own10["onset_event_60s"].nunique()),
        "bursts_with_own_flows_10m_before": int(own10["onset_burst_30m"].nunique()),
        "victims_contacted_by_attacker_in_flows_before_onset": int((seen["attacker_before"] > 0).sum()),
        "victims_contacted_by_attacker_60m_before": int((seen["attacker_60m"] > 0).sum()),
        "median_days_of_flows_before_onset": None if seen.empty else float(seen["days_of_flows_before_onset"].median()),
    }
    return summary, v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "lanl")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "lanl_onset_census.json")
    a = ap.parse_args(argv)
    summary, v = census(a.raw)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "victims": json.loads(v.to_json(orient="records"))}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
