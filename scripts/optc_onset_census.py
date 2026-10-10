"""
Onset census for DARPA OpTC (corrected release): can it support attack-onset forecasting?

Same questions as scripts/ctu13_onset_census.py, for every host the red team reached:

  * onset: start of the host's first red-team process (corrected host ground truth);
  * episodes: red-team process lifetimes merged with the onset study's 30-minute gap rule;
  * lockstep: how many other hosts in the same scenario start within 60 s (one command
    fanned out to many hosts is one decision, not many independent onsets);
  * benign history: days of telemetry for the host before the onset day (from the tar
    index, no download needed), and, when the onset day's flows have been extracted,
    the host's own non-red-team flows in the 10 and 60 minutes before onset.

    python scripts/optc_onset_census.py [--labels data/raw/optc/labels] [--flows data/raw/optc/flows]

Writes reports/optc_onset_census.json. Raw data stays in data/raw/optc (gitignored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegisflow.ml.datasets.optc import load_ground_truth, read_flow_events, label_flows, OptcAdapter  # noqa: E402

EPISODE_GAP = pd.Timedelta(minutes=30)
LOCKSTEP = pd.Timedelta(seconds=60)
HOST_RE = re.compile(r"sysclient(\d{4})")
SCENARIOS = {1: "PowerShell Empire (2019-09-23)", 2: "DeathStar / custom malware (2019-09-24)",
             3: "Malicious software upgrade (2019-09-25)"}


def episodes(intervals: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    rows = intervals.sort_values("start")
    out: list[list[pd.Timestamp]] = []
    for s, e in zip(rows["start"], rows["end"].fillna(rows["start"])):
        e = max(s, e)
        if out and s - out[-1][1] <= EPISODE_GAP:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def onset_table(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, host), g in gt.groupby(["scenario", "host"]):
        eps = episodes(g)
        rows.append({"scenario": int(scen), "host": host, "onset": g["start"].min(),
                     "red_processes": int(g["pid"].nunique()), "episodes": len(eps),
                     "episode_starts": [str(a) for a, _ in eps],
                     "last_red_activity": max(b for _, b in eps)})
    t = pd.DataFrame(rows).sort_values(["scenario", "onset"]).reset_index(drop=True)
    # lockstep batches: chain onsets in the same scenario that are <= 60 s apart
    batch, last, prev_scen = -1, None, None
    ids = []
    for scen, onset in zip(t["scenario"], t["onset"]):
        if scen != prev_scen or onset - last > LOCKSTEP:
            batch += 1
        ids.append(batch)
        last, prev_scen = onset, scen
    t["onset_event"] = ids
    t["batch_size"] = t.groupby("onset_event")["host"].transform("size")
    return t


def history_from_index(t: pd.DataFrame, index_path: Path) -> pd.DataFrame:
    idx = json.loads(index_path.read_text())
    present: dict[str, dict[str, int]] = {}
    for day, members in idx.items():
        for m in members:
            h = HOST_RE.search(m["name"])
            if h:
                present.setdefault(f"sysclient{h.group(1)}", {})[day] = m["size"]
    days = sorted(idx)
    t = t.copy()
    t["benign_days_before"] = [
        sum(1 for d in days if d < str(o.date()) and d in present.get(h, {})) for h, o in zip(t["host"], t["onset"])]
    t["onset_day_file_mb"] = [round(present.get(h, {}).get(str(o.date()), 0) / 1e6, 1)
                              for h, o in zip(t["host"], t["onset"])]
    t["has_telemetry"] = [h in present for h in t["host"]]
    t.attrs["days_indexed"] = days
    t.attrs["hosts_per_day"] = {d: len(idx[d]) for d in days}
    return t


def history_from_flows(t: pd.DataFrame, flows_dir: Path, gt: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    cols = {c: [] for c in ("own_out_10m", "own_out_60m", "inbound_10m", "inbound_60m",
                            "min_since_last_own_out", "red_flows_extracted", "first_red_flow_lag_s",
                            "flows_extracted_until")}
    for host, onset in zip(t["host"], t["onset"]):
        day = str(onset.date())
        files = list(flows_dir.glob(f"{day}/*{host}.flows.json.gz"))
        if not files:
            for v in cols.values():
                v.append(None)
            continue
        raw = read_flow_events(files[0])
        flows = OptcAdapter._to_canonical(raw)
        flows = label_flows(flows, gt)
        benign = flows[flows["dataset_label"] == "Benign"]
        before = benign[benign["timestamp"] < onset]
        out = before[before["optc_direction"] == "outbound"]
        inn = before[before["optc_direction"] == "inbound"]
        win = lambda d, m: int((d["timestamp"] >= onset - pd.Timedelta(minutes=m)).sum())  # noqa: E731
        red = flows[flows["dataset_label"] == "Malicious"]
        cols["own_out_10m"].append(win(out, 10))
        cols["own_out_60m"].append(win(out, 60))
        cols["inbound_10m"].append(win(inn, 10))
        cols["inbound_60m"].append(win(inn, 60))
        cols["min_since_last_own_out"].append(
            round((onset - out["timestamp"].max()).total_seconds() / 60, 2) if len(out) else None)
        cols["red_flows_extracted"].append(int(len(red)))
        cols["first_red_flow_lag_s"].append(
            round((red["timestamp"].min() - onset).total_seconds(), 1) if len(red) else None)
        cols["flows_extracted_until"].append(str(flows["timestamp"].max()))
    for k, v in cols.items():
        t[k] = v
    return t


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "raw" / "optc" / "labels")
    ap.add_argument("--flows", type=Path, default=ROOT / "data" / "raw" / "optc" / "flows")
    ap.add_argument("--index", type=Path, default=ROOT / "reports" / "optc_tar_index.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "optc_onset_census.json")
    a = ap.parse_args(argv)

    gt = load_ground_truth(a.labels)
    t = onset_table(gt)
    if a.index.exists():
        t = history_from_index(t, a.index)
    if a.flows.exists():
        t = history_from_flows(t, a.flows, gt)

    workstations = t[t["host"].str.startswith("sysclient")]
    summary = {
        "host_onsets": int(len(t)),
        "workstation_onsets": int(len(workstations)),
        "distinct_hosts": int(t["host"].nunique()),
        "scenarios": {int(s): {"name": SCENARIOS[int(s)], "host_onsets": int(len(g)),
                               "independent_onset_events": int(g["onset_event"].nunique()),
                               "largest_lockstep_batch": int(g["batch_size"].max()),
                               "onset_clock_range": [str(g["onset"].min().time()), str(g["onset"].max().time())]}
                      for s, g in t.groupby("scenario")},
        "independent_onset_events": int(t["onset_event"].nunique()),
        "episodes": int(t["episodes"].sum()),
    }
    if "benign_days_before" in t:
        summary["workstation_onsets_with_telemetry"] = int(workstations["has_telemetry"].sum())
        summary["workstation_onsets_with_>=3_benign_days"] = int((workstations["benign_days_before"] >= 3).sum())
        summary["hosts_with_telemetry_per_day"] = t.attrs["hosts_per_day"]
    if "own_out_10m" in t and t["own_out_10m"].notna().any():
        have = t[t["own_out_10m"].notna()]
        summary["onsets_with_flows_extracted"] = int(len(have))
        summary["onsets_with_own_outbound_in_10m_before"] = int((have["own_out_10m"] > 0).sum())
        summary["onsets_with_own_outbound_in_60m_before"] = int((have["own_out_60m"] > 0).sum())
        ev = have[have["own_out_10m"] > 0]
        summary["independent_events_with_own_history_10m"] = int(ev["onset_event"].nunique())

    rows = json.loads(t.to_json(orient="records", date_format="iso"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "onsets": rows}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    show = [c for c in ("scenario", "host", "onset", "onset_event", "batch_size", "episodes", "benign_days_before",
                        "onset_day_file_mb", "own_out_10m", "own_out_60m", "inbound_10m",
                        "min_since_last_own_out", "red_flows_extracted", "first_red_flow_lag_s") if c in t]
    print(t[show].to_string())


if __name__ == "__main__":
    main()
