"""
Adapter for DARPA OpTC (Operationally Transparent Cyber, 2019), corrected release.

Opt-in only: selected with ``--dataset optc`` (configs/datasets.yaml). The default
pipeline, demo model and thresholds stay on CIC-IDS2017.

Reads eCAR endpoint telemetry: one JSON object per line, gzip-compressed, one file
per host and day (``AIA-201-225.ecar-2019-09-23-sysclient0201.json.gz`` in the
corrected release, or the ``*.flows.json.gz`` reductions written by
``scripts/optc_remote.py extract``, which keep only FLOW events). Only FLOW events
are used; process, file and registry events are ignored here.

What eCAR can and cannot supply (nothing is invented):
  * One canonical row per flow object: the first ``FLOW START`` event of each objectID,
    i.e. per connection the host's sensor saw open, inbound or outbound. ``src_ip``/``dest_ip`` are the real endpoints, so
    inbound rows have the remote host as ``source_ip``.
  * ``byte_count`` is the sum of the ``size`` of the ``FLOW MESSAGE`` events with the
    same objectID in the same file; NA when the sensor reported none (most UDP).
    Direction of those bytes is not given, so fwd/bwd bytes stay NA.
  * No packet counts, TCP flags, packet sizes, inter-arrival times, TTL or window:
    all NA. ``flow_duration`` stays NA because the unit of the MESSAGE
    ``start_time``/``end_time`` fields is not documented.
  * The sensor ships its telemetry to Kafka (port 9092). Those flows are collection
    infrastructure, not host behaviour, and are dropped unless ``keep_telemetry: true``.
  * Timestamps carry a UTC offset (-04:00, US Eastern daylight time); they are kept as
    naive local clock time, like CIC-IDS2017's.

Labels: ``dataset_label`` is ``Malicious`` when the flow's (host, pid) is a red-team
process in the corrected host ground truth (Majorczyk et al., gitlab.inria.fr/
fmajorcz/a_new_hope_for_darpa_optc, ``labelling/host/ground_truths/
ground_truth_corrected_sc{1,2,3}_updated.csv``) and the flow starts inside that
process's lifetime; otherwise ``Benign``. The scenario (1-3) is kept in
``optc_scenario`` and the sensor host in ``optc_host``.

Official sources: https://github.com/FiveDirections/OpTC-data (original),
doi:10.57745/UXCWOC (corrected, CC BY 4.0).
"""
from __future__ import annotations

import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import DatasetNotFoundError, DatasetValidationError
from ...schema import coerce_canonical_frame, validate_canonical_frame
from . import register
from .base import DatasetAdapter, DiscoveryReport

_FILE_RE = re.compile(r"ecar-(\d{4}-\d{2}-\d{2})-sysclient(\d{4})(\.flows)?\.json\.gz$", re.IGNORECASE)
_PROTOCOLS = {"6": "TCP", "17": "UDP", "1": "ICMP", "58": "ICMPV6"}
TELEMETRY_PORT = 9092
LABEL_FILES = "ground_truth_corrected_sc{n}_updated.csv"


def host_key(hostname: str) -> str:
    """``SysClient0201.systemia.com`` -> ``sysclient0201``; ``DC1.systemia.com`` -> ``dc1``."""
    return str(hostname).split(".")[0].strip().lower()


def load_ground_truth(labels_dir: Path) -> pd.DataFrame:
    """Malicious (host, pid, start, end, scenario) rows from the corrected host ground truth."""
    frames = []
    for n in (1, 2, 3):
        path = labels_dir / LABEL_FILES.format(n=n)
        if not path.exists():
            continue
        df = pd.read_csv(path, header=None, names=["host", "pid", "start", "end"], dtype=str)
        df["scenario"] = n
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["host", "pid", "start", "end", "scenario"])
    gt = pd.concat(frames, ignore_index=True)
    gt["host"] = gt["host"].map(host_key)
    gt["pid"] = pd.to_numeric(gt["pid"], errors="coerce").astype("Int64")
    gt["start"] = parse_local_time(gt["start"])
    gt["end"] = parse_local_time(gt["end"].where(gt["end"].str.lower() != "infinity"))
    return gt.dropna(subset=["pid", "start"])


def parse_local_time(values: pd.Series) -> pd.Series:
    """ISO timestamps with a UTC offset -> naive local clock time (the offset is dropped)."""
    s = values.astype("string").str.slice(0, 23)
    return pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce").fillna(
        pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce"))


def read_flow_events(path: Path, keep_telemetry: bool = False) -> pd.DataFrame:
    """One row per flow object (its first FLOW START) in one eCAR file, with MESSAGE byte totals.

    The sensor repeats START for the same objectID (mostly UDP broadcast and multicast,
    about four STARTs per object), so only the first is kept.
    """
    starts: list[tuple] = []
    sizes: dict[str, float] = {}
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"FLOW"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("object") != "FLOW":
                continue
            p = e.get("properties") or {}
            action = e.get("action")
            if action == "MESSAGE":
                try:
                    sizes[e["objectID"]] = sizes.get(e["objectID"], 0.0) + float(p.get("size", "nan"))
                except (TypeError, ValueError, KeyError):
                    pass
            elif action == "START" and e.get("objectID") not in seen:
                seen.add(e.get("objectID"))
                starts.append((e.get("timestamp"), e.get("hostname"), e.get("pid"), e.get("objectID"),
                               p.get("src_ip"), p.get("src_port"), p.get("dest_ip"), p.get("dest_port"),
                               p.get("l4protocol"), p.get("direction"), p.get("image_path")))
    df = pd.DataFrame(starts, columns=["ts", "hostname", "pid", "objectID", "src_ip", "src_port",
                                       "dest_ip", "dest_port", "l4protocol", "direction", "image_path"])
    df["byte_count"] = df["objectID"].map(sizes).astype("float64")
    if not keep_telemetry:
        df = df[pd.to_numeric(df["dest_port"], errors="coerce") != TELEMETRY_PORT]
    return df.reset_index(drop=True)


def label_flows(flows: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Add dataset_label / optc_scenario: Malicious iff a red-team (host, pid) was alive at flow start."""
    out = flows.copy()
    out["dataset_label"] = "Benign"
    out["optc_scenario"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if gt.empty or out.empty:
        return out
    merged = out.reset_index().merge(gt, left_on=["optc_host", "pid"], right_on=["host", "pid"], how="inner")
    end = merged["end"].fillna(pd.Timestamp.max)
    hit = merged[(merged["timestamp"] >= merged["start"]) & (merged["timestamp"] <= end)]
    hit = hit.drop_duplicates("index")
    out.loc[hit["index"], "dataset_label"] = "Malicious"
    out.loc[hit["index"], "optc_scenario"] = hit["scenario"].to_numpy()
    return out


@register("optc")
class OptcAdapter(DatasetAdapter):
    name = "optc"

    @property
    def labels_dir(self) -> Path:
        return self.cfg.path(self.entry.get("labels_dir", str(Path(self.entry.raw_dir).parent / "labels")))

    def discover(self) -> DiscoveryReport:
        if not self.raw_dir.exists():
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=[
                f"Directory does not exist: {self.raw_dir}"])
        files = sorted(Path(p) for p in glob.glob(str(self.raw_dir / "**" / "*.json.gz"), recursive=True)
                       if _FILE_RE.search(Path(p).name))
        hosts = self.entry.get("hosts")
        if hosts:
            wanted = {int(h) for h in hosts}
            files = [f for f in files if int(_FILE_RE.search(f.name).group(2)) in wanted]
        problems: list[str] = []
        if not any((self.labels_dir / LABEL_FILES.format(n=n)).exists() for n in (1, 2, 3)):
            problems.append(f"No red-team ground truth under {self.labels_dir}: every flow would be labelled Benign.")
        if not files:
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=problems or [
                f"No eCAR '*ecar-YYYY-MM-DD-sysclientNNNN[.flows].json.gz' files under {self.raw_dir}."])
        return DiscoveryReport(self.name, self.raw_dir, found=True, files=files, problems=problems)

    def download_instructions(self) -> str:
        return f"""
DARPA OpTC, corrected release (CC BY 4.0): https://doi.org/10.57745/UXCWOC
Ten tar archives, one per day 2019-09-16 .. 2019-09-25, about 940 GB in all. Each holds
one gzip eCAR file per host and day. Do not download whole archives: fetch single
host-days with HTTP range requests and keep only FLOW events:

   python scripts/optc_remote.py index
   python scripts/optc_remote.py extract --hosts 201 402 --days 2019-09-22 2019-09-23

Files go to {self.raw_dir}. Red-team labels (small CSVs) go to {self.labels_dir}:
   https://gitlab.inria.fr/fmajorcz/a_new_hope_for_darpa_optc/-/tree/main/labelling/host/ground_truths
   (ground_truth_corrected_sc1_updated.csv, _sc2_, _sc3_)
Restrict to some hosts with `hosts: [201, 402]` in the optc entry of configs/datasets.yaml.

   python -m aegisflow validate-dataset --dataset optc

Citation:
{self.entry.citation.strip()}
""".strip()

    def load_raw(self, sample_size: int | None = None) -> pd.DataFrame:
        report = self.discover()
        if not report.found:
            raise DatasetNotFoundError(f"OpTC eCAR files not found.\n\n{self.download_instructions()}")
        keep_telemetry = bool(self.entry.get("keep_telemetry", False))
        frames = [read_flow_events(f, keep_telemetry) for f in report.files]
        raw = pd.concat(frames, ignore_index=True)
        if raw.empty:
            raise DatasetValidationError(f"No FLOW START events found under {self.raw_dir}.")
        if sample_size is not None and sample_size < len(raw):
            rng = np.random.default_rng(int(self.cfg.config.random_seed))
            raw = raw.iloc[np.sort(rng.choice(len(raw), size=sample_size, replace=False))].reset_index(drop=True)
        canonical = self._to_canonical(raw)
        canonical = label_flows(canonical, load_ground_truth(self.labels_dir))
        canonical = coerce_canonical_frame(canonical)
        canonical["source_dataset"] = "optc"
        validate_canonical_frame(canonical, context="optc.load_raw")
        return canonical

    @staticmethod
    def _to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
        port = lambda col: pd.to_numeric(raw[col], errors="coerce").astype("Int64")  # noqa: E731
        proto = raw["l4protocol"].astype("string")
        return pd.DataFrame({
            "timestamp": parse_local_time(raw["ts"]),
            "source_ip": raw["src_ip"].astype("string"),
            "destination_ip": raw["dest_ip"].astype("string"),
            "source_port": port("src_port"),
            "destination_port": port("dest_port"),
            "protocol": proto.map(lambda p: _PROTOCOLS.get(p, p) if isinstance(p, str) else pd.NA).astype("string"),
            "flow_duration": np.nan,
            "packet_count": np.nan,
            "byte_count": raw["byte_count"].astype("float64"),
            "optc_host": raw["hostname"].map(host_key).astype("string"),
            "optc_direction": raw["direction"].astype("string"),
            "pid": pd.to_numeric(raw["pid"], errors="coerce").astype("Int64"),
            "image_path": raw["image_path"].astype("string"),
        })
