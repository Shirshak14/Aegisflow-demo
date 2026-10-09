"""
Adapter for LANL "Comprehensive, Multi-Source Cyber-Security Events" (Kent, 2015), flows only.

Opt-in only: selected with ``--dataset lanl_2015`` (configs/datasets.yaml). The default
pipeline, demo model and thresholds stay on CIC-IDS2017.

Expects ``flows.txt.gz`` (router flow records, 1.1 GB gzipped) and ``redteam.txt.gz``
(749 red-team authentication events) under ``raw_dir``. Flow columns:
time, duration, source computer, source port, destination computer, destination port,
protocol, packet count, byte count. ``?`` means missing.

What the data can and cannot supply (nothing is invented):
  * Computers and most ports are de-identified tokens (``C17693``, ``N10471``). The
    computer token is used as the "IP"; de-identified ports become NA, well-known ports
    (80, 443, ...) stay numeric.
  * Time is seconds since an undisclosed start (epoch 1), so the timestamp is
    ``1970-01-01 + time`` and only *relative* time is meaningful: hour of day and
    weekday are not real clock values.
  * Only total packets and bytes exist; flags, packet sizes, inter-arrival times,
    TTL and window are NA.
  * The red-team ground truth is a list of *authentications* (from auth.txt.gz), not
    flows. A flow is labelled ``RedTeam`` when it goes from a red-team source computer
    to a computer that source authenticated to as red team, at or after that first
    red-team authentication minus ``contact_slack_s`` (default 3600 s). Every other flow
    from a red-team source computer is ``RedTeamSource``; the rest is ``Benign``.
    All three are inferred labels.

The full file has about 129 million flows. ``computers: [...]`` keeps only flows that
touch those computers, and ``time_range_s: [start, end]`` keeps a time slice; both are
applied while streaming, so memory stays small.

Official source: https://csr.lanl.gov/data/cyber1/ (CC0; the page asks for an email
and intended use before download).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import DatasetNotFoundError, DatasetValidationError
from ...schema import coerce_canonical_frame, validate_canonical_frame
from . import register
from .base import DatasetAdapter, DiscoveryReport

FLOW_COLUMNS = ["time", "duration", "src", "src_port", "dst", "dst_port", "protocol", "packets", "bytes"]
_PROTOCOLS = {"6": "TCP", "17": "UDP", "1": "ICMP"}
EPOCH = pd.Timestamp("1970-01-01")


def load_redteam(path: Path) -> pd.DataFrame:
    rt = pd.read_csv(path, header=None, names=["time", "user", "src", "dst"], dtype={"time": "int64"})
    return rt.sort_values("time").reset_index(drop=True)


def victim_onsets(redteam: pd.DataFrame) -> pd.DataFrame:
    """First red-team authentication per destination computer."""
    first = redteam.groupby("dst").agg(onset_s=("time", "min"), attacker=("src", "first"),
                                       red_auths=("time", "size")).reset_index()
    return first.rename(columns={"dst": "computer"}).sort_values("onset_s").reset_index(drop=True)


def iter_flows(path: Path, computers: set[str] | None = None,
               time_range: tuple[int, int] | None = None, chunksize: int = 2_000_000):
    """Stream flows.txt.gz in chunks, keeping only the requested computers / time slice."""
    reader = pd.read_csv(path, header=None, names=FLOW_COLUMNS, dtype=str, chunksize=chunksize,
                         keep_default_na=False)
    for chunk in reader:
        t = pd.to_numeric(chunk["time"], errors="coerce")
        keep = pd.Series(True, index=chunk.index)
        if time_range is not None:
            if t.iloc[0] > time_range[1]:
                break  # the file is sorted by time
            keep &= t.between(*time_range)
        if computers is not None:
            keep &= chunk["src"].isin(computers) | chunk["dst"].isin(computers)
        if keep.any():
            yield chunk[keep]


def label_flows(flows: pd.DataFrame, redteam: pd.DataFrame, contact_slack_s: int = 3600) -> pd.Series:
    t = pd.to_numeric(flows["time"], errors="coerce")
    label = pd.Series("Benign", index=flows.index, dtype="string")
    sources = set(redteam["src"])
    label[flows["src"].isin(sources)] = "RedTeamSource"
    first_pair = redteam.groupby(["src", "dst"])["time"].min()
    pair_start = pd.Series(list(zip(flows["src"], flows["dst"])), index=flows.index).map(first_pair)
    label[pair_start.notna() & (t >= pair_start - contact_slack_s)] = "RedTeam"
    return label


def _port(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.where(~values.str.startswith("N")), errors="coerce").astype("Int64")


@register("lanl_2015")
class Lanl2015Adapter(DatasetAdapter):
    name = "lanl_2015"

    def discover(self) -> DiscoveryReport:
        if not self.raw_dir.exists():
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=[
                f"Directory does not exist: {self.raw_dir}"])
        flows = [p for p in (self.raw_dir / "flows.txt.gz", self.raw_dir / "flows.txt") if p.exists()]
        problems = [] if (self.raw_dir / "redteam.txt.gz").exists() else [
            f"redteam.txt.gz missing under {self.raw_dir}: every flow would be labelled Benign."]
        if not flows:
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=problems + [
                f"flows.txt.gz not found under {self.raw_dir}."])
        return DiscoveryReport(self.name, self.raw_dir, found=True, files=flows[:1], problems=problems)

    def download_instructions(self) -> str:
        return f"""
LANL "Comprehensive, Multi-Source Cyber-Security Events" (CC0):
   {self.entry.official_url}
Fill in the short form on that page (email and intended use), then download
flows.txt.gz (1.1 GB) and redteam.txt.gz (4.8 KB) into

   {self.raw_dir}

auth.txt.gz (7.2 GB) holds the authentications the red-team events come from; it is
not read by this adapter. Restrict loading with `computers: [...]` and/or
`time_range_s: [start, end]` in the lanl_2015 entry of configs/datasets.yaml.

Citation:
{self.entry.citation.strip()}
""".strip()

    def load_raw(self, sample_size: int | None = None) -> pd.DataFrame:
        report = self.discover()
        if not report.found:
            raise DatasetNotFoundError(f"LANL flows not found.\n\n{self.download_instructions()}")
        computers = set(self.entry["computers"]) if self.entry.get("computers") else None
        tr = self.entry.get("time_range_s")
        raw = pd.concat(list(iter_flows(report.files[0], computers, tuple(tr) if tr else None)),
                        ignore_index=True)
        if raw.empty:
            raise DatasetValidationError(f"No LANL flows matched the configured filters in {report.files[0]}.")
        if sample_size is not None and sample_size < len(raw):
            rng = np.random.default_rng(int(self.cfg.config.random_seed))
            raw = raw.iloc[np.sort(rng.choice(len(raw), size=sample_size, replace=False))].reset_index(drop=True)
        rt_path = self.raw_dir / "redteam.txt.gz"
        redteam = load_redteam(rt_path) if rt_path.exists() else pd.DataFrame(columns=["time", "user", "src", "dst"])
        canonical = self._to_canonical(raw)
        canonical["dataset_label"] = label_flows(raw, redteam, int(self.entry.get("contact_slack_s", 3600)))
        canonical = coerce_canonical_frame(canonical)
        canonical["source_dataset"] = "lanl_2015"
        validate_canonical_frame(canonical, context="lanl_2015.load_raw")
        return canonical

    @staticmethod
    def _to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
        num = lambda col: pd.to_numeric(raw[col], errors="coerce").astype("float64")  # noqa: E731
        seconds = pd.to_numeric(raw["time"], errors="coerce")
        return pd.DataFrame({
            "timestamp": EPOCH + pd.to_timedelta(seconds, unit="s"),
            "source_ip": raw["src"].astype("string"),
            "destination_ip": raw["dst"].astype("string"),
            "source_port": _port(raw["src_port"]),
            "destination_port": _port(raw["dst_port"]),
            "protocol": raw["protocol"].map(lambda p: _PROTOCOLS.get(p, p)).astype("string"),
            "flow_duration": num("duration"),
            "packet_count": num("packets"),
            "byte_count": num("bytes"),
            "packet_size_mean": num("bytes") / num("packets").where(num("packets") > 0),
            "lanl_time_s": seconds.astype("Int64"),
        })
