"""
Adapter for CTU-13 (Stratosphere Lab, Czech Technical University, 2011).

Opt-in only: selected with ``--dataset ctu_13`` (configs/datasets.yaml). The default
pipeline, demo model and thresholds stay on CIC-IDS2017.

Expects the per-scenario bidirectional Argus flow files ("*.binetflow", the
``detailed-bidirectional-flow-labels`` folder of each CTU-Malware-Capture-Botnet-42..54
capture), optionally gzip-compressed (``*.binetflow.gz``). Columns:
StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos,
TotPkts, TotBytes, SrcBytes, Label.

What binetflow can and cannot supply (nothing is invented):
  * TotBytes/SrcBytes are Argus byte counts including headers; CICFlowMeter counts
    payload bytes, so byte features are on a different scale across the two datasets.
  * Only total packets exist: fwd/bwd packet counts stay NA.
  * TCP flags come from the Argus ``State`` string (``<src flags>_<dst flags>``), so
    ``syn_count`` etc. are 1 if the flag was seen in either direction, else 0: a lower
    bound, as in the NetFlow ingestion. Non-TCP flows get 0.
  * packet-length std/min/max, inter-arrival statistics, TCP window, TTL and
    retransmissions are absent and stay NA; ``packet_size_mean`` is TotBytes / TotPkts.
  * Timestamps are local capture time in Prague (CEST), kept naive like CIC-IDS2017's.

Labels: CTU-13 has hundreds of fine labels (``flow=From-Botnet-V42-TCP-CC6-...``).
``dataset_label`` is the label family (Botnet / Normal / Background) so that
configs/stages.yaml can map it; the untouched label is kept in ``ctu_label`` and the
scenario number (1-13) in ``ctu_scenario``. "Background" is unverified traffic, not
confirmed benign; stages.yaml marks it as Benign with confidence "inferred".

Official source: https://www.stratosphereips.org/datasets-ctu13
"""
from __future__ import annotations

import glob
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import DatasetNotFoundError, DatasetValidationError
from ...schema import coerce_canonical_frame, validate_canonical_frame
from . import register
from .base import DatasetAdapter, DiscoveryReport
from .cic_ids2017 import spread_sample_plan

_MIRROR = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{n}/detailed-bidirectional-flow-labels/"

# scenario number -> (CTU-Malware-Capture-Botnet id, official binetflow stem)
SCENARIOS: dict[int, tuple[int, str]] = {
    1: (42, "capture20110810"),
    2: (43, "capture20110811"),
    3: (44, "capture20110812"),
    4: (45, "capture20110815"),
    5: (46, "capture20110815-2"),
    6: (47, "capture20110816"),
    7: (48, "capture20110816-2"),
    8: (49, "capture20110816-3"),
    9: (50, "capture20110817"),
    10: (51, "capture20110818"),
    11: (52, "capture20110818-2"),
    12: (53, "capture20110819"),
    13: (54, "capture20110815-3"),
}
_BY_STEM = {stem: s for s, (_, stem) in SCENARIOS.items()}
_BY_CAPTURE = {cap: s for s, (cap, _) in SCENARIOS.items()}

_REQUIRED_RAW = {"StartTime", "Dur", "Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "State",
                 "TotPkts", "TotBytes", "SrcBytes", "Label"}
_FLAG_LETTERS = {"syn_count": "S", "ack_count": "A", "rst_count": "R", "fin_count": "F",
                 "psh_count": "P", "urg_count": "U"}
_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def scenario_of(path: Path) -> int | None:
    """Scenario number (1-13) from an official file name or a ``scenario_NN`` / ``-Botnet-NN`` prefix."""
    name = path.name
    stem = re.sub(r"\.binetflow(\.gz)?$", "", name)
    if stem in _BY_STEM:
        return _BY_STEM[stem]
    m = re.match(r"scenario_(\d+)", name)
    if m and int(m.group(1)) in SCENARIOS:
        return int(m.group(1))
    m = re.search(r"botnet[-_](\d+)", name, flags=re.IGNORECASE)
    if m and int(m.group(1)) in _BY_CAPTURE:
        return _BY_CAPTURE[int(m.group(1))]
    return None


def label_family(label: pd.Series) -> pd.Series:
    """``flow=From-Botnet-V42-...`` -> Botnet; ``...Normal...`` -> Normal; everything else -> Background."""
    s = label.astype("string")
    fam = pd.Series("Background", index=s.index, dtype="string")
    fam[s.str.contains("Normal", na=False)] = "Normal"
    fam[s.str.contains("Botnet", na=False)] = "Botnet"
    return fam


def parse_port(values: pd.Series) -> pd.Series:
    """Argus ports: decimal, hex for ICMP type/code (``0x0303``), or empty for portless protocols."""
    s = values.astype("string").str.strip()
    out = pd.to_numeric(s, errors="coerce")
    hex_mask = s.str.lower().str.startswith("0x", na=False)
    if hex_mask.any():
        out[hex_mask] = s[hex_mask].map(lambda x: int(x, 16))
    return out.astype("Int64")


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" \
        else path.open("r", encoding="utf-8", errors="replace")


def _count_data_rows(path: Path) -> int:
    with _open_text(path) as fh:
        return max(0, sum(1 for _ in fh) - 1)


@register("ctu_13")
class Ctu13Adapter(DatasetAdapter):
    name = "ctu_13"

    def _selected_scenarios(self) -> set[int] | None:
        chosen = self.entry.get("scenarios")
        return {int(s) for s in chosen} if chosen else None

    def discover(self) -> DiscoveryReport:
        if not self.raw_dir.exists():
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=[
                f"Directory does not exist: {self.raw_dir}"
            ])
        files = sorted({Path(p) for pat in ("*.binetflow", "*.binetflow.gz")
                        for p in glob.glob(str(self.raw_dir / "**" / pat), recursive=True)})
        problems: list[str] = []
        wanted = self._selected_scenarios()
        kept: list[Path] = []
        for f in files:
            scen = scenario_of(f)
            if scen is None:
                problems.append(f"{f.name}: cannot tell which CTU-13 scenario this is from its name.")
                continue
            if wanted is not None and scen not in wanted:
                continue
            try:
                with _open_text(f) as fh:
                    header = {c.strip() for c in fh.readline().split(",")}
            except OSError as exc:
                problems.append(f"{f.name}: could not be read ({exc}).")
                continue
            missing = sorted(_REQUIRED_RAW - header)
            if missing:
                problems.append(f"{f.name}: missing binetflow columns {missing}.")
            kept.append(f)
        if not kept:
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=problems or [
                f"No .binetflow or .binetflow.gz files found under {self.raw_dir} (searched recursively)."
            ])
        return DiscoveryReport(self.name, self.raw_dir, found=True, files=kept, problems=problems)

    def download_instructions(self) -> str:
        lines = "\n".join(f"   scenario {s:>2}: {_MIRROR.format(n=cap)}{stem}.binetflow"
                          for s, (cap, stem) in SCENARIOS.items())
        return f"""
CTU-13 is public. Download the bidirectional flow file of each scenario (about
2.7 GB in total uncompressed; gzip them to save space, AegisFlow reads .gz):

{lines}

Put them here, keeping the official names (or scenario_NN_*.binetflow[.gz]):

   {self.expected_directory_structure()}

The all-in-one archive CTU-13-Dataset.tar.bz2 (1.9 GB) from {self.entry.official_url}
also contains them. Restrict to some scenarios with `scenarios: [11, 12]` in the
ctu_13 entry of configs/datasets.yaml. Verify with:

   python -m aegisflow validate-dataset --dataset ctu_13

Citation (cite this if you publish results):
{self.entry.citation.strip()}
""".strip()

    def load_raw(self, sample_size: int | None = None) -> pd.DataFrame:
        report = self.discover()
        if not report.found:
            raise DatasetNotFoundError(f"CTU-13 raw files not found.\n\n{self.download_instructions()}")

        plan: list[np.ndarray | None] = [None] * len(report.files)
        if sample_size is not None:
            counts = [_count_data_rows(f) for f in report.files]
            plan = spread_sample_plan(counts, sample_size, int(self.cfg.config.random_seed))
            self.log.info("spread-out sample plan", sample_size=sample_size, total_rows=sum(counts))

        frames: list[pd.DataFrame] = []
        for fpath, rows in zip(report.files, plan):
            skip = None
            if rows is not None:
                keep = set((rows + 1).tolist())
                skip = lambda i, keep=keep: i != 0 and i not in keep  # noqa: E731
            df = pd.read_csv(fpath, skiprows=skip, dtype=str, keep_default_na=False,
                             encoding="utf-8", encoding_errors="replace")
            df.columns = [c.strip() for c in df.columns]
            df["ctu_scenario"] = scenario_of(fpath)
            frames.append(df)

        raw = pd.concat(frames, ignore_index=True, sort=False)
        if raw.empty:
            raise DatasetValidationError(f"No readable binetflow rows found under {self.raw_dir}.")
        canonical = coerce_canonical_frame(self._to_canonical(raw))
        canonical["source_dataset"] = "ctu_13"
        validate_canonical_frame(canonical, context="ctu_13.load_raw")
        return canonical

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
        num = lambda col: pd.to_numeric(raw[col].str.strip(), errors="coerce")  # noqa: E731
        proto = raw["Proto"].astype("string").str.strip().str.upper()
        packets, total_bytes, src_bytes = num("TotPkts"), num("TotBytes"), num("SrcBytes")

        out = pd.DataFrame({
            "timestamp": pd.to_datetime(raw["StartTime"].str.strip(), format=_TIMESTAMP_FORMAT, errors="coerce"),
            "source_ip": raw["SrcAddr"].str.strip(),
            "destination_ip": raw["DstAddr"].str.strip(),
            "source_port": parse_port(raw["Sport"]),
            "destination_port": parse_port(raw["Dport"]),
            "protocol": proto,
            "flow_duration": num("Dur"),
            "packet_count": packets,
            "byte_count": total_bytes,
            "fwd_byte_count": src_bytes,
            "bwd_byte_count": (total_bytes - src_bytes).clip(lower=0),
            "packet_size_mean": total_bytes / packets.where(packets > 0),
            "dataset_label": label_family(raw["Label"]),
            "ctu_label": raw["Label"].astype("string").str.strip(),
            "ctu_scenario": raw["ctu_scenario"].astype("Int64"),
        })

        state = raw["State"].astype("string").str.strip().fillna("")
        is_tcp = proto == "TCP"
        for col, letter in _FLAG_LETTERS.items():
            seen = state.str.contains(letter, regex=False) & is_tcp
            out[col] = seen.astype("float64")
        return out
