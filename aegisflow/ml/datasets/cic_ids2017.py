"""
Adapter for CIC-IDS2017 (Canadian Institute for Cybersecurity, UNB).

Expects the official "GeneratedLabelledFlows.zip" variant (CICFlowMeter CSV
output with IP/timestamp columns preserved), NOT "MachineLearningCSV.zip"
(which drops IPs/timestamps and is therefore unusable for temporal,
per-host windowing).

Official source: https://www.unb.ca/cic/datasets/ids-2017.html
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import DatasetNotFoundError, DatasetValidationError
from ...schema import coerce_canonical_frame, empty_canonical_frame, validate_canonical_frame
from . import register
from .base import DatasetAdapter, DiscoveryReport

# CICFlowMeter emits headers with inconsistent leading whitespace
# (e.g. " Destination Port"). We normalize by stripping + casing before
# mapping, so the adapter is robust to that quirk across the 8 day-files.
_COLUMN_MAP = {
    "flow id": None,  # unused
    "source ip": "source_ip",
    "src ip": "source_ip",
    "destination ip": "destination_ip",
    "dst ip": "destination_ip",
    "source port": "source_port",
    "src port": "source_port",
    "destination port": "destination_port",
    "dst port": "destination_port",
    "protocol": "protocol",
    "timestamp": "timestamp",
    "flow duration": "flow_duration",
    "total fwd packets": "fwd_packet_count",
    "total backward packets": "bwd_packet_count",
    "total length of fwd packets": "fwd_byte_count",
    "totlen fwd pkts": "fwd_byte_count",
    "total length of bwd packets": "bwd_byte_count",
    "totlen bwd pkts": "bwd_byte_count",
    "average packet size": "packet_size_mean",
    "packet length mean": "packet_size_mean",
    "pkt len mean": "packet_size_mean",
    "packet length std": "packet_size_std",
    "pkt len std": "packet_size_std",
    "min packet length": "packet_size_min",
    "pkt len min": "packet_size_min",
    "max packet length": "packet_size_max",
    "pkt len max": "packet_size_max",
    "flow iat mean": "inter_arrival_mean",
    "flow iat std": "inter_arrival_std",
    "syn flag count": "syn_count",
    "ack flag count": "ack_count",
    "rst flag count": "rst_count",
    "fin flag count": "fin_count",
    "psh flag count": "psh_count",
    "urg flag count": "urg_count",
    "init_win_bytes_forward": "tcp_window_size",
    "init win bytes forward": "tcp_window_size",
    "label": "dataset_label",
}

# protocol number -> name, per IANA (only the ones CIC-IDS2017 actually contains)
_PROTOCOL_NAMES = {"6": "TCP", "17": "UDP", "0": "HOPOPT", "1": "ICMP"}

_TIMESTAMP_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M")

# The TrafficLabelling CSVs use a 12-hour clock with no AM/PM marker: capture days run
# ~08:40-17:15, and afternoon rows read "1:00".."5:xx" (e.g. Friday-...-Afternoon-DDos
# starts at "7/7/2017 3:30"). Hours below this cutoff are therefore PM (+12h);
# hours 8-12 are already correct (12:xx is noon).
_FIRST_MORNING_HOUR = 8

# Rows parsed per read_csv chunk. The chunks of one file are concatenated once, so this bounds the parser's
# working set, not the result; the result is limited to the columns below.
_CSV_CHUNK_ROWS = 250_000

# The only raw columns _to_canonical reads (everything in _COLUMN_MAP that maps to a canonical column) --
# the other ~60 of CICFlowMeter's ~85 columns are never parsed.
_USED_HEADERS = frozenset(k for k, v in _COLUMN_MAP.items() if v)

# Columns read as text. Whole-file inference made these strings; per-chunk inference would infer float64 for a
# chunk that is entirely empty in one of them (trailing blank rows in Thursday-Morning-WebAttacks). The output
# is identical either way today (checked against golden files), so this is a guard that keeps the dtype
# independent of chunk contents, not a fix for an observed difference. `protocol` is deliberately NOT here: it
# is inferred (int64) so it stringifies as "6", exactly as before.
_TEXT_HEADERS = frozenset({"source ip", "src ip", "destination ip", "dst ip", "timestamp", "label"})


def _count_data_rows(path: Path) -> int:
    """Count data lines (excluding the header) without parsing the CSV."""
    newlines, last = 0, b""
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    lines = newlines + (1 if last not in (b"", b"\n") else 0)
    return max(0, lines - 1)


def spread_sample_plan(row_counts: list[int], sample_size: int, seed: int) -> list[np.ndarray | None]:
    """Choose which data rows to read from each file for a spread-out sample.

    One global fraction ``f = sample_size / sum(row_counts)`` is applied to every
    file, so each file keeps ``round(f * n_rows)`` rows and relative time density
    across capture days is preserved. Within a file, rows are drawn uniformly
    without replacement over the WHOLE file (not its head), using a single
    ``numpy.random.default_rng(seed)`` consumed in the given (sorted) file order,
    so the selection is deterministic. Returns sorted 0-based data-row indices per
    file, or ``None`` for a file that should be read in full.
    """
    total = sum(row_counts)
    if total == 0 or sample_size >= total:
        return [None] * len(row_counts)
    frac = sample_size / total
    rng = np.random.default_rng(seed)
    plan: list[np.ndarray | None] = []
    for n in row_counts:
        k = min(n, int(round(n * frac)))
        plan.append(np.sort(rng.choice(n, size=k, replace=False)) if n else np.array([], dtype=int))
    return plan


def _normalize_header(col: str) -> str:
    return col.strip().lower()


def _read_used_columns(fpath: Path, skip) -> pd.DataFrame:
    """Read one CSV, parsing only the columns _to_canonical uses, in chunks, and concatenate once.

    Header spelling is normalised exactly as before (strip + lower). Row selection (``skip``) is the same
    callable the whole-file read used; it indexes file lines, which chunking does not change.
    """
    raw_header = pd.read_csv(fpath, nrows=0, encoding="cp1252").columns
    text_cols = {c: "str" for c in raw_header if _normalize_header(c) in _TEXT_HEADERS}
    chunks = pd.read_csv(
        fpath,
        low_memory=False,
        skiprows=skip,
        encoding="cp1252",
        usecols=lambda c: _normalize_header(c) in _USED_HEADERS,
        dtype=text_cols,
        chunksize=_CSV_CHUNK_ROWS,
    )
    parts = []
    for chunk in chunks:
        chunk.columns = [_normalize_header(c) for c in chunk.columns]
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=sorted(_USED_HEADERS))


@register("cic_ids2017")
class CicIds2017Adapter(DatasetAdapter):
    name = "cic_ids2017"

    def discover(self) -> DiscoveryReport:
        problems: list[str] = []
        if not self.raw_dir.exists():
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=[
                f"Directory does not exist: {self.raw_dir}"
            ])
        pattern = str(self.raw_dir / "**" / "*.csv")
        files = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if not files:
            return DiscoveryReport(self.name, self.raw_dir, found=False, files=[], problems=[
                f"No .csv files found under {self.raw_dir} (searched recursively)."
            ])
        for f in files:
            try:
                head = pd.read_csv(f, nrows=1)
            except Exception as exc:  # noqa: BLE001 - want a readable message, not a stack trace
                problems.append(f"{f.name}: could not be read as CSV ({exc}).")
                continue
            norm = {_normalize_header(c) for c in head.columns}
            if "label" not in norm:
                problems.append(f"{f.name}: no 'Label' column found -- is this the GeneratedLabelledFlows variant?")
        return DiscoveryReport(self.name, self.raw_dir, found=True, files=files, problems=problems)

    def download_instructions(self) -> str:
        return f"""
CIC-IDS2017 is not distributed via an anonymous direct-download link; you must
request/download it from the official CIC dataset portal.

1. Go to: {self.entry.official_url}
2. Download 'GeneratedLabelledFlows.zip' (NOT 'MachineLearningCSV.zip' -- that
   variant drops the IP/timestamp columns AegisFlow needs for per-host,
   temporal windowing).
3. Unzip it so the 8 per-day CSV files end up here:

   {self.expected_directory_structure()}

   The official archive nests them under a 'TrafficLabelling ' folder --
   either keep that folder or flatten it, AegisFlow searches recursively.

4. Verify with:

   python -m aegisflow validate-dataset --dataset cic_ids2017

Citation (cite this if you publish results):
{self.entry.citation.strip()}
""".strip()

    def load_raw(self, sample_size: int | None = None) -> pd.DataFrame:
        report = self.discover()
        if not report.found:
            raise DatasetNotFoundError(
                f"CIC-IDS2017 raw files not found.\n\n{self.download_instructions()}"
            )

        frames: list[pd.DataFrame] = []
        # Sampling: see spread_sample_plan -- same fraction from every file, rows drawn
        # uniformly across each whole file with the project seed. Never a head slice.
        plan: list[np.ndarray | None] = [None] * len(report.files)
        if sample_size is not None and report.files:
            counts = [_count_data_rows(f) for f in report.files]
            plan = spread_sample_plan(counts, sample_size, int(self.cfg.config.random_seed))
            self.log.info("spread-out sample plan", sample_size=sample_size, total_rows=sum(counts),
                          **{f.name: (len(p) if p is not None else n)
                             for f, p, n in zip(report.files, plan, counts)})

        for fpath, rows in zip(report.files, plan):
            skip = None
            if rows is not None:
                keep = set((rows + 1).tolist())  # file line 0 is the header
                skip = lambda i, keep=keep: i != 0 and i not in keep  # noqa: E731
            df = _read_used_columns(fpath, skip)
            # CICFlowMeter CSVs sometimes contain re-embedded header rows
            # (a data row whose 'label' literally reads "Label"); drop them.
            if "label" in df.columns:
                df = df[df["label"].astype(str).str.strip().str.lower() != "label"]
            df["__source_file__"] = fpath.name
            frames.append(df)

        if not frames:
            raise DatasetValidationError(f"No readable CSV rows found under {self.raw_dir}.")

        raw = pd.concat(frames, ignore_index=True, sort=False)
        canonical = self._to_canonical(raw)
        canonical = coerce_canonical_frame(canonical)
        canonical["source_dataset"] = "cic_ids2017"
        validate_canonical_frame(canonical, context="cic_ids2017.load_raw")
        return canonical

    # ------------------------------------------------------------------ #
    def _to_canonical(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = empty_canonical_frame()
        out = out.reindex(range(len(raw)))  # placeholder rows, columns already typed via schema

        rename: dict[str, str] = {}
        for col in raw.columns:
            target = _COLUMN_MAP.get(col)
            if target:
                rename[col] = target

        mapped = raw.rename(columns=rename)

        for canon_col in ["source_ip", "destination_ip", "source_port", "destination_port",
                           "flow_duration", "fwd_packet_count", "bwd_packet_count",
                           "fwd_byte_count", "bwd_byte_count", "packet_size_mean", "packet_size_std",
                           "packet_size_min", "packet_size_max", "inter_arrival_mean", "inter_arrival_std",
                           "syn_count", "ack_count", "rst_count", "fin_count", "psh_count", "urg_count",
                           "tcp_window_size", "dataset_label"]:
            if canon_col in mapped.columns:
                out[canon_col] = mapped[canon_col].values

        # protocol: numeric code -> readable name where known, else the raw code as string
        if "protocol" in mapped.columns:
            proto_str = mapped["protocol"].astype(str).str.strip()
            out["protocol"] = proto_str.map(_PROTOCOL_NAMES).fillna(proto_str)

        # timestamp: CICFlowMeter format is inconsistent across day-files; try known formats,
        # fall back to pandas' general parser, and NEVER silently drop unparsed rows here --
        # they surface as NaT and get counted/dropped explicitly during preprocessing.
        if "timestamp" in mapped.columns:
            ts_raw = mapped["timestamp"].astype(str).str.strip()
            parsed = pd.Series(pd.NaT, index=ts_raw.index, dtype="datetime64[ns]")
            remaining = ts_raw.notna()
            for fmt in _TIMESTAMP_FORMATS:
                if not remaining.any():
                    break
                try_parsed = pd.to_datetime(ts_raw[remaining], format=fmt, errors="coerce")
                filled = try_parsed.notna()
                parsed.loc[try_parsed.index[filled]] = try_parsed[filled]
                remaining.loc[try_parsed.index[filled]] = False
            if remaining.any():
                fallback = pd.to_datetime(ts_raw[remaining], errors="coerce", dayfirst=True)
                parsed.loc[fallback.index] = fallback
            pm = parsed.dt.hour < _FIRST_MORNING_HOUR
            parsed.loc[pm] = parsed.loc[pm] + pd.Timedelta(hours=12)
            out["timestamp"] = parsed

        out["packet_count"] = pd.to_numeric(out["fwd_packet_count"], errors="coerce").fillna(0) + \
            pd.to_numeric(out["bwd_packet_count"], errors="coerce").fillna(0)
        out["byte_count"] = pd.to_numeric(out["fwd_byte_count"], errors="coerce").fillna(0) + \
            pd.to_numeric(out["bwd_byte_count"], errors="coerce").fillna(0)

        # CICFlowMeter reports Flow Duration in microseconds; normalize to seconds
        # so it's comparable across adapters that report seconds natively.
        out["flow_duration"] = pd.to_numeric(out["flow_duration"], errors="coerce") / 1_000_000.0

        # ttl_*, retransmission_count are not present in CICFlowMeter's CIC-IDS2017 CSV output;
        # left as NA by empty_canonical_frame() / coerce_canonical_frame(), never fabricated.
        return out
