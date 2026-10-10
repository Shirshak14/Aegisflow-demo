"""
Read single host files out of the corrected DARPA OpTC release without downloading it.

The corrected release (Majorczyk et al., doi:10.57745/UXCWOC, CC BY 4.0) is ten
uncompressed tar archives, one per day (2019-09-16 .. 2019-09-25, about 940 GB in all).
Each holds one ``AIA-XXX-YYY/AIA-XXX-YYY.ecar-YYYY-MM-DD-sysclient0ZZZ.json.gz`` per
host. The object store honours HTTP Range requests, so this script walks the tar
headers (one 512-byte read per member) to build an index, and then fetches only
the members it is asked for.

    python scripts/optc_remote.py index  [--days 2019-09-16 ...]      -> reports/optc_tar_index.json
    python scripts/optc_remote.py fetch   --hosts 201 402 --days 2019-09-23      -> data/raw/optc/ecar
    python scripts/optc_remote.py extract --hosts 201 --days 2019-09-23 [--until 16:30:00] -> data/raw/optc/flows

Raw files go to data/raw/optc (gitignored). Nothing here is needed to run AegisFlow.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import zlib
from pathlib import Path

import requests

ACCESS = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/{id}"
DATAFILES = {  # day -> Dataverse datafile id (from the dataset's file list)
    "2019-09-16": 713542, "2019-09-17": 713556, "2019-09-18": 713566, "2019-09-19": 713570,
    "2019-09-20": 713571, "2019-09-21": 713572, "2019-09-22": 713573, "2019-09-23": 713574,
    "2019-09-24": 713575, "2019-09-25": 713576,
}
ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports" / "optc_tar_index.json"
HOST_RE = re.compile(r"sysclient(\d{4})\.json\.gz$", re.IGNORECASE)


def _signed_url(session: requests.Session, file_id: int) -> str:
    r = session.get(ACCESS.format(id=file_id), allow_redirects=False, timeout=60)
    if r.status_code not in (301, 302, 303, 307):
        raise RuntimeError(f"datafile {file_id}: expected a redirect, got {r.status_code}")
    return r.headers["location"]  # presigned, valid for 4 hours


def _get_range(session: requests.Session, url: str, start: int, length: int) -> bytes:
    for attempt in range(5):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=120)
            if r.status_code == 206:
                return r.content
            raise RuntimeError(f"HTTP {r.status_code}")
        except (requests.RequestException, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def _octal(field: bytes) -> int:
    field = field.rstrip(b"\x00 ").strip()
    if field and field[0] & 0x80:  # GNU base-256 for very large sizes
        return int.from_bytes(field[1:], "big")
    return int(field or b"0", 8)


def index_tar(session: requests.Session, day: str) -> list[dict]:
    url = _signed_url(session, DATAFILES[day])
    started = time.time()
    members: list[dict] = []
    offset, long_name = 0, None
    while True:
        if time.time() - started > 3.5 * 3600:  # presigned URL lifetime is 4 h
            url, started = _signed_url(session, DATAFILES[day]), time.time()
        hdr = _get_range(session, url, offset, 512)
        if len(hdr) < 512 or hdr == b"\x00" * 512:
            break
        size = _octal(hdr[124:136])
        typeflag = hdr[156:157]
        data_start = offset + 512
        if typeflag == b"L":  # GNU long name: the name is the member's data
            long_name = _get_range(session, url, data_start, size).rstrip(b"\x00").decode()
        else:
            prefix = hdr[345:500].rstrip(b"\x00").decode()
            short = hdr[0:100].rstrip(b"\x00").decode()
            name = long_name or (f"{prefix}/{short}" if prefix else short)
            long_name = None
            if typeflag in (b"0", b"\x00"):
                members.append({"name": name, "offset": data_start, "size": size})
        offset = data_start + (size + 511) // 512 * 512
    return members


def cmd_index(days: list[str]) -> None:
    idx = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    with requests.Session() as s:
        for day in days:
            if day in idx:
                continue
            members = index_tar(s, day)
            idx[day] = members
            INDEX.write_text(json.dumps(idx, indent=1))
            print(day, len(members), "members", round(sum(m["size"] for m in members) / 1e9, 2), "GB", flush=True)


def cmd_fetch(hosts: list[int], days: list[str], out: Path) -> None:
    idx = json.loads(INDEX.read_text())
    out.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        for day in days:
            wanted = [m for m in idx[day] if (h := HOST_RE.search(m["name"])) and int(h.group(1)) in hosts]
            url = _signed_url(s, DATAFILES[day])
            for m in wanted:
                dest = out / day / Path(m["name"]).name
                if dest.exists() and dest.stat().st_size == m["size"]:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                with tmp.open("wb") as fh:
                    pos, chunk = m["offset"], 64 << 20
                    while pos < m["offset"] + m["size"]:
                        n = min(chunk, m["offset"] + m["size"] - pos)
                        fh.write(_get_range(s, url, pos, n))
                        pos += n
                tmp.replace(dest)
                print("fetched", dest, m["size"], flush=True)


def _stream_lines(session: requests.Session, url: str, offset: int, size: int, chunk: int = 8 << 20):
    """Yield decoded lines of a gzip member inside the tar, fetched chunk by chunk."""
    dec, buf, pos, end = zlib.decompressobj(wbits=31), b"", offset, offset + size
    while pos < end:
        n = min(chunk, end - pos)
        data = _get_range(session, url, pos, n)
        pos += n
        while data:
            out = dec.decompress(data)
            data = dec.unused_data
            if data:  # concatenated gzip members
                out += dec.flush()
                dec = zlib.decompressobj(wbits=31)
            buf += out
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line:
                    yield line
    if buf.strip():
        yield buf


def cmd_extract(hosts: list[int], days: list[str], out: Path, until: dict[tuple[int, str], str]) -> None:
    """Keep only FLOW events (still eCAR JSON lines) of each host-day; optionally stop at a time.

    Host-day files are sorted by timestamp, so stopping early is exact. The output
    ``<day>/<member>.flows.json.gz`` is read by aegisflow/ml/datasets/optc.py.
    """
    idx = json.loads(INDEX.read_text())
    with requests.Session() as s:
        for day in days:
            wanted = [m for m in idx.get(day, []) if (h := HOST_RE.search(m["name"])) and int(h.group(1)) in hosts]
            url, started = _signed_url(s, DATAFILES[day]), time.time()
            for m in wanted:
                host = int(HOST_RE.search(m["name"]).group(1))
                stop = until.get((host, day))
                dest = out / day / (Path(m["name"]).name.replace(".json.gz", "") + ".flows.json.gz")
                if dest.exists():
                    continue
                if time.time() - started > 3 * 3600:
                    url, started = _signed_url(s, DATAFILES[day]), time.time()
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                kept = total = 0
                chunk = 8 << 20
                with gzip.open(tmp, "wb") as fh:
                    for line in _stream_lines(s, url, m["offset"], m["size"], chunk):
                        total += 1
                        if b'"object": "FLOW"' not in line and b'"object":"FLOW"' not in line:
                            if stop and total % 5000 == 0 and json.loads(line).get("timestamp", "") > stop:
                                break
                            continue
                        if stop and json.loads(line).get("timestamp", "") > stop:
                            break
                        fh.write(line + b"\n")
                        kept += 1
                tmp.replace(dest)
                print("extracted", dest.name, "events", total, "flows", kept, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--days", nargs="*", default=list(DATAFILES))
    pf = sub.add_parser("fetch")
    pf.add_argument("--hosts", nargs="+", type=int, required=True)
    pf.add_argument("--days", nargs="+", required=True)
    pf.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "optc" / "ecar")
    px = sub.add_parser("extract", help="stream host-days, keep FLOW events only")
    px.add_argument("--hosts", nargs="+", type=int, required=True)
    px.add_argument("--days", nargs="+", required=True)
    px.add_argument("--until", help="local time of day HH:MM:SS to stop each file at (default: whole day)")
    px.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "optc" / "flows")
    a = ap.parse_args(argv)
    if a.cmd == "index":
        cmd_index(a.days)
    elif a.cmd == "fetch":
        cmd_fetch(a.hosts, a.days, a.out)
    else:
        until = {(h, d): f"{d}T{a.until}" for h in a.hosts for d in a.days} if a.until else {}
        cmd_extract(a.hosts, a.days, a.out, until)


if __name__ == "__main__":
    sys.exit(main())
