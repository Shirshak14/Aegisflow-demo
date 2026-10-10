import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("nv", ROOT / "scripts" / "lanl_next_victim_study.py")
nv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nv)


def _write(raw: Path, contact_lead: int = 1200):
    """60 hosts with random benign flows; 3 bursts of 4 victims each, 3 minutes apart; half contacted first."""
    rng = np.random.default_rng(0)
    hosts = [f"C{i}" for i in range(60)]
    n, span = 20000, 6 * 86400
    t = np.sort(rng.integers(1, span, n))
    rows = [(int(tt), 1, hosts[a], "N1", hosts[b], "80", "6", 3, 100)
            for tt, a, b in zip(t, rng.integers(0, 60, n), rng.integers(0, 60, n)) if a != b]
    rt = []
    victims = iter(rng.choice(60, 12, replace=False))
    for day in (2, 3, 4):
        for k in range(4):
            v, on = hosts[next(victims)], day * 86400 + 5000 + 180 * k
            rt.append((on, "U1@DOM1", "C9999", v))
            if k % 2 == 0:
                rows.append((on - contact_lead, 1, "C9999", "N2", v, "445", "6", 5, 500))
    rows.sort()
    raw.mkdir(parents=True)
    with gzip.open(raw / "flows.txt.gz", "wt") as f:
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)
    with gzip.open(raw / "redteam.txt.gz", "wt") as f:
        f.writelines(",".join(map(str, r)) + "\n" for r in sorted(rt))


def test_decision_grid_only_while_active():
    on = np.array([1000, 1200, 50_000])
    grid, burst = nv.decision_grid(on)
    assert grid.min() >= 1000 and (grid % nv.GRID == 0).all()
    last = np.searchsorted(on, grid, side="right") - 1
    assert ((grid - on[last]) <= nv.ACTIVE).all()
    assert set(burst) == {0, 1}


def test_extract_and_rows_are_causal(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write(raw)
    nv.extract(raw, out, chunksize=5000)
    T = nv.load_tables(raw, out)
    R, dec, meta = nv.build_rows(T)
    assert meta["victims"] == 12 and meta["victims_in_flows"] == 12
    assert not (R["onset"] <= R["tau"]).any()             # compromised hosts are never candidates
    assert "C9999" not in set(T["hosts"]["name"].to_numpy()[R["cand"].unique()])
    # contact is visible only once the attacker's minute bin has ended
    contacted = R[R["contact_7d"] > 0]
    assert len(contacted) and (contacted["onset"] - contacted["tau"] < 1200).all()
    c = nv.contact_precision(T)
    assert c["hosts_contacted"] == 6 and c["later_compromised"] == 6


def test_contact_ranker_finds_contacted_victims(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write(raw)
    nv.extract(raw, out, chunksize=5000)
    R, _, _ = nv.build_rows(nv.load_tables(raw, out))
    y = ((R["onset"] > R["tau"]) & (R["onset"] <= R["tau"] + 600)).to_numpy().astype(int)
    res = nv.evaluate(R, y, nv.rank_scores(R, "R-CONTACT"), 42, 600)
    assert res["scored_bursts"] == 3
    assert res["mean_burst_hit10"] > res["null_mean_hit10"]
