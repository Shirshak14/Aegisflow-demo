from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lanl_short_lead_study.py"
spec = importlib.util.spec_from_file_location("lanl_short_lead_study", SCRIPT)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _synthetic(n=24, seed=0):
    rng = np.random.default_rng(seed)
    onsets, contact, rows = {}, {}, []
    for i in range(n):
        v = f"C{i}"
        on = int((9 + i) * 86400 // 2 + 7200 + i * 977) // 60 * 60 + 17
        onsets[v] = on
        if i % 3 == 0:                       # attacker contacts 4 minutes before the logon
            c = on - 240
            contact[v] = c
            for b in range((c // S.BIN), (on - 1) // S.BIN + 1):
                rows.append({"victim": v, "bin": b, "att_flows": 5.0, "in_flows": 5.0, "ctx_flows": 5.0, "in_srcs": 1.0})
        else:
            contact[v] = -1
        for b in rng.integers((on - 86400) // S.BIN, (on - 1) // S.BIN, 50):
            rows.append({"victim": v, "bin": int(b), "own_flows": float(rng.integers(1, 5))})
    bins = pd.DataFrame(rows).fillna(0.0)
    for c in S.ALL:
        if c not in bins:
            bins[c] = 0.0
    bins = bins.groupby(["victim", "bin"], as_index=False).sum()
    return bins, pd.Series(onsets), pd.Series(contact)


def test_pre_flag_is_causal_and_strict():
    bins, on, ct = _synthetic()
    A = S.anchors(bins, on, ct)
    assert (A["t"] <= A["onset"]).all()
    v = "C0"
    m = A["victim"] == v
    c = ct[v]
    vis = (c // S.BIN + 1) * S.BIN
    assert A["pre"][m][A["t"][m] < vis].all()
    assert not A["pre"][m][A["t"][m] >= vis].any()
    assert A["pre"][A["victim"] == "C1"].all()        # never contacted: pre-contact throughout
    # attacker features are zero on every pre-contact anchor of a contacted victim
    j = S.ALL.index("att_flows")
    assert A["X"][m & A["pre"], :, j].sum() == 0


def test_contact_lead_summary():
    _, on, ct = _synthetic()
    s = S.contact_lead_summary(on, ct)
    assert s["contacted"] == 8 and s["lead_ge_min"]["2"] == 8 and s["lead_ge_min"]["5"] == 0


def test_study_runs_end_to_end(tmp_path, monkeypatch):
    bins, on, ct = _synthetic()
    bp, cp = tmp_path / "b.parquet", tmp_path / "c.parquet"
    bins.to_parquet(bp, index=False)
    pd.DataFrame({"victim": on.index, "onset": on.to_numpy(), "contact": ct.to_numpy()}).to_parquet(cp, index=False)
    monkeypatch.setattr(S, "N_PSEUDO", 5)
    monkeypatch.setattr(S, "NEG_SUBSAMPLE", 0.2)
    res = S.study(1, tmp_path / "r.json", bins_path=bp, contacts_path=cp, horizons=(5,))
    assert "detect|logistic|own+inbound+attacker|H5" in res["variants"]
    d = res["variants"]["detect|gboost|own+inbound+attacker|H5"]["verdict"]
    assert d["contacted_onsets"] > 0
