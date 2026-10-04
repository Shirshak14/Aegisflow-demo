"""Score new, unlabeled traffic (e.g. a PCAP) with a trained forecasting model.

flows (canonical schema) -> clean -> flow features -> host windows (same config as training)
-> inference sequences (the last ``sequence_length`` windows of each host, sliding; no future target
is needed) -> preprocessor + model saved by ``train`` -> attack probability for the window after each
sequence.

Training-time sequences (``build_host_sequences``) need a future target window; scoring live or
captured traffic must not, so the newest windows of every host are scored too.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import AegisFlowConfig
from .features.flow_features import compute_flow_features
from .modeling import LSTMForecaster, MODEL_FEATURES
from .preprocessing.cleaning import clean_canonical_frame
from .temporal.windowing import aggregate_host_windows

# Flow-level extras that CICFlowMeter CSVs lack but packet captures provide; summarised per scored
# sequence for analysts. They are NOT model inputs (the model was trained without them).
PACKET_EXTRAS = ("ttl_mean", "ttl_std", "retransmission_count")


def build_inference_sequences(windows: pd.DataFrame, sequence_length: int,
                              feature_names: list[str] | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Every run of ``sequence_length`` consecutive windows per host (no future target required)."""
    names = list(feature_names or MODEL_FEATURES)
    if windows.empty:
        return np.empty((0, sequence_length, len(names)), np.float32), pd.DataFrame()
    missing = set(names) - set(windows.columns)
    if missing:
        raise ValueError(f"windows missing feature columns: {sorted(missing)}")
    w = windows.copy()
    w[names] = w[names].fillna(0.0)  # same NA policy as build_host_sequences
    X, meta = [], []
    for host, g in w.sort_values(["host_id", "window_start"]).groupby("host_id", sort=False):
        feats = g[names].to_numpy(np.float32)
        starts, ends = g["window_start"].to_numpy(), g["window_end"].to_numpy()
        for i in range(0, len(g) - sequence_length + 1):
            X.append(feats[i:i + sequence_length])
            meta.append({"sequence_id": f"{host}_seq_{i}", "host_id": str(host),
                         "seq_start_time": pd.Timestamp(starts[i]),
                         "seq_end_time": pd.Timestamp(ends[i + sequence_length - 1])})
    if not X:
        return np.empty((0, sequence_length, len(names)), np.float32), pd.DataFrame()
    return np.stack(X), pd.DataFrame(meta)


def _host_extras(flows: pd.DataFrame, meta: pd.DataFrame) -> list[dict[str, Any]]:
    present = [c for c in PACKET_EXTRAS if c in flows.columns and flows[c].notna().any()]
    if not present or meta.empty:
        return [{} for _ in range(len(meta))]
    f = flows.assign(source_ip=flows["source_ip"].astype(str))
    out = []
    for r in meta.itertuples(index=False):
        sel = f[(f.source_ip == r.host_id) & (f.timestamp >= r.seq_start_time) & (f.timestamp < r.seq_end_time)]
        d = {}
        if "ttl_mean" in present:
            d["ttl_mean"] = float(sel["ttl_mean"].mean()) if sel["ttl_mean"].notna().any() else None
        if "ttl_std" in present:
            # variance of TTL across ALL packets the host sent in the sequence span, pooled per flow
            d["ttl_std_mean"] = float(sel["ttl_std"].mean()) if sel["ttl_std"].notna().any() else None
        if "retransmission_count" in present:
            d["retransmissions"] = float(sel["retransmission_count"].sum(min_count=1)) \
                if sel["retransmission_count"].notna().any() else None
        out.append(d)
    return out


def score_flows(flows: pd.DataFrame, model_dir: str | Path, cfg: AegisFlowConfig, *,
                threshold: float | None = None) -> pd.DataFrame:
    """Return one row per scored sequence: host, time span, attack probability, alert flag, packet extras."""
    import torch
    model_dir = Path(model_dir)
    meta_json = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    names = list(meta_json["feature_names"])
    L = int(meta_json["input_shape"][0])
    flows = flows.copy()
    for col in ("normalized_attack_class", "attack_stage"):
        if col not in flows.columns:
            flows[col] = pd.Series(pd.NA, index=flows.index, dtype="string")  # unlabeled traffic
    cleaned, _ = clean_canonical_frame(flows)
    windows = aggregate_host_windows(compute_flow_features(cleaned), cfg=cfg)
    X, meta = build_inference_sequences(windows, L, names)
    cols = ["sequence_id", "host_id", "seq_start_time", "seq_end_time", "attack_probability", "predicted_attack",
            "threshold"]
    if len(X) == 0:
        return pd.DataFrame(columns=cols)
    prep = joblib.load(model_dir / "preprocessor.joblib")
    tc = meta_json["training_config"]
    if tc.get("model_type", "lstm") != "lstm":
        raise ValueError("score_flows supports the LSTM forecaster (model_type 'lstm')")
    net = LSTMForecaster(len(names), tc["hidden_size"], tc["dropout"]).net
    net.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)); net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(prep.transform(X)))).numpy().astype(float)
    t = float(meta_json["threshold"] if threshold is None else threshold)
    out = meta.assign(attack_probability=p, predicted_attack=p >= t, threshold=t)
    extras = pd.DataFrame(_host_extras(cleaned, meta), index=out.index)
    return pd.concat([out, extras], axis=1)
