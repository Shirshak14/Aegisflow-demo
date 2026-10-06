"""Temporal graph neural network (opt-in): host-communication graph per window + GRU over time.

For every time window the flows define a communication graph: nodes are hosts, and an edge joins two
hosts that exchanged at least one flow in that window (either direction). Each host-window node carries
its 28 traffic features (the same ``host_windows`` row the LSTM sees). For a host sequence of L
windows, the model sees at every step:

  x_t    the host's own window features
  n_t    the mean features of its graph neighbours in that window (neighbours that sent traffic, so
         have a host-window row); zeros when there are none
  d_t    log(1 + number of distinct neighbours)

and computes a GraphSAGE-style (mean aggregator) layer per step,
  h_t = ReLU(W_self x_t + W_nbr n_t + W_deg d_t),
followed by a GRU over h_1..h_L and a linear attack head (same target, split, class weighting, early
stopping and validation-selected threshold as the LSTM). Neighbour features use their own train-fit
preprocessor. The graph adds information the per-host models cannot see: who a host talked to and
what those peers were doing in the same window.

Limit: on CIC-IDS2017 the graph is dominated by the same attacker host (172.16.0.1) as everything
else, so the GNN can learn the same shortcut through its neighbours.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .modeling import (MODEL_FEATURES, ForecastDataset, SequencePreprocessor, metrics_at, positive_class_weight,
                       select_threshold)


@dataclass
class GraphSequences:
    base: ForecastDataset
    neighbours: np.ndarray   # (n, L, F) raw mean neighbour features
    degree: np.ndarray       # (n, L, 1) distinct neighbours


def window_neighbours(flows: pd.DataFrame, windows: pd.DataFrame, window_size_seconds: float) -> dict:
    """{(host, window_start): set(neighbour hosts)} for every host-window row, from flows in [start, start+w)."""
    w = windows[["host_id", "window_start"]].drop_duplicates()
    starts = np.sort(pd.to_datetime(w["window_start"].unique()))
    f = flows[["timestamp", "source_ip", "destination_ip"]].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f.sort_values("timestamp")
    ts = f["timestamp"].to_numpy()
    src = f["source_ip"].astype(str).to_numpy(); dst = f["destination_ip"].astype(str).to_numpy()
    width = np.timedelta64(int(window_size_seconds * 1e9), "ns")
    out: dict = {}
    for s in starts:
        lo, hi = np.searchsorted(ts, s, "left"), np.searchsorted(ts, s + width, "left")
        nb: dict[str, set] = {}
        for a, b in zip(src[lo:hi], dst[lo:hi]):
            if a != b:
                nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
        for host, peers in nb.items():
            out[(host, pd.Timestamp(s))] = peers
    return out


def build_graph_sequences(sequences: pd.DataFrame, windows: pd.DataFrame, flows: pd.DataFrame,
                          window_size_seconds: float = 60.0) -> GraphSequences:
    base = ForecastDataset.from_frame(sequences)
    names = base.feature_names
    L = base.X.shape[1]
    win = windows.copy()
    win["window_start"] = pd.to_datetime(win["window_start"]); win["window_end"] = pd.to_datetime(win["window_end"])
    win["host_id"] = win["host_id"].astype(str)
    win[names] = win[names].fillna(0.0)
    feats = {(h, s): v for h, s, v in zip(win.host_id, win.window_start, win[names].to_numpy(np.float32))}
    nbrs = window_neighbours(flows, win, window_size_seconds)
    by_host = {h: (g["window_start"].to_numpy(), g["window_end"].to_numpy())
               for h, g in win.sort_values("window_start").groupby("host_id", sort=False)}
    n = len(base.frame)
    NX = np.zeros((n, L, len(names)), np.float32); D = np.zeros((n, L, 1), np.float32)
    seq_start = pd.to_datetime(base.frame["seq_start_time"]).to_numpy()
    seq_end = pd.to_datetime(base.frame["seq_end_time"]).to_numpy()
    for i, (host, s0, s1) in enumerate(zip(base.frame["host_id"].astype(str), seq_start, seq_end)):
        if host not in by_host:
            raise ValueError(f"no host windows for sequence host {host}")
        starts, ends = by_host[host]
        lo = np.searchsorted(starts, s0, "left")
        hi = np.searchsorted(ends, s1, "right")  # windows are sorted by start; ends follow the same order
        steps = starts[lo:hi]
        if len(steps) != L:
            raise ValueError(f"sequence {i} ({host}) maps to {len(steps)} windows, expected {L}; "
                             "are windows and sequences from the same preprocess run?")
        for t, ws in enumerate(steps):
            ws = pd.Timestamp(ws)
            peers = nbrs.get((host, ws), ())
            vecs = [feats[(p, ws)] for p in peers if (p, ws) in feats]
            if vecs:
                NX[i, t] = np.mean(vecs, axis=0)
            D[i, t, 0] = len(peers)
    return GraphSequences(base, NX, D)


class TemporalGNNForecaster:
    def __init__(self, feature_count: int, hidden_size: int = 32, dropout: float = 0.2):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_lin = nn.Linear(feature_count, hidden_size)
                self.nbr_lin = nn.Linear(feature_count, hidden_size, bias=False)
                self.deg_lin = nn.Linear(1, hidden_size, bias=False)
                self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout); self.head = nn.Linear(hidden_size, 1)

            def forward(self, x, nx, deg):
                h = torch.relu(self.self_lin(x) + self.nbr_lin(nx) + self.deg_lin(torch.log1p(deg)))
                _, z = self.gru(h)
                return self.head(self.dropout(z[-1])).squeeze(-1)
        self.net = Net()


def train_tgnn(data: GraphSequences, output: Path, *, seed: int = 42, epochs: int = 20, batch_size: int = 128,
               learning_rate: float = 0.001, hidden_size: int = 32, dropout: float = 0.2, patience: int = 4,
               dataset: str = "cic_ids2017") -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    b = data.base
    tr, va, te = (b.indices(s) for s in ("train", "val", "test"))
    if len(tr) == 0:
        raise ValueError("no training sequences")
    prep = SequencePreprocessor(b.feature_names).fit(b.X[tr])
    nprep = SequencePreprocessor(b.feature_names).fit(data.neighbours[tr])
    X, NX, D = prep.transform(b.X), nprep.transform(data.neighbours), data.degree.astype(np.float32)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"self": prep, "neighbours": nprep}, output / "preprocessor.joblib")
    pos_weight, counts = positive_class_weight(b.attack[tr])
    net = TemporalGNNForecaster(X.shape[-1], hidden_size, dropout).net
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    T = lambda idx: (torch.tensor(X[idx]), torch.tensor(NX[idx]), torch.tensor(D[idx]))
    loader = DataLoader(TensorDataset(*T(tr), torch.tensor(b.attack[tr], dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    vin = T(va); vy = torch.tensor(b.attack[va], dtype=torch.float32)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(epochs):
        net.train(); losses = []
        for x, nx, d, y in loader:
            opt.zero_grad(); loss = loss_fn(net(x, nx, d), y); loss.backward(); opt.step(); losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(*vin), vy).item()) if len(va) else float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": vl})
        if vl < best:
            best, best_epoch, stale = vl, epoch + 1, 0
            torch.save(net.state_dict(), output / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(torch.load(output / "best.pt", weights_only=True)); net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(*vin)).numpy() if len(va) else np.array([])
        pt = torch.sigmoid(net(*T(te))).numpy()
    threshold, reason = select_threshold(b.attack[va], pv)
    training = {"seed": seed, "epochs_requested": epochs, "epochs_run": len(history), "batch_size": batch_size,
                "learning_rate": learning_rate, "hidden_size": hidden_size, "dropout": dropout, "patience": patience,
                "best_checkpoint_epoch": best_epoch, "checkpoint_selection": "minimum validation BCEWithLogitsLoss",
                "model_type": "temporal_gnn"}
    results = {"dataset": dataset, "model": "temporal_gnn", "threshold": threshold, "threshold_selection": reason,
               "splits": {n: {"sequences": int(len(i)), "positive_targets": int(b.attack[i].sum())}
                          for n, i in (("train", tr), ("val", va), ("test", te))},
               "models": {"temporal_gnn": metrics_at(b.attack[te], pt, threshold)} if len(te) else {},
               "graph": {"mean_degree_train": float(data.degree[tr].mean()),
                         "share_steps_with_neighbour_features": float((np.abs(data.neighbours[tr]).sum(-1) > 0).mean())},
               "training": training, "history": history}
    meta = {"model_type": "temporal_gnn", "feature_names": b.feature_names, "input_shape": list(b.X.shape[1:]),
            "threshold": threshold, "class_counts": counts, "training_config": training}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def predict_tgnn(data: GraphSequences, model_dir: str | Path) -> np.ndarray:
    import torch
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("model_type") != "temporal_gnn":
        raise ValueError(f"{model_dir} is not a temporal GNN model directory")
    preps = joblib.load(model_dir / "preprocessor.joblib")
    tc = meta["training_config"]
    net = TemporalGNNForecaster(len(meta["feature_names"]), tc["hidden_size"], tc["dropout"]).net
    net.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(preps["self"].transform(data.base.X)),
                                 torch.tensor(preps["neighbours"].transform(data.neighbours)),
                                 torch.tensor(data.degree))).numpy()
