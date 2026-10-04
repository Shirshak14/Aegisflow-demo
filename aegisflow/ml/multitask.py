"""Multi-task forecaster: attack in the next window, its attack stage, and K-step future network state.

One LSTM encoder over the 10-window input sequence feeds three heads:

  attack  binary logit, attack present in the horizon-1 target window (same target as the Phase 3 LSTM)
  stage   softmax over ``stages.yaml: stages_order`` (Benign, Reconnaissance, ..., Impact) for the
          horizon-1 target window's dominant stage. Stages are the documented label-to-stage PROXY.
  state   regression of the 28 traffic features of each of the next K target windows (K-step future
          network state), in preprocessed units (signed log1p + robust scaling, train-fit).

The K future windows follow the same definition as ``build_host_sequences``: horizon k is the k-th
same-host window whose start is strictly after the last input window ends. They are looked up from
``host_windows.parquet`` (``multi_horizon_targets``); a missing horizon (host stops sending) is masked
out of the loss and the metrics.

Evaluation is honest by construction: stage accuracy is reported overall AND on attack-positive targets
only (benign dominates), next to a majority-class baseline; future-state error is reported next to a
persistence baseline (the last input window). Nothing here is tuned on test.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .modeling import (MODEL_FEATURES, ForecastDataset, SequencePreprocessor, metrics_at,
                       positive_class_weight, select_threshold)
from .temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

DEFAULT_STAGES = ("Benign", "Reconnaissance", "Initial Access", "Lateral Movement",
                  "Command and Control", "Exfiltration", "Impact")


# ---------------------------------------------------------------------------------------------- targets
def multi_horizon_targets(sequences: pd.DataFrame, windows: pd.DataFrame, horizons: int,
                          feature_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw feature vectors of the next ``horizons`` eligible same-host windows for each sequence.

    Returns (targets (n, K, F) float32 with NaN where missing, mask (n, K) bool).
    """
    names = list(feature_names or MODEL_FEATURES)
    if horizons < 1:
        raise ValueError("horizons must be >= 1")
    missing = set(names) - set(windows.columns)
    if missing:
        raise ValueError(f"host windows missing feature columns: {sorted(missing)}")
    n = len(sequences)
    out = np.full((n, horizons, len(names)), np.nan, dtype=np.float32)
    mask = np.zeros((n, horizons), dtype=bool)
    w = windows.assign(window_start=pd.to_datetime(windows["window_start"]))
    w[names] = w[names].fillna(0.0)  # same NA policy as build_host_sequences
    groups = {str(h): g.sort_values("window_start") for h, g in w.groupby("host_id", sort=False)}
    seq_end = pd.to_datetime(sequences["seq_end_time"]).to_numpy()
    hosts = sequences["host_id"].astype(str).to_numpy()
    for host in np.unique(hosts):
        g = groups.get(host)
        if g is None:
            continue
        starts = g["window_start"].to_numpy()
        feats = g[names].to_numpy(np.float32)
        rows = np.flatnonzero(hosts == host)
        first = np.searchsorted(starts, seq_end[rows], side="right")  # first start strictly > seq_end
        for k in range(horizons):
            idx = first + k
            ok = idx < len(starts)
            out[rows[ok], k] = feats[idx[ok]]
            mask[rows[ok], k] = True
    return out, mask


@dataclass
class MultiTaskData:
    base: ForecastDataset
    stage_index: np.ndarray        # (n,) int, index into stages
    stages: list[str]
    future: np.ndarray             # (n, K, F) raw, NaN where missing
    future_mask: np.ndarray        # (n, K) bool

    @classmethod
    def build(cls, sequences: pd.DataFrame, windows: pd.DataFrame | None, horizons: int = 1,
              stages: list[str] | tuple[str, ...] = DEFAULT_STAGES) -> "MultiTaskData":
        base = ForecastDataset.from_frame(sequences)
        stages = list(stages)
        unknown = sorted(set(base.stages) - set(stages))
        if unknown:
            raise ValueError(f"target stages not in stages_order: {unknown}")
        stage_index = np.array([stages.index(s) for s in base.stages], dtype=np.int64)
        if windows is None:
            if horizons != 1:
                raise ValueError("horizons > 1 needs host_windows.parquet")
            raw = list(STANDARD_NUMERIC_WINDOW_FEATURES)
            cols = [raw.index(n) for n in base.feature_names]
            future = base.future_state[:, cols][:, None, :]
            mask = np.ones((len(base.frame), 1), dtype=bool)
        else:
            future, mask = multi_horizon_targets(base.frame, windows, horizons, base.feature_names)
            # Horizon 1 must agree with the sequence table's own target window.
            raw = list(STANDARD_NUMERIC_WINDOW_FEATURES)
            expect = base.future_state[:, [raw.index(n) for n in base.feature_names]]
            if mask[:, 0].any() and not np.allclose(future[mask[:, 0], 0], expect[mask[:, 0]], equal_nan=True, rtol=1e-4,
                                                    atol=1e-3):
                raise ValueError("host windows disagree with the sequence table's horizon-1 targets; "
                                 "were they produced by the same preprocess run?")
        return cls(base, stage_index, stages, future.astype(np.float32), mask)

    @classmethod
    def from_dir(cls, processed_dir: str | Path, horizons: int = 1,
                 stages: list[str] | tuple[str, ...] = DEFAULT_STAGES) -> "MultiTaskData":
        d = Path(processed_dir)
        seq = pd.read_parquet(d / "sequences.parquet")
        windows = pd.read_parquet(d / "host_windows.parquet") if horizons > 1 else None
        return cls.build(seq, windows, horizons, stages)


# ---------------------------------------------------------------------------------------------- model
class MultiTaskForecaster:
    def __init__(self, feature_count: int, n_stages: int, horizons: int, hidden_size: int = 32,
                 dropout: float = 0.2):
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.attack = nn.Linear(hidden_size, 1)
                self.stage = nn.Linear(hidden_size, n_stages)
                self.state = nn.Linear(hidden_size, horizons * feature_count)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                z = self.dropout(h[-1])
                return (self.attack(z).squeeze(-1), self.stage(z),
                        self.state(z).view(-1, horizons, feature_count))
        self.net = Net()


def inverse_transform(prep: SequencePreprocessor, x: np.ndarray) -> np.ndarray:
    """Map preprocessed feature vectors (..., F) back to raw traffic units."""
    shape = x.shape
    raw = prep.scaler.inverse_transform(np.asarray(x, dtype=np.float64).reshape(-1, shape[-1])).reshape(shape)
    for i, name in enumerate(prep.feature_names):
        if name in prep.LOG_FEATURES:
            raw[..., i] = np.sign(raw[..., i]) * np.expm1(np.abs(raw[..., i]))
    return raw


def _stage_metrics(y: np.ndarray, pred: np.ndarray, stages: list[str], attack: np.ndarray,
                   train_y: np.ndarray) -> dict[str, Any]:
    present = sorted(set(y.tolist()) | set(pred.tolist()))
    majority = int(np.bincount(train_y, minlength=len(stages)).argmax()) if len(train_y) else 0
    pos = attack == 1
    out = {"labels": [stages[i] for i in present],
           "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
           "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)) if len(y) else None,
           "attack_target_count": int(pos.sum()),
           "attack_stage_accuracy": float(accuracy_score(y[pos], pred[pos])) if pos.any() else None,
           "majority_baseline": {"stage": stages[majority],
                                 "accuracy": float(np.mean(y == majority)) if len(y) else None,
                                 "attack_stage_accuracy": float(np.mean(y[pos] == majority)) if pos.any() else None},
           "confusion_matrix": confusion_matrix(y, pred, labels=present).tolist() if len(y) else [],
           "support": {stages[i]: int(np.sum(y == i)) for i in range(len(stages))},
           "train_support": {stages[i]: int(np.sum(train_y == i)) for i in range(len(stages))}}
    absent = [s for s, c in out["train_support"].items() if c == 0]
    if absent:
        out["note"] = f"No training examples for {absent}; the model cannot learn to predict these stages."
    return out


def _state_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, last_input: np.ndarray) -> dict[str, Any]:
    per = []
    for k in range(target.shape[1]):
        m = mask[:, k]
        if not m.any():
            per.append({"horizon": k + 1, "count": 0}); continue
        err = pred[m, k] - target[m, k]; perr = last_input[m] - target[m, k]
        per.append({"horizon": k + 1, "count": int(m.sum()),
                    "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "persistence_mae": float(np.mean(np.abs(perr))),
                    "persistence_rmse": float(np.sqrt(np.mean(perr ** 2)))})
    return {"units": "preprocessed (signed log1p + train-fit robust scaling), averaged over the 28 features",
            "baseline": "persistence: the last input window repeated", "per_horizon": per}


def train_multitask(data: MultiTaskData, output: Path, *, seed: int = 42, epochs: int = 20, batch_size: int = 128,
                    learning_rate: float = 0.001, hidden_size: int = 32, dropout: float = 0.2, patience: int = 4,
                    stage_weight: float = 1.0, state_weight: float = 0.5, dataset: str = "cic_ids2017") -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    b = data.base
    tr, va, te = (b.indices(s) for s in ("train", "val", "test"))
    if len(tr) == 0:
        raise ValueError("no training sequences")
    prep = SequencePreprocessor(b.feature_names).fit(b.X[tr])
    X = prep.transform(b.X)
    K, F = data.future.shape[1], data.future.shape[2]
    fut = np.where(data.future_mask[..., None], prep.transform(np.nan_to_num(data.future)), 0.0).astype(np.float32)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    prep.save(output / "preprocessor.joblib")

    pos_weight, counts = positive_class_weight(b.attack[tr])
    n_st = len(data.stages)
    freq = np.bincount(data.stage_index[tr], minlength=n_st).astype(np.float64)
    # Inverse-frequency stage weights (clipped) so rare attack stages are not ignored; unseen stages get 0.
    sw = np.where(freq > 0, freq.sum() / (n_st * np.maximum(freq, 1)), 0.0)
    sw = np.clip(sw, 0, 100.0)

    net = MultiTaskForecaster(F, n_st, K, hidden_size, dropout).net
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(sw, dtype=torch.float32))

    def loss_fn(out, ya, ys, yf, m):
        a, s, f = out
        mse = ((f - yf) ** 2).mean(dim=-1)
        state = (mse * m).sum() / m.sum().clamp(min=1.0)
        return bce(a, ya) + stage_weight * ce(s, ys) + state_weight * state

    def tensors(idx):
        return (torch.tensor(X[idx]), torch.tensor(b.attack[idx], dtype=torch.float32),
                torch.tensor(data.stage_index[idx]), torch.tensor(fut[idx]),
                torch.tensor(data.future_mask[idx], dtype=torch.float32))

    loader = DataLoader(TensorDataset(*tensors(tr)), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val = tensors(va) if len(va) else None
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(epochs):
        net.train(); losses = []
        for xb, ya, ys, yf, m in loader:
            opt.zero_grad(); loss = loss_fn(net(xb), ya, ys, yf, m); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(val[0]), *val[1:]).item()) if val else float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": vl})
        if vl < best:
            best, best_epoch, stale = vl, epoch + 1, 0
            torch.save(net.state_dict(), output / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(torch.load(output / "best.pt", weights_only=True)); net.eval()

    def run(idx):
        with torch.no_grad():
            a, s, f = net(torch.tensor(X[idx]))
        return torch.sigmoid(a).numpy(), torch.softmax(s, -1).numpy(), f.numpy()

    pv, _, _ = run(va) if len(va) else (np.array([]), None, None)
    threshold, reason = select_threshold(b.attack[va], pv)
    pa, ps, pf = run(te)
    stage_pred = ps.argmax(axis=1)
    results = {
        "dataset": dataset, "model": "multitask_lstm", "horizons": K,
        "splits": {n: {"sequences": int(len(i)), "positive_targets": int(b.attack[i].sum())}
                   for n, i in (("train", tr), ("val", va), ("test", te))},
        "threshold": threshold, "threshold_selection": reason,
        "attack": metrics_at(b.attack[te], pa, threshold) if len(te) else None,
        "stage": _stage_metrics(data.stage_index[te], stage_pred, data.stages, b.attack[te], data.stage_index[tr]),
        "future_state": _state_metrics(pf, fut[te], data.future_mask[te], X[te][:, -1, :]),
        "stage_class_weights": {data.stages[i]: float(sw[i]) for i in range(n_st)},
        "training": {"seed": seed, "epochs_requested": epochs, "epochs_run": len(history), "batch_size": batch_size,
                     "learning_rate": learning_rate, "hidden_size": hidden_size, "dropout": dropout,
                     "patience": patience, "stage_weight": stage_weight, "state_weight": state_weight,
                     "best_checkpoint_epoch": best_epoch,
                     "checkpoint_selection": "minimum validation multi-task loss"},
        "history": history,
        "notes": ["Stage labels are the label-to-stage proxy from configs/stages.yaml, not ground truth."],
    }
    meta = {"model_type": "multitask_lstm", "feature_names": b.feature_names, "input_shape": list(b.X.shape[1:]),
            "stages": data.stages, "horizons": K, "threshold": threshold, "class_counts": counts,
            "training_config": results["training"]}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------------------------- inference
class MultiTaskPredictor:
    """Load a trained multi-task model directory and predict attack, stage and K-step future state."""

    def __init__(self, model_dir: str | Path):
        import torch
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.meta.get("model_type") != "multitask_lstm":
            raise ValueError(f"{model_dir} is not a multi-task model directory")
        self.prep: SequencePreprocessor = joblib.load(self.model_dir / "preprocessor.joblib")
        self.stages = list(self.meta["stages"])
        self.feature_names = list(self.meta["feature_names"])
        self.horizons = int(self.meta["horizons"])
        tc = self.meta["training_config"]
        self.net = MultiTaskForecaster(len(self.feature_names), len(self.stages), self.horizons,
                                       tc["hidden_size"], tc["dropout"]).net
        self.net.load_state_dict(torch.load(self.model_dir / "best.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        self.threshold = float(self.meta["threshold"])

    def predict(self, X_raw: np.ndarray) -> dict[str, np.ndarray]:
        import torch
        x = self.prep.transform(np.asarray(X_raw))
        with torch.no_grad():
            a, s, f = self.net(torch.tensor(x))
        probs = torch.softmax(s, -1).numpy()
        return {"attack_probability": torch.sigmoid(a).numpy(), "stage_probabilities": probs,
                "stage": np.array(self.stages, dtype=object)[probs.argmax(1)],
                "stage_probability": probs.max(1), "future_state": inverse_transform(self.prep, f.numpy())}

    def rows(self, X_raw: np.ndarray, sequence_ids: list[str], *, stage_threshold: float = 0.5,
             uncertain_label: str = "UNCERTAIN", mitre_lookup=None) -> list[dict[str, Any]]:
        p = self.predict(X_raw)
        out = []
        for i, sid in enumerate(sequence_ids):
            stage = str(p["stage"][i]) if p["stage_probability"][i] >= stage_threshold else uncertain_label
            row = {"sequence_id": sid, "attack_probability": float(p["attack_probability"][i]),
                   "predicted_attack": bool(p["attack_probability"][i] >= self.threshold), "threshold": self.threshold,
                   "predicted_stage": stage, "stage_probability": float(p["stage_probability"][i]),
                   "stage_distribution": {s: round(float(v), 4) for s, v in zip(self.stages, p["stage_probabilities"][i])},
                   "future_state": [{"horizon": k + 1, **{n: float(v) for n, v in zip(self.feature_names, p["future_state"][i, k])}}
                                    for k in range(self.horizons)]}
            if mitre_lookup is not None and stage != uncertain_label:
                m = mitre_lookup(stage)
                row["mitre"] = None if m is None else {"tactic_id": m.get("tactic_id"), "tactic_name": m.get("tactic_name"),
                                                       "techniques": m.get("techniques", [])}
            out.append(row)
        return out
