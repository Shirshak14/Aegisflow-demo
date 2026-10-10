"""Phase 3 forecasting baselines for the existing temporal sequence table."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
import joblib

from .temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

# These two host-window values are computed from dataset labels, not traffic
# observations. They are useful annotations but unavailable to an unlabeled
# deployment input, so they must not enter the forecasting tensor.
LABEL_DERIVED_FEATURES = {"attack_flow_ratio", "benign_flow_ratio"}
MODEL_FEATURES = [name for name in STANDARD_NUMERIC_WINDOW_FEATURES if name not in LABEL_DERIVED_FEATURES]


@dataclass
class ForecastDataset:
    frame: pd.DataFrame
    X: np.ndarray
    attack: np.ndarray
    classes: np.ndarray
    stages: np.ndarray
    future_state: np.ndarray
    feature_names: list[str]
    host_relative: bool = False

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> "ForecastDataset":
        return cls.from_frame(pd.read_parquet(path), **kwargs)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, host_relative: bool = False, **hr_kwargs: Any) -> "ForecastDataset":
        required = {
            "sequence_features",
            "target_features",
            "target_attack_present",
            "target_dominant_class",
            "target_dominant_stage",
            "split",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Sequence table missing required columns: {sorted(missing)}")
        raw_names = list(STANDARD_NUMERIC_WINDOW_FEATURES)
        names = list(MODEL_FEATURES)
        # PyArrow round-trips list<list<float>> as a NumPy object array of row arrays.
        X = np.stack([np.asarray(np.asarray(x).tolist(), dtype=np.float32) for x in frame.sequence_features])
        ystate = np.stack([np.asarray(np.asarray(x).tolist(), dtype=np.float32) for x in frame.target_features])
        if X.ndim != 3 or X.shape[2] != len(raw_names) or ystate.shape != (len(frame), len(raw_names)):
            raise ValueError(f"Unexpected feature tensor shapes: X={X.shape}, target={ystate.shape}")
        X = X[:, :, [raw_names.index(name) for name in names]]
        frame = frame.reset_index(drop=True)
        if host_relative:
            # Must see every split of a host so its baseline is its own past, whatever the split.
            X = host_relative_inputs(frame, X, names, **hr_kwargs)
        return cls(
            frame,
            X,
            frame.target_attack_present.to_numpy(np.int64),
            frame.target_dominant_class.astype(str).to_numpy(),
            frame.target_dominant_stage.astype(str).to_numpy(),
            ystate,
            names,
            host_relative,
        )

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.frame.split.to_numpy() == split)


def _signed_log(X: np.ndarray, names: list[str], log_features: set[str]) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64).copy()
    x[~np.isfinite(x)] = np.nan
    for i, name in enumerate(names):
        if name in log_features:
            x[..., i] = np.sign(x[..., i]) * np.log1p(np.abs(x[..., i]))
    return x


def host_relative_inputs(
    frame: pd.DataFrame,
    X: np.ndarray,
    names: list[str],
    *,
    window_seconds: float = 60.0,
    lookback: int = 240,
    min_history: int = 5,
) -> np.ndarray:
    """Express every input window relative to the same host's own *past* (B1).

    For a sequence starting at ``t`` the baseline is the per-feature median of that host's
    earlier sequence-first-windows that had already ended by ``t`` (``start + window_seconds
    <= t``), capped to the last ``lookback`` of them. Nothing at or after ``t`` is used, so
    the baseline is causal. Output is ``log-space window - baseline``; a host with fewer than
    ``min_history`` earlier windows gets a zero delta (no evidence it differs from itself).
    Quiet hosts therefore look alike, and a host that is always noisy (e.g. an attacker) is
    no longer recognisable merely by its level.
    """
    x = _signed_log(X, names, SequencePreprocessor.LOG_FEATURES)
    out = np.zeros_like(x)
    starts_all = pd.to_datetime(frame["seq_start_time"]).to_numpy("datetime64[ns]")
    hosts = frame["host_id"].astype(str).to_numpy()
    w = np.timedelta64(int(window_seconds * 1e9), "ns")
    for host in pd.unique(hosts):
        ix = np.flatnonzero(hosts == host)
        ix = ix[np.argsort(starts_all[ix], kind="stable")]
        starts = starts_all[ix]
        first = x[ix, 0, :]  # (n, F) first window of each sequence
        done = np.searchsorted(starts + w, starts, side="right")  # earlier windows ended by start_i
        for pos, i in enumerate(ix):
            k = min(done[pos], pos)  # never include itself or later ones
            if k < min_history:
                continue
            base = np.nanmedian(first[max(0, k - lookback) : k], axis=0)
            out[i] = x[i] - np.nan_to_num(base, nan=0.0)
    return out.astype(np.float32)


class SequencePreprocessor:
    """Train-only median imputation, signed log1p for skewed counts/rates, RobustScaler."""

    LOG_FEATURES = {
        "flow_count",
        "packet_count_sum",
        "byte_count_sum",
        "syn_count_sum",
        "ack_count_sum",
        "rst_count_sum",
        "fin_count_sum",
        "psh_count_sum",
        "unique_destination_ips",
        "unique_destination_ports",
        "unique_source_ports",
        "connection_burst_max",
    }

    def __init__(self, feature_names: list[str], prelogged: bool = False):
        self.feature_names = list(feature_names)
        self.prelogged = prelogged  # host-relative inputs are already log-space deltas
        self.medians: np.ndarray | None = None
        self.scaler = RobustScaler()

    def _finite(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64).copy()
        x[~np.isfinite(x)] = np.nan
        return x

    def _log(self, X: np.ndarray) -> np.ndarray:
        x = X.copy()
        if getattr(self, "prelogged", False):
            return x
        for i, name in enumerate(self.feature_names):
            if name in self.LOG_FEATURES:
                x[..., i] = np.sign(x[..., i]) * np.log1p(np.abs(x[..., i]))
        return x

    def fit(self, X: np.ndarray) -> "SequencePreprocessor":
        x = self._log(self._finite(X))
        self.medians = np.nanmedian(x.reshape(-1, x.shape[-1]), axis=0)
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        x = np.where(np.isfinite(x), x, self.medians)
        self.scaler.fit(x.reshape(-1, x.shape[-1]))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("Preprocessor must be fit on training data first")
        x = self._log(self._finite(X))
        x = np.where(np.isfinite(x), x, self.medians)
        shape = x.shape
        return self.scaler.transform(x.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def positive_class_weight(y: np.ndarray) -> tuple[float, dict[str, int]]:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        return 1.0, {"positive": positives, "negative": negatives}
    return negatives / positives, {"positive": positives, "negative": negatives}


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if np.sum(y) else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "positive_predictions": int(pred.sum()),
        "actual_positives": int(np.sum(y)),
        "actual_negatives": int(np.sum(y == 0)),
    }


_THRESHOLD_GRID = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, str]:
    """Pick the validation threshold that maximises F1 (prediction rule: ``p >= t``).

    Candidates are the fixed grid plus every distinct validation probability. Ties in F1 resolve to the
    lowest candidate, as in the original loop over ``sorted(candidates)``. F1 for all candidates comes from one
    ``precision_recall_curve`` call: its thresholds are the distinct probabilities, and a grid value ``t`` that is
    not one of them predicts exactly what the smallest curve threshold >= ``t`` predicts (none if ``t`` is above
    every probability, which scores 0).
    """
    if len(y) == 0 or not np.any(y == 1):
        return 0.5, "validation set has no positive targets; default 0.5 retained"
    y = np.asarray(y)
    p = np.asarray(p)
    n_pos = int((y == 1).sum())
    precision, recall, thr = precision_recall_curve(y, p)
    thr = thr.astype(float)
    precision, recall = precision[:-1], recall[:-1]  # drop the final (precision=1, recall=0) point; it has no threshold
    # Rebuild exact integer counts so F1 = 2TP / (predicted + positives) is computed without float drift.
    tp = np.rint(recall * n_pos)
    predicted = np.where(tp > 0, np.rint(tp / np.where(precision > 0, precision, 1.0)), 0.0)
    f1_at_thr = np.where(tp > 0, 2 * tp / (predicted + n_pos), 0.0)
    candidates = np.unique(np.concatenate([np.asarray(_THRESHOLD_GRID), p.astype(float)]))  # ascending, distinct
    idx = np.searchsorted(thr, candidates, side="left")
    f1 = np.where(idx < len(thr), f1_at_thr[np.minimum(idx, len(thr) - 1)], 0.0)
    return float(candidates[int(np.argmax(f1))]), "validation F1 maximization"  # argmax -> first = lowest threshold


class LSTMForecaster:
    # hidden_size default matches configs/config.yaml `model.hidden_size` (the committed demo model).
    def __init__(self, feature_count: int, hidden_size: int = 16, dropout: float = 0.2):
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.head(self.dropout(h[-1])).squeeze(-1)

        self.net = Net()


def _attention_pool_class():
    """Additive (Bahdanau-style) attention over time steps: one weight per input window, summing to 1."""
    from torch import nn

    class AttentionPool(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))

        def forward(self, h):  # h: (n, L, dim)
            w = self.score(h).squeeze(-1).softmax(dim=-1)  # (n, L)
            return (w.unsqueeze(-1) * h).sum(dim=1), w

    return AttentionPool


class AttentionLSTMForecaster:
    """LSTM over all windows + attention pooling. ``net.attention(x)`` returns per-window weights (n, L)."""

    def __init__(self, feature_count: int, hidden_size: int = 32, dropout: float = 0.2):
        from torch import nn

        AttentionPool = _attention_pool_class()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
                self.pool = AttentionPool(hidden_size)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def _pooled(self, x):
                out, _ = self.lstm(x)
                return self.pool(out)

            def forward(self, x):
                ctx, _ = self._pooled(x)
                return self.head(self.dropout(ctx)).squeeze(-1)

            def attention(self, x):
                return self._pooled(x)[1]

        self.net = Net()


class TransformerForecaster:
    """Temporal Transformer encoder over the window sequence, then attention pooling over time.

    Input windows are projected to ``hidden_size`` (d_model), a learned positional embedding marks
    each window's place in the sequence, and ``num_layers`` pre-norm encoder layers with ``num_heads``
    heads mix information across windows. ``net.attention(x)`` returns the pooling weights (n, L).
    """

    def __init__(
        self,
        feature_count: int,
        hidden_size: int = 32,
        dropout: float = 0.2,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 64,
    ):
        import torch
        from torch import nn

        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        AttentionPool = _attention_pool_class()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(feature_count, hidden_size)
                self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_size))
                nn.init.normal_(self.pos, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    hidden_size,
                    num_heads,
                    dim_feedforward=2 * hidden_size,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
                self.pool = AttentionPool(hidden_size)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def _pooled(self, x):
                if x.shape[1] > max_len:
                    raise ValueError(f"sequence length {x.shape[1]} exceeds max_len {max_len}")
                h = self.encoder(self.proj(x) + self.pos[:, : x.shape[1]])
                return self.pool(h)

            def forward(self, x):
                ctx, _ = self._pooled(x)
                return self.head(self.dropout(ctx)).squeeze(-1)

            def attention(self, x):
                return self._pooled(x)[1]

        self.net = Net()


MODEL_TYPES = ("lstm", "attention_lstm", "transformer")


def build_model(model_type: str, feature_count: int, hidden_size: int = 32, dropout: float = 0.2):
    """Return the torch module for ``model_type`` (one of MODEL_TYPES)."""
    if model_type == "lstm":
        return LSTMForecaster(feature_count, hidden_size, dropout).net
    if model_type == "attention_lstm":
        return AttentionLSTMForecaster(feature_count, hidden_size, dropout).net
    if model_type == "transformer":
        return TransformerForecaster(feature_count, hidden_size, dropout).net
    raise ValueError(f"unknown model type {model_type!r}; choose from {MODEL_TYPES}")


def load_model(model_dir: Path, map_location: str = "cpu"):
    """Load ``best.pt`` from ``model_dir`` with the architecture recorded in metadata.json (default lstm)."""
    import torch

    meta = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    tc = meta["training_config"]
    net = build_model(tc.get("model_type", "lstm"), len(meta["feature_names"]), tc["hidden_size"], tc["dropout"])
    net.load_state_dict(torch.load(Path(model_dir) / "best.pt", map_location=map_location, weights_only=True))
    net.eval()
    return net, meta


def train_experiment(
    data: ForecastDataset,
    output: Path,
    *,
    seed=42,
    epochs=8,
    batch_size=256,
    learning_rate=0.001,
    hidden_size=16,
    dropout=0.2,
    patience=3,
    model_type: str = "lstm",
) -> dict[str, Any]:
    """Train the baselines and write artifacts to ``output``.

    Defaults equal configs/config.yaml `model:` (the committed demo model). Side effect: this seeds
    the Python/NumPy/torch RNGs and turns on torch deterministic algorithms for the whole process.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_i, val_i, test_i = (data.indices(s) for s in ("train", "val", "test"))
    prep = SequencePreprocessor(data.feature_names, prelogged=data.host_relative).fit(data.X[train_i])
    Xt = prep.transform(data.X[train_i])
    Xv = prep.transform(data.X[val_i])
    Xq = prep.transform(data.X[test_i])
    output.mkdir(parents=True, exist_ok=True)
    prep.save(output / "preprocessor.joblib")
    # Classical baseline uses flattened normalized historical windows.
    flat_train, flat_val, flat_test = (x.reshape(len(x), -1) for x in (Xt, Xv, Xq))
    prior = float(np.mean(data.attack[train_i])) if len(train_i) else 0.0
    prior_probs = np.full(len(test_i), prior)
    classical = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    classical.fit(flat_train, data.attack[train_i])
    cp = classical.predict_proba(flat_test)[:, 1]
    threshold, threshold_reason = select_threshold(
        data.attack[val_i], classical.predict_proba(flat_val)[:, 1] if len(val_i) else np.array([])
    )
    # For common model comparison, validation threshold is chosen for LSTM as well.
    counts = positive_class_weight(data.attack[train_i])
    pos_weight = counts[0]
    model = build_model(model_type, data.X.shape[-1], hidden_size, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    loader = DataLoader(
        TensorDataset(torch.tensor(Xt), torch.tensor(data.attack[train_i], dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    vx = torch.tensor(Xv, device=device)
    vy = torch.tensor(data.attack[val_i], dtype=torch.float32, device=device)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vx), vy).item()) if len(val_i) else float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), output / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    torch.save(model.state_dict(), output / "last.pt")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        lp = torch.sigmoid(model(torch.tensor(Xq, device=device))).cpu().numpy()
        lv = torch.sigmoid(model(vx)).cpu().numpy() if len(val_i) else np.array([])
    lstm_threshold, lstm_reason = select_threshold(data.attack[val_i], lv)
    results = {
        "dataset": "cic_ids2017",
        "splits": {},
        "positive_class_weight": pos_weight,
        "train_class_counts": counts[1],
        "threshold_selection": lstm_reason,
        "threshold": lstm_threshold,
        "models": {
            "majority": metrics_at(data.attack[test_i], prior_probs, 0.5),
            "logistic_regression": metrics_at(data.attack[test_i], cp, threshold),
            model_type: metrics_at(data.attack[test_i], lp, lstm_threshold),
        },
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_run": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "patience": patience,
            "device": str(device),
            "best_checkpoint_epoch": best_epoch,
            "checkpoint_selection": "minimum validation BCEWithLogitsLoss",
        },
        "history": history,
    }
    if model_type != "lstm":  # the default LSTM's metadata stays byte-for-byte what it was
        results["training"]["model_type"] = model_type
    for name, idx in zip(("train", "val", "test"), (train_i, val_i, test_i)):
        results["splits"][name] = {
            "sequences": len(idx),
            "positive_targets": int(data.attack[idx].sum()),
            "benign_targets": int((data.attack[idx] == 0).sum()),
        }
    results["stage_proxy"] = {
        "status": "Not scored: too few attack-positive test targets per stage to support meaningful stage forecasting.",
        "test_distribution": pd.Series(data.stages[test_i]).value_counts().to_dict(),
    }
    target_start = pd.to_datetime(data.frame.iloc[test_i].target_window_start)
    seq_end = pd.to_datetime(data.frame.iloc[test_i].seq_end_time)
    lead = (target_start - seq_end).dt.total_seconds() if len(test_i) else pd.Series(dtype=float)
    results["forecast_lead_time_seconds"] = {
        "count": int(len(lead)),
        "median": float(lead.median()) if len(lead) else None,
        "min": float(lead.min()) if len(lead) else None,
        "max": float(lead.max()) if len(lead) else None,
        "nonpositive_count": int((lead <= 0).sum()),
        "positive_count": int((lead > 0).sum()),
        "interpretation": "Derived as target_window_start - seq_end_time. Nonpositive values mean the target window starts before the input window ends; these examples are not leakage-free forecasts.",
    }
    # Regression is descriptive persistence baseline for future network state, independently labeled.
    results["future_network_state"] = {
        "status": "Not trained in Phase 3; sparse attack targets and primary binary objective prioritized."
    }
    metadata = {
        "feature_names": data.feature_names,
        "input_shape": list(data.X.shape[1:]),
        "label_mapping": {"benign": 0, "attack": 1},
        "host_relative": data.host_relative,
        "threshold": lstm_threshold,
        "class_counts": counts[1],
        "training_config": results["training"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    joblib.dump(classical, output / "logistic.joblib")
    return results


def predict_sequences(path: Path, model_dir: Path, threshold: float | None = None) -> list[dict[str, Any]]:
    import torch

    model, meta = load_model(model_dir)
    data = ForecastDataset.from_parquet(path, host_relative=bool(meta.get("host_relative", False)))
    prep = joblib.load(model_dir / "preprocessor.joblib")
    x = prep.transform(data.X)
    model_type = meta["training_config"].get("model_type", "lstm")
    with torch.no_grad():
        xt = torch.tensor(x)
        probs = torch.sigmoid(model(xt)).numpy()
        attn = model.attention(xt).numpy() if hasattr(model, "attention") else None
    t = float(meta["threshold"] if threshold is None else threshold)
    name = "cic_ids2017-lstm-phase3" if model_type == "lstm" else f"cic_ids2017-{model_type}"
    rows = [
        {
            "sequence_id": str(data.frame.iloc[i].get("sequence_id", "")),
            "attack_probability": float(probs[i]),
            "predicted_attack": bool(probs[i] >= t),
            "threshold": t,
            "model": name,
        }
        for i in range(len(probs))
    ]
    if attn is not None:  # per-input-window attention weights, oldest window first
        for row, w in zip(rows, attn):
            row["attention"] = [round(float(v), 6) for v in w]
    return rows
