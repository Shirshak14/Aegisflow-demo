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
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
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

    @classmethod
    def from_parquet(cls, path: str | Path) -> "ForecastDataset":
        return cls.from_frame(pd.read_parquet(path))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ForecastDataset":
        required = {"sequence_features", "target_features", "target_attack_present",
                    "target_dominant_class", "target_dominant_stage", "split"}
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
        return cls(frame.reset_index(drop=True), X, frame.target_attack_present.to_numpy(np.int64),
                   frame.target_dominant_class.astype(str).to_numpy(),
                   frame.target_dominant_stage.astype(str).to_numpy(), ystate, names)

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.frame.split.to_numpy() == split)


class SequencePreprocessor:
    """Train-only median imputation, signed log1p for skewed counts/rates, RobustScaler."""
    LOG_FEATURES = {"flow_count", "packet_count_sum", "byte_count_sum", "syn_count_sum",
                    "ack_count_sum", "rst_count_sum", "fin_count_sum", "psh_count_sum",
                    "unique_destination_ips", "unique_destination_ports", "unique_source_ports",
                    "connection_burst_max"}

    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.medians: np.ndarray | None = None
        self.scaler = RobustScaler()

    def _finite(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64).copy()
        x[~np.isfinite(x)] = np.nan
        return x

    def _log(self, X: np.ndarray) -> np.ndarray:
        x = X.copy()
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
    positives = int(np.sum(y == 1)); negatives = int(np.sum(y == 0))
    if positives == 0:
        return 1.0, {"positive": positives, "negative": negatives}
    return negatives / positives, {"positive": positives, "negative": negatives}


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "pr_auc": float(average_precision_score(y, p)) if np.sum(y) else None,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "positive_predictions": int(pred.sum()), "actual_positives": int(np.sum(y)),
            "actual_negatives": int(np.sum(y == 0))}


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, str]:
    if len(y) == 0 or not np.any(y == 1):
        return 0.5, "validation set has no positive targets; default 0.5 retained"
    candidates = sorted(set([0.01, 0.05, 0.1, 0.2, 0.3, 0.5] + list(map(float, p))))
    scores = [(f1_score(y, p >= t, zero_division=0), t) for t in candidates]
    return float(max(scores, key=lambda z: (z[0], -z[1]))[1]), "validation F1 maximization"


class LSTMForecaster:
    def __init__(self, feature_count: int, hidden_size: int = 32, dropout: float = 0.2):
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__(); self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout); self.head = nn.Linear(hidden_size, 1)
            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.head(self.dropout(h[-1])).squeeze(-1)
        self.net = Net()


def train_experiment(data: ForecastDataset, output: Path, *, seed=42, epochs=20, batch_size=128,
                     learning_rate=0.001, hidden_size=32, dropout=0.2, patience=4) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_i, val_i, test_i = (data.indices(s) for s in ("train", "val", "test"))
    prep = SequencePreprocessor(data.feature_names).fit(data.X[train_i])
    Xt = prep.transform(data.X[train_i]); Xv = prep.transform(data.X[val_i]); Xq = prep.transform(data.X[test_i])
    output.mkdir(parents=True, exist_ok=True)
    prep.save(output / "preprocessor.joblib")
    # Classical baseline uses flattened normalized historical windows.
    flat_train, flat_val, flat_test = (x.reshape(len(x), -1) for x in (Xt, Xv, Xq))
    prior = float(np.mean(data.attack[train_i])) if len(train_i) else 0.0
    prior_probs = np.full(len(test_i), prior)
    classical = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    classical.fit(flat_train, data.attack[train_i])
    cp = classical.predict_proba(flat_test)[:, 1]
    threshold, threshold_reason = select_threshold(data.attack[val_i],
        classical.predict_proba(flat_val)[:, 1] if len(val_i) else np.array([]))
    # For common model comparison, validation threshold is chosen for LSTM as well.
    counts = positive_class_weight(data.attack[train_i])
    pos_weight = counts[0]
    model = LSTMForecaster(data.X.shape[-1], hidden_size, dropout).net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    loader = DataLoader(TensorDataset(torch.tensor(Xt), torch.tensor(data.attack[train_i], dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    vx = torch.tensor(Xv, device=device); vy = torch.tensor(data.attack[val_i], dtype=torch.float32, device=device)
    best_loss=float("inf"); best_epoch=0; stale=0; history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device); optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad(): val_loss=float(loss_fn(model(vx),vy).item()) if len(val_i) else float(np.mean(losses))
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss=val_loss; best_epoch=epoch+1; stale=0; torch.save(model.state_dict(), output / "best.pt")
        else:
            stale+=1
            if stale>=patience: break
    torch.save(model.state_dict(), output / "last.pt")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)); model.eval()
    with torch.no_grad():
        lp=torch.sigmoid(model(torch.tensor(Xq,device=device))).cpu().numpy()
        lv=torch.sigmoid(model(vx)).cpu().numpy() if len(val_i) else np.array([])
    lstm_threshold, lstm_reason = select_threshold(data.attack[val_i], lv)
    results={"dataset":"cic_ids2017", "splits":{}, "positive_class_weight":pos_weight,
             "train_class_counts":counts[1], "threshold_selection":lstm_reason,
             "threshold":lstm_threshold, "models":{"majority":metrics_at(data.attack[test_i],prior_probs,0.5),
             "logistic_regression":metrics_at(data.attack[test_i],cp,threshold),
             "lstm":metrics_at(data.attack[test_i],lp,lstm_threshold)},
             "training":{"seed":seed,"epochs_requested":epochs,"epochs_run":len(history),"batch_size":batch_size,
                         "learning_rate":learning_rate,"hidden_size":hidden_size,"dropout":dropout,"patience":patience,
                         "device":str(device),"best_checkpoint_epoch":best_epoch,
                         "checkpoint_selection":"minimum validation BCEWithLogitsLoss"}, "history":history}
    for name, idx in zip(("train","val","test"),(train_i,val_i,test_i)):
        results["splits"][name]={"sequences":len(idx),"positive_targets":int(data.attack[idx].sum()),"benign_targets":int((data.attack[idx]==0).sum())}
    results["stage_proxy"]={"status":"Not scored: too few attack-positive test targets per stage to support meaningful stage forecasting.",
                            "test_distribution":pd.Series(data.stages[test_i]).value_counts().to_dict()}
    target_start=pd.to_datetime(data.frame.iloc[test_i].target_window_start)
    seq_end=pd.to_datetime(data.frame.iloc[test_i].seq_end_time)
    lead=(target_start-seq_end).dt.total_seconds() if len(test_i) else pd.Series(dtype=float)
    results["forecast_lead_time_seconds"]={"count":int(len(lead)), "median":float(lead.median()) if len(lead) else None,
        "min":float(lead.min()) if len(lead) else None, "max":float(lead.max()) if len(lead) else None,
        "nonpositive_count":int((lead<=0).sum()), "positive_count":int((lead>0).sum()),
        "interpretation":"Derived as target_window_start - seq_end_time. Nonpositive values mean the target window starts before the input window ends; these examples are not leakage-free forecasts."}
    # Regression is descriptive persistence baseline for future network state, independently labeled.
    results["future_network_state"]={"status":"Not trained in Phase 3; sparse attack targets and primary binary objective prioritized."}
    metadata={"feature_names":data.feature_names,"input_shape":list(data.X.shape[1:]),"label_mapping":{"benign":0,"attack":1},
              "threshold":lstm_threshold,"class_counts":counts[1],"training_config":results["training"]}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (output/"metrics.json").write_text(json.dumps(results,indent=2,default=str),encoding="utf-8")
    joblib.dump(classical,output/"logistic.joblib")
    return results


def predict_sequences(path: Path, model_dir: Path, threshold: float | None = None) -> list[dict[str, Any]]:
    import torch
    data=ForecastDataset.from_parquet(path); meta=json.loads((model_dir/"metadata.json").read_text(encoding="utf-8"))
    prep=joblib.load(model_dir/"preprocessor.joblib"); x=prep.transform(data.X)
    model=LSTMForecaster(len(meta["feature_names"]),meta["training_config"]["hidden_size"],meta["training_config"]["dropout"]).net
    model.load_state_dict(torch.load(model_dir/"best.pt",map_location="cpu",weights_only=True)); model.eval()
    with torch.no_grad(): probs=torch.sigmoid(model(torch.tensor(x))).numpy()
    t=float(meta["threshold"] if threshold is None else threshold)
    return [{"sequence_id":str(data.frame.iloc[i].get("sequence_id","")),"attack_probability":float(probs[i]),
             "predicted_attack":bool(probs[i]>=t),"threshold":t,"model":"cic_ids2017-lstm-phase3"} for i in range(len(probs))]
