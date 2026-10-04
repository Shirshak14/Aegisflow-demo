"""Per-prediction explanations for the Phase 3 forecasting models.

Two attribution methods for the LSTM, both computed in the model's input space (the
preprocessed tensor the network actually sees) and explaining the LSTM *logit*
(log-odds of an attack in the target window), not the sigmoid probability:

  shap                  SHAP values via ``shap.GradientExplainer`` (expected gradients) against a
                        background sample of TRAINING sequences. Additivity holds in expectation:
                        sum(attributions) ~= f(x) - E_background[f].
  integrated_gradients  Integrated Gradients (Sundararajan et al. 2017) from a baseline equal to the
                        element-wise median background sequence (a "typical" training window;
                        the mean is dragged far out by heavy-tailed traffic counts). Completeness: sum(attributions) ~= f(x) - f(baseline).
                        Pure PyTorch, so it works without the optional ``shap`` package.

For the logistic-regression baseline the SHAP values are exact and closed-form
(independent features): phi_j = w_j * (x_j - E_background[x_j]), explaining the decision function.

Every attribution has shape (sequence_length, n_features). It is summarised per feature (summed
over time steps) and per time step (summed over features), and the top features are reported
with their sign: positive pushes the prediction towards "attack", negative towards "benign".
Explanations are faithful to the model they explain; they do not make a weak model accurate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .modeling import LSTMForecaster, SequencePreprocessor

METHODS = ("shap", "integrated_gradients")


@dataclass
class Explanation:
    """Attribution of one model output to its (time step, feature) inputs."""
    sequence_id: str
    model: str
    method: str
    target: str
    output: float
    reference_output: float
    attributions: np.ndarray            # (sequence_length, n_features)
    feature_names: list[str]
    raw_last_window: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def feature_totals(self) -> np.ndarray:
        return self.attributions.sum(axis=0)

    @property
    def timestep_totals(self) -> np.ndarray:
        return self.attributions.sum(axis=1)

    @property
    def additivity_gap(self) -> float:
        """|sum(attributions) - (output - reference_output)|; small means the attributions add up."""
        return float(abs(self.attributions.sum() - (self.output - self.reference_output)))

    def top_features(self, k: int = 5) -> list[dict[str, Any]]:
        totals = self.feature_totals
        order = np.argsort(-np.abs(totals))[:k]
        out = []
        for i in order:
            item = {"feature": self.feature_names[i], "attribution": round(float(totals[i]), 6),
                    "direction": "towards attack" if totals[i] > 0 else "towards benign" if totals[i] < 0 else "none",
                    "most_influential_step": int(np.argmax(np.abs(self.attributions[:, i])))}
            if self.raw_last_window is not None:
                item["last_window_value"] = float(self.raw_last_window[i])
            out.append(item)
        return out

    def to_dict(self, k: int = 5, include_matrix: bool = False) -> dict[str, Any]:
        d = {"sequence_id": self.sequence_id, "model": self.model, "method": self.method,
             "target": self.target, "output": round(self.output, 6),
             "reference_output": round(self.reference_output, 6),
             "attribution_sum": round(float(self.attributions.sum()), 6),
             "additivity_gap": round(self.additivity_gap, 6),
             "top_features": self.top_features(k),
             "timestep_attribution": [round(float(v), 6) for v in self.timestep_totals],
             "notes": list(self.notes)}
        if include_matrix:
            d["feature_names"] = list(self.feature_names)
            d["attributions"] = np.round(self.attributions, 6).tolist()
        return d


def _logit_fn(net):
    import torch

    def f(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(x, dtype=np.float32))).cpu().numpy().astype(np.float64)
    return f


def integrated_gradients(net, x: np.ndarray, baseline: np.ndarray, steps: int = 64) -> np.ndarray:
    """IG attributions for a batch ``x`` (n, L, F) w.r.t. one ``baseline`` (L, F); trapezoid rule."""
    import torch
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    b_t = torch.as_tensor(np.asarray(baseline, dtype=np.float32)).unsqueeze(0).expand_as(x_t)
    alphas = torch.linspace(0.0, 1.0, steps + 1)
    grads = []
    # cuDNN RNN backward needs train mode on GPU; on CPU eval mode is fine and keeps dropout off.
    for a in alphas:
        point = (b_t + a * (x_t - b_t)).detach().requires_grad_(True)
        out = net(point).sum()
        (g,) = torch.autograd.grad(out, point)
        grads.append(g.detach())
    g = torch.stack(grads)                        # (steps+1, n, L, F)
    avg = (g[:-1] + g[1:]).mean(dim=0) / 2.0      # trapezoid
    return ((x_t - b_t) * avg).numpy().astype(np.float64)


def shap_values(net, x: np.ndarray, background: np.ndarray, nsamples: int = 200, seed: int = 0) -> np.ndarray:
    """SHAP values via ``shap.GradientExplainer`` (raises ImportError if shap is not installed)."""
    import shap
    import torch
    torch.manual_seed(seed)
    explainer = shap.GradientExplainer(_Unsqueeze(net), torch.as_tensor(np.asarray(background, dtype=np.float32)))
    vals = explainer.shap_values(torch.as_tensor(np.asarray(x, dtype=np.float32)), nsamples=nsamples,
                                 rseed=seed)
    vals = np.asarray(vals[0] if isinstance(vals, list) else vals, dtype=np.float64)
    if vals.ndim == x.ndim + 1:      # newer shap appends an output axis of size 1
        vals = vals[..., 0]
    return vals


class _Unsqueeze:
    """Wrap a (n,)-output net so it returns (n, 1), the shape GradientExplainer expects."""
    def __new__(cls, net):
        from torch import nn

        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__(); self.net = net
            def forward(self, x):
                return self.net(x).unsqueeze(-1)
        return Wrapped().eval()


def linear_shap(coef: np.ndarray, x_flat: np.ndarray, background_flat: np.ndarray) -> np.ndarray:
    """Exact SHAP values of a linear model with independent features: w * (x - E[x])."""
    return np.asarray(coef, dtype=np.float64).ravel() * (np.asarray(x_flat, dtype=np.float64)
                                                        - np.asarray(background_flat, dtype=np.float64).mean(axis=0))


class ModelExplainer:
    """Explain the saved Phase 3 models in ``model_dir`` (metadata.json, preprocessor.joblib, best.pt)."""

    def __init__(self, model_dir: str | Path, background_raw: np.ndarray, *, background_size: int = 100,
                 seed: int = 0):
        import torch
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.prep: SequencePreprocessor = joblib.load(self.model_dir / "preprocessor.joblib")
        self.feature_names = list(self.meta["feature_names"])
        tc = self.meta["training_config"]
        self.net = LSTMForecaster(len(self.feature_names), tc["hidden_size"], tc["dropout"]).net
        self.net.load_state_dict(torch.load(self.model_dir / "best.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        lr_path = self.model_dir / "logistic.joblib"
        self.logistic = joblib.load(lr_path) if lr_path.exists() else None
        rng = np.random.default_rng(seed)
        bg = np.asarray(background_raw)
        if len(bg) == 0:
            raise ValueError("background sample is empty; pass training-split sequences")
        if len(bg) > background_size:
            bg = bg[np.sort(rng.choice(len(bg), background_size, replace=False))]
        self.background = self.prep.transform(bg)
        self.seed = seed
        self._f = _logit_fn(self.net)

    def explain(self, X_raw: np.ndarray, sequence_ids: list[str] | None = None, *, model: str = "lstm",
                method: str = "shap", nsamples: int = 200, steps: int = 128) -> list[Explanation]:
        X_raw = np.asarray(X_raw)
        if X_raw.ndim == 2:
            X_raw = X_raw[None]
        ids = list(sequence_ids) if sequence_ids is not None else [str(i) for i in range(len(X_raw))]
        x = self.prep.transform(X_raw)
        notes = ["Attributions are in log-odds units, taken with respect to the preprocessed inputs "
                 "(signed log1p + robust scaling); last_window_value is the raw value."]
        if model == "logistic_regression":
            if self.logistic is None:
                raise FileNotFoundError(self.model_dir / "logistic.joblib")
            flat, bgf = x.reshape(len(x), -1), self.background.reshape(len(self.background), -1)
            phi = linear_shap(self.logistic.coef_, flat, bgf).reshape(x.shape)
            out = self.logistic.decision_function(flat)
            ref = float(self.logistic.decision_function(bgf.mean(axis=0, keepdims=True))[0])
            outs, refs, used = out, np.full(len(x), ref), "shap_linear_exact"
        elif model == "lstm":
            if method == "shap":
                phi = shap_values(self.net, x, self.background, nsamples=nsamples, seed=self.seed)
                refs = np.full(len(x), float(self._f(self.background).mean()))
                notes.append("SHAP (expected gradients) is sampled: attributions add up to output minus the "
                             "background mean only approximately; raise nsamples to tighten.")
            elif method == "integrated_gradients":
                base = np.median(self.background, axis=0)
                phi = integrated_gradients(self.net, x, base, steps=steps)
                refs = np.full(len(x), float(self._f(base[None])[0]))
            else:
                raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
            outs, used = self._f(x), method
        else:
            raise ValueError(f"unknown model {model!r}; choose 'lstm' or 'logistic_regression'")
        return [Explanation(ids[i], model, used, "log-odds of attack in target window", float(outs[i]),
                            float(refs[i]), phi[i], self.feature_names, X_raw[i, -1], notes)
                for i in range(len(x))]


def explain_parquet(sequences_path: str | Path, model_dir: str | Path, sequence_ids: list[str] | None = None, *,
                    model: str = "lstm", method: str = "shap", limit: int = 5, background_size: int = 100,
                    nsamples: int = 200, seed: int = 0) -> list[Explanation]:
    """Explain sequences from a sequences.parquet with a ``split`` column (background = train split)."""
    import pandas as pd

    from .modeling import ForecastDataset
    data = ForecastDataset.from_parquet(sequences_path)
    train = data.indices("train")
    if len(train) == 0:
        raise ValueError("sequence table has no train split to draw the SHAP background from")
    explainer = ModelExplainer(model_dir, data.X[train], background_size=background_size, seed=seed)
    sid = data.frame["sequence_id"].astype(str) if "sequence_id" in data.frame else pd.Series(map(str, range(len(data.frame))))
    if sequence_ids:
        missing = set(sequence_ids) - set(sid)
        if missing:
            raise KeyError(f"sequence ids not found: {sorted(missing)[:5]}")
        idx = np.flatnonzero(sid.isin(sequence_ids).to_numpy())
    else:
        idx = data.indices("test")[:limit]
    return explainer.explain(data.X[idx], sid.iloc[idx].tolist(), model=model, method=method, nsamples=nsamples)
