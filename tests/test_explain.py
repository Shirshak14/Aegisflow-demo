import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aegisflow.ml.explain import (Explanation, ModelExplainer, explain_parquet, integrated_gradients,
                                  linear_shap, shap_values)
from aegisflow.ml.modeling import MODEL_FEATURES, train_experiment, ForecastDataset
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

SIGNAL = "syn_count_sum"


def synthetic_frame(n=240, length=4, seed=0):
    """Attack target iff the SIGNAL feature spikes in the last input window; other features are noise."""
    rng = np.random.default_rng(seed)
    F = len(STANDARD_NUMERIC_WINDOW_FEATURES)
    X = rng.gamma(2.0, 5.0, size=(n, length, F)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    j = STANDARD_NUMERIC_WINDOW_FEATURES.index(SIGNAL)
    X[y == 1, -1, j] += 500.0
    split = np.array(["train"] * int(n * 0.6) + ["val"] * int(n * 0.2) + ["test"] * (n - int(n * 0.6) - int(n * 0.2)))
    return pd.DataFrame({"sequence_id": [f"h{i % 7}_seq_{i}" for i in range(n)], "host_id": [f"h{i % 7}" for i in range(n)],
                         "seq_end_time": pd.date_range("2017-07-03", periods=n, freq="30s"),
                         "target_window_start": pd.date_range("2017-07-03 00:00:30", periods=n, freq="30s"),
                         "target_window_end": pd.date_range("2017-07-03 00:01:30", periods=n, freq="30s"),
                         "sequence_features": [x.tolist() for x in X], "target_features": X[:, -1].tolist(),
                         "target_attack_present": y, "target_dominant_class": np.where(y, "Denial of Service", "Benign"),
                         "target_dominant_stage": np.where(y, "Impact", "Benign"), "split": split})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("explain")
    frame = synthetic_frame()
    seq = root / "sequences.parquet"
    frame.to_parquet(seq)
    model_dir = root / "model"
    train_experiment(ForecastDataset.from_frame(frame), model_dir, epochs=15, batch_size=32, hidden_size=8,
                     learning_rate=0.01, patience=15)
    return seq, model_dir, frame


class LastWindowLinear(torch.nn.Module):
    """f(x) = 3 * x[:, -1, j]: all attribution must land on (last step, feature j)."""
    def __init__(self, j):
        super().__init__(); self.j = j
    def forward(self, x):
        return 3.0 * x[:, -1, self.j]


def test_integrated_gradients_exact_on_linear_model():
    net = LastWindowLinear(2).eval()
    x = np.random.default_rng(1).normal(size=(3, 4, 5)).astype(np.float32)
    base = np.zeros((4, 5), dtype=np.float32)
    a = integrated_gradients(net, x, base, steps=8)
    expected = np.zeros_like(a); expected[:, -1, 2] = 3.0 * x[:, -1, 2]
    assert np.allclose(a, expected, atol=1e-5)


def test_shap_puts_attribution_on_the_only_input_that_matters():
    pytest.importorskip("shap")
    net = LastWindowLinear(2).eval()
    rng = np.random.default_rng(2)
    bg = rng.normal(size=(50, 4, 5)).astype(np.float32)
    x = rng.normal(size=(2, 4, 5)).astype(np.float32)
    phi = shap_values(net, x, bg, nsamples=100)
    assert phi.shape == x.shape
    mask = np.zeros_like(phi, dtype=bool); mask[:, -1, 2] = True
    assert np.abs(phi[~mask]).max() < 1e-6
    # With a single-point background, expected gradients on a linear model are exact: 3 * (x - bg).
    one = np.repeat(bg[:1], 10, axis=0)
    phi1 = shap_values(net, x, one, nsamples=20)
    assert np.allclose(phi1[:, -1, 2], 3.0 * (x[:, -1, 2] - one[0, -1, 2]), atol=1e-4)


def test_linear_shap_is_exact():
    coef = np.array([[1.0, -2.0, 0.5]])
    bg = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    x = np.array([[3.0, 3.0, 3.0]])
    phi = linear_shap(coef, x, bg)
    assert np.allclose(phi, [[2.0, -2.0, 0.0]])
    # efficiency: sum(phi) = f(x) - f(E[bg]) for a linear decision function
    assert np.isclose(phi.sum(), (coef @ x.T).item() - (coef @ bg.mean(0)).item())


@pytest.mark.parametrize("model,method,tol", [("lstm", "integrated_gradients", 1e-2),
                                             ("logistic_regression", "shap", 1e-6),
                                             ("lstm", "shap", 0.25)])
def test_attributions_add_up_to_the_model_output(trained, model, method, tol):
    if method == "shap" and model == "lstm":
        pytest.importorskip("shap")
    seq, model_dir, _ = trained
    ex = explain_parquet(seq, model_dir, model=model, method=method, limit=4, nsamples=300)
    assert len(ex) == 4
    for e in ex:
        assert e.attributions.shape == (4, len(MODEL_FEATURES))
        assert e.additivity_gap <= tol * max(1.0, abs(e.output - e.reference_output))


def test_trained_model_explanation_finds_the_planted_signal(trained):
    seq, model_dir, frame = trained
    test = frame[(frame.split == "test") & (frame.target_attack_present == 1)]
    ex = explain_parquet(seq, model_dir, test.sequence_id.tolist()[:5], method="integrated_gradients")
    for e in ex:
        top = e.top_features(1)[0]
        assert top["feature"] == SIGNAL and top["direction"] == "towards attack"
        assert top["most_influential_step"] == 3


def test_explanation_serialises_and_rejects_unknown_ids(trained):
    seq, model_dir, frame = trained
    sid = frame[frame.split == "test"].sequence_id.iloc[0]
    d = explain_parquet(seq, model_dir, [sid], method="integrated_gradients")[0].to_dict(k=3, include_matrix=True)
    json.dumps(d)
    assert d["sequence_id"] == sid and len(d["top_features"]) == 3 and len(d["timestep_attribution"]) == 4
    assert np.array(d["attributions"]).shape == (4, len(MODEL_FEATURES))
    with pytest.raises(KeyError):
        explain_parquet(seq, model_dir, ["nope"])
    with pytest.raises(ValueError):
        ModelExplainer(model_dir, np.empty((0, 4, len(MODEL_FEATURES))))


def test_cli_explain_prints_one_json_line_per_sequence(trained, capsys):
    from aegisflow.cli import main
    seq, model_dir, _ = trained
    assert main(["explain", "--input", str(seq), "--model-dir", str(model_dir), "--limit", "2",
                 "--method", "integrated_gradients", "--top", "2"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2 and all(len(l["top_features"]) == 2 for l in lines)


def test_explain_endpoint(trained, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app.replay as replay
    from backend.app.main import create_app
    seq, model_dir, frame = trained
    data_dir = tmp_path / "data"; data_dir.mkdir()
    (data_dir / "sequences.parquet").write_bytes(seq.read_bytes())
    monkeypatch.setattr(replay, "DATA_DIR", data_dir)
    monkeypatch.setattr(replay, "MODEL_DIR", model_dir)
    monkeypatch.setattr(replay, "model_version", lambda: "test")
    client = TestClient(create_app(tmp_path / "l.db", warm_up=False))
    sid = frame[frame.split == "test"].sequence_id.iloc[0]
    r = client.get(f"/explain/{sid}", params={"method": "integrated_gradients", "top": 3})
    assert r.status_code == 200, r.text
    assert r.json()["sequence_id"] == sid and len(r.json()["top_features"]) == 3
    assert client.get("/explain/not-a-sequence", params={"method": "integrated_gradients"}).status_code == 404
    assert client.get(f"/explain/{sid}", params={"method": "bogus"}).status_code == 422
    # explaining does not write alerts
    assert client.get("/health").json()["alerts_in_ledger"] == 0
