"""Opt-in architectures: attention-LSTM and temporal Transformer (the default LSTM is unchanged)."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aegisflow.cli import build_parser, main, resolve_model_type, train_hyperparameters
from aegisflow.config import load_config
from aegisflow.errors import ConfigError
from aegisflow.ml.modeling import (MODEL_TYPES, ForecastDataset, TransformerForecaster, build_model,
                                   load_model, predict_sequences, train_experiment)
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES

SIGNAL = "syn_count_sum"


def synthetic_frame(n=240, length=6, seed=0):
    """Attack target iff SIGNAL spikes in input window 2 (not the last one), so attention must look back."""
    rng = np.random.default_rng(seed)
    X = rng.gamma(2.0, 5.0, size=(n, length, len(STANDARD_NUMERIC_WINDOW_FEATURES))).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    X[y == 1, 2, STANDARD_NUMERIC_WINDOW_FEATURES.index(SIGNAL)] += 500.0
    split = ["train"] * 144 + ["val"] * 48 + ["test"] * (n - 192)
    t = pd.date_range("2017-07-03", periods=n, freq="30s")
    return pd.DataFrame({"sequence_id": [f"s{i}" for i in range(n)], "host_id": "h", "seq_end_time": t,
                         "target_window_start": t + pd.Timedelta("30s"), "sequence_features": [x.tolist() for x in X],
                         "target_features": X[:, -1].tolist(), "target_attack_present": y,
                         "target_dominant_class": "Benign", "target_dominant_stage": "Benign", "split": split})


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_forward_shapes_and_attention_is_a_distribution(model_type):
    net = build_model(model_type, 28, 16, 0.1).eval()
    x = torch.randn(5, 10, 28)
    assert net(x).shape == (5,)
    if model_type == "lstm":
        assert not hasattr(net, "attention")
    else:
        w = net.attention(x)
        assert w.shape == (5, 10) and torch.all(w >= 0)
        assert torch.allclose(w.sum(dim=1), torch.ones(5), atol=1e-5)


def test_bad_configs_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        TransformerForecaster(28, 18, 0.1)
    with pytest.raises(ValueError, match="unknown model type"):
        build_model("gnn", 28)
    with pytest.raises(ValueError, match="max_len"):
        build_model("transformer", 28, 16).eval()(torch.zeros(1, 65, 28))


@pytest.fixture(scope="module")
def seq_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("seq") / "sequences.parquet"
    synthetic_frame().to_parquet(p)
    return p


@pytest.mark.parametrize("model_type", ["attention_lstm", "transformer"])
def test_train_save_load_predict_round_trip(model_type, seq_path, tmp_path):
    out = tmp_path / model_type
    res = train_experiment(ForecastDataset.from_parquet(seq_path), out, epochs=25, batch_size=32, hidden_size=16,
                           learning_rate=0.005, dropout=0.0, patience=25, model_type=model_type)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["training_config"]["model_type"] == model_type
    net, _ = load_model(out)
    assert type(net).__qualname__.startswith(("AttentionLSTMForecaster", "TransformerForecaster"))
    rows = predict_sequences(seq_path, out)
    assert len(rows) == 240 and rows[0]["model"] == f"cic_ids2017-{model_type}"
    assert all(len(r["attention"]) == 6 and abs(sum(r["attention"]) - 1) < 1e-4 for r in rows)
    # The planted signal is learnable: test ROC-AUC is computed, not asserted to any target number,
    # beyond being clearly better than chance on this easy synthetic task.
    assert "lstm" not in res["models"]
    assert res["models"][model_type]["roc_auc"] > 0.75
    if model_type == "transformer":
        # Encoder positions are tied to their own window, so pooling attention on predicted-attack test
        # sequences should favour the window carrying the signal. (Not asserted for attention_lstm: an
        # LSTM state at step t already summarises steps <= t, so later steps can legitimately win.)
        pos = [r for r in rows[192:] if r["predicted_attack"]]
        assert pos and np.mean([np.argmax(r["attention"]) == 2 for r in pos]) > 0.5


def test_default_lstm_metadata_has_no_model_type_and_loads(seq_path, tmp_path):
    train_experiment(ForecastDataset.from_parquet(seq_path), tmp_path, epochs=1, batch_size=64, hidden_size=4)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert "model_type" not in meta["training_config"]
    net, _ = load_model(tmp_path)
    rows = predict_sequences(seq_path, tmp_path)
    assert rows[0]["model"] == "cic_ids2017-lstm-phase3" and "attention" not in rows[0]


def test_cli_model_choice_and_default_output_dir(monkeypatch, tmp_path):
    cfg = load_config()
    args = build_parser().parse_args(["train"])
    assert resolve_model_type(cfg, args) == "lstm"
    assert resolve_model_type(cfg, build_parser().parse_args(["train", "--model", "transformer"])) == "transformer"
    assert resolve_model_type(load_config(overrides=["model.type=attention_lstm"]), args) == "attention_lstm"
    with pytest.raises(ConfigError):
        train_hyperparameters(load_config(overrides=["model.type=gru"]), args)

    seen = {}
    import aegisflow.ml.modeling as modeling
    monkeypatch.setattr(modeling.ForecastDataset, "from_parquet", classmethod(lambda cls, p: None))
    monkeypatch.setattr(modeling, "train_experiment",
                        lambda data, out, **kw: seen.update(out=out, **kw) or {"models": {}})
    assert main(["train", "--model", "transformer", "--epochs", "1"]) == 0
    assert seen["model_type"] == "transformer" and seen["out"].name == "cic_ids2017_transformer"
    assert main(["train", "--epochs", "1"]) == 0
    assert seen["model_type"] == "lstm" and seen["out"].name == "cic_ids2017"
