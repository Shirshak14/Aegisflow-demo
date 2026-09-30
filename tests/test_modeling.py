import numpy as np
import pandas as pd
import pytest

from aegisflow.ml.modeling import (ForecastDataset, LSTMForecaster,
                                   LABEL_DERIVED_FEATURES, MODEL_FEATURES,
                                   SequencePreprocessor, metrics_at,
                                   positive_class_weight, select_threshold)
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES


def fixture_frame(n=6):
    values = np.arange(n * 3 * len(STANDARD_NUMERIC_WINDOW_FEATURES), dtype=np.float32).reshape(n, 3, -1)
    return pd.DataFrame({"sequence_id": [f"s{i}" for i in range(n)], "host_id": ["h"] * n,
        "sequence_features": list(values), "target_features": list(values[:, -1]),
        "target_attack_present": [0, 1, 0, 0, 1, 0], "target_dominant_class": ["Benign"] * n,
        "target_dominant_stage": ["Benign"] * n, "split": ["train"] * 4 + ["val", "test"]})


def test_tensor_shape_order_and_targets():
    ds = ForecastDataset.from_frame(fixture_frame())
    assert ds.X.shape == (6, 3, 28)
    assert ds.feature_names == MODEL_FEATURES
    assert LABEL_DERIVED_FEATURES.isdisjoint(ds.feature_names)
    assert ds.attack.tolist() == [0, 1, 0, 0, 1, 0]


def test_preprocessor_fit_only_when_called_on_training_and_deterministic():
    frame = fixture_frame(); ds = ForecastDataset.from_frame(frame)
    p = SequencePreprocessor(ds.feature_names).fit(ds.X[ds.indices("train")])
    out1 = p.transform(ds.X[ds.indices("val")]); out2 = p.transform(ds.X[ds.indices("val")])
    assert np.array_equal(out1, out2)
    assert out1.shape == (1, 3, 28)
    transformed_train = p._log(ds.X[:4].astype(float))
    assert transformed_train.shape[-1] == 28
    assert np.allclose(p.medians, np.median(transformed_train.reshape(-1, 28), axis=0))


def test_weight_and_validation_threshold_no_positive():
    weight, counts = positive_class_weight(np.array([0, 0, 1]))
    assert weight == 2 and counts == {"positive": 1, "negative": 2}
    assert select_threshold(np.array([0, 0]), np.array([0.1, 0.9])) == (0.5, "validation set has no positive targets; default 0.5 retained")


def test_lstm_forward_shape():
    torch = pytest.importorskip("torch")
    model = LSTMForecaster(30, 8, 0.1).net
    model.eval()
    assert model(torch.zeros((4, 10, 30))).shape == (4,)


def test_model_state_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    model = LSTMForecaster(30, 8, 0.1).net
    model.eval()
    expected = model(torch.ones((2, 3, 30))).detach()
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = LSTMForecaster(30, 8, 0.1).net
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    assert torch.equal(expected, restored(torch.ones((2, 3, 30))).detach())


def test_threshold_evaluation_uses_requested_threshold_and_no_ids_as_features():
    got = metrics_at(np.array([0, 1]), np.array([0.4, 0.6]), 0.55)
    assert got["threshold"] == 0.55
    assert got["confusion_matrix"] == [[1, 0], [0, 1]]
    ds = ForecastDataset.from_frame(fixture_frame())
    assert "host_id" not in ds.feature_names and "sequence_id" not in ds.feature_names
    changed = fixture_frame()
    changed["target_attack_present"] = 1 - changed["target_attack_present"]
    changed["target_dominant_class"] = "future-only-label"
    changed["target_dominant_stage"] = "future-only-stage"
    assert np.array_equal(ds.X, ForecastDataset.from_frame(changed).X)
