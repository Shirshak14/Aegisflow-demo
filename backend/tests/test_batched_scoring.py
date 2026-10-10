"""Batched LSTM scoring must give the same probabilities as one big forward pass."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aegisflow.ml.modeling import LSTMForecaster, score_batched  # noqa: E402


def _net_and_data(n=1000, seq=10, feats=28):
    torch.manual_seed(0)
    net = LSTMForecaster(feats, 16, 0.2).net.eval()
    x = np.random.default_rng(0).normal(size=(n, seq, feats)).astype(np.float32)
    return net, x


@pytest.mark.parametrize("batch_size", [1, 7, 256, 999, 1000, 4096])
def test_batched_scores_equal_single_call(batch_size):
    net, x = _net_and_data()
    with torch.no_grad():
        single = torch.sigmoid(net(torch.tensor(x))).numpy()
    batched = score_batched(lambda t: torch.sigmoid(net(t)), x, batch_size=batch_size)
    assert batched.shape == single.shape and batched.dtype == single.dtype
    # LSTM matmuls can differ in the last bits with batch shape; allow float32 rounding only.
    np.testing.assert_allclose(batched, single, rtol=1e-5, atol=1e-6)


def test_empty_input_returns_empty():
    net, x = _net_and_data(n=0)
    assert score_batched(lambda t: torch.sigmoid(net(t)), x).shape == (0,)
