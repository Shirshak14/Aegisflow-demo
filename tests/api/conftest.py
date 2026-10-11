import pytest

AUTH = {"X-API-Key": "test-key"}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """POST /alerts is disabled unless a key is configured; the tests configure one."""
    monkeypatch.setenv("AEGISFLOW_API_KEY", "test-key")
    monkeypatch.delenv("AEGISFLOW_LEDGER_KEY", raising=False)


def build_synthetic_sequences(seed: int = 7, seq_len: int = 5):
    """A small sequences table with the real column layout and a learnable signal, built in memory.

    Attack sequences have much larger traffic counts than benign ones, so a tiny model separates them. Every
    split (train/val/test) has several positives, which the threshold selection and replay assertions need.
    """
    import numpy as np
    import pandas as pd

    from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES as F

    rng = np.random.default_rng(seed)
    names = list(F)
    n_feat = len(names)
    sizes = {"train": 240, "val": 80, "test": 120}
    rows, t0 = [], pd.Timestamp("2017-07-05 09:00:00")
    step = 0
    for split, n in sizes.items():
        for i in range(n):
            attack = int(rng.random() < 0.25)
            base = rng.gamma(2.0, 2.0, size=(seq_len + 1, n_feat)).astype(np.float32)
            for k in ("flow_count", "packet_count_sum", "byte_count_sum"):
                base[:, names.index(k)] *= 40.0 if attack else 1.0
            # label-derived columns: attack traffic ratio, nonzero only while an attack is under way
            base[:, names.index("attack_flow_ratio")] = (rng.random(seq_len + 1) < 0.5) * attack
            start = t0 + pd.Timedelta(seconds=30 * step)
            step += 1
            rows.append({
                "sequence_id": f"h{i % 3}_{split}_{i}", "host_id": f"10.0.0.{i % 3 + 1}",
                "sequence_features": [r.tolist() for r in base[:seq_len]], "target_features": base[seq_len].tolist(),
                "target_attack_present": attack,
                "target_dominant_class": "DoS" if attack else "Benign",
                "target_dominant_stage": "Impact" if attack else "Benign", "split": split,
                "seq_end_time": start, "target_window_start": start + pd.Timedelta(seconds=30),
                "target_window_end": start + pd.Timedelta(seconds=90)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_replay_env(tmp_path_factory):
    """Trains a tiny real model on synthetic sequences; returns (data_dir, model_dir)."""
    pytest.importorskip("torch")
    from aegisflow.ml.modeling import ForecastDataset, train_experiment

    root = tmp_path_factory.mktemp("synthetic_replay")
    data_dir, model_dir = root / "data", root / "model"
    data_dir.mkdir()
    frame = build_synthetic_sequences()
    frame.to_parquet(data_dir / "sequences.parquet", index=False)
    train_experiment(ForecastDataset.from_frame(frame), model_dir, seed=42, epochs=4, batch_size=64, hidden_size=8)
    return data_dir, model_dir


@pytest.fixture()
def synthetic_replay(synthetic_replay_env, monkeypatch):
    """Point the replay engine at the synthetic data and model instead of data/processed and artifacts/models."""
    data_dir, model_dir = synthetic_replay_env
    from backend.app import replay
    monkeypatch.setattr(replay, "DATA_DIR", data_dir)
    monkeypatch.setattr(replay, "MODEL_DIR", model_dir)
    monkeypatch.setattr(replay, "model_version", lambda: "synthetic-test-model")
    return data_dir, model_dir
