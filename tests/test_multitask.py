"""Multi-task model: attack + stage + K-step future state, built from the real sequence pipeline."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aegisflow.ml.multitask import (DEFAULT_STAGES, MultiTaskData, MultiTaskPredictor, inverse_transform,
                                    multi_horizon_targets, train_multitask)
from aegisflow.ml.modeling import MODEL_FEATURES, SequencePreprocessor
from aegisflow.ml.temporal.sequences import STANDARD_NUMERIC_WINDOW_FEATURES, build_host_sequences

PORTS = "unique_destination_ports"
PPS = "syn_count_sum"


def synthetic_windows(hosts=12, n=60, seed=0):
    """Non-overlapping 60 s windows. Each host has one attack episode (Recon = port fan-out, or
    Impact = packet flood) lasting 15 windows; flow_count follows a per-host sine (forecastable)."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2017-07-03 09:00:00")
    for h in range(hosts):
        stage = "Reconnaissance" if h % 2 == 0 else "Impact"
        start = rng.integers(10, n - 25)
        for i in range(n):
            r = {f: float(rng.gamma(2.0, 2.0)) for f in STANDARD_NUMERIC_WINDOW_FEATURES}
            r["flow_count"] = 20 + 10 * np.sin(i / 3 + h)
            attack = start <= i < start + 15
            if attack:
                r[PORTS if stage == "Reconnaissance" else PPS] += 400.0
            r.update(host_id=f"10.0.0.{h}", window_start=t0 + pd.Timedelta(minutes=i),
                     window_end=t0 + pd.Timedelta(minutes=i + 1), attack_present=int(attack),
                     attack_flow_ratio=float(attack), benign_flow_ratio=float(not attack),
                     dominant_class="Attack" if attack else "Benign", dominant_stage=stage if attack else "Benign",
                     stage_distribution="{}", label_confidence="inferred" if attack else "ground_truth")
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pipeline():
    w = synthetic_windows(hosts=30)
    seq = build_host_sequences(w, sequence_length=5, forecast_horizon=1)
    hosts = sorted(seq.host_id.unique())
    # Host-disjoint split so the test cannot pass by memorising hosts.
    split = {h: "train" if i < 20 else "val" if i < 25 else "test" for i, h in enumerate(hosts)}
    seq["split"] = seq.host_id.map(split)
    return w, seq


def test_multi_horizon_targets_follow_the_sequence_horizon_definition(pipeline):
    w, seq = pipeline
    fut, mask = multi_horizon_targets(seq, w, 3)
    for k in (1, 2, 3):
        ref = build_host_sequences(w, sequence_length=5, forecast_horizon=k).set_index("sequence_id")
        for i in [0, 7, 100, len(seq) - 1]:
            sid = seq.sequence_id.iloc[i]
            if sid in ref.index:
                exp = np.asarray(ref.loc[sid, "target_features"])[[STANDARD_NUMERIC_WINDOW_FEATURES.index(f) for f in MODEL_FEATURES]]
                assert mask[i, k - 1] and np.allclose(fut[i, k - 1], exp, rtol=1e-5)
            else:  # host ran out of future windows
                assert not mask[i, k - 1]
    assert not mask[:, 2].all()  # the last sequences of each host have no 3rd future window


def test_build_rejects_mismatched_windows(pipeline):
    w, seq = pipeline
    bad = w.copy(); bad["flow_count"] += 1000
    with pytest.raises(ValueError, match="disagree"):
        MultiTaskData.build(seq, bad, horizons=2)
    with pytest.raises(ValueError, match="host_windows"):
        MultiTaskData.build(seq, None, horizons=2)
    odd = seq.copy(); odd.loc[0, "target_dominant_stage"] = "Teleportation"
    with pytest.raises(ValueError, match="stages_order"):
        MultiTaskData.build(odd, None, horizons=1)


def test_inverse_transform_round_trips():
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 50.0, size=(30, 4, len(MODEL_FEATURES)))
    p = SequencePreprocessor(MODEL_FEATURES).fit(X)
    assert np.allclose(inverse_transform(p, p.transform(X)), X, rtol=1e-4, atol=1e-3)


@pytest.fixture(scope="module")
def trained(pipeline, tmp_path_factory):
    w, seq = pipeline
    data = MultiTaskData.build(seq, w, horizons=2)
    out = tmp_path_factory.mktemp("mt")
    res = train_multitask(data, out, epochs=40, batch_size=64, hidden_size=32, learning_rate=0.005,
                          dropout=0.1, patience=10)
    return data, out, res


def test_heads_learn_attack_stage_and_future_state(trained):
    data, out, res = trained
    # Attack onsets are unpredictable by design, so attack/stage are well above chance but not perfect.
    assert res["attack"]["roc_auc"] > 0.8
    st = res["stage"]
    # Stage on attack-positive test targets (unseen hosts): far above the majority (Benign) baseline of 0.
    assert st["attack_target_count"] > 0 and st["majority_baseline"]["attack_stage_accuracy"] == 0.0
    assert st["attack_stage_accuracy"] > 0.7
    assert "Exfiltration" in st["note"]  # stages with no training data are called out
    for h in res["future_state"]["per_horizon"]:
        assert h["count"] > 0 and h["mae"] < h["persistence_mae"]


def test_artifacts_and_predictor(trained):
    data, out, res = trained
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["model_type"] == "multitask_lstm" and meta["stages"] == list(DEFAULT_STAGES) and meta["horizons"] == 2
    assert json.loads((out / "metrics.json").read_text())["stage"]["accuracy"] == res["stage"]["accuracy"]
    pred = MultiTaskPredictor(out)
    te = data.base.indices("test")
    p = pred.predict(data.base.X[te])
    assert p["future_state"].shape == (len(te), 2, len(MODEL_FEATURES))
    assert np.allclose(p["stage_probabilities"].sum(1), 1, atol=1e-5)
    # future flow_count is back in raw units (the synthetic sine lives in [10, 30])
    fc = p["future_state"][:, 0, MODEL_FEATURES.index("flow_count")]
    assert 5 < np.median(fc) < 35
    rows = pred.rows(data.base.X[te[:3]], ["a", "b", "c"], stage_threshold=1.01,
                     mitre_lookup=lambda s: {"tactic_id": "X"})
    assert all(r["predicted_stage"] == "UNCERTAIN" and "mitre" not in r for r in rows)
    pos = te[data.base.attack[te] == 1][:3]
    rows = pred.rows(data.base.X[pos], ["x"] * len(pos), stage_threshold=0.0,
                     mitre_lookup={"Reconnaissance": {"tactic_id": "TA0043"}, "Impact": {"tactic_id": "TA0040"}}.get)
    assert all(r["predicted_stage"] in DEFAULT_STAGES for r in rows)
    assert any(r.get("mitre") and r["mitre"]["tactic_id"] in {"TA0043", "TA0040"} for r in rows)


def test_non_multitask_dir_is_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"feature_names": []}))
    with pytest.raises(ValueError, match="not a multi-task"):
        MultiTaskPredictor(tmp_path)


def test_cli_forecast(trained, tmp_path, capsys):
    from aegisflow.cli import main
    data, out, _ = trained
    p = tmp_path / "sequences.parquet"
    data.base.frame.to_parquet(p)
    assert main(["forecast", "--input", str(p), "--model-dir", str(out), "--limit", "2"]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(rows) == 2 and len(rows[0]["future_state"]) == 2 and "stage_distribution" in rows[0]


def test_replay_uses_stage_model_only_when_configured(trained, tmp_path, monkeypatch):
    import backend.app.replay as replay
    from backend.app.audit import Ledger
    from aegisflow.ml.modeling import ForecastDataset, train_experiment
    data, mt_dir, _ = trained
    data_dir = tmp_path / "data"; data_dir.mkdir()
    frame = data.base.frame.copy()
    frame["target_window_end"] = pd.to_datetime(frame.target_window_start) + pd.Timedelta(minutes=1)
    frame.to_parquet(data_dir / "sequences.parquet")
    lstm_dir = tmp_path / "lstm"
    train_experiment(ForecastDataset.from_frame(frame), lstm_dir, epochs=3, batch_size=32, hidden_size=8)
    monkeypatch.setattr(replay, "DATA_DIR", data_dir)
    monkeypatch.setattr(replay, "MODEL_DIR", lstm_dir)
    monkeypatch.setattr(replay, "model_version", lambda: "test")

    def run(stage_dir):
        cfg = replay.load_config(overrides=[f"replay.stage_model_dir={stage_dir}"] if stage_dir else [])
        monkeypatch.setattr(replay, "load_config", lambda: cfg)
        eng = replay.ReplayEngine(Ledger(tmp_path / f"{bool(stage_dir)}.db"))
        eng.prepare(); eng.session_id = "t"
        for i in range(len(eng.events)):
            eng._process(eng.events.iloc[i])
        return eng, eng.ledger.list(limit=100000)

    orig = replay.load_config
    eng, alerts = run(None)
    assert "pred_stage" not in eng.events and {a["predicted_stage"] for a in alerts} <= {"UNCERTAIN"}
    assert all(json.loads(a["risk_components"])["stage_severity"] == 0 for a in alerts)
    monkeypatch.setattr(replay, "load_config", orig)
    eng, alerts = run(str(mt_dir))
    assert alerts, "the trained LSTM should alert on the synthetic attacks"
    stages = {a["predicted_stage"] for a in alerts}
    assert stages - {"UNCERTAIN"}, stages
    assert any(json.loads(a["risk_components"])["stage_severity"] > 0 for a in alerts)
    assert eng.risk_config()["max_reachable_risk"] > 75
    assert eng.ledger.verify()["status"] == "VERIFIED"
