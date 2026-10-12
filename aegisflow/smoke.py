"""Tiny synthetic end-to-end smoke test: flows -> windows -> sequences -> train -> manifest -> replay -> ledger.

Runs without CIC-IDS2017 or any downloaded file, in a temporary directory, in well under a minute:

    python -m aegisflow.smoke            # prints a JSON report, exit 1 on any failed check

The traffic is generated here with a fixed seed: benign hosts with steady traffic and one "attacker" host that sends
DoS-labelled bursts in episodes. The numbers it prints say nothing about detection quality on real data; they only
show that every stage of the pipeline runs and hands valid data to the next one. Nothing in the repository's data/,
artifacts/ or reports/ is touched.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, load_config


def synthetic_flows(seed: int = 7, hours: float = 3.0) -> pd.DataFrame:
    """Cleaned canonical flows (the schema ingestion produces) with labels the repository's stage mapping knows."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2017-07-05 09:00:00")
    rows = []
    hosts = [f"10.0.0.{i}" for i in range(1, 9)]
    for h in hosts:  # steady benign traffic, a few flows per minute
        n = int(hours * 60 * 4)
        ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, hours * 3600, n)), unit="s")
        for t in ts:
            pk = float(rng.integers(2, 30))
            rows.append((t, h, "10.0.1.1", 40000 + int(rng.integers(0, 2000)), 80, "TCP",
                         float(rng.uniform(0.01, 2.0)), pk, pk * float(rng.uniform(60, 120)), "BENIGN"))
    attacker = "172.16.0.9"  # benign background plus three DoS episodes, spread over train, validation and test
    for k in range(int(hours * 60)):
        t = t0 + pd.Timedelta(seconds=k * 60 + int(rng.integers(0, 60)))
        rows.append((t, attacker, "10.0.1.1", 50000, 80, "TCP", 0.5, 10.0, 800.0, "BENIGN"))
    for start_min in (int(hours * 60 * f) for f in (0.2, 0.67, 0.9)):
        for s in range(0, 12 * 60):
            if rng.random() < 0.8:
                t = t0 + pd.Timedelta(seconds=start_min * 60 + s)
                rows.append((t, attacker, "10.0.1.1", 50000 + s % 500, 80, "TCP", 0.001, 4.0, 2000.0, "DoS Hulk"))
    df = pd.DataFrame(rows, columns=["timestamp", "source_ip", "destination_ip", "source_port", "destination_port",
                                     "protocol", "flow_duration", "packet_count", "byte_count", "dataset_label"])
    # The model expects every optional sensor column CIC-IDS2017 supplies; fill them with plausible values.
    pk, by = df["packet_count"], df["byte_count"]
    df["packet_size_mean"], df["packet_size_std"] = by / pk, (by / pk) * 0.1
    df["packet_size_min"], df["packet_size_max"] = (by / pk) * 0.5, (by / pk) * 1.5
    df["inter_arrival_mean"], df["inter_arrival_std"] = df["flow_duration"] / pk, df["flow_duration"] / pk * 0.2
    for c, f in (("syn_count", 0.1), ("ack_count", 0.8), ("rst_count", 0.0), ("fin_count", 0.05), ("psh_count", 0.2),
                 ("urg_count", 0.0), ("fwd_packet_count", 0.6), ("bwd_packet_count", 0.4)):
        df[c] = pk * f
    df["fwd_byte_count"], df["bwd_byte_count"] = by * 0.6, by * 0.4
    df["ttl_mean"], df["ttl_std"], df["tcp_window_size"], df["retransmission_count"] = 64.0, 1.0, 8192.0, 0.0
    attack = df["dataset_label"] != "BENIGN"
    df.loc[attack, "syn_count"] = pk[attack] * 0.9  # SYN-flood-like
    for c in ("source_port", "destination_port"):
        df[c] = df[c].astype("Int64")
    for c in ("source_ip", "destination_ip", "protocol", "dataset_label"):
        df[c] = df[c].astype("string")
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _check(report: dict, name: str, ok: bool, detail: object) -> None:
    report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})


def run_smoke(workdir: Path | None = None, *, keep: bool = False) -> dict:
    """Run every stage in a scratch repository root; returns the report (``report["ok"]`` is the verdict)."""
    from backend.app import replay
    from backend.app.audit import Ledger
    from .manifest import verify_manifest, write_manifest
    from .ml.modeling import ForecastDataset, train_experiment
    from .ml.preprocessing.labels import apply_stage_mapping
    from .ml.temporal.pipeline import run_preprocessing

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="aegisflow_smoke_"))
    report: dict = {"workdir": str(tmp), "checks": [], "ok": False}
    try:
        shutil.copytree(REPO_ROOT / "configs", tmp / "configs", dirs_exist_ok=True)
        cfg = load_config(root=tmp)
        flows = apply_stage_mapping(synthetic_flows(), cfg, "cic_ids2017")
        flows["source_dataset"] = "synthetic_smoke"
        interim = tmp / "data" / "interim"
        interim.mkdir(parents=True)
        flows.to_parquet(interim / "cic_ids2017.parquet", index=False)
        _check(report, "synthetic flows generated", len(flows) > 1000,
               {"flows": len(flows), "attack_flows": int((flows["normalized_attack_class"] != "Benign").sum())})

        res = run_preprocessing(cfg, "cic_ids2017")
        seq = pd.read_parquet(res.sequences_path, columns=["split", "target_attack_present"])
        by_split = seq.groupby("split")["target_attack_present"].agg(["size", "sum"]).to_dict("index")
        _check(report, "windows and sequences built, all three splits non-empty",
               res.windows_count > 0 and {"train", "val", "test"} <= set(by_split),
               {"windows": res.windows_count, "sequences": res.sequences_count,
                "per_split": {k: {"sequences": int(v["size"]), "positives": int(v["sum"])} for k, v in by_split.items()}})

        model_dir = tmp / "artifacts" / "models" / "cic_ids2017"
        out = train_experiment(ForecastDataset.from_parquet(res.sequences_path), model_dir, seed=42, epochs=3,
                               batch_size=64, hidden_size=8)
        _check(report, "model trained and artifacts written",
               all((model_dir / f).exists() for f in ("best.pt", "logistic.joblib", "preprocessor.joblib", "metadata.json",
                                                      "metrics.json")), sorted(p.name for p in model_dir.iterdir()))

        write_manifest(model_dir, root=tmp, config=cfg, split_metadata=res.split_metadata_path)
        v = verify_manifest(model_dir)
        _check(report, "manifest written and verifies", v["ok"], v)
        (model_dir / "metadata.json").write_text((model_dir / "metadata.json").read_text() + " ", encoding="utf-8")
        _check(report, "manifest detects a modified file", verify_manifest(model_dir)["changed"] == ["metadata.json"],
               "appended one byte to metadata.json")

        old = (replay.DATA_DIR, replay.MODEL_DIR, replay.model_version)
        replay.DATA_DIR, replay.MODEL_DIR = res.sequences_path.parent, model_dir
        replay.model_version = lambda: "synthetic-smoke-model"
        try:
            engine = replay.ReplayEngine(Ledger(tmp / "smoke_ledger.db"))
            engine.prepare()
            engine.session_id = "smoke"
            for i in range(len(engine.events)):
                engine._process(engine.events.iloc[i])
            alerts = engine.ledger.list(limit=100000)
            verdict = engine.ledger.verify()
            cp = tmp / "checkpoint.json"
            engine.ledger.export_checkpoint(cp)
            checkpoint = engine.ledger.verify_checkpoint(cp)
            engine.ledger.close()
        finally:
            replay.DATA_DIR, replay.MODEL_DIR, replay.model_version = old
        _check(report, "replay scored the test split and wrote alerts to the ledger", len(engine.events) > 0,
               {"test_sequences_replayed": len(engine.events), "alerts": len(alerts)})
        _check(report, "ledger hash chain verifies", verdict["status"] == "VERIFIED", verdict)
        _check(report, "external ledger checkpoint matches", checkpoint["status"] == "MATCHES", checkpoint)
        report["note"] = ("Synthetic data: these counts show the pipeline runs end to end, not how well it detects "
                          "attacks. Real results come from the CIC-IDS2017 walkthrough in docs/release.md.")
        report["ok"] = all(c["ok"] for c in report["checks"])
        return report
    finally:
        if not keep and workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    report = run_smoke()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
