"""
AegisFlow command-line interface.

    python -m aegisflow <command> [options]

Implemented commands:
- ingest: Load, clean, and map raw dataset into data/interim/<name>.parquet
- validate-dataset: Check dataset presence and readable formats
- feature-registry: Print machine-readable canonical feature catalog
- eda: Exploratory analysis on ingested datasets
- list-datasets: Display available dataset adapters
- preprocess: End-to-end canonical feature engineering, host windowing, sequences and chronological split
- rebuild-temporal: regenerate host windows/sequences/reports from existing engineered flows
- window: host-level temporal window aggregation
- train: train the majority / logistic-regression / LSTM baselines
- predict: run the trained LSTM on a sequences Parquet file
- doctor: list the files the demo needs that are missing, and the command that creates each

The replay engine, audit ledger and dashboard are served by the FastAPI app, not by this CLI:
    uvicorn backend.app.main:app --port 8000
The `evaluate`, `replay` and `serve` subcommands below are placeholders that only report "not implemented".
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .errors import AegisFlowError, ConfigError, NotImplementedPhaseError
from .logging_setup import configure_logging, get_logger

log = get_logger("CLI")

# train CLI flag (argparse dest) -> (config.model key, type)
_TRAIN_PARAMS = {
    "epochs": ("epochs", int),
    "batch_size": ("batch_size", int),
    "learning_rate": ("learning_rate", float),
    "hidden_size": ("hidden_size", int),
    "dropout": ("dropout", float),
    "patience": ("early_stopping_patience", int),
}


def train_hyperparameters(cfg, args) -> dict:
    """Resolve training hyperparameters: an explicit CLI flag wins, else ``config.model.*``."""
    model_cfg = cfg.config.model
    resolve_model_type(cfg, args)  # validate early
    out = {}
    for flag, (key, cast) in _TRAIN_PARAMS.items():
        value = getattr(args, flag, None)
        out[flag] = cast(value if value is not None else model_cfg[key])
    return out


def resolve_model_type(cfg, args) -> str:
    """``--model`` wins, else ``config.model.type``; must be one of modeling.MODEL_TYPES."""
    from .ml.modeling import MODEL_TYPES
    model_type = str(getattr(args, "model", None) or cfg.config.model.type).lower()
    if model_type not in MODEL_TYPES:
        raise ConfigError(
            f"model type '{model_type}' is not implemented; choose one of {', '.join(MODEL_TYPES)} "
            "(--model, or model.type in configs/config.yaml)."
        )
    return model_type


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Override a config.yaml key, e.g. --set model.batch_size=64 (repeatable)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aegisflow", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Dataset discovery & validation
    p_list = sub.add_parser("list-datasets", help="List dataset adapters and their status.")
    _common_args(p_list)

    p_val = sub.add_parser("validate-dataset", help="Check that a dataset's raw files are present and readable.")
    p_val.add_argument("--dataset", required=True, help="Key from configs/datasets.yaml, e.g. cic_ids2017")
    p_val.add_argument("--no-sample-load", action="store_true", help="Skip the small real-load check.")
    _common_args(p_val)

    # 2. Ingestion & EDA
    p_ing = sub.add_parser("ingest", help="Load, clean and label a dataset; write data/interim/<name>.parquet")
    p_ing.add_argument("--dataset", required=True)
    p_ing.add_argument("--sample-size", type=int, default=None,
                        help="Load only this many rows total (fast dev iteration). Omit for the full dataset.")
    _common_args(p_ing)

    p_eda = sub.add_parser("eda", help="Compute summary stats + figures from an ingested dataset.")
    p_eda.add_argument("--dataset", required=True)
    _common_args(p_eda)

    p_reg = sub.add_parser("feature-registry", help="Print the canonical feature registry as a Markdown table.")
    p_reg.add_argument("--level", choices=["flow", "host-window", "sequence"], default=None,
                       help="Filter registry table by aggregation level.")
    _common_args(p_reg)

    # 3. Phase 2 Preprocessing & Temporal Windowing
    p_prep = sub.add_parser("preprocess", help="Run Phase 2 feature engineering, host temporal windows, sequences & split.")
    p_prep.add_argument("--dataset", required=True, help="Dataset key, e.g. cic_ids2017")
    p_prep.add_argument("--sample-size", type=int, default=None, help="Row limit for fast dev iteration.")
    p_prep.add_argument("--window-size", type=float, default=None, help="Temporal window duration in seconds (e.g. 60).")
    p_prep.add_argument("--stride", type=float, default=None, help="Window stride in seconds (e.g. 30).")
    p_prep.add_argument("--seq-len", type=int, default=None, help="Number of steps in sequence input (e.g. 10).")
    p_prep.add_argument("--horizon", type=int, default=None, help="Future forecast horizon in steps (e.g. 1).")
    p_prep.add_argument("--reingest", action="store_true", help="Force re-running Phase 1 ingestion before preprocessing.")
    _common_args(p_prep)

    p_win = sub.add_parser("window", help="Run host-level temporal window aggregation on canonical flows.")
    p_win.add_argument("--dataset", required=True, help="Dataset key, e.g. cic_ids2017")
    p_win.add_argument("--sample-size", type=int, default=None, help="Row limit for fast dev iteration.")
    p_win.add_argument("--window-size", type=float, default=None, help="Temporal window duration in seconds.")
    p_win.add_argument("--stride", type=float, default=None, help="Window stride in seconds.")
    _common_args(p_win)

    p_rebuild = sub.add_parser("rebuild-temporal", help="Rebuild Phase 2 windows/sequences from saved flows.parquet; no raw re-ingestion.")
    p_rebuild.add_argument("--dataset", required=True, help="Dataset key, e.g. cic_ids2017")
    _common_args(p_rebuild)

    p_train = sub.add_parser("train", help="Train Phase 3 CIC-IDS2017 forecasting baselines.")
    p_train.add_argument("--dataset", default="cic_ids2017")
    # Defaults come from config.yaml `model:`; pass a flag to override it for one run.
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--learning-rate", type=float, default=None)
    p_train.add_argument("--hidden-size", type=int, default=None)
    p_train.add_argument("--dropout", type=float, default=None)
    p_train.add_argument("--patience", type=int, default=None)
    p_train.add_argument("--model", choices=["lstm", "attention_lstm", "transformer"], default=None,
                         help="Architecture (default: config model.type = lstm). Non-LSTM models are written to "
                              "artifacts/models/<dataset>_<model> so the demo model is never overwritten.")
    p_train.add_argument("--output-dir", default=None, help="Override the model output directory.")
    _common_args(p_train)

    p_predict = sub.add_parser("predict", help="Run Phase 3 model inference on a sequences Parquet file "
                               "(attention models also print per-window attention weights).")
    p_predict.add_argument("--input", required=True)
    p_predict.add_argument("--model-dir", default="artifacts/models/cic_ids2017")
    p_predict.add_argument("--threshold", type=float, default=None)
    _common_args(p_predict)

    p_explain = sub.add_parser("explain", help="Explain predictions per feature and time step (SHAP / Integrated Gradients).")
    p_explain.add_argument("--dataset", default="cic_ids2017")
    p_explain.add_argument("--input", default=None, help="sequences Parquet (default: data/processed/<dataset>/sequences.parquet)")
    p_explain.add_argument("--model-dir", default=None, help="default: artifacts/models/<dataset>")
    p_explain.add_argument("--model", choices=["lstm", "logistic_regression"], default="lstm")
    p_explain.add_argument("--method", choices=["shap", "integrated_gradients"], default="shap",
                           help="LSTM attribution method; logistic regression always uses exact linear SHAP")
    p_explain.add_argument("--sequence-id", action="append", default=[], help="repeatable; default: first --limit test sequences")
    p_explain.add_argument("--limit", type=int, default=5)
    p_explain.add_argument("--top", type=int, default=5, help="top features to print per sequence")
    _common_args(p_explain)
    p_mt = sub.add_parser("train-multitask", help="Train the multi-task model: next-window attack, attack stage, "
                                                  "and K-step future network state.")
    p_mt.add_argument("--dataset", default="cic_ids2017")
    p_mt.add_argument("--horizons", type=int, default=3, help="K future windows to forecast (needs host_windows.parquet if > 1)")
    p_mt.add_argument("--epochs", type=int, default=None)
    p_mt.add_argument("--batch-size", type=int, default=None)
    p_mt.add_argument("--learning-rate", type=float, default=None)
    p_mt.add_argument("--hidden-size", type=int, default=None)
    p_mt.add_argument("--dropout", type=float, default=None)
    p_mt.add_argument("--patience", type=int, default=None)
    p_mt.add_argument("--output-dir", default=None, help="default: artifacts/models/<dataset>_multitask")
    _common_args(p_mt)

    p_fc = sub.add_parser("forecast", help="Predict attack, stage (+ MITRE tactic) and K-step future state per sequence.")
    p_fc.add_argument("--dataset", default="cic_ids2017")
    p_fc.add_argument("--input", default=None, help="sequences Parquet (default: data/processed/<dataset>/sequences.parquet)")
    p_fc.add_argument("--model-dir", default=None, help="default: artifacts/models/<dataset>_multitask")
    p_fc.add_argument("--sequence-id", action="append", default=[])
    p_fc.add_argument("--limit", type=int, default=5, help="first N test sequences when no --sequence-id")
    _common_args(p_fc)
    p_ip = sub.add_parser("ingest-pcap", help="Convert a .pcap/.pcapng into canonical bidirectional flows (Parquet or CSV).")
    p_ip.add_argument("--input", required=True)
    p_ip.add_argument("--output", required=True, help="*.parquet or *.csv")
    p_ip.add_argument("--reader", choices=["scapy", "pyshark"], default="scapy")
    p_ip.add_argument("--idle-timeout", type=float, default=120.0, help="seconds")
    p_ip.add_argument("--active-timeout", type=float, default=3600.0, help="seconds")
    p_ip.add_argument("--labels", default=None, help="optional CSV: source_ip,start,end,label (labels from stages.yaml)")
    p_ip.add_argument("--label-map", default="cic_ids2017", help="stages.yaml section the --labels use")
    _common_args(p_ip)

    p_sp = sub.add_parser("score-pcap", help="Score a .pcap/.pcapng with a trained model: attack probability per host sequence.")
    p_sp.add_argument("--input", required=True)
    p_sp.add_argument("--model-dir", default="artifacts/models/cic_ids2017")
    p_sp.add_argument("--reader", choices=["scapy", "pyshark"], default="scapy")
    p_sp.add_argument("--threshold", type=float, default=None)
    p_sp.add_argument("--output", default=None, help="optional CSV of all scored sequences")
    _common_args(p_sp)

    p_inf = sub.add_parser("ingest-netflow", help="Convert NetFlow v5/v9/IPFIX (capture of exports, or nfdump CSV) "
                                                 "into canonical flows.")
    p_inf.add_argument("--input", required=True, help=".pcap/.pcapng of export traffic, or `nfdump -o csv` output")
    p_inf.add_argument("--format", choices=["auto", "capture", "nfdump-csv"], default="auto")
    p_inf.add_argument("--output", required=True, help="*.parquet or *.csv")
    _common_args(p_inf)

    p_snf = sub.add_parser("score-netflow", help="Score NetFlow/IPFIX input with a trained model, per host sequence.")
    p_snf.add_argument("--input", required=True)
    p_snf.add_argument("--format", choices=["auto", "capture", "nfdump-csv"], default="auto")
    p_snf.add_argument("--model-dir", default="artifacts/models/cic_ids2017")
    p_snf.add_argument("--threshold", type=float, default=None)
    p_snf.add_argument("--output", default=None, help="optional CSV of all scored sequences")
    _common_args(p_snf)
    p_st = sub.add_parser("stream", help="Score a capture or flow file as a stream: windows are scored as they close.")
    p_st.add_argument("--input", required=True, help=".pcap/.pcapng, or canonical flows .parquet/.csv")
    p_st.add_argument("--model-dir", default="artifacts/models/cic_ids2017")
    p_st.add_argument("--chunk-seconds", type=float, default=30.0, help="push flows in chunks of this much traffic time")
    p_st.add_argument("--lateness", type=float, default=0.0, help="allowed out-of-order seconds")
    p_st.add_argument("--alerts-only", action="store_true")
    _common_args(p_st)
    p_gnn = sub.add_parser("train-gnn", help="Train the temporal GNN (host-communication graph per window + GRU).")
    p_gnn.add_argument("--dataset", default="cic_ids2017")
    for flag, typ in (("--epochs", int), ("--batch-size", int), ("--learning-rate", float), ("--hidden-size", int),
                      ("--dropout", float), ("--patience", int)):
        p_gnn.add_argument(flag, type=typ, default=None)
    p_gnn.add_argument("--output-dir", default=None, help="default: artifacts/models/<dataset>_tgnn")
    _common_args(p_gnn)
    p_ex = sub.add_parser("export-alerts", help="Export ledger alerts as CEF, RFC 5424 syslog or JSON lines (SIEM).")
    p_ex.add_argument("--db", default=None, help="ledger SQLite file (default: config database.url)")
    p_ex.add_argument("--format", choices=["cef", "syslog", "jsonl"], default="cef")
    p_ex.add_argument("--since-id", type=int, default=0)
    p_ex.add_argument("--limit", type=int, default=10000)
    p_ex.add_argument("--output", default=None, help="write to this file instead of stdout")
    p_ex.add_argument("--syslog-host", default=None, help="also send each line via UDP syslog to this host")
    p_ex.add_argument("--syslog-port", type=int, default=514)
    _common_args(p_ex)

    p_doc = sub.add_parser("doctor", help="Check that the data, model and report files the demo needs exist; "
                                          "exit 1 and print the fix command for each missing one.")
    _common_args(p_doc)

    # Planned later phases
    for name, phase in [
        ("evaluate", "Phase 3/4 (evaluation report against held-out test split)"),
        ("replay", "Phase 6 (replay engine + hash-chained audit log)"),
        ("serve", "Phase 6 (FastAPI backend)"),
    ]:
        p = sub.add_parser(name, help=f"[NOT IMPLEMENTED YET -- {phase}]")
        _common_args(p)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(overrides=args.overrides)
    configure_logging(level=args.log_level, json_file=cfg.path(cfg.config.logging.json_file))

    try:
        return _dispatch(args, cfg)
    except AegisFlowError as exc:
        log.error(str(exc))
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, cfg) -> int:
    if args.command == "doctor":
        from .doctor import demo_ready, format_report, run_checks
        results = run_checks(cfg.root)
        print(format_report(results))
        return 0 if demo_ready(results) else 1
    if args.command == "list-datasets":
        return _cmd_list_datasets(cfg)
    if args.command == "validate-dataset":
        return _cmd_validate_dataset(cfg, args)
    if args.command == "ingest":
        return _cmd_ingest(cfg, args)
    if args.command == "eda":
        return _cmd_eda(cfg, args)
    if args.command == "feature-registry":
        return _cmd_feature_registry(args)
    if args.command == "preprocess":
        return _cmd_preprocess(cfg, args)
    if args.command == "window":
        return _cmd_window(cfg, args)
    if args.command == "rebuild-temporal":
        from .ml.temporal.pipeline import rebuild_temporal_outputs_from_flows
        result = rebuild_temporal_outputs_from_flows(cfg, args.dataset)
        print(f"Temporal outputs rebuilt from existing flows for '{args.dataset}'")
        print(f"  Flows          : {result.flows_count:,} (unchanged) -> {result.flows_path}")
        print(f"  Host windows   : {result.windows_count:,} -> {result.windows_path}")
        print(f"  Sequences      : {result.sequences_count:,} -> {result.sequences_path}")
        print(f"  Split metadata : {result.split_metadata_path}")
        print(f"  Quality report : {result.report_md_path}")
        return 0
    if args.command == "train":
        from .ml.modeling import ForecastDataset, train_experiment
        path = cfg.path(cfg.config.paths.data_processed, args.dataset, "sequences.parquet")
        hp = train_hyperparameters(cfg, args)
        model_type = resolve_model_type(cfg, args)
        default_dir = args.dataset if model_type == "lstm" else f"{args.dataset}_{model_type}"
        out = cfg.path(args.output_dir) if args.output_dir else cfg.path("artifacts", "models", default_dir)
        log.info("training hyperparameters", model_type=model_type, **hp)
        result = train_experiment(ForecastDataset.from_parquet(path), out, seed=int(cfg.config.random_seed),
                                  model_type=model_type, **hp)
        print(f"Phase 3 training complete: {out}")
        for name, metrics in result["models"].items(): print(f"  {name}: {metrics}")
        print(f"  Metrics: {out / 'metrics.json'}")
        return 0
    if args.command == "predict":
        from .ml.modeling import predict_sequences
        rows = predict_sequences(cfg.path(args.input), cfg.path(args.model_dir), args.threshold)
        for row in rows: print(__import__("json").dumps(row))
        return 0
    if args.command == "explain":
        import json
        from .ml.explain import explain_parquet
        path = cfg.path(args.input) if args.input else cfg.path(cfg.config.paths.data_processed, args.dataset, "sequences.parquet")
        model_dir = cfg.path(args.model_dir) if args.model_dir else cfg.path("artifacts", "models", args.dataset)
        for e in explain_parquet(path, model_dir, args.sequence_id or None, model=args.model, method=args.method,
                                 limit=args.limit, seed=int(cfg.config.random_seed)):
            print(json.dumps(e.to_dict(k=args.top)))
        return 0
    if args.command == "train-multitask":
        from .ml.multitask import MultiTaskData, train_multitask
        hp = train_hyperparameters(cfg, args)
        data = MultiTaskData.from_dir(cfg.path(cfg.config.paths.data_processed, args.dataset), args.horizons,
                                      list(cfg.stages.stages_order))
        out = cfg.path(args.output_dir) if args.output_dir else cfg.path("artifacts", "models", f"{args.dataset}_multitask")
        res = train_multitask(data, out, seed=int(cfg.config.random_seed), dataset=args.dataset, **hp)
        st = res["stage"]
        print(f"Multi-task training complete: {out}")
        print(f"  attack : {res['attack']}")
        print(f"  stage  : accuracy {st['accuracy']}, on attack targets {st['attack_stage_accuracy']} "
              f"(majority baseline {st['majority_baseline']['accuracy']}, {st['majority_baseline']['attack_stage_accuracy']})")
        for h in res["future_state"]["per_horizon"]:
            print(f"  state  : {h}")
        print(f"  Metrics: {out / 'metrics.json'}")
        return 0
    if args.command == "forecast":
        import json

        import numpy as np
        import yaml

        from .ml.modeling import ForecastDataset
        from .ml.multitask import MultiTaskPredictor
        path = cfg.path(args.input) if args.input else cfg.path(cfg.config.paths.data_processed, args.dataset, "sequences.parquet")
        model_dir = cfg.path(args.model_dir) if args.model_dir else cfg.path("artifacts", "models", f"{args.dataset}_multitask")
        data = ForecastDataset.from_parquet(path)
        sid = data.frame["sequence_id"].astype(str)
        idx = (np.flatnonzero(sid.isin(args.sequence_id).to_numpy()) if args.sequence_id
               else data.indices("test")[:args.limit])
        mitre = yaml.safe_load(cfg.path("configs", "mitre_mapping.yaml").read_text(encoding="utf-8"))
        rows = MultiTaskPredictor(model_dir).rows(
            data.X[idx], sid.iloc[idx].tolist(), stage_threshold=float(cfg.config.confidence.stage_prediction_threshold),
            uncertain_label=str(cfg.config.confidence.uncertain_label), mitre_lookup=mitre.get)
        for row in rows: print(json.dumps(row))
        return 0
    if args.command == "ingest-pcap":
        from .ml.ingestion.pcap import apply_label_file, pcap_to_flows
        flows = pcap_to_flows(cfg.path(args.input), reader=args.reader, idle_timeout=args.idle_timeout,
                              active_timeout=args.active_timeout)
        if args.labels:
            flows = apply_label_file(flows, cfg.path(args.labels), dict(cfg.stages[args.label_map]))
        out = cfg.path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        flows.to_csv(out, index=False) if out.suffix.lower() == ".csv" else flows.to_parquet(out, index=False)
        print(f"{len(flows):,} flows from {flows.source_ip.nunique()} source hosts -> {out}")
        print(f"  TTL present: {int(flows.ttl_mean.notna().sum()):,} flows; "
              f"TCP retransmissions: {int(flows.retransmission_count.fillna(0).sum()):,}")
        return 0
    if args.command == "ingest-netflow":
        from .ml.ingestion.netflow import netflow_to_flows
        flows = netflow_to_flows(cfg.path(args.input), args.format)
        out = cfg.path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        flows.to_csv(out, index=False) if out.suffix.lower() == ".csv" else flows.to_parquet(out, index=False)
        print(f"{len(flows):,} NetFlow records from {flows.source_ip.nunique()} source hosts -> {out}")
        return 0
    if args.command in {"score-pcap", "score-netflow"}:
        from .ml.scoring import score_flows
        if args.command == "score-pcap":
            from .ml.ingestion.pcap import pcap_to_flows
            flows = pcap_to_flows(cfg.path(args.input), reader=args.reader)
        else:
            from .ml.ingestion.netflow import netflow_to_flows
            flows = netflow_to_flows(cfg.path(args.input), args.format)
        scored = score_flows(flows, cfg.path(args.model_dir), cfg, threshold=args.threshold)
        print(f"{len(flows):,} flows -> {len(scored):,} scored host sequences, "
              f"{int(scored.predicted_attack.sum()) if len(scored) else 0} above threshold")
        if args.output:
            scored.to_csv(cfg.path(args.output), index=False)
        for row in scored.sort_values("attack_probability", ascending=False).head(10).to_dict("records"):
            print(__import__("json").dumps(row, default=str))
        return 0
    if args.command == "stream":
        import json

        import pandas as pd

        from .ml.streaming import StreamScorer
        src = cfg.path(args.input)
        if src.suffix.lower() in {".pcap", ".pcapng", ".cap"}:
            from .ml.ingestion.pcap import pcap_to_flows
            flows = pcap_to_flows(src)
        else:
            flows = pd.read_parquet(src) if src.suffix.lower() == ".parquet" else pd.read_csv(src, parse_dates=["timestamp"])
        flows = flows.sort_values("timestamp", kind="stable")
        st = StreamScorer(cfg.path(args.model_dir), cfg, allowed_lateness=args.lateness)
        chunk = ((flows.timestamp - flows.timestamp.min()).dt.total_seconds() // args.chunk_seconds).astype(int)
        for _, part in flows.groupby(chunk, sort=True):
            for r in st.push(part):
                if r["predicted_attack"] or not args.alerts_only:
                    print(json.dumps(r))
        for r in st.flush():
            if r["predicted_attack"] or not args.alerts_only:
                print(json.dumps(r))
        print(json.dumps({"status": st.status()}))
        return 0
    if args.command == "train-gnn":
        import pandas as pd

        from .ml.graph import build_graph_sequences, train_tgnn
        hp = train_hyperparameters(cfg, args)
        d = cfg.path(cfg.config.paths.data_processed, args.dataset)
        data = build_graph_sequences(pd.read_parquet(d / "sequences.parquet"), pd.read_parquet(d / "host_windows.parquet"),
                                     pd.read_parquet(d / "flows.parquet", columns=["timestamp", "source_ip", "destination_ip"]),
                                     float(cfg.config.windowing.window_size_seconds))
        out = cfg.path(args.output_dir) if args.output_dir else cfg.path("artifacts", "models", f"{args.dataset}_tgnn")
        res = train_tgnn(data, out, seed=int(cfg.config.random_seed), dataset=args.dataset, **hp)
        print(f"Temporal GNN training complete: {out}")
        print(f"  graph : {res['graph']}")
        for name, m in res["models"].items(): print(f"  {name}: {m}")
        return 0
    if args.command == "export-alerts":
        from backend.app.audit import Ledger
        from backend.app.siem import alerts_since, render, send_syslog_udp
        db = cfg.path(args.db) if args.db else cfg.path(str(cfg.config.database.url).replace("sqlite:///", ""))
        if not db.exists():
            raise AegisFlowError(f"ledger not found: {db}")
        lines = render(alerts_since(Ledger(db), args.since_id, args.limit), args.format)
        text = "\n".join(lines) + ("\n" if lines else "")
        if args.output:
            cfg.path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        if args.syslog_host:
            sent = send_syslog_udp(render(alerts_since(Ledger(db), args.since_id, args.limit), "syslog"),
                                   args.syslog_host, args.syslog_port)
            print(f"sent {sent} syslog datagrams to {args.syslog_host}:{args.syslog_port}", file=sys.stderr)
        return 0
    if args.command in {"train", "evaluate", "replay", "serve"}:
        raise NotImplementedPhaseError(
            f"'{args.command}' is planned but not implemented in the current phase. "
            f"See docs/architecture.md for the phase plan and run --help for what's available now."
        )
    parser = build_parser()
    parser.print_help()
    return 1


def _cmd_list_datasets(cfg) -> int:
    print(f"{'key':<14} {'status':<10} {'adapter':<14} display_name")
    for key, entry in cfg.datasets.items():
        print(f"{key:<14} {entry.get('status', '?'):<10} {entry.get('adapter', '?'):<14} {entry.get('display_name', '')}")
    return 0


def _cmd_validate_dataset(cfg, args) -> int:
    from .ml.datasets.validate import validate_dataset

    result = validate_dataset(cfg, args.dataset, try_sample_load=not args.no_sample_load)
    ok = result.report.ok and result.sample_load_ok
    print(f"\n{'PASSED' if ok else 'FAILED'}: dataset '{args.dataset}'")
    print(f"  files found: {len(result.report.files)}")
    if result.report.problems:
        print("  problems:")
        for p in result.report.problems:
            print(f"    - {p}")
    if not result.sample_load_ok:
        print(f"  sample load error: {result.sample_load_error}")
    return 0 if ok else 1


def _cmd_ingest(cfg, args) -> int:
    from .ml.ingestion.pipeline import run_ingestion

    result = run_ingestion(cfg, args.dataset, sample_size=args.sample_size)
    print(f"\nIngestion complete for '{args.dataset}'")
    print(f"  raw rows loaded   : {result.rows_raw}")
    print(f"  cleaned rows kept : {result.cleaning.rows_out}")
    print(f"  written to        : {result.output_path}")
    print("  normalized_attack_class distribution:")
    for k, v in sorted(result.class_distribution.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} {v}")
    return 0


def _cmd_eda(cfg, args) -> int:
    from .ml.eda import run_eda

    summary = run_eda(cfg, args.dataset)
    print(f"\nEDA complete for '{args.dataset}': {summary['n_rows']} rows, "
          f"{summary['n_unique_hosts_src']} unique source hosts.")
    print(f"  attack_stage counts: {summary['attack_stage_counts']}")
    return 0


def _cmd_feature_registry(args) -> int:
    from .ml.features.registry import registry_as_markdown

    print(registry_as_markdown(level=args.level))
    return 0


def _cmd_preprocess(cfg, args) -> int:
    from .ml.temporal.pipeline import run_preprocessing

    res = run_preprocessing(
        cfg=cfg,
        dataset_key=args.dataset,
        sample_size=args.sample_size,
        window_size_seconds=args.window_size,
        stride_seconds=args.stride,
        sequence_length=args.seq_len,
        forecast_horizon=args.horizon,
        reingest=args.reingest,
    )
    print(f"\nPhase 2 Preprocessing complete for '{args.dataset}'")
    print(f"  Engineered flows : {res.flows_count:,} -> {res.flows_path}")
    print(f"  Host windows     : {res.windows_count:,} -> {res.windows_path}")
    print(f"  Sequences        : {res.sequences_count:,} -> {res.sequences_path}")
    print(f"  Split metadata   : {res.split_metadata_path}")
    print(f"  Quality report   : {res.report_md_path}")
    print(f"\nChronological Split Breakdown:")
    print(f"  Train sequences  : {res.metrics.split_info.get('train_sequences', 0):,}")
    print(f"  Val sequences    : {res.metrics.split_info.get('val_sequences', 0):,}")
    print(f"  Test sequences   : {res.metrics.split_info.get('test_sequences', 0):,}")
    print(f"  Boundary excluded: {res.metrics.split_info.get('boundary_excluded_sequences', 0):,}")
    return 0


def _cmd_window(cfg, args) -> int:
    from .ml.temporal.pipeline import run_preprocessing

    res = run_preprocessing(
        cfg=cfg,
        dataset_key=args.dataset,
        sample_size=args.sample_size,
        window_size_seconds=args.window_size,
        stride_seconds=args.stride,
    )
    print(f"\nHost-level temporal windowing complete for '{args.dataset}'")
    print(f"  Host windows generated: {res.windows_count:,}")
    print(f"  Output location       : {res.windows_path}")
    return 0
