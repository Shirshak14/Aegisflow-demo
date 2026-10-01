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
    model_type = str(model_cfg.type).lower()
    if model_type != "lstm":
        raise ConfigError(
            f"model.type is '{model_cfg.type}' but only 'lstm' is implemented. "
            "Set model.type to 'lstm' in configs/config.yaml (or --set model.type=lstm)."
        )
    out = {}
    for flag, (key, cast) in _TRAIN_PARAMS.items():
        value = getattr(args, flag, None)
        out[flag] = cast(value if value is not None else model_cfg[key])
    return out


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
    _common_args(p_train)

    p_predict = sub.add_parser("predict", help="Run Phase 3 LSTM inference on a sequences Parquet file.")
    p_predict.add_argument("--input", required=True)
    p_predict.add_argument("--model-dir", default="artifacts/models/cic_ids2017")
    p_predict.add_argument("--threshold", type=float, default=None)
    _common_args(p_predict)

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
        out = cfg.path("artifacts", "models", args.dataset)
        hp = train_hyperparameters(cfg, args)
        log.info("training hyperparameters", **hp)
        result = train_experiment(ForecastDataset.from_parquet(path), out, seed=int(cfg.config.random_seed), **hp)
        print(f"Phase 3 training complete: {out}")
        for name, metrics in result["models"].items(): print(f"  {name}: {metrics}")
        print(f"  Metrics: {out / 'metrics.json'}")
        return 0
    if args.command == "predict":
        from .ml.modeling import predict_sequences
        rows = predict_sequences(cfg.path(args.input), cfg.path(args.model_dir), args.threshold)
        for row in rows: print(__import__("json").dumps(row))
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
