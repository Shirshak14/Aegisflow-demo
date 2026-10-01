"""Phase-1 exploratory data analysis: summary stats + figures from the interim parquet."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import AegisFlowConfig
from ..errors import DatasetNotFoundError
from ..logging_setup import get_logger

log = get_logger("EDA")


def run_eda(cfg: AegisFlowConfig, dataset_key: str) -> dict:
    """Load data/interim/<dataset_key>.parquet, compute summary stats, save figures + JSON report."""
    interim_path = cfg.path(cfg.config.paths.data_interim) / f"{dataset_key}.parquet"
    if not interim_path.exists():
        raise DatasetNotFoundError(
            f"{interim_path} does not exist. Run 'python -m aegisflow ingest --dataset {dataset_key}' first."
        )
    df = pd.read_parquet(interim_path)
    log.info("loaded interim parquet", rows=len(df), path=str(interim_path))

    figures_dir = cfg.path(cfg.config.paths.reports) / "figures"
    eda_dir = cfg.path(cfg.config.paths.reports) / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset_key,
        "n_rows": int(len(df)),
        "n_unique_hosts_src": int(df["source_ip"].nunique()),
        "n_unique_hosts_dst": int(df["destination_ip"].nunique()),
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "protocol_counts": df["protocol"].value_counts(dropna=False).to_dict(),
        "normalized_attack_class_counts": df["normalized_attack_class"].value_counts(dropna=False).to_dict(),
        "attack_stage_counts": df["attack_stage"].value_counts(dropna=False).to_dict(),
        "label_confidence_counts": df["label_confidence"].value_counts(dropna=False).to_dict(),
        "missing_fraction_by_column": (df.isna().mean().round(4)).to_dict(),
    }

    # Figure 1: attack-stage class distribution (log scale -- benign dwarfs everything else)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = df["attack_stage"].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color="#1F3864")
    ax.set_yscale("log")
    ax.set_ylabel("Flow count (log scale)")
    ax.set_title(f"{dataset_key}: attack_stage distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{dataset_key}_stage_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 2: flow volume over time (hourly), split benign vs attack
    ts = df.set_index("timestamp").copy()
    ts["is_attack"] = ts["attack_stage"].ne("Benign")
    hourly = ts.groupby([pd.Grouper(freq="1h"), "is_attack"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    hourly.plot(ax=ax, color={False: "#6FAE8C", True: "#E8710C"})
    ax.set_ylabel("Flows / hour")
    ax.set_title(f"{dataset_key}: traffic volume over time")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{dataset_key}_traffic_timeline.png", dpi=150)
    plt.close(fig)

    out_json = eda_dir / f"{dataset_key}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("eda complete", summary_path=str(out_json))
    return summary
