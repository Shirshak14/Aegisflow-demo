"""`train` takes its hyperparameters from config.yaml `model:`, with CLI flags overriding."""
from __future__ import annotations

from argparse import Namespace

import pytest

from aegisflow.cli import build_parser, train_hyperparameters
from aegisflow.config import load_config
from aegisflow.errors import ConfigError

# Hyperparameters of the committed demo model (artifacts/models/cic_ids2017/metadata.json).
COMMITTED = {"epochs": 8, "batch_size": 256, "learning_rate": 0.001, "hidden_size": 16,
             "dropout": 0.2, "patience": 3}


def _args(*argv: str) -> Namespace:
    return build_parser().parse_args(["train", *argv])


def test_no_flags_reproduces_the_committed_model_hyperparameters():
    assert train_hyperparameters(load_config(), _args()) == COMMITTED


def test_cli_flag_overrides_config():
    hp = train_hyperparameters(load_config(), _args("--hidden-size", "32", "--epochs", "20"))
    assert hp["hidden_size"] == 32 and hp["epochs"] == 20
    assert hp["batch_size"] == 256  # untouched flag still comes from config


def test_set_override_reaches_training():
    hp = train_hyperparameters(load_config(overrides=["model.hidden_size=8", "model.early_stopping_patience=5"]), _args())
    assert hp["hidden_size"] == 8 and hp["patience"] == 5


def test_types_are_cast():
    hp = train_hyperparameters(load_config(), _args())
    assert isinstance(hp["epochs"], int) and isinstance(hp["learning_rate"], float)


def test_unimplemented_model_type_is_rejected():
    with pytest.raises(ConfigError, match="only 'lstm'"):
        train_hyperparameters(load_config(overrides=["model.type=gru"]), _args())
