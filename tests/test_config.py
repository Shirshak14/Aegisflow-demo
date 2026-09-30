from __future__ import annotations

import pytest

from aegisflow.config import load_config
from aegisflow.errors import ConfigError


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.config.active_dataset == "cic_ids2017"
    assert cfg.config.windowing.window_size_seconds == 60


def test_override_applies():
    cfg = load_config(overrides=["model.batch_size=64"])
    assert cfg.config.model.batch_size == 64
    assert isinstance(cfg.config.model.batch_size, int)


def test_override_rejects_unknown_key():
    with pytest.raises(ConfigError):
        load_config(overrides=["model.totally_made_up=1"])


def test_override_rejects_bad_syntax():
    with pytest.raises(ConfigError):
        load_config(overrides=["no-equals-sign"])


def test_active_dataset_entry_found():
    cfg = load_config()
    entry = cfg.active_dataset_entry()
    assert entry.display_name == "CIC-IDS2017"


def test_active_dataset_entry_missing_raises():
    cfg = load_config(overrides=[])
    cfg.config["active_dataset"] = "does_not_exist"
    with pytest.raises(ConfigError):
        cfg.active_dataset_entry()
