"""
Configuration loading.

Every tunable pipeline parameter lives in configs/*.yaml, never hardcoded in
source. This module loads and merges those files into typed, dotted-access
objects, and supports ``--set key.path=value`` overrides from the CLI.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent


class DotDict(dict):
    """A dict that also supports attribute access, recursively.

    Example: ``cfg.windowing.window_size_seconds`` as well as
    ``cfg["windowing"]["window_size_seconds"]``.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"AegisFlow expects it at this exact path relative to the repo root."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a YAML mapping at the top level.")
    return data


def _set_by_dotted_path(cfg: dict, dotted_key: str, value: str) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise ConfigError(f"Cannot override '{dotted_key}': '{p}' is not an existing config section.")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node:
        raise ConfigError(f"Cannot override '{dotted_key}': key does not exist in config.")
    parsed = yaml.safe_load(value)
    node[leaf] = parsed


@dataclass
class AegisFlowConfig:
    """Container bundling the four config files plus the resolved repo root."""

    root: Path
    config: DotDict
    datasets: DotDict
    stages: DotDict
    mitre: DotDict
    overrides: list[str] = field(default_factory=list)

    def path(self, *parts: str) -> Path:
        """Resolve a path from ``paths.*`` in config.yaml relative to repo root."""
        return self.root / Path(*parts)

    def active_dataset_entry(self) -> DotDict:
        name = self.config.active_dataset
        if name not in self.datasets:
            raise ConfigError(
                f"active_dataset '{name}' (set in configs/config.yaml) has no entry in "
                f"configs/datasets.yaml. Known datasets: {list(self.datasets.keys())}"
            )
        return self.datasets[name]


def load_config(
    root: Path | None = None,
    config_dir: str = "configs",
    overrides: list[str] | None = None,
) -> AegisFlowConfig:
    """Load all four config files and apply ``key.path=value`` overrides.

    Raises:
        ConfigError: if a file is missing, malformed, or an override targets
            a key that doesn't exist.
    """
    root = root or REPO_ROOT
    cdir = root / config_dir

    raw_config = _load_yaml(cdir / "config.yaml")
    raw_datasets = _load_yaml(cdir / "datasets.yaml")
    raw_stages = _load_yaml(cdir / "stages.yaml")
    raw_mitre = _load_yaml(cdir / "mitre_mapping.yaml")

    raw_config = copy.deepcopy(raw_config)
    for ov in overrides or []:
        if "=" not in ov:
            raise ConfigError(f"Invalid --set override '{ov}', expected key.path=value")
        key, value = ov.split("=", 1)
        _set_by_dotted_path(raw_config, key.strip(), value.strip())

    return AegisFlowConfig(
        root=root,
        config=_to_dotdict(raw_config),
        datasets=_to_dotdict(raw_datasets),
        stages=_to_dotdict(raw_stages),
        mitre=_to_dotdict(raw_mitre),
        overrides=list(overrides or []),
    )
