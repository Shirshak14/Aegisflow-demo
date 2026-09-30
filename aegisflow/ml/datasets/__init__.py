"""Dataset adapters: raw files -> canonical flow schema (aegisflow.schema)."""
from __future__ import annotations

from ...config import AegisFlowConfig, DotDict
from ...errors import ConfigError
from .base import DatasetAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(name: str):
    def _wrap(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_adapter(name: str, cfg: AegisFlowConfig) -> DatasetAdapter:
    """Instantiate the adapter registered under ``name`` (see configs/datasets.yaml: adapter)."""
    # Import submodules for their side-effecting @register decorators.
    from . import cic_ids2017  # noqa: F401

    if name not in _REGISTRY:
        raise ConfigError(
            f"No dataset adapter registered as '{name}'. Available: {sorted(_REGISTRY)}"
        )
    entry: DotDict = cfg.datasets[_dataset_key_for_adapter(cfg, name)]
    return _REGISTRY[name](cfg=cfg, entry=entry)


def _dataset_key_for_adapter(cfg: AegisFlowConfig, adapter_name: str) -> str:
    for key, entry in cfg.datasets.items():
        if entry.get("adapter") == adapter_name:
            return key
    raise ConfigError(f"No dataset entry in configs/datasets.yaml uses adapter '{adapter_name}'.")
