"""Exception hierarchy. Every pipeline failure raises one of these with an actionable message."""
from __future__ import annotations


class AegisFlowError(Exception):
    """Base class for all expected AegisFlow failures."""


class ConfigError(AegisFlowError):
    """Configuration files are missing or invalid."""


class DatasetNotFoundError(AegisFlowError):
    """Raw dataset files were not found where expected."""


class DatasetValidationError(AegisFlowError):
    """Dataset files exist but their content is not what the adapter requires."""


class SchemaError(AegisFlowError):
    """A frame does not conform to the canonical flow schema."""


class LabelMappingError(AegisFlowError):
    """A dataset label has no entry in the attack-stage mapping."""


class MissingDependencyError(AegisFlowError):
    """An optional runtime dependency is not installed."""


class NotImplementedPhaseError(AegisFlowError):
    """Feature exists in the plan but is NOT IMPLEMENTED in the current phase."""
