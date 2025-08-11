"""Public API for the forvoice package (Fake Or Real Voice).

Exports:
- ExperimentConfig: simple, self-validating configuration object.
- ValidationMixin, ConfigError: validation helpers and exception.
- CreateExperiment: creates the experiment folder layout and manifest.
- PrepareDataset: ZIP-only data preparer (load -> split -> save -> transform -> update manifest).
"""

from .config import ExperimentConfig
from .validatorsforvoice import ValidationMixin, ConfigError
from .experiment import CreateExperiment
from .prepare_dataset import PrepareDataset

__all__ = [
    "ExperimentConfig",
    "ValidationMixin",
    "ConfigError",
    "CreateExperiment",
    "PrepareDataset",
]