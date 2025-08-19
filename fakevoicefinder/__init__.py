"""
fakevoicefinder package

Public API
----------
- ExperimentConfig : Single-file configuration class (validates itself).
- ConfigError      : Exception raised on invalid configuration or setup.
- CreateExperiment : Creates the experiment layout under ./outputs/<run_name>,
                     builds the in-memory manifest, and provides data preparation
                     (split + transforms) via `prepare_data(...)`.
- ModelLoader      : Registers models in the experiment:
                        * Benchmarks (torchvision): 'scratch' / 'pretrain'
                        * User models (TorchScript ONLY): copied as-is into
                          outputs/<exp>/models/loaded and recorded as 'usermodel_jit'.
- Trainer          : Trains all registered models across prepared transforms and
                     updates the experiment manifest with trained checkpoints.

Notes
-----
• User models are supported **only** as TorchScript archives created with
  `torch.jit.script` or `torch.jit.trace` + `torch.jit.save`.
• Benchmarks use torchvision and are adapted to 2 classes (real/fake).
"""

from .config import ExperimentConfig
from .validatorsforvoice import ConfigError
from .experiment import CreateExperiment
from .model_loader import ModelLoader
from .trainer import Trainer

__all__ = [
    "ExperimentConfig",
    "ConfigError",
    "CreateExperiment",
    "ModelLoader",
    "Trainer",
]

__version__ = "0.1.0"
