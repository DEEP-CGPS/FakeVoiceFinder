# existing exports...
from .config import ExperimentConfig
from .validatorsforvoice import ConfigError
from .experiment import CreateExperiment
from .prepare_dataset import PrepareDataset
from .model_loader import ModelLoader
from .trainer import Trainer
from .duration_probe import shortest_audio_seconds

# NEW: metrics reporter
from .metrics import MetricsReporter

__all__ = [
    "ExperimentConfig",
    "ConfigError",
    "CreateExperiment",
    "PrepareDataset",
    "ModelLoader",
    "Trainer",
    "MetricsReporter",
    "shortest_audio_seconds",
]
