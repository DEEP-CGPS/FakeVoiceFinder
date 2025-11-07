"""Central experiment configuration for FakeVoiceFinder.

This module defines a single configuration container, `ExperimentConfig`,
responsible for describing *what* to train/infer and *where* to read/write
artifacts.

Inputs (how you fill it):
- You can create it empty and assign attributes in Python code.
- You can load it from a plain dict (`ExperimentConfig.from_dict(...)`).
- You can load it from a Python file with variables
  (`ExperimentConfig.from_pyfile("my_config.py")`).

Outputs (what you can get from it):
- A plain dictionary ready to be stored or inspected (`to_dict()`).
- A JSON file on disk for reproducible runs (`to_json(path)`).
- A human-readable multi-line summary for debugging (`summary()`).

Notes:
- Validation is *opt-in*: the class inherits from `ValidationMixin`, but you
  call `validate()` explicitly when you want to enforce constraints.
- All experiment artifacts are placed under `<repo_root>/outputs`. The field
  `outputs_path` is kept only for backward compatibility with older scripts and
  is not used by `CreateExperiment` nor by the current path validation logic.
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .validatorsforvoice import ValidationMixin, ConfigError


class ExperimentConfig(ValidationMixin):
    """Configuration object describing data, models, transforms and runtime.

    This class acts as a single source of truth for a FakeVoiceFinder experiment.
    It stores:
    - Input locations (dataset, models directory, zip names).
    - What to run (which models, which transforms, which training mode).
    - How to run it (device, batch size, workers, optimizer).
    - How to persist results (run name, saving policy).

    Typical flow:
    1. Build or load an instance (`__init__`, `from_dict`, or `from_pyfile`).
    2. Optionally normalize/validate using the mixin.
    3. Pass the resulting object to the components that prepare data, train,
       evaluate and run inference.

    The goal is to keep this object simple, explicit and serializable.
    """

    # Accepted values for several config fields; can be overridden if needed.
    ALLOWED_TYPE_TRAIN = {"scratch", "pretrain", "both"}
    ALLOWED_DEVICE = {"cpu", "gpu"}
    ALLOWED_OPTIM = {"adam", "sgd"}
    ALLOWED_TRANSFORMS = {"mel", "log", "dwt", "cqt"}
    ALLOWED_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}

    def __init__(self) -> None:
        # User-provided repository-relative paths.
        self.models_path: str = "../models"
        self.data_path: str = "../dataset"

        # Backward-compat field; actual outputs are fixed to the repo root.
        self.outputs_path: str = "outputs"

        # Model selection and training control.
        self.models_list: List[str] = []
        self.flag_train: bool = True
        self.type_train: str = "both"  # 'scratch' | 'pretrain' | 'both'

        # List of audio-to-image transforms to be created and consumed.
        self.transform_list: List[str] = ["mel", "log"]

        # Optional per-transform configuration.
        self.mel_params: Optional[Dict[str, Any]] = None
        self.log_params: Optional[Dict[str, Any]] = None
        self.dwt_params: Optional[Dict[str, Any]] = None
        self.cqt_params: Optional[Dict[str, Any]] = None

        # Training/evaluation hyperparameters.
        self.epochs: int = 20
        self.batch_size: int = 32
        self.optimizer: str = "Adam"  # 'Adam' | 'SGD'
        self.learning_rate: float = 0.003
        self.patience: int = 10
        self.eval_metric: List[str] = ["accuracy", "F1"]

        # Runtime setup.
        self.device: str = "gpu"  # 'gpu' | 'cpu'
        self.seed: int = 23
        self.num_workers: int = 4

        # Persistence and caching behavior.
        self.save_models: bool = True
        self.save_best_only: bool = True
        self.cache_features: bool = True
        self.run_name: Optional[str] = None

        # Dataset file names expected under `data_path`.
        self.real_zip: str = "real.zip"
        self.fake_zip: str = "fake.zip"

        # Optional clip duration in seconds. If None, dataset code supplies the default.
        self.clip_seconds: Optional[float] = None

        # Optional target image size for spectrogram-like transforms.
        self.image_size: Optional[int] = None

    # ---------- Convenience constructors ----------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create a configuration from a dictionary of parameters.

        Input:
            d: mapping of attribute names to values. Extra keys are ignored.
               Key 'epoch' is accepted as an alias for 'epochs'.

        Output:
            An `ExperimentConfig` instance with the passed values applied and
            basic normalization (lowercasing, lists normalized).
        """
        cfg = cls()
        d = dict(d)
        if "epoch" in d and "epochs" not in d:
            d["epochs"] = d.pop("epoch")
        for key, val in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

        # Basic normalization to keep consumers predictable.
        cfg.type_train = str(cfg.type_train).lower()
        cfg.device = str(cfg.device).lower()
        if str(cfg.optimizer).lower() in {"adam", "sgd"}:
            cfg.optimizer = str(cfg.optimizer).capitalize()
        cfg.transform_list = [str(x).lower() for x in (cfg.transform_list or [])]
        cfg.eval_metric = [str(x) for x in (cfg.eval_metric or [])]
        return cfg

    @classmethod
    def from_pyfile(cls, path: str | Path) -> "ExperimentConfig":
        """Create a configuration by executing a Python file with variables.

        Input:
            path: path to a .py file that defines config variables (e.g. models_list,
                  data_path, transform_list, etc.).

        Output:
            An `ExperimentConfig` instance populated with the variables found
            in the file. Variables with names starting with '__' are ignored.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        ns = runpy.run_path(str(path))
        user_vars = {k: v for k, v in ns.items() if not k.startswith("__")}
        return cls.from_dict(user_vars)

    # ---------- Small utilities ----------
    def to_dict(self) -> Dict[str, Any]:
        """Return the current configuration as a plain serializable dictionary.

        Output:
            dict with all configuration fields, suitable for JSON dumping or
            logging.
        """
        return dict(self.__dict__)

    def to_json(self, path: str | Path) -> None:
        """Write the current configuration to a JSON file.

        Input:
            path: target path where the JSON will be stored. Parent directories
                  are created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """Build a human-readable report of all configuration fields.

        Output:
            Multiline string listing every field and its current value, sorted
            by field name. Useful for CLI tools and debug logs.
        """
        items = sorted(self.to_dict().items(), key=lambda kv: kv[0])
        key_w = max(len(k) for k, _ in items)
        lines = ["ExperimentConfig:"]
        for k, v in items:
            lines.append(f"  {k.ljust(key_w)} : {v}")
        return "\n".join(lines)
