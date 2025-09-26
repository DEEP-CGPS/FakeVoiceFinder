"""Simple experiment configuration for forvoice that inherits validations.

- Single class `ExperimentConfig` inheriting from `ValidationMixin`.
- Set attributes directly; call `validate()` only when you want.
- Pure standard library (plus `runpy` for from_pyfile helper).

NOTE: The experiment artifacts are always stored under `<repo_root>/outputs`.
`outputs_path` is kept only for backward compatibility and is **not used**
by `CreateExperiment` nor by path validation (which fix outputs to repo root).
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .validatorsforvoice import ValidationMixin, ConfigError


class ExperimentConfig(ValidationMixin):
    """Very small configuration object with opt-in validation."""

    # Allowed sets can be overridden here if you need to customize domains.
    ALLOWED_TYPE_TRAIN = {"scratch", "pretrain", "both"}
    ALLOWED_DEVICE = {"cpu", "gpu"}
    ALLOWED_OPTIM = {"adam", "sgd"}
    ALLOWED_TRANSFORMS = {"mel", "log", "dwt"}
    ALLOWED_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}

    def __init__(self) -> None:
        # --- Paths (user-controlled)
        self.models_path: str = "../models"
        self.data_path: str = "../dataset"

        # (kept for backward-compat only; ignored by CreateExperiment/validation)
        self.outputs_path: str = "outputs"

        # --- Models
        self.models_list: List[str] = []
        self.flag_train: bool = True
        self.type_train: str = "both"  # 'scratch' | 'pretrain' | 'both'

        # --- Transforms
        self.transform_list: List[str] = ["mel", "log"]

        # --- (NEW) Optional overrides for transform hyperparameters
        self.mel_params: Optional[Dict[str, Any]] = None
        self.log_params: Optional[Dict[str, Any]] = None
        self.dwt_params: Optional[Dict[str, Any]] = None

        # --- Training / Eval
        self.epochs: int = 20
        self.batch_size: int = 32
        self.optimizer: str = "Adam"  # 'Adam' | 'SGD'
        self.learning_rate: float = 0.003
        self.patience: int = 10
        self.eval_metric: List[str] = ["accuracy", "F1"]

        # --- Runtime
        self.device: str = "gpu"  # 'gpu' | 'cpu'
        self.seed: int = 23
        self.num_workers: int = 4

        # --- Persistence / cache
        self.save_models: bool = True
        self.save_best_only: bool = True
        self.cache_features: bool = True
        self.run_name: Optional[str] = None

        # --- Data files (inside data_path)
        self.real_zip: str = "real.zip"
        self.fake_zip: str = "fake.zip"

        # --- (NEW) Optional clip window (seconds). If None, defaults to 3.0s in PrepareDataset
        self.clip_seconds: Optional[float] = None

        # --- (NEW) Optional image size for transforms (e.g., 224 for ViT). If None, no resize in mel/log.
        self.image_size: Optional[int] = None

    # ---------- Convenience constructors ----------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create a config from a plain dict. Unknown keys are ignored."""
        cfg = cls()
        d = dict(d)  # shallow copy
        # Accept 'epoch' alias
        if "epoch" in d and "epochs" not in d:
            d["epochs"] = d.pop("epoch")
        for key, val in d.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

        # Light normalization
        cfg.type_train = str(cfg.type_train).lower()
        cfg.device = str(cfg.device).lower()
        if str(cfg.optimizer).lower() in {"adam", "sgd"}:
            cfg.optimizer = str(cfg.optimizer).capitalize()
        cfg.transform_list = [str(x).lower() for x in (cfg.transform_list or [])]
        cfg.eval_metric = [str(x) for x in (cfg.eval_metric or [])]
        return cfg

    @classmethod
    def from_pyfile(cls, path: str | Path) -> "ExperimentConfig":
        """Create a config by executing a Python file (e.g., 'config.py')."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        ns = runpy.run_path(str(path))
        user_vars = {k: v for k, v in ns.items() if not k.startswith("__")}
        return cls.from_dict(user_vars)

    # ---------- Small utilities ----------
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary form of the config."""
        return dict(self.__dict__)

    def to_json(self, path: str | Path) -> None:
        """Serialize the configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """Human-readable multi-line summary of all fields."""
        items = sorted(self.to_dict().items(), key=lambda kv: kv[0])
        key_w = max(len(k) for k, _ in items)
        lines = ["ExperimentConfig:"]
        for k, v in items:
            lines.append(f"  {k.ljust(key_w)} : {v}")
        return "\n".join(lines)
