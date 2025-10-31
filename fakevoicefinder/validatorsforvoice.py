"""
validatorsforvoice.py — Validation helpers for the FakeVoiceFinder stack.

Purpose
-------
Provide a lightweight, opt-in validation layer for configuration objects
(typically `ExperimentConfig`). The mixin exposes a single entrypoint
`validate(...)` that runs several focused checks on paths, data files, lists,
allowed values, numeric ranges, and cross-field semantics.

Inputs (expected config attributes)
-----------------------------------
A consumer (e.g. ExperimentConfig) is expected to expose at least:
    models_path: str               # where user TorchScript models live
    data_path: str                 # where real.zip and fake.zip live
    models_list: list[str]         # benchmark model names to instantiate
    flag_train: bool               # if False, we still need at least one model to evaluate
    type_train: str                # 'scratch' | 'pretrain' | 'both'
    transform_list: list[str]      # e.g. ['mel', 'log', 'dwt']
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    num_workers: int
    eval_metric: list[str]         # evaluation metrics names
    device: str                    # 'cpu' | 'gpu'
    real_zip: str                  # filename under data_path
    fake_zip: str                  # filename under data_path

Optionally, the config can also define:
    outputs_path: str              # to override the default <repo_root>/outputs
    ALLOWED_* sets                 # to override default domains

Outputs / behavior
------------------
- On success: does nothing and returns None.
- On failure: raises `ConfigError` with a concrete, actionable message.
- If `create_dirs=True`, it will try to create missing outputs/models folders.
- If `device='gpu'` but CUDA is not available, it will silently downgrade to
  'cpu' (with a warning) when `check_cuda=True`.
"""
from __future__ import annotations

from pathlib import Path
import warnings


class ConfigError(ValueError):
    """Raised when configuration values are invalid or incomplete."""


# Defaults (subclasses may override them)
DEFAULT_ALLOWED_TYPE_TRAIN = {"scratch", "pretrain", "both"}
DEFAULT_ALLOWED_DEVICE = {"cpu", "gpu"}
DEFAULT_ALLOWED_OPTIM = {"adam", "sgd"}
DEFAULT_ALLOWED_TRANSFORMS = {"mel", "log", "dwt"}
DEFAULT_ALLOWED_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}


class ValidationMixin:
    """Tiny, modular validator. Nothing runs until you call `validate()`."""

    # -------- Orchestrator --------
    def validate(
        self,
        *,
        check_paths: bool = True,
        check_data: bool = True,
        check_lists: bool = True,
        check_domains: bool = True,
        check_numbers: bool = True,
        check_semantics: bool = True,
        create_dirs: bool = True,
        check_cuda: bool = True,
    ) -> None:
        """Run all selected validations; raise ConfigError on the first failure.

        Parameters
        ----------
        check_paths:
            Ensure `outputs/`, `models_path` and `data_path` exist. Can create
            outputs/models if `create_dirs=True`.
        check_data:
            Ensure `real_zip` and `fake_zip` exist under `data_path`.
        check_lists:
            Ensure list-like fields (transforms/models) are not empty.
        check_domains:
            Validate that string-valued fields are within the allowed sets.
        check_numbers:
            Validate basic numeric ranges (epochs, batch_size, learning_rate...).
        check_semantics:
            Validate cross-field logic, such as "pretrain" needing actual models.
        create_dirs:
            If True and `check_paths` is on, create outputs/models when missing.
        check_cuda:
            If True and `device=='gpu'`, try to check CUDA and downgrade to CPU
            with a warning when not available.
        """
        if check_domains:
            self._check_domains()
        if check_numbers:
            self._check_numbers()
        if check_lists:
            self._check_lists()
        if check_paths:
            self._check_paths(create_dirs=create_dirs)
        if check_data:
            self._check_data_presence()
        if check_semantics:
            self._check_semantics(check_cuda=check_cuda)

    # -------- Checks --------
    def _check_paths(self, *, create_dirs: bool) -> None:
        # Keep outputs anchored to the repository root unless cfg.outputs_path says otherwise
        repo_root = Path(__file__).resolve().parents[1]  # parent of the package root

        # Honor user-provided outputs_path (relative to repo or absolute)
        outputs_cfg = getattr(self, "outputs_path", None)
        if outputs_cfg:
            out_path = Path(outputs_cfg)
            outputs = out_path if out_path.is_absolute() else (repo_root / out_path)
        else:
            outputs = repo_root / "outputs"

        models = Path(self.models_path)
        data = Path(self.data_path)

        if not outputs.exists():
            if create_dirs:
                try:
                    outputs.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ConfigError(f"Could not create outputs dir: {outputs} (error: {e})")
            if not outputs.exists():
                raise ConfigError(f"outputs dir does not exist: {outputs}")

        if not models.exists():
            if create_dirs:
                try:
                    models.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ConfigError(f"Could not create models_path: {models} (error: {e})")
            if not models.exists():
                raise ConfigError(f"models_path does not exist: {models}")

        if not data.exists() or not data.is_dir():
            raise ConfigError(f"data_path must exist and be a folder: {data}")

    def _check_data_presence(self) -> None:
        data = Path(self.data_path)
        real = data / self.real_zip
        fake = data / self.fake_zip
        if not real.is_file():
            raise ConfigError(f"Missing '{self.real_zip}' in '{data}'.")
        if not fake.is_file():
            raise ConfigError(f"Missing '{self.fake_zip}' in '{data}'.")

    def _check_lists(self) -> None:
        if not getattr(self, "transform_list", None):
            raise ConfigError("transform_list cannot be empty (e.g., ['mel','log','dwt']).")

        if not bool(getattr(self, "flag_train", True)):
            if not self._has_custom_models() and not getattr(self, "models_list", []):
                raise ConfigError(
                    "No models available to evaluate: models_list is empty and models_path has no .pt/.pth, "
                    "while flag_train=False."
                )

    def _has_custom_models(self) -> bool:
        models = Path(self.models_path)
        if not models.exists():
            return False
        return any(models.glob("**/*.pt")) or any(models.glob("**/*.pth"))

    def _check_domains(self) -> None:
        allowed_t = getattr(self, "ALLOWED_TYPE_TRAIN", DEFAULT_ALLOWED_TYPE_TRAIN)
        allowed_d = getattr(self, "ALLOWED_DEVICE", DEFAULT_ALLOWED_DEVICE)
        allowed_o = getattr(self, "ALLOWED_OPTIM", DEFAULT_ALLOWED_OPTIM)
        allowed_tf = getattr(self, "ALLOWED_TRANSFORMS", DEFAULT_ALLOWED_TRANSFORMS)
        allowed_m = getattr(self, "ALLOWED_METRICS", DEFAULT_ALLOWED_METRICS)

        if str(self.type_train).lower() not in allowed_t:
            raise ConfigError(f"type_train must be one of: {sorted(allowed_t)}")
        if str(self.device).lower() not in allowed_d:
            raise ConfigError(f"device must be one of: {sorted(allowed_d)}")
        if str(self.optimizer).lower() not in allowed_o:
            raise ConfigError("optimizer must be 'Adam' or 'SGD'.")

        bad_tf = [t for t in self.transform_list if str(t).lower() not in allowed_tf]
        if bad_tf:
            raise ConfigError(f"Unsupported transforms: {bad_tf}. Supported: {sorted(allowed_tf)}")

        bad_m = [m for m in self.eval_metric if str(m).lower() not in allowed_m]
        if bad_m:
            raise ConfigError(f"Unsupported metrics: {bad_m}. Supported: {sorted(allowed_m)}")

    def _check_numbers(self) -> None:
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ConfigError("epochs must be an integer > 0.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ConfigError("batch_size must be an integer > 0.")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ConfigError("num_workers must be an integer ≥ 0.")
        try:
            lr = float(self.learning_rate)
        except Exception:
            raise ConfigError("learning_rate must be numeric > 0.")
        if lr <= 0:
            raise ConfigError("learning_rate must be > 0.")
        if not isinstance(self.patience, int) or self.patience < 0:
            raise ConfigError("patience must be an integer ≥ 0.")

    def _check_semantics(self, *, check_cuda: bool) -> None:
        if str(self.type_train).lower() == "pretrain":
            if not self.models_list and not self._has_custom_models():
                raise ConfigError(
                    "type_train='pretrain' requires at least one available model "
                    "(a name in models_list or a .pt/.pth in models_path)."
                )

        if str(self.device).lower() == "gpu" and check_cuda:
            # Best-effort check: downgrade to CPU if we cannot confirm CUDA
            try:
                import importlib
                spec = importlib.util.find_spec("torch")
                if spec is None:
                    warnings.warn("'device'='gpu' but PyTorch is not installed. Falling back to 'cpu'.", RuntimeWarning)
                    self.device = "cpu"
                else:
                    import torch  # type: ignore
                    if not torch.cuda.is_available():
                        warnings.warn("'device'='gpu' but CUDA is not available. Falling back to 'cpu'.", RuntimeWarning)
                        self.device = "cpu"
            except Exception as e:
                warnings.warn(f"Could not verify CUDA availability: {e}", RuntimeWarning)
