"""Validation mixin for the forvoice project (Fake Or Real Voice).

Simple design:
- `ConfigError` for actionable validation failures.
- `ValidationMixin.validate(...)` with boolean flags to toggle checks.
- Internal `_check_*` helpers that assume the subclass exposes a few attributes.

Expected subclass attributes (ExperimentConfig provides these):
    models_path: str
    data_path: str
    models_list: list[str]
    flag_train: bool
    type_train: str
    transform_list: list[str]
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    num_workers: int
    eval_metric: list[str]
    device: str
    real_zip: str
    fake_zip: str

Allowed sets can be overridden by the subclass as attributes:
    ALLOWED_TYPE_TRAIN, ALLOWED_DEVICE, ALLOWED_OPTIM,
    ALLOWED_TRANSFORMS, ALLOWED_METRICS
"""
from __future__ import annotations

from pathlib import Path
import warnings


class ConfigError(ValueError):
    """Raised when the configuration is invalid with an actionable message."""


# Defaults (subclass may override)
DEFAULT_ALLOWED_TYPE_TRAIN = {"scratch", "pretrain", "both"}
DEFAULT_ALLOWED_DEVICE = {"cpu", "gpu"}
DEFAULT_ALLOWED_OPTIM = {"adam", "sgd"}
DEFAULT_ALLOWED_TRANSFORMS = {"mel", "log", "dwt"}
DEFAULT_ALLOWED_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}


class ValidationMixin:
    """Tiny validation mixin. Call `validate()` only when you want.

    Toggle checks via boolean parameters. If you never call validate(), nothing runs.
    """

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
        """Validate the configuration. Raises ConfigError on failure.

        Args:
            check_paths: Verify path existence; optionally create outputs/models.
            check_data: Verify required data files exist under data_path.
            check_lists: Validate list fields (e.g., transform_list).
            check_domains: Validate enumerated/domain fields.
            check_numbers: Validate numeric ranges.
            check_semantics: Cross-field logic (e.g., pretrain requires models).
            create_dirs: If True and check_paths, create outputs/models when missing.
            check_cuda: If True and device=='gpu', try to verify CUDA and fallback.
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
        # Always keep outputs under the repository root
        repo_root = Path(__file__).resolve().parents[1]  # parent of 'forvoice'
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
                    "No models available to evaluate (models_list empty and no .pt/.pth in models_path) while flag_train=False."
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
                raise ConfigError("type_train='pretrain' requires available models (.pt/.pth in models_path or benchmarks in models_list).")

        if str(self.device).lower() == "gpu" and check_cuda:
            # Best-effort CUDA check: fallback to CPU with a warning if unavailable.
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
