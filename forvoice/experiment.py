"""
CreateExperiment: prepare an experiment folder and a lightweight manifest.

Key points
----------
- Always builds under: <repo_root>/outputs/<experiment_slug>/
- Stores ONLY repo-relative paths in experiment.json (portable)
  e.g., "outputs/exp_demo/datasets/train/transforms/mel"
- `original_dataset.path` now points INSIDE the experiment:
    outputs/<exp>/datasets/{train|test}/original
  and we initialize `num_items` with 0 (filled later by PrepareDataset.save_original()).

This module does NOT train models or generate transforms; it only prepares structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
import re
import os
from datetime import datetime

from .config import ExperimentConfig
from .validatorsforvoice import ConfigError


class CreateExperiment:
    """Create and manage the on-disk layout and manifest for one experiment."""

    def __init__(self, cfg: ExperimentConfig, experiment_name: Optional[str] = None) -> None:
        """
        Args:
            cfg: The ExperimentConfig (validated or not; your choice).
            experiment_name: Optional. If not provided, uses cfg.run_name.
        """
        self.cfg = cfg
        self.exp_name = self._require_name(experiment_name or getattr(cfg, "run_name", None))
        self.slug = self._slugify(self.exp_name)

        # Repo root and outputs base (fixed)
        self.repo_root = Path(__file__).resolve().parents[1]  # parent of 'forvoice'
        outputs_base = self.repo_root / "outputs"
        outputs_base.mkdir(parents=True, exist_ok=True)

        # Experiment root under outputs/
        self.root = outputs_base / self.slug

        # Precompute key folders
        self.models_root = self.root / "models"
        self.loaded_models = self.models_root / "loaded"
        self.trained_models = self.models_root / "trained"

        self.ds_root = self.root / "datasets"
        self.train_root = self.ds_root / "train"
        self.test_root = self.ds_root / "test"
        self.train_orig = self.train_root / "original"
        self.test_orig = self.test_root / "original"
        self.train_tf_root = self.train_root / "transforms"
        self.test_tf_root = self.test_root / "transforms"

        self.experiment_dict: Optional[Dict[str, Any]] = None  # set by build()

    # ---------------- public API ----------------

    def build(
        self,
        *,
        make_dirs: bool = True,
        include_original_subfolders: bool = True,
    ) -> Dict[str, Any]:
        """Create folders (optionally) and return the experiment dictionary."""
        if make_dirs:
            for p in [
                self.root,
                self.models_root, self.loaded_models, self.trained_models,
                self.ds_root, self.train_root, self.test_root,
                self.train_tf_root, self.test_tf_root,
            ]:
                p.mkdir(parents=True, exist_ok=True)
            if include_original_subfolders:
                self.train_orig.mkdir(parents=True, exist_ok=True)
                self.test_orig.mkdir(parents=True, exist_ok=True)
            # Per-transform subfolders
            for tf in (self.cfg.transform_list or []):
                (self.train_tf_root / str(tf).lower()).mkdir(parents=True, exist_ok=True)
                (self.test_tf_root / str(tf).lower()).mkdir(parents=True, exist_ok=True)

        # Models section (one entry per model in config)
        models_section: Dict[str, Any] = {}
        for raw_name in (self.cfg.models_list or []):
            mname = self._normalize_model_name(raw_name)
            models_section[mname] = {
                "loaded_path": None,   # e.g., outputs/<EXP>/models/loaded/<file>.pth
                "trained_path": None,  # e.g., outputs/<EXP>/models/trained/<file>.pth
                "train_parameters": {
                    "epochs": self.cfg.epochs,
                    "learning_rate": self.cfg.learning_rate,
                    "batch_size": self.cfg.batch_size,
                    "optimizer": self.cfg.optimizer,
                    "patience": self.cfg.patience,
                    "device": self.cfg.device,
                    "seed": self.cfg.seed,
                    "type_train": self.cfg.type_train,
                    "num_workers": self.cfg.num_workers,
                    "transform": None,  # set when the checkpoint is produced
                },
            }

        # Train/Test dataset sections (repo-relative; originals live inside the experiment)
        train_transforms: Dict[str, Any] = {}
        test_transforms: Dict[str, Any] = {}
        for tf in (self.cfg.transform_list or []):
            tf_key = str(tf).lower()
            train_transforms[tf_key] = {
                "path": self._repo_rel(self.train_tf_root / tf_key),
                "params": {},
            }
            test_transforms[tf_key] = {
                "path": self._repo_rel(self.test_tf_root / tf_key),
                "params": {},
            }

        experiment_dict: Dict[str, Any] = {
            "models": models_section,
            "train_data": {
                "original_dataset": {
                    "path": self._repo_rel(self.train_orig),  # outputs/<exp>/datasets/train/original
                    "num_items": 0,  # will be filled after save_original()
                },
                "transforms_dataset": train_transforms,
            },
            "test_data": {
                "original_dataset": {
                    "path": self._repo_rel(self.test_orig),   # outputs/<exp>/datasets/test/original
                    "num_items": 0,  # will be filled after save_original()
                },
                "transforms_dataset": test_transforms,
            },
        }

        self.experiment_dict = experiment_dict

        if make_dirs:
            self._write_manifest(experiment_dict)

        return experiment_dict

    def checkpoint_filename(self, model_name: str, seed: int, transform: str,
                            epoch: int, key_metric: str, metric_value: float) -> str:
        """Format a trained checkpoint filename.
        <Model>_seed<SEED>_<TRANSFORM>_epoch{EEE}_{KEY}{VAL}.pth
        Example: AlexNet_seed23_mel_epoch020_acc0.91.pth
        """
        m = self._normalize_model_name(model_name)
        t = str(transform).lower()
        e = f"{int(epoch):03d}"
        k = re.sub(r"[^A-Za-z0-9_]+", "", str(key_metric).lower())
        v = f"{float(metric_value):.2f}"
        return f"{m}_seed{seed}_{t}_epoch{e}_{k}{v}.pth"

    def trained_checkpoint_path(self, filename: str) -> Path:
        """Return the absolute path for a file under models/trained/."""
        return (self.trained_models / filename).resolve()

    def update_manifest(self) -> None:
        """Rewrite experiment.json using the current in-memory dictionary."""
        if self.experiment_dict is None:
            raise RuntimeError("No experiment_dict in memory. Call build() first.")
        self._write_manifest(self.experiment_dict)

    # ---------------- internals ----------------

    @staticmethod
    def _require_name(name: Optional[str]) -> str:
        if not name or not str(name).strip():
            raise ConfigError("You must provide an experiment name or set cfg.run_name.")
        return str(name).strip()

    @staticmethod
    def _slugify(name: str) -> str:
        name = name.strip().lower().replace(" ", "_")
        return re.sub(r"[^a-z0-9_\-]+", "", name)

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")

    def _repo_rel(self, p: Path) -> str:
        """Return repo-relative POSIX path (always with '/')."""
        rel = os.path.relpath(str(Path(p).resolve()), str(self.repo_root.resolve()))
        return rel.replace(os.sep, "/")

    def _write_manifest(self, experiment: Dict[str, Any]) -> None:
        payload = {
            "_created_at": datetime.utcnow().isoformat() + "Z",
            "_experiment_root": self._repo_rel(self.root),  # repo-relative
            "experiment": experiment,
        }
        out = self.root / "experiment.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
