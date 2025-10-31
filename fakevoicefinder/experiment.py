# experiment.py
"""
CreateExperiment: define, create and persist the on-disk structure of a single
FakeVoiceFinder experiment, plus a lightweight manifest (`experiment.json`).

Inputs
------
- An `ExperimentConfig` instance that already knows:
  - which models to consider (`cfg.models_list`)
  - which transforms to produce (`cfg.transform_list`)
  - training/eval parameters (epochs, batch size, optimizer, etc.)
  - where the dataset lives (`cfg.data_path`, ZIP names)
  - optional `cfg.run_name`
- Optionally, an explicit experiment name.

Outputs
-------
- A concrete experiment folder under: <repo_root>/outputs/<experiment_slug>/
- A JSON manifest (`experiment.json`) with ONLY repo-relative paths so the
  experiment can be moved or shared.
- A pre-created folder structure for:
  - models/ (with loaded/ and trained/)
  - datasets/train/ and datasets/test/ (original + per-transform)
  - reports/

What is recorded
----------------
- For each model name in `cfg.models_list`, an entry with placeholders for:
  - `loaded_path`
  - `trained_path`
  - and the training parameters used to generate that checkpoint
- For train/test, two datasets:
  - `original_dataset` (inside the experiment; num_items=0 initially)
  - `transforms_dataset` with one sub-entry per transform
- A `reports` section pointing to the experiment's reports folder

Design note
-----------
Model names are stored exactly as written by the user (no normalization). A
minimal sanitizer is applied only when building checkpoint filenames, so they
are safe across operating systems.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
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
            cfg: Experiment configuration object that indicates models, transforms
                 and training/runtime parameters.
            experiment_name: Optional external name for the experiment. If omitted,
                 `cfg.run_name` is used.
        """
        self.cfg = cfg
        self.exp_name = self._require_name(experiment_name or getattr(cfg, "run_name", None))
        self.slug = self._slugify(self.exp_name)

        # Locate the repository root and the base outputs folder.
        self.repo_root = Path(__file__).resolve().parents[1]  # parent of package root

        outputs_cfg = getattr(self.cfg, "outputs_path", None)
        if outputs_cfg:
            out_path = Path(outputs_cfg)
            outputs_base = out_path if out_path.is_absolute() else (self.repo_root / out_path)
        else:
            outputs_base = self.repo_root / "outputs"
        outputs_base.mkdir(parents=True, exist_ok=True)

        # Root folder for this specific experiment.
        self.root = outputs_base / self.slug

        # Precomputed subfolders for models and datasets.
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

        # Reports folder for metrics/data visualizations.
        self.reports_root = self.root / "reports"

        self.experiment_dict: Optional[Dict[str, Any]] = None  # set by build()

    # ---------------- public API ----------------

    def build(
        self,
        *,
        make_dirs: bool = True,
        include_original_subfolders: bool = True,
    ) -> Dict[str, Any]:
        """Create the experiment folder structure and return the manifest dict.

        If `make_dirs=True`, all directories are created on disk. The returned
        dict is also written to `experiment.json` so other components can find
        datasets, transforms, models and reports.
        """
        if make_dirs:
            for p in [
                self.root,
                self.models_root, self.loaded_models, self.trained_models,
                self.ds_root, self.train_root, self.test_root,
                self.train_tf_root, self.test_tf_root,
                self.reports_root,
            ]:
                p.mkdir(parents=True, exist_ok=True)
            if include_original_subfolders:
                self.train_orig.mkdir(parents=True, exist_ok=True)
                self.test_orig.mkdir(parents=True, exist_ok=True)
            # One subfolder per transform, for train and test.
            for tf in (self.cfg.transform_list or []):
                (self.train_tf_root / str(tf).lower()).mkdir(parents=True, exist_ok=True)
                (self.test_tf_root / str(tf).lower()).mkdir(parents=True, exist_ok=True)

        # Build the models section: one entry per model in the configuration.
        models_section: Dict[str, Any] = {}
        for model_name in (self.cfg.models_list or []):
            # Store model name exactly as provided by the user.
            models_section[str(model_name)] = {
                "loaded_path": None,
                "trained_path": None,
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
                    "transform": None,
                },
            }

        # Train/Test dataset sections with repo-relative paths and empty counts.
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
                    "path": self._repo_rel(self.train_orig),
                    "num_items": 0,
                },
                "transforms_dataset": train_transforms,
            },
            "test_data": {
                "original_dataset": {
                    "path": self._repo_rel(self.test_orig),
                    "num_items": 0,
                },
                "transforms_dataset": test_transforms,
            },
            "reports": {
                "path": self._repo_rel(self.reports_root)
            },
        }

        self.experiment_dict = experiment_dict

        if make_dirs:
            self._write_manifest(experiment_dict)

        return experiment_dict

    def checkpoint_filename(self, model_name: str, seed: int, transform: str,
                            epoch: int, key_metric: str, metric_value: float) -> str:
        """Return a standardized, readable checkpoint filename.

        Filenames are sanitized just enough to be portable, but they still
        include model name, seed, transform, epoch and a key metric value.
        """
        m = self._safe_for_filename(model_name)
        t = str(transform).lower()
        e = f"{int(epoch):03d}"
        k = "".join(ch for ch in str(key_metric).lower() if ch.isalnum() or ch == "_")
        v = f"{float(metric_value):.2f}"
        return f"{m}_seed{seed}_{t}_epoch{e}_{k}{v}.pth"

    def trained_checkpoint_path(self, filename: str) -> Path:
        """Return the absolute path to a file under `models/trained/`."""
        return (self.trained_models / filename).resolve()

    def update_manifest(self) -> None:
        """Rewrite `experiment.json` from the current in-memory manifest."""
        if self.experiment_dict is None:
            raise RuntimeError("No experiment_dict in memory. Call build() first.")
        self._write_manifest(self.experiment_dict)

    def prepare_data(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
        transforms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end data preparation helper for one experiment.

        Steps:
            1) Load ZIP members from `cfg.data_path` (real.zip, fake.zip).
            2) Split into train/test using a given ratio.
            3) Save originals inside the experiment and update counts.
            4) Generate spectrogram-like transforms and update the manifest.

        Returns:
            A dictionary with per-step summaries (load, split, save, transforms).
        """
        from .prepare_dataset import PrepareDataset

        sd = int(self.cfg.seed if seed is None else seed)
        tlist = list(self.cfg.transform_list or []) if transforms is None else list(transforms)

        prep = PrepareDataset(self)

        summary: Dict[str, Any] = {}
        summary["load"] = prep.load_data()
        summary["split"] = prep.split(train_ratio=train_ratio, seed=sd)
        summary["save_original"] = prep.save_original()

        summary["transforms"] = {}
        for t in tlist:
            res = prep.transform(t)
            prep.update_experiment_json(t)
            summary["transforms"][str(t).lower()] = res

        return summary

    # ---------------- internals ----------------

    @staticmethod
    def _require_name(name: Optional[str]) -> str:
        if not name or not str(name).strip():
            raise ConfigError("You must provide an experiment name or set cfg.run_name.")
        return str(name).strip()

    @staticmethod
    def _slugify(name: str) -> str:
        """Turn the experiment name into a filesystem-friendly slug."""
        name = name.strip().lower().replace(" ", "_")
        return re.sub(r"[^a-z0-9_\-]+", "", name)

    @staticmethod
    def _safe_for_filename(s: str) -> str:
        """Keep alnum/dash/underscore and replace everything else with '_'."""
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))

    def _repo_rel(self, p: Path) -> str:
        """Return a repo-relative POSIX path for anything created in the experiment."""
        rel = os.path.relpath(str(Path(p).resolve()), str(self.repo_root.resolve()))
        return rel.replace(os.sep, "/")

    def _write_manifest(self, experiment: Dict[str, Any]) -> None:
        payload = {
            "_created_at": datetime.utcnow().isoformat() + "Z",
            "_experiment_root": self._repo_rel(self.root),
            "experiment": experiment,
        }
        out = self.root / "experiment.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

