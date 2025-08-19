"""
trainer.py — Training utilities for fakevoicefinder.

This module trains all models registered in `experiment.json` across all prepared
data transforms (e.g., "mel", "log").

Supported model variants
------------------------
1) Benchmark PICKLED nn.Module:
   - "scratch" / "pretrain"
   Directly load the pickled module (no architecture rebuild), train, and save
   the best checkpoint as a PICKLED nn.Module with `.pt` extension.

2) User TorchScript ONLY:
   - "usermodel_jit"
   Load a TorchScript module via `torch.jit.load`, train it as-is (no internal
   layer surgery), and save the best checkpoint as a TorchScript archive (.pt).

Data expectations
-----------------
Prepared datasets must exist under:
  outputs/<EXP>/datasets/{train|test}/transforms/<transform>/{reals|fakes}/*.npy

Each `.npy` contains an array of shape [H, W] or [C, H, W], where C ∈ {1, 3}.
Labels: reals→0, fakes→1 (also accepts folders "real"/"fake").

Manifest update
---------------
After training, updates:
  experiment.models[<name>].trained_variants[<transform>][<variant>] = <repo_rel_path>

Security note
-------------
Loading pickled modules requires trusting the file (executes Python on load).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .experiment import CreateExperiment
from .validatorsforvoice import ConfigError
from .model_loader import _safe_filename

# ----------------------- Dataset -----------------------

_CLASS_DIRS = [("reals", 0), ("real", 0), ("fakes", 1), ("fake", 1)]


class NpyFolderDataset(Dataset):
    """Minimal dataset for .npy spectrogram tensors saved under class folders."""
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Transform folder not found: {self.root}")

        self.items: List[Tuple[Path, int]] = []
        for cls_name, label in _CLASS_DIRS:
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for p in cls_dir.glob("*.npy"):
                    self.items.append((p, label))

        if not self.items:
            raise RuntimeError(f"No .npy files found under {self.root} (expected 'reals'/'fakes').")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, label = self.items[idx]
        arr = np.load(str(p))  # [H, W] or [C, H, W]
        if arr.ndim == 2:
            arr = arr[None, ...]  # [1, H, W]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            pass
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} in {p.name}.")
        x = torch.from_numpy(arr).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ----------------------- Trainer -----------------------

class Trainer:
    """
    Train benchmarks (pickled nn.Module) and user TorchScript models across all transforms.

    Typical usage:
        trainer = Trainer(exp)              # 'exp' is a built CreateExperiment
        trainer.train_all()                 # trains each variant on each transform

    Args:
        exp: A `CreateExperiment` instance with a non-None `experiment_dict`.
        device: Optional device override. If None, uses config.device
                ("gpu"/"cpu") and falls back to CPU if CUDA is unavailable.
    """

    def __init__(self, exp: CreateExperiment, device: Optional[str] = None) -> None:
        if exp.experiment_dict is None:
            raise RuntimeError("CreateExperiment has no experiment_dict. Call exp.build() first.")

        self.exp = exp
        self.cfg = exp.cfg
        self.repo_root = exp.repo_root
        self.root = exp.root

        # Resolve device
        d = (self.cfg.device or "cpu").strip().lower()
        if device is not None:
            d = device
        if d == "gpu" and not torch.cuda.is_available():
            print("⚠️ CUDA not available; falling back to CPU.")
            d = "cpu"
        self.device = torch.device("cuda" if d in ("gpu", "cuda") else "cpu")

        # Seed
        torch.manual_seed(int(self.cfg.seed))

        # Load manifest
        self.manifest_path = self.root / "experiment.json"
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest: Dict = json.load(f)

    # ----------- public API -----------

    def train_all(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Train every model variant for each available transform.

        Returns:
            {
              "<model_name>": {
                "<transform>": { "<variant>": "<repo_rel_path>", ... },
                ...
              },
              ...
            }
        """
        transforms = self._list_transforms()
        if not transforms:
            raise RuntimeError("No transforms found in manifest to train on.")

        models = self.manifest.get("experiment", {}).get("models", {})
        results: Dict[str, Dict[str, Dict[str, str]]] = {}

        for model_name, entry in models.items():
            loaded_variants: Dict[str, str] = entry.get("loaded_variants", {})
            if not loaded_variants:
                continue

            # Hyperparameters (per-model overrides fallback to config defaults)
            params = entry.get("train_parameters", {})
            epochs = int(params.get("epochs", self.cfg.epochs))
            lr = float(params.get("learning_rate", self.cfg.learning_rate))
            bs = int(params.get("batch_size", self.cfg.batch_size))
            optim_name = str(params.get("optimizer", self.cfg.optimizer))
            patience = int(params.get("patience", self.cfg.patience))
            seed = int(params.get("seed", self.cfg.seed))
            num_workers = int(params.get("num_workers", self.cfg.num_workers))

            for transform in transforms:
                train_loader, test_loader = self._build_loaders(transform, batch_size=bs, num_workers=num_workers)
                trained_for_transform: Dict[str, str] = {}

                for variant, ckpt_rel in loaded_variants.items():
                    ckpt_path = self.repo_root / ckpt_rel

                    if variant in ("scratch", "pretrain"):
                        # Benchmark PICKLED nn.Module: load as module (unsafe but requested)
                        model = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                        model.to(self.device)
                        torch.manual_seed(seed)

                        best_path = self._train_one(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            epochs=epochs,
                            lr=lr,
                            optimizer_name=optim_name,
                            patience=patience,
                            model_name=model_name,
                            variant=variant,
                            transform=transform,
                            seed=seed,
                            save_as="pickle",  # saved as .pt (pickled)
                        )
                        trained_for_transform[variant] = self._repo_rel(best_path)

                    elif variant == "usermodel_jit":
                        # TorchScript user model: load module and train as-is (no surgery)
                        model = torch.jit.load(str(ckpt_path), map_location="cpu")
                        model.to(self.device)
                        torch.manual_seed(seed)

                        best_path = self._train_one(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            epochs=epochs,
                            lr=lr,
                            optimizer_name=optim_name,
                            patience=patience,
                            model_name=model_name,
                            variant=variant,
                            transform=transform,
                            seed=seed,
                            save_as="jit",     # saved as .pt (TorchScript)
                        )
                        trained_for_transform[variant] = self._repo_rel(best_path)

                    else:
                        # Unknown variant; ignore
                        continue

                if trained_for_transform:
                    self._update_trained_variants(model_name, transform, trained_for_transform)
                    results.setdefault(model_name, {})[transform] = trained_for_transform

        self._write_manifest()
        return results

    # ----------- helpers -----------

    def _list_transforms(self) -> List[str]:
        """Return the list of transform names found in the manifest."""
        exp = self.manifest.get("experiment", {})
        tr = exp.get("train_data", {}).get("transforms_dataset", {})
        return list(tr.keys())

    def _build_loaders(self, transform: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
        """
        Build DataLoaders for train/test of a given transform.

        Raises:
            RuntimeError: If the transform is missing for either split.
        """
        exp = self.manifest.get("experiment", {})
        train_tf = exp.get("train_data", {}).get("transforms_dataset", {}).get(transform, {})
        test_tf = exp.get("test_data", {}).get("transforms_dataset", {}).get(transform, {})

        if not train_tf or not test_tf:
            raise RuntimeError(f"Transform '{transform}' not found in manifest for both train and test.")

        train_root = self.repo_root / train_tf["path"]
        test_root = self.repo_root / test_tf["path"]

        ds_train = NpyFolderDataset(train_root)
        ds_test = NpyFolderDataset(test_root)

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        return dl_train, dl_test

    @torch.no_grad()
    def _eval_acc(self, model, loader: DataLoader) -> float:
        """
        Compute classification accuracy on a DataLoader.

        The model may be an nn.Module or a TorchScript module.
        The forward may return a tensor or a (tensor, ...) tuple; the first element is used.
        """
        model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=False)
            y = y.to(self.device, non_blocking=False)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return (correct / max(total, 1)) if total else 0.0

    def _train_one(
        self,
        *,
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        lr: float,
        optimizer_name: str,
        patience: int,
        model_name: str,
        variant: str,
        transform: str,
        seed: int,
        save_as: str = "pickle",  # "pickle" | "jit"
    ) -> Path:
        """
        Train a single model instance with early stopping on test accuracy.

        Raises:
            RuntimeError: If the model has no trainable parameters.
        """
        params = list(model.parameters()) if hasattr(model, "parameters") else []
        if not params:
            raise RuntimeError(f"Model '{model_name}' variant '{variant}' has no trainable parameters.")

        criterion = nn.CrossEntropyLoss()
        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(params, lr=lr)

        best_acc = -1.0
        best_epoch = -1
        epochs_no_improve = 0
        best_model_snapshot = None  # for pickled save

        for epoch in range(1, int(epochs) + 1):
            model.train()
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            acc = self._eval_acc(model, test_loader)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                epochs_no_improve = 0
                if save_as == "pickle":
                    best_model_snapshot = model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(patience):
                    break

        acc_tag = f"{best_acc:.2f}"
        base_name = f"{_safe_filename(model_name)}_{variant}_{transform}_seed{seed}_epoch{best_epoch:03d}_acc{acc_tag}"

        out_path = self.exp.trained_models / f"{base_name}.pt"
        if save_as == "jit":
            try:
                model.cpu()
                torch.jit.save(model, str(out_path))
            finally:
                model.to(self.device)
        else:
            to_save = best_model_snapshot if best_model_snapshot is not None else model
            torch.save(to_save, str(out_path))

        return out_path

    def _update_trained_variants(self, model_name: str, transform: str, variant_map: Dict[str, str]) -> None:
        """Merge/update the manifest's trained_variants for a model/transform."""
        ex = self.exp.experiment_dict
        if ex is None:
            return
        models = ex.setdefault("models", {})
        entry = models.setdefault(model_name, {})
        tv = entry.get("trained_variants", {})
        tv[transform] = dict(variant_map)
        entry["trained_variants"] = tv

    def _write_manifest(self) -> None:
        """Persist the current in-memory experiment_dict to experiment.json."""
        self.exp.update_manifest()

    def _repo_rel(self, p: Path) -> str:
        """Convert an absolute path to a repository-relative POSIX path."""
        rel = os.path.relpath(str(Path(p).resolve()), str(self.repo_root.resolve()))
        return rel.replace(os.sep, "/")
