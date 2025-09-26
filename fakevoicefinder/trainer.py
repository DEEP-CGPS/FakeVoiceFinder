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
  outputs/<EXP>/datasets/{train|test}/transforms/<transform>/{real|fake}/*.npy

Each `.npy` contains an array of shape [H, W] or [C, H, W], where C ∈ {1, 3}.
Labels: real→0, fake→1 (also accepts folders "real"/"fake").

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

_CLASS_DIRS = [("real", 0), ("real", 0), ("fake", 1), ("fake", 1)]


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
            raise RuntimeError(f"No .npy files found under {self.root} (expected 'real'/'fake').")

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
        print(f"[Trainer] Using device: {self.device.type}")
        print(f"[Trainer] Transforms to train: {transforms}")
        print(f"[Trainer] Models found: {list(models.keys())}")

        results: Dict[str, Dict[str, Dict[str, str]]] = {}

        for model_name, entry in models.items():
            print(f"\n=== MODEL: {model_name} ===")
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
            print(f"[{model_name}] Hyperparams -> epochs={epochs}, lr={lr}, bs={bs}, "
                  f"optimizer={optim_name}, patience={patience}, seed={seed}, num_workers={num_workers}")

            for transform in transforms:
                train_loader, test_loader = self._build_loaders(transform, batch_size=bs, num_workers=num_workers)
                print(f"[{model_name}][{transform}] Dataset sizes -> train: {len(train_loader.dataset)}, "
                      f"test: {len(test_loader.dataset)}")
                print(f"[{model_name}][{transform}] Batches -> train: {len(train_loader)}, test: {len(test_loader)}")

                trained_for_transform: Dict[str, str] = {}

                for variant, ckpt_rel in loaded_variants.items():
                    ckpt_path = self.repo_root / ckpt_rel
                    print(f"[{model_name}][{transform}][{variant}] Loading checkpoint: {ckpt_path}")

                    # Load model (benchmark pickled or user TorchScript)
                    if variant in ("scratch", "pretrain"):
                        model = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                        print(f"[{model_name}][{transform}][{variant}] Loaded pickled module.")
                    elif variant == "usermodel_jit":
                        model = torch.jit.load(str(ckpt_path), map_location="cpu")
                        print(f"[{model_name}][{transform}][{variant}] Loaded TorchScript module.")
                    else:
                        print(f"[{model_name}][{transform}][{variant}] Unknown variant, skipping.")
                        continue

                    model.to(self.device)
                    torch.manual_seed(seed)
                    print(f"[{model_name}][{transform}][{variant}] Start training for {epochs} epochs")

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
                        save_as=("jit" if variant == "usermodel_jit" else "pickle"),
                    )
                    print(f"[{model_name}][{transform}] Saved best checkpoint -> {best_path.name}")

                    trained_for_transform[variant] = self._repo_rel(best_path)

                if trained_for_transform:
                    self._update_trained_variants(model_name, transform, trained_for_transform)
                    results.setdefault(model_name, {})[transform] = trained_for_transform

        self._write_manifest()
        print("\n[Trainer] All training finished.")
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
    def _eval_metrics(self, model, loader: DataLoader) -> Tuple[float, np.ndarray]:
        """
        Compute accuracy and 2x2 confusion matrix on a DataLoader.

        Returns:
            (acc, cm) where cm is:
                [[TN, FP],
                 [FN, TP]]
            with class 0 = 'real', class 1 = 'fake'.
        """
        model.eval()
        y_true_list, y_pred_list = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=False)
            y = y.to(self.device, non_blocking=False)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)
            y_true_list.append(y.cpu())
            y_pred_list.append(pred.cpu())

        if not y_true_list:
            return 0.0, np.zeros((2, 2), dtype=int)

        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()
        acc = float((y_true == y_pred).mean())

        # Build 2x2 confusion matrix (binary)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in (0, 1) and p in (0, 1):
                cm[t, p] += 1
        return acc, cm

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
            running_loss = 0.0
            batches = 0
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
                running_loss += float(loss.detach().item())
                batches += 1

            epoch_loss = running_loss / max(1, batches)
            acc, cm = self._eval_metrics(model, test_loader)

            # Pretty-print confusion matrix
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            print(f"[{model_name}][{transform}][{variant}] Epoch {epoch}/{epochs} - "
                  f"loss={epoch_loss:.4f} acc={acc:.4f}")
            print(f"[{model_name}][{transform}][{variant}] Confusion matrix (test):")
            print(f"[[TN={tn:4d}, FP={fp:4d}],")
            print(f" [FN={fn:4d}, TP={tp:4d}]]")

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_snapshot = model
                print(f"[{model_name}][{transform}][{variant}] ✅ New best acc={best_acc:.4f} at epoch {epoch}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(patience):
                    print(f"[{model_name}][{transform}][{variant}] Early stopping at epoch {epoch}")
                    break

        acc_tag = f"{best_acc:.2f}"
        base_name = f"{_safe_filename(model_name)}_{variant}_{transform}_seed{seed}_epoch{best_epoch:03d}_acc{acc_tag}"
        out_path = self.exp.trained_models / f"{base_name}.pt"

        # Save best checkpoint
        if save_as == "jit":
            try:
                best_model_snapshot.cpu()
                torch.jit.save(best_model_snapshot, str(out_path))
            finally:
                best_model_snapshot.to(self.device)
        else:
            torch.save(best_model_snapshot, str(out_path))

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

