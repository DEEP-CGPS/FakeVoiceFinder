"""
metrics.py — Evaluation and plotting utilities for FakeVoiceFinder.

This module reads the trained checkpoints registered in `experiment.json`,
evaluates them on the prepared **test** transforms, builds a metrics DataFrame,
and exposes three plotting helpers:

1) plot_architectures_for_transform:
   Bar chart of (model + variant) performance for a single transform.

2) plot_variants_for_model:
   Bar chart of one (model, variant) across all available transforms.

3) plot_heatmap_models_transforms:
   Heatmap where rows = "model (variant)" and columns = transforms.

Key principles
--------------
- Read-only: this module does NOT write back to experiment.json.
- Reports: figures/CSVs are written to `outputs/<exp>/reports/`.
- Variants: supports 'scratch', 'pretrain' (pickled nn.Module) and
  'usermodel_jit' (TorchScript).
- Loading: we call `torch.load(..., weights_only=False)` for pickled modules
  to avoid PyTorch 2.6+ errors.

Dependencies
------------
- numpy, pandas, matplotlib
- scikit-learn (for metrics)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import (
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required for metrics. Install: pip install scikit-learn"
    ) from e

from .experiment import CreateExperiment


# ---------------------------------------------------------------------
# Dataset to read test .npy features saved by the data preparation step
# ---------------------------------------------------------------------

# Expected classes and numeric labels (duplicated to scan potential folders)
_CLASS_DIRS = [("real", 0), ("real", 0), ("fake", 1), ("fake", 1)]


class NpyFolderDataset(Dataset):
    """
    Minimal dataset for spectrogram-like `.npy` files saved per class.

    Expected directory layout:
        <root>/<class>/*.npy, with <class> in {'real', 'fake'}.

    Each .npy must be:
        - 2D: [H, W] → converted to [1, H, W], or
        - 3D: [C, H, W] with C in {1, 3}.
    """

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


# ---------------------------------------------------------------------
# MetricsReporter
# ---------------------------------------------------------------------

class MetricsReporter:
    """
    Evaluate trained checkpoints on test transforms and produce metrics/plots.

    Typical flow:
        cfg = ExperimentConfig()
        cfg.run_name = "exp_name"
        exp = CreateExperiment(cfg, experiment_name=cfg.run_name)
        exp.build(make_dirs=False)

        rep = MetricsReporter(exp)
        df  = rep.evaluate_all("metrics.csv")

        rep.plot_architectures_for_transform(df, transform="mel", metric="accuracy")
        rep.plot_variants_for_model(df, model="resnet18", variant="pretrain", metric="f1")
        rep.plot_heatmap_models_transforms(df, metric="accuracy")

    Notes
    -----
    - If the in-memory experiment dict is missing, it is loaded from
      `outputs/<exp>/experiment.json`.
    - Output artifacts (CSVs, figures) go to the experiment's `reports/` folder.
    """

    def __init__(self, exp: CreateExperiment, device: Optional[str] = None) -> None:
        # Base experiment objects/paths
        self.exp = exp
        self.cfg = exp.cfg
        self.repo_root = exp.repo_root
        self.root = exp.root

        manifest_path = self.root / "experiment.json"
        if self.exp.experiment_dict is None:
            if not manifest_path.exists():
                raise RuntimeError(f"experiment.json not found at {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                full = json.load(f)
            self.exp.experiment_dict = full.get("experiment", {})
        self.manifest: Dict = self.exp.experiment_dict

        # Reports dir (local-only, no manifest writeback)
        self.reports_dir = self.root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        d = (self.cfg.device or "cpu").strip().lower()
        if device is not None:
            d = device
        if d in ("gpu", "cuda") and not torch.cuda.is_available():
            print("⚠️ CUDA not available; falling back to CPU.")
            d = "cpu"
        self.device = torch.device("cuda" if d in ("gpu", "cuda") else "cpu")

        # DataLoader cache per transform
        self._test_loader_cache: Dict[str, DataLoader] = {}

    # -------------------- public API --------------------

    def evaluate_all(self, save_csv_name: Optional[str] = None) -> pd.DataFrame:
        """
        Evaluate every trained variant found in the manifest and optionally
        persist the result as CSV inside `reports/`.

        Args:
            save_csv_name: filename for the CSV (e.g. "metrics.csv").

        Returns:
            pd.DataFrame with one row per (model, variant, transform).
        """
        df = self.summarize()
        if save_csv_name:
            out = self.reports_dir / str(save_csv_name)
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[metrics] CSV saved -> {out}")
        return df

    def summarize(self) -> pd.DataFrame:
        """
        Build a DataFrame with one row per (model, variant, transform).

        Columns:
            ['model', 'variant', 'transform',
             'accuracy', 'f1', 'f1_macro', 'f1_micro', 'precision', 'recall',
             'checkpoint']

        It uses only the **test** split for each transform.
        """
        rows: List[Dict] = []

        models = self.manifest.get("models", {})
        if not models:
            raise RuntimeError("No 'models' section found in experiment.json.")

        for model_name, meta in models.items():
            trained = meta.get("trained_variants", {})
            if not trained:
                continue

            for transform, var_map in trained.items():
                loader = self._get_test_loader(transform)

                for variant, ckpt_rel in var_map.items():
                    ckpt_path = self.repo_root / ckpt_rel

                    model = self._load_model_variant(ckpt_path, variant)
                    acc, f1_main, f1_macro, f1_micro, prec, rec = self._eval_metrics(model, loader)

                    rows.append(
                        {
                            "model": str(model_name),
                            "variant": str(variant),
                            "transform": str(transform),
                            "accuracy": round(float(acc) * 100.0, 2),
                            "f1": round(float(f1_main) * 100.0, 2),
                            "f1_macro": round(float(f1_macro) * 100.0, 2),
                            "f1_micro": round(float(f1_micro) * 100.0, 2),
                            "precision": round(float(prec) * 100.0, 2),
                            "recall": round(float(rec) * 100.0, 2),
                            "checkpoint": ckpt_rel,
                        }
                    )

        if not rows:
            raise RuntimeError("No trained variants found to evaluate.")

        df = pd.DataFrame(rows)
        df = df.sort_values(by=["transform", "model", "variant"]).reset_index(drop=True)
        return df

    # -------------------- plotting --------------------

    def plot_architectures_for_transform(
        self,
        df: pd.DataFrame,
        *,
        transform: str,
        metric: str = "accuracy",
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        out_name: Optional[str] = None,
    ) -> None:
        """
        Plot all (model, variant) pairs for a given transform as a bar chart.

        Args:
            df: DataFrame returned by summarize()/evaluate_all().
            transform: target transform (e.g. "mel", "log").
            metric: column to plot (e.g. "accuracy").
            y_min, y_max: optional y-axis limits in percentage.
            out_name: optional filename (saved to reports/).
        """
        mcol = metric.lower()
        allowed = {"accuracy", "f1", "f1_macro", "f1_micro", "precision", "recall"}
        if mcol not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}.")

        sub = df[df["transform"].str.lower() == transform.lower()].copy()
        if sub.empty:
            raise RuntimeError(f"No rows for transform='{transform}' in the DataFrame.")

        sub["label"] = sub["model"].astype(str) + " (" + sub["variant"].astype(str) + ")"
        sub = sub.sort_values(by=mcol, ascending=False)

        plt.figure(figsize=(12, 5))
        bars = plt.bar(sub["label"], sub[mcol])
        plt.ylabel(f"{mcol.replace('_', ' ').title()} (%)")
        plt.title(f"Architectures/variants — Transform: {transform}")
        if y_min is not None or y_max is not None:
            ymin = y_min if y_min is not None else plt.ylim()[0]
            ymax = y_max if y_max is not None else plt.ylim()[1]
            plt.ylim(ymin, ymax)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()

        out = self._resolve_savepath(out_name=out_name, default_name=f"fig_architectures_{transform}_{mcol}.png")
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"[metrics] Figure saved -> {out}")

    def plot_variants_for_model(
        self,
        df: pd.DataFrame,
        *,
        model: str,
        variant: str,
        metric: str = "accuracy",
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        out_name: Optional[str] = None,
    ) -> None:
        """
        Plot all transforms for a single (model, variant) as a bar chart.

        Args:
            df: DataFrame from summarize()/evaluate_all().
            model: model name exactly as in experiment.json.
            variant: one of {"scratch", "pretrain", "usermodel_jit"}.
            metric: metric column to use.
            y_min, y_max: optional y-axis limits.
            out_name: optional filename (saved to reports/).
        """
        mcol = metric.lower()
        allowed = {"accuracy", "f1", "f1_macro", "f1_micro", "precision", "recall"}
        if mcol not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}.")

        sub = df[(df["model"].str.lower() == model.lower()) &
                 (df["variant"].str.lower() == variant.lower())].copy()
        if sub.empty:
            raise RuntimeError(f"No rows for model='{model}', variant='{variant}' in the DataFrame.")

        sub = sub.sort_values(by="transform")
        plt.figure(figsize=(9, 4.5))
        bars = plt.bar(sub["transform"], sub[mcol])
        plt.ylabel(f"{mcol.replace('_', ' ').title()} (%)")
        plt.title(f"{model} — variant: {variant}")
        if y_min is not None or y_max is not None:
            ymin = y_min if y_min is not None else plt.ylim()[0]
            ymax = y_max if y_max is not None else plt.ylim()[1]
            plt.ylim(ymin, ymax)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()

        out = self._resolve_savepath(out_name=out_name, default_name=f"fig_transforms_{model}_{variant}_{mcol}.png")
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"[metrics] Figure saved -> {out}")

    def plot_heatmap_models_transforms(
        self,
        df: pd.DataFrame,
        *,
        metric: str = "accuracy",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        out_name: Optional[str] = None,
    ) -> None:
        """
        Heatmap where rows are "model (variant)" and columns are transforms.

        Default behavior:
        - Color range: from the worst observed value up to 100%.
        - Colormap: 'RdYlGn' (red = low score, green = high score).
        - Each cell shows its value as a percentage.
        """
        mcol = metric.lower()
        allowed = {"accuracy", "f1", "f1_macro", "f1_micro", "precision", "recall"}
        if mcol not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}.")

        tmp = df.copy()
        tmp["row"] = tmp["model"].astype(str) + " (" + tmp["variant"].astype(str) + ")"
        pivot = tmp.pivot_table(index="row", columns="transform", values=mcol, aggfunc="max")

        A = pivot.to_numpy(dtype=float)

        worst = float(np.nanmin(A)) if A.size else 0.0
        vmin_eff = float(vmin) if vmin is not None else worst
        vmax_eff = float(vmax) if vmax is not None else 100.0

        plt.figure(figsize=(12, 7))
        im = plt.imshow(
            A,
            aspect="auto",
            interpolation="nearest",
            cmap="RdYlGn",
            vmin=vmin_eff,
            vmax=vmax_eff,
        )
        plt.colorbar(im, label=f"{mcol.replace('_', ' ').title()} (%)")

        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
        plt.title(f"Models+variants × transforms — {mcol.replace('_', ' ').title()}")

        def _text_color(value: float) -> str:
            r, g, b, _ = im.cmap(im.norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "black" if luminance > 0.5 else "white"

        n_rows, n_cols = A.shape
        for i in range(n_rows):
            for j in range(n_cols):
                val = A[i, j]
                if np.isnan(val):
                    plt.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
                else:
                    plt.text(
                        j,
                        i,
                        f"{val:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=_text_color(val),
                    )

        plt.tight_layout()
        out = self._resolve_savepath(out_name=out_name, default_name=f"fig_heatmap_{mcol}.png")
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"[metrics] Figure saved -> {out}")

    # -------------------- internals --------------------

    def _resolve_savepath(self, *, out_name: Optional[str], default_name: str) -> Path:
        """
        Build a path under `reports/` using the given name or the provided default.
        """
        name = out_name or default_name
        return self.reports_dir / name

    def _get_test_loader(self, transform: str) -> DataLoader:
        """Return and cache a DataLoader for the **test** split of the given transform."""
        key = transform.lower()
        if key in self._test_loader_cache:
            return self._test_loader_cache[key]

        exp = self.manifest
        test_tf = exp.get("test_data", {}).get("transforms_dataset", {}).get(key, {})
        if not test_tf:
            raise RuntimeError(f"Transform '{transform}' not found in manifest's test_data.transforms_dataset.")

        test_root = self.repo_root / test_tf["path"]
        ds_test = NpyFolderDataset(test_root)

        bs = int(getattr(self.cfg, "batch_size", 32))
        nw = int(getattr(self.cfg, "num_workers", 0))
        loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)
        self._test_loader_cache[key] = loader
        return loader

    def _load_model_variant(self, path: Path, variant: str):
        """
        Load a checkpoint depending on its variant:
        - 'usermodel_jit' → torch.jit.load
        - otherwise → torch.load(..., weights_only=False)
        """
        if variant.lower() == "usermodel_jit":
            return torch.jit.load(str(path), map_location=self.device)
        return torch.load(str(path), map_location=self.device, weights_only=False)

    @torch.no_grad()
    def _eval_metrics(self, model, loader: DataLoader) -> Tuple[float, float, float, float, float, float]:
        """
        Run the model on the given test loader and compute a set of classification metrics.

        Returns:
            (accuracy,
             f1_main,   # binary F1 (pos_label=1) if possible; otherwise weighted F1
             f1_macro,
             f1_micro,
             precision, # binary precision for class 1
             recall)    # binary recall for class 1
        """
        model.eval()
        all_preds: List[int] = []
        all_true: List[int] = []

        for x, y in loader:
            x = x.to(self.device, non_blocking=False)
            y = y.to(self.device, non_blocking=False)

            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)

            all_preds.extend(pred.detach().cpu().tolist())
            all_true.extend(y.detach().cpu().tolist())

        all_preds_arr = np.array(all_preds, dtype=np.int32)
        all_true_arr  = np.array(all_true,  dtype=np.int32)

        # Basic accuracy
        total = max(len(all_true_arr), 1)
        acc = float((all_preds_arr == all_true_arr).sum()) / float(total)

        # Which classes appear in GT and prediction
        labels_true = set(np.unique(all_true_arr).tolist())
        labels_pred = set(np.unique(all_preds_arr).tolist())
        has_both_true = {0, 1}.issubset(labels_true)
        has_pos_pred  = 1 in labels_pred

        # Per-class F1 and support (useful for weighted fallback)
        _, _, f1_per_class, support = precision_recall_fscore_support(
            all_true_arr, all_preds_arr, labels=[0, 1], zero_division=0
        )
        support = np.asarray(support, dtype=np.float64)
        if support.sum() > 0:
            f1_weighted = float(np.average(f1_per_class, weights=support))
        else:
            f1_weighted = 0.0

        # Main F1: use binary if both classes appear and positive is predicted;
        # otherwise fall back to the weighted version.
        if has_both_true and has_pos_pred:
            f1_main = float(f1_score(all_true_arr, all_preds_arr, average="binary", pos_label=1, zero_division=0))
        else:
            f1_main = f1_weighted

        # Macro & micro F1
        f1_macro = float(f1_score(all_true_arr, all_preds_arr, average="macro", zero_division=0))
        f1_micro = float(f1_score(all_true_arr, all_preds_arr, average="micro", zero_division=0))

        # Precision/recall for the positive class = 1
        precision = float(precision_score(all_true_arr, all_preds_arr, pos_label=1, zero_division=0))
        recall    = float(recall_score(all_true_arr, all_preds_arr,    pos_label=1, zero_division=0))

        return acc, f1_main, f1_macro, f1_micro, precision, recall




