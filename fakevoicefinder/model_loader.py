"""
model_loader.py — Load and prepare models for FakeVoiceFinder.

What this module does
---------------------
- Creates ready-to-train **benchmark** models from torchvision with 2 output
  classes (real/fake).
- Discovers and registers **user** models, but only when they are provided as
  TorchScript archives.

User models (TorchScript-only)
------------------------------
- Only TorchScript is accepted for user-provided models.
- Put your TorchScript file(s) (exported with `torch.jit.save`) under
  `../models/`.
- Each valid file is loaded with `torch.jit.load` and re-saved **unchanged** to:
    outputs/<EXP>/models/loaded/<basename>_usermodel_jit.pt
- The experiment manifest (`experiment.json`) records these under
  `loaded_variants["usermodel_jit"]`.

Benchmark models (torchvision)
------------------------------
- For each model name in `cfg.models_list`, we can build:
    - a "scratch" version (random init)
    - a "pretrain" version (ImageNet weights, when available)
- We adjust the first conv layer when the experiment requests 1-channel inputs.
- We replace the final classifier/head so it outputs **2** classes.
- We **do not** add Softmax here; the trainer will handle that on export.
- These are stored as **pickled** `nn.Module` objects under:
    outputs/<EXP>/models/loaded/

Security note
-------------
Loading pickled `nn.Module` requires that you trust the source file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import os

import torch
import torch.nn as nn

try:
    from torchvision import models as tvm
except Exception as e:
    raise RuntimeError(
        "torchvision is required for benchmark models. Install: pip install torchvision"
    ) from e

from .experiment import CreateExperiment
from .validatorsforvoice import ConfigError


# ---------------------------- small utils ----------------------------

def _safe_filename(s: str) -> str:
    """Return a filesystem-friendly name: keep alnum/dash/underscore, replace the rest with '_'."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))


def _find_first_conv(module: nn.Module) -> Optional[Tuple[nn.Module, str, nn.Conv2d]]:
    """
    Try to locate the first Conv2d in common torchvision classifiers.

    Returns:
        (parent_module, attribute_name, conv_layer) or None if no Conv2d is found.
    """
    if hasattr(module, "conv1") and isinstance(module.conv1, nn.Conv2d):
        return module, "conv1", module.conv1
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        if name == "features" and isinstance(child, nn.Sequential):
            for subname, layer in child.named_children():
                if isinstance(layer, nn.Conv2d):
                    return child, subname, layer

    for _, child in module.named_children():
        res = _find_first_conv(child)
        if res is not None:
            return res

    return None


def _adapt_first_conv_in_channels(model: nn.Module, target_in: int) -> None:
    """
    Replace the first Conv2d to match `target_in` channels.

    - If `target_in` is 1 and the original conv has 3 channels, we average RGB
      weights to create the 1-channel kernel.
    - Only 1 or 3 channels are supported.
    """
    if target_in in (None, 0) or target_in == 3:
        return
    if target_in not in (1, 3):
        raise ConfigError(f"Unsupported input_channels={target_in}. Use 1 or 3.")

    hit = _find_first_conv(model)
    if hit is None:
        return
    parent, attr, conv = hit
    if conv.in_channels == target_in:
        return

    new_conv = nn.Conv2d(
        in_channels=target_in,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1 if conv.groups == conv.in_channels else conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if conv.in_channels == 3 and target_in == 1 and hasattr(conv, "weight"):
            w = conv.weight.data  # [out, 3, k, k]
            new_conv.weight.data = w.mean(dim=1, keepdim=True)
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.data = conv.bias.data.clone()

    setattr(parent, attr, new_conv)


def _find_last_linear(module: nn.Module):
    """
    Walk the module tree and return the last encountered nn.Linear.

    Returns:
        (parent_module, attribute_name, linear_layer) or None.
    """
    last = None
    for _, child in module.named_children():
        res = _find_last_linear(child)
        if res is not None:
            last = res
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            last = (module, name, child)
    return last


def _replace_final_layer_for_arch(model: nn.Module, arch: str, num_classes: int = 2) -> nn.Module:
    """
    Replace the classifier/head so the model outputs `num_classes`.

    We branch on the known torchvision families first and fall back to
    "last Linear in the tree" if no specific rule matches.
    No Softmax is added here.
    """
    a = arch.lower()

    if any(k in a for k in ["resnet", "resnext", "wide_resnet"]):
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
            return model

    if "densenet" in a and hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes, bias=True)
        return model

    if any(k in a for k in ["vgg", "alexnet", "mobilenet", "efficientnet", "convnext"]):
        if hasattr(model, "classifier") and isinstance(model, nn.Sequential):
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes, bias=True)
                    return model

    if "squeezenet" in a and hasattr(model, "classifier"):
        for i in range(len(model.classifier)):
            if isinstance(model.classifier[i], nn.Conv2d):
                in_ch = model.classifier[i].in_channels
                k = model.classifier[i].kernel_size
                model.classifier[i] = nn.Conv2d(in_ch, num_classes, kernel_size=k)
                return model

    if "vit" in a:
        heads = getattr(model, "heads", None)
        if heads is not None and hasattr(heads, "head") and isinstance(heads.head, nn.Linear):
            heads.head = nn.Linear(heads.head.in_features, num_classes, bias=True)
            return model
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes, bias=True)
            return model

    if "googlenet" in a:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        for aux_name in ("aux1", "aux2"):
            aux = getattr(model, aux_name, None)
            if aux is not None and hasattr(aux, "fc") and isinstance(aux.fc, nn.Linear):
                aux.fc = nn.Linear(aux.fc.in_features, num_classes, bias=True)
        return model

    if "inception" in a:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        aux = getattr(model, "AuxLogits", None)
        if aux is not None and hasattr(aux, "fc") and isinstance(aux.fc, nn.Linear):
            aux.fc = nn.Linear(aux.fc.in_features, num_classes, bias=True)
        return model

    last = _find_last_linear(model)
    if last is not None:
        parent, attr, lin = last
        setattr(parent, attr, nn.Linear(lin.in_features, num_classes, bias=True))
        return model

    raise ConfigError(f"Could not locate a final classifier layer for arch '{arch}'.")


def _instantiate_torchvision(arch: str, pretrained: bool) -> nn.Module:
    """
    Build a torchvision model by name, with or without pretrained weights.

    The name is normalized to match known torchvision entry points.
    Some architectures (e.g. GoogLeNet, Inception) require extra kwargs.
    """
    name = arch.strip().lower()
    name_map = {
        "alexnet": "alexnet",
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "vgg16": "vgg16",
        "vgg19": "vgg19",
        "densenet121": "densenet121",
        "mobilenet_v2": "mobilenet_v2",
        "efficientnet_b0": "efficientnet_b0",
        "squeezenet1_0": "squeezenet1_0",
        "vit_b_16": "vit_b_16",
        "googlenet": "googlenet",
        "inception_v3": "inception_v3",
        # ConvNeXt family
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
    }
    key = name.replace("-", "_").replace(" ", "_")
    fn_name = name_map.get(key)
    if fn_name is None or not hasattr(tvm, fn_name):
        raise ConfigError(f"Unsupported torchvision model: '{arch}'. Update name_map in model_loader.py.")
    fn = getattr(tvm, fn_name)

    extra_kwargs = {}
    if key in ("googlenet", "inception_v3"):
        extra_kwargs["aux_logits"] = True

    try:
        if pretrained:
            return fn(weights="DEFAULT", **extra_kwargs)
        else:
            return fn(weights=None, **extra_kwargs)
    except TypeError:
        # Fallback for older torchvision versions
        return fn(pretrained=bool(pretrained), **extra_kwargs)


# ---------------------------- main class ----------------------------

class ModelLoader:
    """
    Build and register models under `outputs/<EXP>/models/loaded/` for one experiment.

    - Torchvision benchmarks → pickled `nn.Module` with 2 outputs.
    - User TorchScript models → re-saved, unchanged, with the `_usermodel_jit.pt`
      suffix, and recorded in the manifest.
    """

    def __init__(self, exp: CreateExperiment) -> None:
        if exp.experiment_dict is None:
            raise RuntimeError("CreateExperiment has no experiment_dict. Call exp.build() first.")
        self.exp = exp
        self.cfg = exp.cfg

    # ---- Benchmarks --------------------------------------------------

    def prepare_benchmarks(
        self,
        *,
        add_softmax: bool = False,             # kept for backward signature; not used
        input_channels: Optional[int] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Create torchvision benchmark models according to `cfg.type_train`
        ('scratch' | 'pretrain' | 'both') and store them in models/loaded/.

        Returns:
            {
                "<model_name>": {
                    "scratch": "<repo_rel_path>.pt",
                    "pretrain": "<repo_rel_path>.pt",
                },
                ...
            }
        """
        results: Dict[str, Dict[str, str]] = {}
        type_train = (self.cfg.type_train or "both").strip().lower()
        want_scratch = type_train in ("scratch", "both")
        want_pretrain = type_train in ("pretrain", "both")

        in_ch = input_channels if input_channels is not None else getattr(self.cfg, "input_channels", 3)

        for model_name in (self.cfg.models_list or []):
            saved: Dict[str, str] = {}
            name_lower = str(model_name).lower()

            if want_scratch:
                m = _instantiate_torchvision(model_name, pretrained=False)
                _adapt_first_conv_in_channels(m, in_ch)
                if in_ch == 1 and ("googlenet" in name_lower or "inception" in name_lower):
                    if hasattr(m, "transform_input"):
                        m.transform_input = False
                    if "inception" in name_lower:
                        if hasattr(m, "AuxLogits"):
                            m.AuxLogits = None
                        if hasattr(m, "aux_logits"):
                            m.aux_logits = False
                m = _replace_final_layer_for_arch(m, model_name, num_classes=2)
                fname = f"{_safe_filename(model_name)}_scratch.pt"
                fpath = self.exp.loaded_models / fname
                torch.save(m, str(fpath))  # pickled module
                saved["scratch"] = self._repo_rel(fpath)

            if want_pretrain:
                m = _instantiate_torchvision(model_name, pretrained=True)
                _adapt_first_conv_in_channels(m, in_ch)
                if in_ch == 1 and ("googlenet" in name_lower or "inception" in name_lower):
                    if hasattr(m, "transform_input"):
                        m.transform_input = False
                    if "inception" in name_lower:
                        if hasattr(m, "AuxLogits"):
                            m.AuxLogits = None
                        if hasattr(m, "aux_logits"):
                            m.aux_logits = False
                m = _replace_final_layer_for_arch(m, model_name, num_classes=2)
                fname = f"{_safe_filename(model_name)}_pretrain.pt"
                fpath = self.exp.loaded_models / fname
                torch.save(m, str(fpath))  # pickled module
                saved["pretrain"] = self._repo_rel(fpath)

            results[str(model_name)] = saved
            self._update_manifest_for_benchmark(model_name, saved)

        self.exp.update_manifest()
        return results

    # ---- User models (TorchScript ONLY) ------------------------------

    def prepare_user_models(
        self,
        *,
        add_softmax: bool = False,             # ignored for JIT, kept for symmetry
        input_channels: Optional[int] = None,  # ignored for JIT
        weights_only: bool = True,             # ignored for JIT
    ) -> Dict[str, str]:
        """
        Find TorchScript archives in `cfg.models_path`, load them and re-save them
        into the experiment's `models/loaded/` folder, adding the `_usermodel_jit.pt`
        suffix. Manifest is updated so the trainer can discover these models.

        Returns:
            { original_filename: repo_relative_saved_path }
        """
        models_dir = Path(self.cfg.models_path)
        if not models_dir.exists():
            raise ConfigError(f"models_path does not exist: {models_dir}")

        paths = [p for p in models_dir.iterdir() if p.suffix.lower() in (".pt", ".pth") and p.is_file()]
        results: Dict[str, str] = {}

        for p in paths:
            try:
                jit_model = torch.jit.load(str(p), map_location="cpu")
            except Exception:
                # Not a valid TorchScript archive → skip silently
                continue

            dst = self.exp.loaded_models / f"{_safe_filename(p.stem)}_usermodel_jit.pt"
            jit_model.save(str(dst))
            results[p.name] = self._repo_rel(dst)
            self._update_manifest_for_user_model_jit(p.name, self._repo_rel(dst))

        self.exp.update_manifest()
        return results

    # ---------------- manifest updates ----------------

    def _update_manifest_for_benchmark(self, model_name: str, saved_variants: Dict[str, str]) -> None:
        """
        Insert/overwrite the model entry with its loaded benchmark variants.
        """
        ex = self.exp.experiment_dict
        if ex is None:
            return
        models = ex.setdefault("models", {})
        entry = models.setdefault(str(model_name), {
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
        })
        entry.pop("loaded_path", None)
        entry["loaded_variants"] = dict(saved_variants)

    def _update_manifest_for_user_model_jit(self, original_name: str, repo_rel_path: str) -> None:
        """
        Register a user TorchScript model under a synthetic key
        'usermodel_<original_name>' with a single 'usermodel_jit' variant.
        """
        ex = self.exp.experiment_dict
        if ex is None:
            return
        models = ex.setdefault("models", {})
        key = f"usermodel_{original_name}"
        entry = models.get(key)
        if entry is None:
            entry = {
                "trained_path": None,
                "train_parameters": {
                    "epochs": self.cfg.epochs,
                    "learning_rate": self.cfg.learning_rate,
                    "batch_size": self.cfg.batch_size,
                    "optimizer": self.cfg.optimizer,
                    "patience": self.cfg.patience,
                    "device": self.cfg.device,
                    "seed": self.cfg.seed,
                    "type_train": "usermodel_jit",
                    "num_workers": self.cfg.num_workers,
                    "transform": None,
                },
            }
            models[key] = entry
        entry.pop("loaded_path", None)
        lv = entry.get("loaded_variants", {})
        lv["usermodel_jit"] = repo_rel_path
        entry["loaded_variants"] = lv

    # ---------------- misc ----------------

    def _repo_rel(self, p: Path) -> str:
        """
        Convert an absolute path inside the repo to a POSIX-style repo-relative path.
        """
        rel = os.path.relpath(str(Path(p).resolve()), str(self.exp.repo_root.resolve()))
        return rel.replace(os.sep, "/")

