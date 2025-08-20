"""
model_loader.py — Load/prepare models for fakevoicefinder.

User models policy (simplified)
-------------------------------
- ONLY TorchScript is supported for user models.
- Place TorchScript archive(s) (created via `torch.jit.save`) under `../models/`.
- We load them with `torch.jit.load` and re-save them (unchanged) into:
    outputs/<EXP>/models/loaded/<basename>_usermodel_jit.pt
- The manifest (experiment.json) records ONLY `loaded_variants["usermodel_jit"]`.

Benchmark models
----------------
- Torchvision architectures: 'scratch' / 'pretrain'.
- We adapt input channels (if needed) and replace the final classifier with 2 outputs.
- We **DO NOT** append Softmax here; the trainer will append Softmax when saving
  the best-trained checkpoint for inference.
- Saved as PICKLED `nn.Module` (extension `.pt`) into outputs/<EXP>/models/loaded/.

Security note
-------------
Loading pickled nn.Module requires trusting the file on load (executes Python).
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
    """Keep alnum, dash, underscore; replace others with '_'."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))


def _find_first_conv(module: nn.Module) -> Optional[Tuple[nn.Module, str, nn.Conv2d]]:
    """Heuristically locate the first Conv2d in common torchvision classifiers."""
    if hasattr(module, "conv1") and isinstance(module.conv1, nn.Conv2d):
        return module, "conv1", module.conv1
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        if name == "features" and isinstance(child, nn.Sequential):
            for i, layer in enumerate(child):
                if isinstance(layer, nn.Conv2d):
                    return child, str(i), layer
    for _, child in module.named_children():
        res = _find_first_conv(child)
        if res is not None:
            return res
    return None


def _adapt_first_conv_in_channels(model: nn.Module, target_in: int) -> None:
    """Replace first Conv2d to match `target_in` channels. If shrinking 3->1, average RGB weights."""
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
    """Locate the last nn.Linear leaf in a module tree; return (parent, attr_name, layer) or None."""
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
    Replace the final classifier depending on torchvision architecture,
    falling back to the last Linear if needed. No Softmax is added here.
    """
    a = arch.lower()

    if any(k in a for k in ["resnet", "resnext", "wide_resnet"]):
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
            return model

    if "densenet" in a and hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes, bias=True)
        return model

    if any(k in a for k in ["vgg", "alexnet", "mobilenet", "efficientnet"]):
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
        aux = getattr(model, "AuxLogLogits", None)
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
    """Instantiate a torchvision model by name with/without weights."""
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
        return fn(pretrained=bool(pretrained), **extra_kwargs)


# ---------------------------- main class ----------------------------

class ModelLoader:
    """
    Prepare models according to config and save them under models/loaded/.

    - Benchmarks (torchvision): 'scratch' / 'pretrain' — saved as **pickled nn.Module**
      with `.pt` extension. Final head set to 2 outputs; **no Softmax added here**.
    - User models (TorchScript ONLY): copied (re-saved) unchanged as
      <basename>_usermodel_jit.pt.
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
        add_softmax: bool = False,             # <-- now False by default; ignored (no softmax here)
        input_channels: Optional[int] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Prepare torchvision benchmark models based on cfg.type_train ('scratch'|'pretrain'|'both').

        Returns:
            { model_name: { 'scratch'?: path, 'pretrain'?: path } }
        """
        results: Dict[str, Dict[str, str]] = {}
        type_train = (self.cfg.type_train or "both").strip().lower()
        want_scratch = type_train in ("scratch", "both")
        want_pretrain = type_train in ("pretrain", "both")

        in_ch = input_channels if input_channels is not None else getattr(self.cfg, "input_channels", 3)

        for model_name in (self.cfg.models_list or []):
            saved: Dict[str, str] = {}

            if want_scratch:
                m = _instantiate_torchvision(model_name, pretrained=False)
                _adapt_first_conv_in_channels(m, in_ch)
                m = _replace_final_layer_for_arch(m, model_name, num_classes=2)
                # No Softmax here
                fname = f"{_safe_filename(model_name)}_scratch.pt"
                fpath = self.exp.loaded_models / fname
                torch.save(m, str(fpath))  # PICKLED MODULE, .pt extension
                saved["scratch"] = self._repo_rel(fpath)

            if want_pretrain:
                m = _instantiate_torchvision(model_name, pretrained=True)
                _adapt_first_conv_in_channels(m, in_ch)
                m = _replace_final_layer_for_arch(m, model_name, num_classes=2)
                # No Softmax here
                fname = f"{_safe_filename(model_name)}_pretrain.pt"
                fpath = self.exp.loaded_models / fname
                torch.save(m, str(fpath))  # PICKLED MODULE, .pt extension
                saved["pretrain"] = self._repo_rel(fpath)

            results[str(model_name)] = saved
            self._update_manifest_for_benchmark(model_name, saved)

        self.exp.update_manifest()
        return results

    # ---- User models (TorchScript ONLY) ------------------------------

    def prepare_user_models(
        self,
        *,
        add_softmax: bool = False,             # ignored for JIT (kept for signature symmetry)
        input_channels: Optional[int] = None,  # ignored for JIT
        weights_only: bool = True,             # ignored for JIT
    ) -> Dict[str, str]:
        """
        Discover user TorchScript archives under cfg.models_path, load & re-save them
        unchanged into outputs/<EXP>/models/loaded/ as <basename>_usermodel_jit.pt.

        Returns:
            { original_basename: saved_path_repo_relative }

        Notes:
            - Only .pt/.pth files that can be opened by `torch.jit.load` are accepted.
            - Files that are NOT valid TorchScript archives are skipped.
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
                # Not a TorchScript archive -> skip silently
                continue

            dst = self.exp.loaded_models / f"{_safe_filename(p.stem)}_usermodel_jit.pt"
            jit_model.save(str(dst))
            results[p.name] = self._repo_rel(dst)
            self._update_manifest_for_user_model_jit(p.name, self._repo_rel(dst))

        self.exp.update_manifest()
        return results

    # ---------------- manifest updates ----------------

    def _update_manifest_for_benchmark(self, model_name: str, saved_variants: Dict[str, str]) -> None:
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
        """Return repo-relative POSIX path for consistency with experiment.json."""
        rel = os.path.relpath(str(Path(p).resolve()), str(self.exp.repo_root.resolve()))
        return rel.replace(os.sep, "/")
