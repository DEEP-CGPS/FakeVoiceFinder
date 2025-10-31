# inference.py
"""
Inference utilities for FakeVoiceFinder.

This module provides a minimal, self-contained runner that:

- Loads a trained checkpoint (either TorchScript or a pickled `nn.Module`).
- Loads a single audio file from disk.
- Applies one of the supported transforms (`"mel"`, `"log"`, or `"dwt"`) using
  parameters that mirror the ones used in `prepare_dataset.py`.
- Runs the model to obtain logits, converts them to probabilities if needed,
  and returns class percentages for `{'real', 'fake'}`.

Inputs:
- A model file path (TorchScript or pickled module).
- An audio file path to classify.
- A transform name and, optionally, transform-specific parameters.
- An optional device override.

Outputs:
- A dictionary with:
    {
        "real": <percent 0–100>,
        "fake": <percent 0–100>,
        "pred_label": "real" | "fake",
        "confidence": <percent 0–100 of predicted class>
    }

Usage:
    from fakevoicefinder.inference import InferenceRunner

    runner = InferenceRunner(
        model_path="outputs/exp1/models/trained/resnet18_pretrain_mel.pt",
        transform="mel",
        transform_params={
            # optional overrides; defaults shown below
            "sample_rate": 16000,
            "clip_seconds": 3.0,
            "image_size": 224,         # only used for resizing MEL/LOG/DWT
            # MEL specific (used if transform == "mel"):
            "n_mels": 128, "n_fft": 1024, "hop_length": 256,
            "win_length": None, "fmin": 0, "fmax": None,
            # LOG specific (used if transform == "log"):
            # "n_fft": 1024, "hop_length": 256, "win_length": None,
            # DWT specific (used if transform == "dwt"):
            # "wavelet": "db4", "level": 4, "mode": "symmetric"
        },
        device=None  # 'cuda'|'cpu'|None (auto). Default: None
    )
    scores = runner.predict("sample.wav")
    # scores -> {'real': 12.34, 'fake': 87.66, 'pred_label': 'fake', 'confidence': 87.66}
"""

from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path
from typing import Iterable, Tuple  # new types for visualization helper
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms

import numpy as np
import torch
import torch.nn as nn

# librosa is required for I/O and MEL/LOG transforms
try:
    import librosa
except Exception as e:
    raise RuntimeError(
        "librosa is required for inference. Install: pip install librosa soundfile"
    ) from e


class InferenceRunner:
    """Single-audio inference runner that applies a selected audio-to-image transform."""

    # ------ defaults aligned with prepare_dataset.py ------
    _DEFAULT_SR = 16000
    _DEFAULT_CLIP_S = 3.0
    _DEFAULT_IMG = None  # For MEL/LOG/DWT; if None, MEL/LOG keep native size; DWT falls back to 224

    _DEFAULT_MEL = dict(n_mels=128, n_fft=1024, hop_length=256, win_length=None, fmin=0, fmax=None)
    _DEFAULT_LOG = dict(n_fft=1024, hop_length=256, win_length=None)
    _DEFAULT_DWT = dict(wavelet="db4", level=4, mode="symmetric")

    def __init__(
        self,
        *,
        model_path: str | Path,
        transform: str,
        transform_params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.transform = str(transform).lower().strip()
        if self.transform not in {"mel", "log", "dwt"}:
            raise ValueError("transform must be one of {'mel','log','dwt'}.")

        self.params = dict(transform_params or {})

        # Global audio controls
        self.sample_rate = int(self.params.pop("sample_rate", self._DEFAULT_SR))
        self.clip_seconds = float(self.params.pop("clip_seconds", self._DEFAULT_CLIP_S))
        img_sz = self.params.pop("image_size", self._DEFAULT_IMG)
        self.image_size = (int(img_sz) if img_sz is not None else None)

        # Per-transform params (merge user overrides only for the selected transform)
        self.mel_params = {**self._DEFAULT_MEL, **({} if self.transform != "mel" else self.params)}
        self.log_params = {**self._DEFAULT_LOG, **({} if self.transform != "log" else self.params)}
        self.dwt_params = {**self._DEFAULT_DWT, **({} if self.transform != "dwt" else self.params)}

        # Device selection (explicit > auto)
        d = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
        self.device = torch.device("cuda" if d in ("cuda", "gpu") and torch.cuda.is_available() else "cpu")

        # Load model (TorchScript or pickled nn.Module)
        self.model = self._load_model(self.model_path).to(self.device).eval()

    # ---------------- public API ----------------

    @torch.no_grad()
    def predict(self, audio_path: str | Path) -> Dict[str, float]:
        """
        Run inference on a single audio file and return class percentages.

        Returns:
            dict with keys:
                - 'real': float, percent in [0, 100]
                - 'fake': float, percent in [0, 100]
                - 'pred_label': str, 'real' or 'fake'
                - 'confidence': float, percent in [0, 100] for the predicted class
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1) Load mono audio at target sample rate
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        # 2) Fix length (clip/pad)
        target_len = int(self.sample_rate * self.clip_seconds)
        y = librosa.util.fix_length(y, size=target_len)

        # 3) Transform into 2D feature (H, W) in dB (for MEL/LOG) or DWT dB
        feat_2d = self._apply_transform(y, sr=self.sample_rate, tkey=self.transform)  # (H, W)

        # 4) To tensor [1, C=1, H, W]
        x = torch.from_numpy(feat_2d.astype(np.float32))[None, None, ...].to(self.device)

        # 5) Forward
        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 1:
            logits = logits[None, :]

        # 6) Ensure probabilities (apply softmax if outputs are not already probs)
        probs = self._ensure_probabilities(logits)  # [B, 2]
        p_real = float(probs[0, 0].item())
        p_fake = float(probs[0, 1].item())

        out = {
            "real": round(p_real * 100.0, 2),
            "fake": round(p_fake * 100.0, 2),
        }
        if p_fake >= p_real:
            out["pred_label"] = "fake"
            out["confidence"] = out["fake"]
        else:
            out["pred_label"] = "real"
            out["confidence"] = out["real"]
        return out

    # ---------------- internals ----------------

    def _load_model(self, path: Path) -> nn.Module:
        """
        Try to load the model as TorchScript first; if that fails, fall back to
        a pickled PyTorch module. This mirrors `metrics.py` behavior
        (`torch.load(..., weights_only=False)`).
        """
        try:
            return torch.jit.load(str(path), map_location=self.device)
        except Exception:
            # Not a TorchScript archive -> try pickled Module
            return torch.load(str(path), map_location=self.device, weights_only=False)

    def _ensure_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Normalize model outputs to probabilities.

        If the given tensor already looks like probabilities (all non-negative
        and rows sum approximately to 1), it is returned as-is. Otherwise
        a softmax is applied along dim=1.
        """
        with torch.no_grad():
            sums = logits.float().sum(dim=1, keepdim=True)
            nonneg = (logits >= 0).all().item()
            approx_one = torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)
            if nonneg and approx_one:
                return logits.float()
            return torch.softmax(logits.float(), dim=1)

    # ----- small helpers copied/adapted from prepare_dataset.py -----

    def _resize_2d(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize a 2D array (H, W) to (target_h, target_w) using 1D interpolation on rows and columns."""
        h, w = img.shape
        x_old = np.linspace(0.0, 1.0, num=w, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=target_w, dtype=np.float32)
        tmp = np.empty((h, target_w), dtype=np.float32)
        for i in range(h):
            tmp[i, :] = np.interp(x_new, x_old, img[i, :])

        if h == 1:
            return np.repeat(tmp, repeats=target_h, axis=0).astype(np.float32)

        y_old = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
        y_new = np.linspace(0.0, 1.0, num=target_h, dtype=np.float32)
        out = np.empty((target_h, target_w), dtype=np.float32)
        for j in range(target_w):
            out[:, j] = np.interp(y_new, y_old, tmp[:, j])
        return out.astype(np.float32)

    def _apply_transform(self, y: np.ndarray, sr: int, tkey: str) -> np.ndarray:
        if tkey == "mel":
            S = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_mels=self.mel_params["n_mels"],
                n_fft=self.mel_params["n_fft"],
                hop_length=self.mel_params["hop_length"],
                win_length=self.mel_params["win_length"],
                fmin=self.mel_params["fmin"],
                fmax=self.mel_params["fmax"],
                power=2.0,
            )
            out = librosa.power_to_db(S, ref=np.max).astype(np.float32)
            if self.image_size:
                out = self._resize_2d(out, self.image_size, self.image_size)
            return out

        elif tkey == "log":
            D = librosa.stft(
                y=y,
                n_fft=self.log_params["n_fft"],
                hop_length=self.log_params["hop_length"],
                win_length=self.log_params["win_length"],
            )
            amp = np.abs(D)
            out = librosa.amplitude_to_db(amp, ref=np.max).astype(np.float32)
            if self.image_size:
                out = self._resize_2d(out, self.image_size, self.image_size)
            return out

        elif tkey == "dwt":
            try:
                import pywt
            except Exception as e:
                raise RuntimeError(
                    "PyWavelets (pywt) is required for DWT. Install: pip install PyWavelets"
                ) from e

            wavelet = self.dwt_params.get("wavelet", "db4")
            level   = int(self.dwt_params.get("level", 4))
            mode    = self.dwt_params.get("mode", "symmetric")

            # Target size for DWT: use image_size if provided, else 224
            target_size = int(self.image_size) if (self.image_size is not None) else 224
            TARGET_H, TARGET_W = target_size, target_size

            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode=mode)  # [cA_L, cD_L, ..., cD_1]
            coeffs_abs = [np.abs(c).astype(np.float32, copy=False) for c in coeffs]

            # Resample each band directly to TARGET_W (avoid zero-padding artifacts)
            x_new = np.linspace(0.0, 1.0, num=TARGET_W, dtype=np.float32)
            rows = []
            for c in coeffs_abs:
                if c.size == 0:
                    rows.append(np.zeros(TARGET_W, dtype=np.float32))
                    continue
                x_old = np.linspace(0.0, 1.0, num=c.shape[-1], dtype=np.float32)
                rows.append(np.interp(x_new, x_old, c).astype(np.float32))
            scalogram = np.stack(rows, axis=0)  # (levels+1, TARGET_W)

            # Convert to dB for stability/scale alignment with MEL/LOG
            scalogram_db = librosa.amplitude_to_db(scalogram, ref=np.max, top_db=80.0).astype(np.float32)

            # Resize height to TARGET_H
            scalogram_db = self._resize_2d(scalogram_db, TARGET_H, TARGET_W)
            return scalogram_db

        else:
            raise ValueError(f"Unsupported transform '{tkey}'.")


# ======================== Visualization helper ========================


class FakeProbabilityGauge:
    """
    Lightweight Matplotlib-based gauge to visualize the predicted probability
    of the 'fake' class, using the dict returned by `InferenceRunner.predict(...)`.

    Features:
    - Color gradient green → yellow → red.
    - Optional 3-band annotations (low/medium/high).
    - Can take a raw percentage or a model scores dict.
    - Saves to file or returns (fig, ax).
    """

    def __init__(
        self,
        *,
        bands: Iterable[Tuple[float, float, str]] = ((0, 50, "low"), (50, 75, "medium"), (75, 100, "high")),
        title: str = "Fake probability",
        xlabel: str = "Probability (%)",
        figsize: Tuple[float, float] = (9, 2.0),
        dpi: int = 160,
        show_threshold: bool = False,
        threshold_value: float = 90.0,
    ) -> None:
        self.bands = tuple((float(lo), float(hi), str(lbl)) for lo, hi, lbl in bands)
        self.title = title
        self.xlabel = xlabel
        self.figsize = figsize
        self.dpi = int(dpi)
        self.show_threshold = bool(show_threshold)
        self.threshold_value = float(threshold_value)

        # Prebuild the green→yellow→red colormap and gradient
        self._cmap = LinearSegmentedColormap.from_list(
            "gyr", [(0.0, "#1a9850"), (0.5, "#ffd166"), (1.0, "#d73027")]
        )
        self._grad = np.linspace(0, 1, 500)[None, :]  # 1x500 gradient image

    # ----- public API -----

    def plot_from_scores(self, scores: Dict, *, prefer_key: str = "fake", save_path: Optional[str] = None):
        """Create the gauge using the score dictionary returned by the inference runner."""
        p = self._get_fake_percent_from_scores(scores, prefer=prefer_key)
        return self.plot(p, save_path=save_path)

    def plot(self, p: float, *, save_path: Optional[str] = None):
        """Draw the gauge for a percentage in [0, 100]."""
        p = float(np.clip(p, 0, 100))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        # Title as suptitle (keeps band labels centered and on their own row)
        fig.suptitle(self.title, y=1.14, fontsize=12, fontweight="bold")

        # Gradient bar
        ax.imshow(self._grad, aspect="auto", extent=(0, 100, 0, 1),
                  cmap=self._cmap, origin="lower", interpolation="bicubic")

        # Band labels row (independent from title)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        band_y = 1.06
        for lo, hi, label in self.bands:
            if hi < 100:
                ax.axvline(hi, 0.18, 0.82, linestyle="--", linewidth=1.2, color="black", alpha=0.35)
            ax.text((lo + hi) / 2, band_y, label, transform=trans, va="bottom", ha="center", fontsize=10)

        # Axes formatting
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel(self.xlabel, labelpad=10)

        # Optional threshold marker
        if self.show_threshold:
            t = float(np.clip(self.threshold_value, 0, 100))
            ax.axvline(t, 0, 1, linewidth=2, color="black", alpha=0.8)

        # Current value marker + label
        ax.axvline(p, 0, 1, linewidth=2.5, color="black")
        ax.plot(p, 0.5, marker="s", markersize=8, color="white",
                markeredgecolor="black", zorder=5)
        tx = p + 2 if p <= 96 else p - 2
        ha = "left" if p <= 96 else "right"
        ax.text(tx, 0.5, f"{p:0.2f}%", va="center", ha=ha, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        # Clean frame & reserve space for title/bands
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_alpha(0.6)
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, ax

    # ----- helpers -----

    @staticmethod
    def _to_percent(x: float) -> float:
        """Normalize a value to the percentage range [0, 100]."""
        x = float(x)
        return float(np.clip(x * 100, 0, 100)) if 0.0 <= x <= 1.0 else float(np.clip(x, 0, 100))

    def _get_fake_percent_from_scores(self, scores: Dict, prefer: str = "fake") -> float:
        """Extract the 'fake' probability from a scores dict, or fall back to the most confident number."""
        if not isinstance(scores, dict):
            raise TypeError("scores must be a dict like {'real':..., 'fake':..., 'confidence':...}")
        if prefer in scores:
            return self._to_percent(scores[prefer])
        if "confidence" in scores:
            return self._to_percent(scores["confidence"])
        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        return self._to_percent(max(numeric)) if numeric else 0.0


