# prepare_dataset.py
"""
PrepareDataset (zip-only, manifest-aware pipeline)

Purpose
-------
This helper takes the two audio archives declared in the experiment config
(`real.zip` and `fake.zip`), builds the train/test split, materializes the
original audio files inside the current experiment, and generates the feature
tensors (MEL, LOG, DWT, CQT) that the training code consumes.

What it reads
-------------
- Two ZIP files located at:  <cfg.data_path>/<cfg.real_zip>  and
                             <cfg.data_path>/<cfg.fake_zip>
  These are expected to be audio-only ZIPs.

What it writes
--------------
- Originals (preserving filenames) into:
    outputs/<EXP>/datasets/{train|test}/original/{real|fake}/<filename>.<ext>
- Transformed features (saved as .npy) into:
    outputs/<EXP>/datasets/<split>/transforms/<transform>/{real|fake}/<basename>.npy
  DWT outputs are always resized to a square image (cfg.image_size or 224) to
  stay compatible with CNN / ViT style backbones.

Key traits
----------
- Works strictly from ZIPs (no loose folders).
- Keeps the experiment layout created by CreateExperiment.
- All paths stored in the manifest remain repo-relative.

Dependencies
------------
    pip install numpy librosa soundfile scikit-learn PyWavelets
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import zipfile
import numpy as np

try:
    import librosa
except Exception as e:
    raise RuntimeError(
        "librosa is required for audio I/O and transforms. Install: pip install librosa soundfile"
    ) from e

try:
    import pywt  # DWT
except Exception as e:
    raise RuntimeError(
        "PyWavelets (pywt) is required for the DWT transform. Install: pip install PyWavelets"
    ) from e

from .experiment import CreateExperiment
from .validatorsforvoice import ConfigError


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Types
ZipMember = Tuple[Path, str]            # (zip_path, member_name)
LabeledMember = Tuple[ZipMember, int]   # ((zip_path, member), label_int) — 0=real, 1=fake


class PrepareDataset:
    """
    ZIP-based dataset builder that fills an already-created experiment structure.

    Inputs (from cfg):
      - cfg.data_path : root folder where real/fake ZIPs live
      - cfg.real_zip  : name of the ZIP that contains real audios
      - cfg.fake_zip  : name of the ZIP that contains fake audios
      - optional per-transform params: cfg.mel_params / cfg.log_params / cfg.dwt_params / cfg.cqt_params
      - optional window/time control: cfg.clip_seconds
      - optional spatial size for spectrogram-like features: cfg.image_size

    Outputs (written under this experiment):
      - /datasets/train/original/{real,fake}/...
      - /datasets/test/original/{real,fake}/...
      - /datasets/<split>/transforms/<mel|log|dwt|cqt>/{real,fake}/... .npy
      - experiment.json is updated with counters and transform params
    """

    def __init__(self, exp: CreateExperiment) -> None:
        if exp.experiment_dict is None:
            raise RuntimeError("CreateExperiment has no experiment_dict. Call exp.build() first.")

        self.exp = exp
        self.cfg = exp.cfg
        self.repo_root = exp.repo_root

        # Zips expected under cfg.data_path
        data_root = Path(self.cfg.data_path)
        self.real_zip = data_root / self.cfg.real_zip
        self.fake_zip = data_root / self.cfg.fake_zip

        # Pools and splits
        self._real_members: List[ZipMember] = []
        self._fake_members: List[ZipMember] = []
        self.train_items: List[LabeledMember] = []
        self.test_items: List[LabeledMember] = []

        # Transform params (defaults)
        self.sample_rate = 16000  # sr = 16 kHz (standard TTS/V2V)

        self.mel_params = dict(
            n_mels=128,
            n_fft=1024,
            hop_length=256,
            win_length=None,
            fmin=0,
            fmax=None,
        )
        self.log_params = dict(n_fft=1024, hop_length=256, win_length=None)
        self.dwt_params = dict(wavelet="db4", level=4, mode="symmetric")  # DWT → resize to image_size or 224

        # CQT defaults (tabla):
        #   hop_length = 256
        #   n_bins     ≈ 84–120  → usamos 96 por defecto (término medio)
        #   bins_per_octave = 12 o 24 → usamos 24 para mayor detalle en formantes
        #   fmin = C1 (puedes cambiar a C2 vía cfg.cqt_params["fmin"])
        #   scale = True (distribución espectral más estable)
        self.cqt_params = dict(
            hop_length=256,
            fmin=float(librosa.note_to_hz("C1")),
            bins_per_octave=24,
            n_bins=96,
            scale=True,
        )

        # ---- Optional overrides coming from cfg (minimal change) ----
        user_mel = getattr(self.cfg, "mel_params", None)
        if isinstance(user_mel, dict) and user_mel:
            self.mel_params.update(user_mel)

        user_log = getattr(self.cfg, "log_params", None)
        if isinstance(user_log, dict) and user_log:
            self.log_params.update(user_log)

        user_dwt = getattr(self.cfg, "dwt_params", None)
        if isinstance(user_dwt, dict) and user_dwt:
            self.dwt_params.update(user_dwt)

        user_cqt = getattr(self.cfg, "cqt_params", None)
        if isinstance(user_cqt, dict) and user_cqt:
            self.cqt_params.update(user_cqt)

        # Clip/pad window in seconds. Default is 3.0 if not specified.
        clip_val = getattr(self.cfg, "clip_seconds", None)
        try:
            cs = float(clip_val) if clip_val is not None else 3.0
        except Exception:
            cs = 3.0
        if not (cs > 0):
            cs = 3.0
        self.clip_seconds = cs

        # Optional image side for MEL/LOG/DWT/CQT (e.g. 224 for ViT).
        # None = DWT→224 and no resize for MEL/LOG/CQT.
        img_sz = getattr(self.cfg, "image_size", None)
        try:
            self.image_size = int(img_sz) if img_sz is not None else None
        except Exception:
            self.image_size = None
        if self.image_size is not None and self.image_size <= 0:
            self.image_size = None

    # 1) LOAD ----------------------------------------------------------------

    def load_data(self) -> Dict[str, int]:
        """Scan real.zip and fake.zip; collect their audio members into in-memory lists."""
        if not self.real_zip.is_file():
            raise ConfigError(f"Missing real zip: {self.real_zip}")
        if not self.fake_zip.is_file():
            raise ConfigError(f"Missing fake zip: {self.fake_zip}")

        self._real_members = self._scan_zip(self.real_zip)
        self._fake_members = self._scan_zip(self.fake_zip)
        return {"real": len(self._real_members), "fake": len(self._fake_members)}

    def _scan_zip(self, zip_path: Path) -> List[ZipMember]:
        out: List[ZipMember] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if Path(info.filename).suffix.lower() in AUDIO_EXTS:
                    out.append((zip_path.resolve(), info.filename))
        if not out:
            raise ConfigError(f"No audio files found in {zip_path}")
        return out

    # 2) SPLIT ---------------------------------------------------------------

    def split(self, train_ratio: float = 0.8, seed: int = 23) -> Dict[str, Dict[str, int]]:
        """Stratified split (two classes) on top of the ZIP members using scikit-learn."""
        try:
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for stratified splits. Install: pip install scikit-learn"
            ) from e

        real = list(self._real_members)
        fake = list(self._fake_members)
        if not real or not fake:
            raise ConfigError("Both real.zip and fake.zip must contain at least one audio file.")

        items: List[ZipMember] = real + fake
        labels: List[int] = [0] * len(real) + [1] * len(fake)

        x_train, x_test, y_train, y_test = train_test_split(
            items,
            labels,
            test_size=(1.0 - float(train_ratio)),
            random_state=int(seed),
            stratify=labels,
            shuffle=True,
        )

        self.train_items = list(zip(x_train, y_train))
        self.test_items = list(zip(x_test, y_test))

        return {
            "train": {
                "total": len(self.train_items),
                "real": sum(1 for _, y in self.train_items if y == 0),
                "fake": sum(1 for _, y in self.train_items if y == 1),
            },
            "test": {
                "total": len(self.test_items),
                "real": sum(1 for _, y in self.test_items if y == 0),
                "fake": sum(1 for _, y in self.test_items if y == 1),
            },
        }

    # 3) SAVE ORIGINALS ------------------------------------------------------

    def save_original(self) -> Dict[str, int]:
        """Extract the selected ZIP members to the experiment/original folders, keeping their filenames."""
        out_train = self.exp.train_orig
        out_test = self.exp.test_orig
        (out_train / "real").mkdir(parents=True, exist_ok=True)
        (out_train / "fake").mkdir(parents=True, exist_ok=True)
        (out_test / "real").mkdir(parents=True, exist_ok=True)
        (out_test / "fake").mkdir(parents=True, exist_ok=True)

        n_train = self._extract_members(self.train_items, out_train)
        n_test = self._extract_members(self.test_items, out_test)

        # Update num_items in experiment.json (count audios under originals)
        ex = self.exp.experiment_dict
        if ex:
            ex["train_data"]["original_dataset"]["num_items"] = self._count_audio_files_in(out_train)
            ex["test_data"]["original_dataset"]["num_items"] = self._count_audio_files_in(out_test)
            self.exp.update_manifest()

        return {"train": n_train, "test": n_test}

    def _extract_members(self, items: List[LabeledMember], dst_root: Path) -> int:
        """Extract ZIP members into dst_root/{real|fake}/<filename>, preserving the original ZIP name."""
        n = 0
        for (zip_path, member), label in items:
            cls = "real" if label == 0 else "fake"
            out_path = dst_root / cls / Path(member).name
            if out_path.exists():
                n += 1
                continue
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(member) as src:
                    data = src.read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(data)
            n += 1
        return n

    def _count_audio_files_in(self, root: Path) -> int:
        """Count how many audio files exist below {real,fake} in the given root."""
        cnt = 0
        for sub in ("real", "fake"):
            d = root / sub
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    cnt += 1
        return cnt

    # 4) TRANSFORM -----------------------------------------------------------

    def transform(self, transform_name: str) -> Dict[str, int]:
        """
        Apply one transform ('mel', 'log', 'dwt' or 'cqt') to train/test originals and
        save the result as .npy in the corresponding transforms folder.
        """
        tkey = transform_name.lower()
        if tkey not in {"mel", "log", "dwt", "cqt"}:
            raise ValueError("transform_name must be 'mel', 'log', 'dwt' o 'cqt'.")

        # Ensure destinations
        (self.exp.train_tf_root / tkey / "real").mkdir(parents=True, exist_ok=True)
        (self.exp.train_tf_root / tkey / "fake").mkdir(parents=True, exist_ok=True)
        (self.exp.test_tf_root / tkey / "real").mkdir(parents=True, exist_ok=True)
        (self.exp.test_tf_root / tkey / "fake").mkdir(parents=True, exist_ok=True)

        n_train = self._transform_split(self.exp.train_orig, self.exp.train_tf_root / tkey, tkey)
        n_test = self._transform_split(self.exp.test_orig, self.exp.test_tf_root / tkey, tkey)

        # Persist transform hyperparameters to the manifest
        self.update_experiment_json(tkey)

        return {"train": n_train, "test": n_test}

    def _transform_split(self, orig_root: Path, out_root: Path, tkey: str) -> int:
        """Read audios from orig_root/{real|fake} and write feature arrays to out_root/{real|fake}."""
        n = 0
        for src_cls, dst_cls in (("real", "real"), ("fake", "fake")):
            src_dir = orig_root / src_cls
            if not src_dir.exists():
                continue
            for wav in src_dir.rglob("*"):
                if not (wav.is_file() and wav.suffix.lower() in AUDIO_EXTS):
                    continue
                y, sr = librosa.load(str(wav), sr=self.sample_rate, mono=True)

                # Configurable window (default 3.0 s)
                y = librosa.util.fix_length(y, size=int(sr * self.clip_seconds))

                arr = self._apply_transform(y, sr, tkey)
                out_path = out_root / dst_cls / (wav.stem + ".npy")
                np.save(str(out_path), arr.astype(np.float32))
                n += 1
        return n

    # ---------- Helper: 2D resize (no external deps) ------------------------
    def _resize_2d(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Resize a 2D array (H, W) → (target_h, target_w) by doing 1D interpolation
        on rows first (width) and then on columns (height).
        """
        h, w = img.shape
        # Resize width (row-wise)
        x_old = np.linspace(0.0, 1.0, num=w, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=target_w, dtype=np.float32)
        tmp = np.empty((h, target_w), dtype=np.float32)
        for i in range(h):
            tmp[i, :] = np.interp(x_new, x_old, img[i, :])

        # Resize height (column-wise)
        if h == 1:
            # Edge case: single-row input → replicate to target_h
            out = np.repeat(tmp, repeats=target_h, axis=0)
            return out.astype(np.float32)

        y_old = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
        y_new = np.linspace(0.0, 1.0, num=target_h, dtype=np.float32)
        out = np.empty((target_h, target_w), dtype=np.float32)
        for j in range(target_w):
            out[:, j] = np.interp(y_new, y_old, tmp[:, j])
        return out.astype(np.float32)

    def _apply_transform(self, y: np.ndarray, sr: int, tkey: str) -> np.ndarray:
        if tkey == "mel":
            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.mel_params["n_mels"],
                n_fft=self.mel_params["n_fft"],
                hop_length=self.mel_params["hop_length"],
                win_length=self.mel_params["win_length"],
                fmin=self.mel_params["fmin"],
                fmax=self.mel_params["fmax"],
                power=2.0,
            )
            out = librosa.power_to_db(S, ref=np.max)
            # Optional square resize for ViT / image-like backbones
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
            out = librosa.amplitude_to_db(amp, ref=np.max)
            # Optional square resize for ViT / image-like backbones
            if self.image_size:
                out = self._resize_2d(out, self.image_size, self.image_size)
            return out

        elif tkey == "cqt":
            # Constant-Q transform → magnitude in dB
            fmin = self.cqt_params.get("fmin")
            if fmin is None:
                fmin = float(librosa.note_to_hz("C1"))
            else:
                fmin = float(fmin)

            scale = bool(self.cqt_params.get("scale", True))

            C = librosa.cqt(
                y,
                sr=sr,
                hop_length=int(self.cqt_params["hop_length"]),
                fmin=fmin,
                n_bins=int(self.cqt_params["n_bins"]),
                bins_per_octave=int(self.cqt_params["bins_per_octave"]),
                scale=scale,
            )
            C_mag = np.abs(C)
            out = librosa.amplitude_to_db(C_mag, ref=np.max)
            if self.image_size:
                out = self._resize_2d(out, self.image_size, self.image_size)
            return out

        elif tkey == "dwt":
            # DWT scalogram 2D → rows = [cA_L, cD_L, ..., cD_1] → dB → resize to (image_size or 224)
            wavelet = self.dwt_params.get("wavelet", "db4")
            level = int(self.dwt_params.get("level", 4))
            mode = self.dwt_params.get("mode", "symmetric")

            # Final size: cfg.image_size if provided, otherwise 224
            target_size = int(self.image_size) if (self.image_size is not None) else 224
            TARGET_H, TARGET_W = target_size, target_size

            # Wavelet coefficients: [cA_L, cD_L, ..., cD_1]
            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode=mode)
            coeffs_abs = [np.abs(c).astype(np.float32, copy=False) for c in coeffs]

            # Resample each band to the target width (avoid flat bands due to padding)
            x_new = np.linspace(0.0, 1.0, num=TARGET_W, dtype=np.float32)
            rows: List[np.ndarray] = []
            for c in coeffs_abs:
                if c.size == 0:
                    rows.append(np.zeros(TARGET_W, dtype=np.float32))
                    continue
                x_old = np.linspace(0.0, 1.0, num=c.shape[-1], dtype=np.float32)
                rows.append(np.interp(x_new, x_old, c).astype(np.float32))
            scalogram = np.stack(rows, axis=0)  # (levels+1, TARGET_W)

            # Log/dB compression for a scale similar to MEL/LOG
            scalogram_db = librosa.amplitude_to_db(scalogram, ref=np.max, top_db=80.0)

            # Resize height to the target (width is already TARGET_W)
            scalogram_db = self._resize_2d(scalogram_db, TARGET_H, TARGET_W)

            return scalogram_db

        else:
            raise ValueError(f"Unsupported transform '{tkey}'.")

    # 5) UPDATE MANIFEST -----------------------------------------------------

    def update_experiment_json(self, transform_name: str) -> None:
        """Write back the transform params that were actually used to experiment.json."""
        params = self._params_for_transform(transform_name.lower())
        ex = self.exp.experiment_dict
        if not ex:
            return
        for split_key in ("train_data", "test_data"):
            tf = ex[split_key]["transforms_dataset"]
            key = transform_name.lower()
            if key in tf:
                tf[key]["params"] = params
        self.exp.update_manifest()

    def _params_for_transform(self, tkey: str) -> Dict[str, Any]:
        # Always store sample_rate, clip_seconds and image_size together with the transform params
        base = {
            "sample_rate": self.sample_rate,
            "clip_seconds": self.clip_seconds,
            "image_size": self.image_size,
        }
        if tkey == "mel":
            return {**base, **self.mel_params}
        if tkey == "log":
            return {**base, **self.log_params}
        if tkey == "dwt":
            return {**base, **self.dwt_params}
        if tkey == "cqt":
            return {**base, **self.cqt_params}
        return {}




