# prepare_dataset.py
"""
PrepareDataset (zip-only, manifest-driven counters):
- Read TWO zips from cfg.data_path (reals.zip and fakes.zip).
- Stratified split (scikit-learn).
- Extract selected originals into the experiment:
    outputs/<EXP>/datasets/{train|test}/original/{reals|fakes}/<filename>.<ext>
- Apply transforms ('mel', 'log', 'dwt') and save features as .npy under:
    outputs/<EXP>/datasets/<split>/transforms/<transform>/{real|fake}/<basename>.npy
  * DWT se reescala a 224x224 para compatibilidad con CNN/ViT (AlexNet, VGG, ResNet, ViT).
- Update experiment.json:
    - Fill num_items for original_dataset (train/test).
    - Store transform params (light metadata).

Dependencies:
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
LabeledMember = Tuple[ZipMember, int]   # ((zip_path, member), label_int) — 0=reals, 1=fakes


class PrepareDataset:
    """ZIP-based preparer that writes into an existing experiment layout."""

    def __init__(self, exp: CreateExperiment) -> None:
        if exp.experiment_dict is None:
            raise RuntimeError("CreateExperiment has no experiment_dict. Call exp.build() first.")

        self.exp = exp
        self.cfg = exp.cfg
        self.repo_root = exp.repo_root

        # Zips expected under cfg.data_path
        data_root = Path(self.cfg.data_path)
        self.reals_zip = data_root / self.cfg.reals_zip
        self.fakes_zip = data_root / self.cfg.fakes_zip

        # Pools and splits
        self._real_members: List[ZipMember] = []
        self._fake_members: List[ZipMember] = []
        self.train_items: List[LabeledMember] = []
        self.test_items: List[LabeledMember] = []

        # Transform params (tweak as needed)
        self.sample_rate = 16000
        self.mel_params = dict(n_mels=128, n_fft=1024, hop_length=256, win_length=None, fmin=0, fmax=None)
        self.log_params = dict(n_fft=1024, hop_length=256, win_length=None)
        self.dwt_params = dict(wavelet="db4", level=4, mode="symmetric")  # NEW

        # === MINIMAL CHANGE: allow optional overrides from cfg ===
        user_mel = getattr(self.cfg, "mel_params", None)
        if isinstance(user_mel, dict) and user_mel:
            self.mel_params.update(user_mel)

        user_log = getattr(self.cfg, "log_params", None)
        if isinstance(user_log, dict) and user_log:
            self.log_params.update(user_log)

        user_dwt = getattr(self.cfg, "dwt_params", None)
        if isinstance(user_dwt, dict) and user_dwt:
            self.dwt_params.update(user_dwt)
        # === end overrides ===

    # 1) LOAD ----------------------------------------------------------------

    def load_data(self) -> Dict[str, int]:
        """Scan reals.zip and fakes.zip; build member lists."""
        if not self.reals_zip.is_file():
            raise ConfigError(f"Missing reals zip: {self.reals_zip}")
        if not self.fakes_zip.is_file():
            raise ConfigError(f"Missing fakes zip: {self.fakes_zip}")

        self._real_members = self._scan_zip(self.reals_zip)
        self._fake_members = self._scan_zip(self.fakes_zip)
        return {"reals": len(self._real_members), "fakes": len(self._fake_members)}

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
        """Stratified split with scikit-learn on zip members."""
        try:
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for stratified splits. Install: pip install scikit-learn"
            ) from e

        real = list(self._real_members)
        fake = list(self._fake_members)
        if not real or not fake:
            raise ConfigError("Both reals.zip and fakes.zip must contain at least one audio file.")

        items: List[ZipMember] = real + fake
        labels: List[int] = [0] * len(real) + [1] * len(fake)

        x_train, x_test, y_train, y_test = train_test_split(
            items, labels, test_size=(1.0 - float(train_ratio)),
            random_state=int(seed), stratify=labels, shuffle=True
        )

        self.train_items = list(zip(x_train, y_train))
        self.test_items = list(zip(x_test, y_test))

        return {
            "train": {"total": len(self.train_items), "reals": sum(1 for _, y in self.train_items if y == 0),
                      "fakes": sum(1 for _, y in self.train_items if y == 1)},
            "test":  {"total": len(self.test_items),  "reals": sum(1 for _, y in self.test_items  if y == 0),
                      "fakes": sum(1 for _, y in self.test_items  if y == 1)},
        }

    # 3) SAVE ORIGINALS ------------------------------------------------------

    def save_original(self) -> Dict[str, int]:
        """Extract selected members to experiment/original folders, preserving filenames."""
        out_train = self.exp.train_orig
        out_test  = self.exp.test_orig
        (out_train / "reals").mkdir(parents=True, exist_ok=True)
        (out_train / "fakes").mkdir(parents=True, exist_ok=True)
        (out_test  / "reals").mkdir(parents=True, exist_ok=True)
        (out_test  / "fakes").mkdir(parents=True, exist_ok=True)

        n_train = self._extract_members(self.train_items, out_train)
        n_test  = self._extract_members(self.test_items,  out_test)

        # Update num_items in experiment.json (counts audios under originals)
        ex = self.exp.experiment_dict
        if ex:
            ex["train_data"]["original_dataset"]["num_items"] = self._count_audio_files_in(out_train)
            ex["test_data"]["original_dataset"]["num_items"]  = self._count_audio_files_in(out_test)
            self.exp.update_manifest()

        return {"train": n_train, "test": n_test}

    def _extract_members(self, items: List[LabeledMember], dst_root: Path) -> int:
        """Extract zip members into dst_root/{reals|fakes}/<filename> (preserve names)."""
        n = 0
        for (zip_path, member), label in items:
            cls = "reals" if label == 0 else "fakes"
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
        """Count audio files under {reals,fakes} folders."""
        cnt = 0
        for sub in ("reals", "fakes"):
            d = root / sub
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    cnt += 1
        return cnt

    # 4) TRANSFORM -----------------------------------------------------------

    def transform(self, transform_name: str) -> Dict[str, int]:
        """Apply 'mel', 'log' o 'dwt' to originals (train/test) and save .npy under transforms/."""
        tkey = transform_name.lower()
        if tkey not in {"mel", "log", "dwt"}:
            raise ValueError("transform_name must be 'mel', 'log' o 'dwt'.")

        # Ensure destinations
        (self.exp.train_tf_root / tkey / "real").mkdir(parents=True, exist_ok=True)
        (self.exp.train_tf_root / tkey / "fake").mkdir(parents=True, exist_ok=True)
        (self.exp.test_tf_root  / tkey / "real").mkdir(parents=True, exist_ok=True)
        (self.exp.test_tf_root  / tkey / "fake").mkdir(parents=True, exist_ok=True)

        n_train = self._transform_split(self.exp.train_orig, self.exp.train_tf_root / tkey, tkey)
        n_test  = self._transform_split(self.exp.test_orig,  self.exp.test_tf_root  / tkey, tkey)

        # MINIMAL ADD: write transform hyperparameters to experiment.json
        self.update_experiment_json(tkey)

        return {"train": n_train, "test": n_test}

    def _transform_split(self, orig_root: Path, out_root: Path, tkey: str) -> int:
        """Read audios from orig_root/{reals|fakes} and write arrays to out_root/{real|fake}."""
        n = 0
        for src_cls, dst_cls in (("reals", "real"), ("fakes", "fake")):
            src_dir = orig_root / src_cls
            if not src_dir.exists():
                continue
            for wav in src_dir.rglob("*"):
                if not (wav.is_file() and wav.suffix.lower() in AUDIO_EXTS):
                    continue
                y, sr = librosa.load(str(wav), sr=self.sample_rate, mono=True)

                # Duración fija de 3 segundos (recorte/zero-pad)
                y = librosa.util.fix_length(y, size=int(sr * 3.0))

                arr = self._apply_transform(y, sr, tkey)
                out_path = out_root / dst_cls / (wav.stem + ".npy")
                np.save(str(out_path), arr.astype(np.float32))
                n += 1
        return n

    # ---------- Helper: 2D resize (sin dependencias externas) ---------------
    def _resize_2d(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Redimensiona img (H,W) a (target_h, target_w) mediante interpolación 1D
        primero en ancho (por filas) y luego en alto (por columnas).
        """
        h, w = img.shape
        # Redimensionar ancho (por filas)
        x_old = np.linspace(0.0, 1.0, num=w, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=target_w, dtype=np.float32)
        tmp = np.empty((h, target_w), dtype=np.float32)
        for i in range(h):
            tmp[i, :] = np.interp(x_new, x_old, img[i, :])

        # Redimensionar alto (por columnas)
        if h == 1:
            # Caso borde: si solo hay 1 fila, replicar
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
                y=y, sr=sr,
                n_mels=self.mel_params["n_mels"],
                n_fft=self.mel_params["n_fft"],
                hop_length=self.mel_params["hop_length"],
                win_length=self.mel_params["win_length"],
                fmin=self.mel_params["fmin"],
                fmax=self.mel_params["fmax"],
                power=2.0,
            )
            return librosa.power_to_db(S, ref=np.max)

        elif tkey == "log":
            D = librosa.stft(
                y=y,
                n_fft=self.log_params["n_fft"],
                hop_length=self.log_params["hop_length"],
                win_length=self.log_params["win_length"],
            )
            amp = np.abs(D)
            return librosa.amplitude_to_db(amp, ref=np.max)

        elif tkey == "dwt":
            # Escalograma DWT 2D (filas = [cA_L, cD_L, ..., cD_1]) + resize 224x224
            wavelet = self.dwt_params.get("wavelet", "db4")
            level   = int(self.dwt_params.get("level", 4))
            mode    = self.dwt_params.get("mode", "symmetric")

            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode=mode)  # [cA_L, cD_L, ..., cD_1]
            coeffs_abs = [np.abs(c) for c in coeffs]

            # Alinear longitudes: padding con ceros a la derecha
            max_len = max(c.shape[-1] for c in coeffs_abs)
            rows: List[np.ndarray] = []
            for c in coeffs_abs:
                if c.shape[-1] < max_len:
                    c = np.pad(c, (0, max_len - c.shape[-1]), mode="constant", constant_values=0)
                rows.append(c.astype(np.float32))

            scalogram = np.stack(rows, axis=0)  # (niveles+1, ancho)

            # 🔧 Paso clave: redimensionar para compatibilidad CNN/ViT
            TARGET_H, TARGET_W = 224, 224
            scalogram = self._resize_2d(scalogram, TARGET_H, TARGET_W)

            return scalogram

        else:
            raise ValueError(f"Unsupported transform '{tkey}'.")

    # 5) UPDATE MANIFEST -----------------------------------------------------

    def update_experiment_json(self, transform_name: str) -> None:
        """Store transform params in experiment.json and persist it."""
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
        if tkey == "mel":
            return dict(sample_rate=self.sample_rate, **self.mel_params)
        if tkey == "log":
            return dict(sample_rate=self.sample_rate, **self.log_params)
        if tkey == "dwt":
            return dict(sample_rate=self.sample_rate, **self.dwt_params)
        return {}


