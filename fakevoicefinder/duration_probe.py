# fakevoicefinder/duration_probe.py
"""
Optional utility to inspect the audio duration inside the ZIP files
(`real.zip` and `fake.zip`) and return the shortest duration in seconds.

Typical usage in a notebook:
    from fakevoicefinder.duration_probe import shortest_audio_seconds

    # Using an experiment config (recommended):
    min_sec = shortest_audio_seconds(cfg)
    print("Shortest duration (s):", min_sec)

    # Or passing paths directly:
    min_sec = shortest_audio_seconds(
        data_path="../dataset",
        real_zip="real.zip",
        fake_zip="fake.zip",
    )

Notes:
- Reads audio files directly from the ZIPs, without unpacking the whole dataset.
- Tries to read headers with `soundfile` first (fast for WAV/FLAC/OGG, etc.).
- If the format is not supported by `soundfile` (e.g. MP3/M4A), it falls back
  to `librosa` by writing a temporary file.
- This function never modifies the experiment state; it only inspects durations.
"""
from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import soundfile as sf  # fast header-based duration for WAV/FLAC/OGG, etc.
except Exception as e:
    raise RuntimeError(
        "duration_probe requiere 'soundfile'. Instala: pip install soundfile"
    ) from e

try:
    import librosa  # fallback for formats not supported by soundfile (e.g. MP3/M4A)
except Exception as e:
    raise RuntimeError(
        "duration_probe requiere 'librosa'. Instala: pip install librosa"
    ) from e


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _iter_audio_members(zip_path: Path) -> Iterable[str]:
    """Yield audio member names found inside the given ZIP file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if Path(info.filename).suffix.lower() in AUDIO_EXTS:
                yield info.filename


def _duration_via_soundfile(bytes_data: bytes) -> Optional[float]:
    """Return duration in seconds using soundfile, or None if unsupported."""
    bio = io.BytesIO(bytes_data)
    try:
        with sf.SoundFile(bio) as f:
            frames = len(f)
            sr = f.samplerate or 0
            if frames > 0 and sr > 0:
                return float(frames) / float(sr)
            return None
    except Exception:
        return None


def _duration_via_librosa_tmp(bytes_data: bytes, suffix: str) -> Optional[float]:
    """Return duration in seconds using librosa and a temporary file."""
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(bytes_data)
        tmp.flush()
        tmp.close()
        return float(librosa.get_duration(path=tmp.name))
    except Exception:
        return None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


def _duration_from_zip_member(zip_path: Path, member_name: str) -> Optional[float]:
    """Compute the duration (in seconds) of a single audio file inside a ZIP."""
    suffix = Path(member_name).suffix.lower()
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as fp:
            data = fp.read()

    # 1) Try fast path with soundfile.
    dur = _duration_via_soundfile(data)
    if dur is not None:
        return dur

    # 2) Fallback with librosa for formats not supported by soundfile.
    return _duration_via_librosa_tmp(data, suffix=suffix)


def shortest_audio_seconds(
    cfg: Optional[object] = None,
    *,
    data_path: Optional[str | Path] = None,
    real_zip: Optional[str | Path] = None,
    fake_zip: Optional[str | Path] = None,
) -> float:
    """
    Return the shortest audio duration (in seconds) found in the ZIP files.

    You can provide:
      - an experiment-like object (`cfg`) with attributes: `data_path`,
        `real_zip`, `fake_zip`; or
      - the explicit paths via keyword arguments.

    Raises:
      FileNotFoundError: if any of the ZIPs cannot be found.
      ValueError: if no valid audio duration could be computed.
    """
    if cfg is not None:
        data_root = Path(getattr(cfg, "data_path"))
        rz = data_root / getattr(cfg, "real_zip")
        fz = data_root / getattr(cfg, "fake_zip")
    else:
        if data_path is None or real_zip is None or fake_zip is None:
            raise ValueError("Provee cfg o (data_path, real_zip, fake_zip).")
        data_root = Path(data_path)
        rz = data_root / real_zip
        fz = data_root / fake_zip

    if not rz.is_file():
        raise FileNotFoundError(f"real zip no encontrado: {rz}")
    if not fz.is_file():
        raise FileNotFoundError(f"fake zip no encontrado: {fz}")

    min_dur = np.inf
    found_any = False

    # Scan both ZIP files.
    for zip_path in (rz, fz):
        for member in _iter_audio_members(zip_path):
            dur = _duration_from_zip_member(zip_path, member)
            if dur is None:
                continue
            found_any = True
            if dur < min_dur:
                min_dur = dur

    if not found_any or not np.isfinite(min_dur):
        raise ValueError("No se pudo calcular duración de ningún audio en los zips.")

    return float(min_dur)
