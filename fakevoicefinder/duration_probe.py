# fakevoicefinder/duration_probe.py
"""
Utilidad opcional para inspeccionar la duración de audios en los ZIPs
(real.zip y fake.zip) y obtener la duración mínima en segundos.

Uso típico en notebook:
    from fakevoicefinder.duration_probe import shortest_audio_seconds

    # Vía cfg (recomendado)
    min_sec = shortest_audio_seconds(cfg)
    print("Duración mínima (s):", min_sec)

    # O pasando rutas directamente:
    min_sec = shortest_audio_seconds(
        data_path="../dataset",
        real_zip="real.zip",
        fake_zip="fake.zip",
    )

Notas:
- Lee los audios directamente desde los ZIPs.
- Intenta primero leer encabezados con soundfile (rápido).
- Si el formato no está soportado por soundfile (p. ej., MP3), hace fallback
  a librosa escribiendo un temporal.
- No modifica nada del experimento: es solo consulta.
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
    import soundfile as sf  # lectura rápida de encabezados (WAV/FLAC/OGG, etc.)
except Exception as e:
    raise RuntimeError(
        "duration_probe requiere 'soundfile'. Instala: pip install soundfile"
    ) from e

try:
    import librosa  # fallback para formatos no soportados por soundfile (p. ej., MP3/M4A)
except Exception as e:
    raise RuntimeError(
        "duration_probe requiere 'librosa'. Instala: pip install librosa"
    ) from e


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _iter_audio_members(zip_path: Path) -> Iterable[str]:
    """Devuelve nombres de miembros de audio dentro del zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if Path(info.filename).suffix.lower() in AUDIO_EXTS:
                yield info.filename


def _duration_via_soundfile(bytes_data: bytes) -> Optional[float]:
    """Duración (s) leyendo encabezado con soundfile. None si formato no soportado."""
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
    """Duración (s) con librosa mediante archivo temporal (para MP3/M4A, etc.)."""
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
    """Obtiene duración (s) de un miembro de audio dentro del zip."""
    suffix = Path(member_name).suffix.lower()
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as fp:
            data = fp.read()

    # 1) Intento rápido con soundfile
    dur = _duration_via_soundfile(data)
    if dur is not None:
        return dur

    # 2) Fallback con librosa (temporal) para formatos no soportados
    return _duration_via_librosa_tmp(data, suffix=suffix)


def shortest_audio_seconds(
    cfg: Optional[object] = None,
    *,
    data_path: Optional[str | Path] = None,
    real_zip: Optional[str | Path] = None,
    fake_zip: Optional[str | Path] = None,
) -> float:
    """
    Retorna la duración (en segundos) del audio más corto encontrado en los ZIPs.

    Puedes pasar:
      - cfg con atributos: cfg.data_path, cfg.real_zip, cfg.fake_zip
      - o bien data_path/real_zip/fake_zip directamente (kwargs)

    Raises:
      FileNotFoundError si no encuentra los zips.
      ValueError si no hay audios válidos o no se pudo calcular ninguna duración.
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

    # Escanear ambos zips
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
