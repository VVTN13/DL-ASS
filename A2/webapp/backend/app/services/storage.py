"""Locations of files written at runtime under ``DATA_DIR``."""

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from ..core.config import get_settings

CHUNK = 1 << 20


class UploadTooLarge(Exception):
    pass


def recording_dir(recording_id: uuid.UUID) -> Path:
    return get_settings().DATA_DIR / "recordings" / str(recording_id)


def source_path(recording_id: uuid.UUID) -> Path:
    return recording_dir(recording_id) / "source.edf"


def hypnogram_path(recording_id: uuid.UUID) -> Path:
    return recording_dir(recording_id) / "hypnogram.edf"


def waveform_path(recording_id: uuid.UUID) -> Path:
    return recording_dir(recording_id) / "display_100hz.npy"


def temp_dir() -> Path:
    path = get_settings().DATA_DIR / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_stream(stream: BinaryIO, max_bytes: int) -> tuple[Path, str, int]:
    """Copy an upload to a temporary file; return ``(path, sha256, size)``."""
    path = temp_dir() / f"{uuid.uuid4()}.edf"  # MNE picks its reader by extension
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as target:
            while chunk := stream.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge
                digest.update(chunk)
                target.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, digest.hexdigest(), size


def move_into_place(temporary: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(temporary, target)


def delete_recording_files(recording_id: uuid.UUID) -> None:
    shutil.rmtree(recording_dir(recording_id), ignore_errors=True)
