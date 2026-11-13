"""Create recordings and jobs. Shared by the upload route, the jobs route and seeding."""

import logging
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..ml.registry import ModelRelease
from ..models import Job, JobStatus, Recording, ReferenceStage
from ..schemas import JobPublic, RecordingPublic
from . import storage
from .edf import EdfHeader, parse_hypnogram, read_header, validate_recording

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


def expires_at(recording: Recording, settings: Settings) -> datetime | None:
    if recording.is_sample or settings.RETENTION_DAYS <= 0:
        return None
    return recording.created_at + timedelta(days=settings.RETENTION_DAYS)


def to_public(recording: Recording, settings: Settings) -> RecordingPublic:
    computed = {"jobs", "expires_at"}
    return RecordingPublic(
        **{name: getattr(recording, name)
           for name in RecordingPublic.model_fields if name not in computed},
        jobs=[JobPublic.model_validate(job) for job in recording.jobs],
        expires_at=expires_at(recording, settings),
    )


def active_job(session: Session, recording_id: uuid.UUID, model_id: str) -> Job | None:
    return session.scalars(
        select(Job).where(Job.recording_id == recording_id, Job.model_id == model_id,
                          Job.status != JobStatus.FAILED)
    ).first()


def enqueue_job(session: Session, recording: Recording, release: ModelRelease,
                settings: Settings) -> Job:
    queued = session.scalar(
        select(func.count()).select_from(Job).where(Job.status == JobStatus.QUEUED))
    if queued >= settings.MAX_QUEUED_JOBS:
        raise QueueFull
    job = Job(recording_id=recording.id, model_id=release.id)
    session.add(job)
    session.flush()
    return job


def attach_hypnogram(session: Session, recording: Recording, temporary: Path) -> None:
    """Parse first so an invalid file never leaves partial reference rows behind."""
    stages = parse_hypnogram(temporary, recording.n_epochs)
    session.add_all(
        ReferenceStage(recording_id=recording.id, epoch_index=index, stage=stage,
                       in_benchmark=in_benchmark)
        for index, (stage, in_benchmark) in stages.items()
    )
    storage.move_into_place(temporary, storage.hypnogram_path(recording.id))
    recording.has_hypnogram = True


def create_recording(session: Session, *, temporary: Path, header: EdfHeader, filename: str,
                     sha256: str, size_bytes: int, is_sample: bool = False) -> Recording:
    recording = Recording(
        id=uuid.uuid4(), filename=filename, sha256=sha256, size_bytes=size_bytes,
        duration_seconds=header.duration_seconds, n_epochs=header.n_epochs,
        dropped_seconds=header.dropped_seconds, channels=header.channels(),
        is_sample=is_sample,
    )
    session.add(recording)
    session.flush()
    storage.move_into_place(temporary, storage.source_path(recording.id))
    return recording


def seed_samples(session: Session, settings: Settings) -> None:
    """Register the bundled excerpt so the app is usable without owning an EDF file."""
    for psg in sorted(settings.SAMPLES_DIR.glob("*-PSG.edf")):
        with psg.open("rb") as stream:
            temporary, digest, size = storage.save_stream(stream, settings.MAX_UPLOAD_BYTES)
        if session.scalar(select(Recording.id).where(Recording.sha256 == digest)):
            temporary.unlink()
            continue
        header = read_header(temporary)
        validate_recording(header)
        recording = create_recording(session, temporary=temporary, header=header,
                                     filename=psg.name, sha256=digest, size_bytes=size,
                                     is_sample=True)
        hypnogram = psg.with_name(psg.name.replace("-PSG.edf", "-Hypnogram.edf"))
        if hypnogram.is_file():
            copy = storage.temp_dir() / f"{uuid.uuid4()}.edf"
            shutil.copyfile(hypnogram, copy)
            attach_hypnogram(session, recording, copy)
        session.commit()
        logger.info("Seeded sample recording %s", psg.name)
