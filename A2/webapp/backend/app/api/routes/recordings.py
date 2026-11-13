import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from ...models import Recording, ReferenceStage
from ...schemas import EpochPublic, RecordingPublic, RecordingsPublic
from ...services import recordings as service
from ...services import storage
from ...services.edf import CHANNELS, EPOCH_SECONDS, EdfError, read_header, validate_recording
from ...services.recordings import QueueFull, to_public
from ..deps import ApiError, ReleasesDep, SessionDep, SettingsDep

router = APIRouter(prefix="/recordings", tags=["recordings"])

DISPLAY_SAMPLING_RATE = 100


def get_recording(session: SessionDep, recording_id: uuid.UUID) -> Recording:
    recording = session.get(Recording, recording_id)
    if recording is None:
        raise ApiError(404, "RECORDING_NOT_FOUND", "Recording not found.")
    return recording


@router.get("", response_model=RecordingsPublic)
def list_recordings(session: SessionDep, settings: SettingsDep):
    """Every recording, samples first, then newest uploads."""
    recordings = session.scalars(
        select(Recording).order_by(Recording.is_sample.desc(), Recording.created_at.desc())
    ).all()
    return RecordingsPublic(data=[to_public(item, settings) for item in recordings],
                            count=len(recordings))


@router.post("", response_model=RecordingPublic, status_code=status.HTTP_201_CREATED,
             responses={200: {"description": "This file was uploaded before; it is reused."}})
def create_recording(
    session: SessionDep, settings: SettingsDep, releases: ReleasesDep, response: Response,
    edf: Annotated[UploadFile, File(description="Sleep-EDF compatible PSG recording")],
    hypnogram: Annotated[UploadFile | None, File(description="Optional EDF+ hypnogram")] = None,
    model_ids: Annotated[list[str] | None, Form(description="Models to run right away")] = None,
):
    """Upload a recording, optionally with its hypnogram, and queue the selected models.

    Uploading the same file again returns the existing recording and only queues models
    that have not been run on it yet.
    """
    model_ids = list(dict.fromkeys(model_ids or []))
    for model_id in model_ids:
        if model_id not in releases:
            raise ApiError(404, "MODEL_NOT_FOUND", f"Unknown model {model_id!r}.")
        if not releases[model_id].available:
            raise ApiError(503, "MODEL_UNAVAILABLE",
                           f"Checkpoints for {model_id} are not installed on this server.")

    try:
        temporary, digest, size = storage.save_stream(edf.file, settings.MAX_UPLOAD_BYTES)
    except storage.UploadTooLarge:
        limit = settings.MAX_UPLOAD_BYTES >> 20
        raise ApiError(413, "UPLOAD_TOO_LARGE",
                       f"Recordings are limited to {limit} MiB.") from None
    hypnogram_temporary = None
    created = None  # Set only when this request stores a new recording.
    try:
        header = read_header(temporary)
        validate_recording(header)
        recording = session.scalars(select(Recording).where(Recording.sha256 == digest)).first()
        if recording is not None:
            response.status_code = status.HTTP_200_OK
            temporary.unlink()
        else:
            uploads = session.scalar(
                select(func.count()).select_from(Recording).where(~Recording.is_sample))
            if uploads >= settings.MAX_RECORDINGS:
                raise ApiError(507, "STORAGE_FULL",
                               "The server has reached its recording limit. Try again later.")
            recording = created = service.create_recording(
                session, temporary=temporary, header=header,
                filename=(edf.filename or "recording.edf")[-255:], sha256=digest, size_bytes=size)

        if hypnogram is not None and hypnogram.filename and not recording.has_hypnogram:
            try:
                hypnogram_temporary, _, _ = storage.save_stream(
                    hypnogram.file, settings.MAX_HYPNOGRAM_BYTES)
            except storage.UploadTooLarge:
                raise ApiError(413, "UPLOAD_TOO_LARGE",
                               "The hypnogram file is too large.") from None
            service.attach_hypnogram(session, recording, hypnogram_temporary)

        for model_id in model_ids:
            if service.active_job(session, recording.id, model_id) is None:
                service.enqueue_job(session, recording, releases[model_id], settings)
        session.commit()
    except Exception as error:
        session.rollback()
        temporary.unlink(missing_ok=True)
        if hypnogram_temporary is not None:
            hypnogram_temporary.unlink(missing_ok=True)
        if created is not None:
            storage.delete_recording_files(created.id)
        if isinstance(error, EdfError):
            raise ApiError(422, error.code, str(error)) from error
        if isinstance(error, QueueFull):
            raise ApiError(429, "QUEUE_FULL", "Too many jobs are waiting. Try again later.") \
                from error
        if isinstance(error, IntegrityError):
            raise ApiError(409, "UPLOAD_CONFLICT",
                           "The same file is being uploaded right now. Try again.") from error
        raise
    session.refresh(recording)
    return to_public(recording, settings)


@router.get("/{recording_id}", response_model=RecordingPublic)
def read_recording(recording_id: uuid.UUID, session: SessionDep, settings: SettingsDep):
    return to_public(get_recording(session, recording_id), settings)


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording(recording_id: uuid.UUID, session: SessionDep):
    """Delete an uploaded recording with its jobs, predictions and files."""
    recording = get_recording(session, recording_id)
    if recording.is_sample:
        raise ApiError(403, "SAMPLE_READ_ONLY", "Sample recordings cannot be deleted.")
    session.delete(recording)
    session.commit()
    storage.delete_recording_files(recording_id)


@router.get("/{recording_id}/epochs/{epoch_index}", response_model=EpochPublic)
def read_epoch(recording_id: uuid.UUID, epoch_index: int, session: SessionDep):
    """Thirty seconds of the filtered EEG that the models saw, at 100 Hz in microvolts."""
    recording = get_recording(session, recording_id)
    if not 0 <= epoch_index < recording.n_epochs:
        raise ApiError(404, "EPOCH_NOT_FOUND",
                       f"Epoch index must be between 0 and {recording.n_epochs - 1}.")
    if not recording.waveform_ready:
        raise ApiError(409, "WAVEFORM_NOT_READY",
                       "The waveform is available after the first job has preprocessed it.")
    samples = EPOCH_SECONDS * DISPLAY_SAMPLING_RATE
    waveform = np.load(storage.waveform_path(recording_id), mmap_mode="r")
    window = np.asarray(waveform[:, epoch_index * samples:(epoch_index + 1) * samples])
    reference = session.get(ReferenceStage, (recording_id, epoch_index))
    return EpochPublic(
        recording_id=recording_id, epoch_index=epoch_index,
        onset_seconds=epoch_index * EPOCH_SECONDS, sampling_rate=DISPLAY_SAMPLING_RATE,
        unit="uV",
        channels=[{"label": label, "samples": np.round(row, 2).tolist()}
                  for label, row in zip(CHANNELS, window, strict=True)],
        reference_stage=None if reference is None else reference.stage,
    )
