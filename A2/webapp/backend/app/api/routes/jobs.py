import json
import re
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ...ml.registry import CLASS_NAMES
from ...models import Job, JobStatus, Prediction, Recording, ReferenceStage
from ...schemas import JobCreate, JobPublic, MetricsPublic, PredictionsPublic
from ...services import exports, metrics
from ...services.edf import EPOCH_SECONDS
from ...services.recordings import QueueFull, active_job, enqueue_job
from ..deps import ApiError, ReleasesDep, SessionDep, SettingsDep

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_job(session: SessionDep, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise ApiError(404, "JOB_NOT_FOUND", "Job not found.")
    return job


def get_finished_job(session: SessionDep, job_id: uuid.UUID) -> Job:
    job = get_job(session, job_id)
    if job.status != JobStatus.SUCCEEDED:
        raise ApiError(409, "JOB_NOT_FINISHED", f"The job is {job.status}; results are published "
                       "only after it succeeds.")
    return job


def reference_stages(session: SessionDep, recording_id: uuid.UUID, *,
                     benchmark_only: bool = False) -> dict[int, int]:
    query = select(ReferenceStage.epoch_index, ReferenceStage.stage).where(
        ReferenceStage.recording_id == recording_id)
    if benchmark_only:
        query = query.where(ReferenceStage.in_benchmark)
    return dict(session.execute(query).all())


@router.post("", response_model=JobPublic, status_code=status.HTTP_202_ACCEPTED)
def create_job(body: JobCreate, session: SessionDep, settings: SettingsDep,
               releases: ReleasesDep):
    """Queue one model on one recording. The worker picks it up; poll the job for progress."""
    recording = session.get(Recording, body.recording_id)
    if recording is None:
        raise ApiError(404, "RECORDING_NOT_FOUND", "Recording not found.")
    release = releases.get(body.model_id)
    if release is None:
        raise ApiError(404, "MODEL_NOT_FOUND", f"Unknown model {body.model_id!r}.")
    if not release.available:
        raise ApiError(503, "MODEL_UNAVAILABLE",
                       f"Checkpoints for {release.id} are not installed on this server.")
    existing = active_job(session, recording.id, release.id)
    if existing is not None:
        raise ApiError(409, "JOB_EXISTS", f"This model is already {existing.status} for this "
                       "recording.", job_id=str(existing.id))
    try:
        job = enqueue_job(session, recording, release, settings)
        session.commit()
    except QueueFull:
        raise ApiError(429, "QUEUE_FULL", "Too many jobs are waiting. Try again later.") from None
    except IntegrityError:  # Lost a race against an identical request.
        session.rollback()
        raise ApiError(409, "JOB_EXISTS",
                       "This model is already queued for this recording.") from None
    return job


@router.get("/{job_id}", response_model=JobPublic)
def read_job(job_id: uuid.UUID, session: SessionDep):
    return get_job(session, job_id)


@router.get("/{job_id}/predictions", response_model=PredictionsPublic)
def read_job_predictions(job_id: uuid.UUID, session: SessionDep):
    """All epochs as column arrays: ``stages[i]`` and ``probabilities[i]`` describe epoch ``i``."""
    job = get_finished_job(session, job_id)
    rows = session.execute(
        select(Prediction.stage, Prediction.p_w, Prediction.p_n1, Prediction.p_n2,
               Prediction.p_n3, Prediction.p_rem)
        .where(Prediction.job_id == job_id).order_by(Prediction.epoch_index)
    ).all()
    reference = None
    if job.recording.has_hypnogram:
        scored = reference_stages(session, job.recording_id)
        reference = [scored.get(index) for index in range(len(rows))]
    return PredictionsPublic(
        job_id=job.id, recording_id=job.recording_id, model_id=job.model_id,
        class_names=list(CLASS_NAMES), epoch_seconds=EPOCH_SECONDS,
        stages=[row[0] for row in rows],
        probabilities=[[round(value, 5) for value in row[1:]] for row in rows],
        reference_stages=reference,
    )


@router.get("/{job_id}/metrics", response_model=MetricsPublic)
def read_job_metrics(job_id: uuid.UUID, session: SessionDep,
                     scope: Literal["all", "benchmark"] = "all"):
    """Agreement with the uploaded hypnogram.

    ``all`` scores every epoch the hypnogram labels. ``benchmark`` keeps only epochs within
    30 minutes of scored sleep, the window used for the metrics published in the A2 report.
    """
    job = get_finished_job(session, job_id)
    if not job.recording.has_hypnogram:
        raise ApiError(409, "NO_HYPNOGRAM", "Upload a hypnogram to compute agreement metrics.")
    reference = reference_stages(session, job.recording_id, benchmark_only=scope == "benchmark")
    predicted = dict(session.execute(
        select(Prediction.epoch_index, Prediction.stage).where(Prediction.job_id == job_id)
    ).all())
    epochs = sorted(set(reference) & set(predicted))
    if not epochs:
        raise ApiError(409, "NO_SCORED_EPOCHS", "No scored epochs fall inside this scope.")
    matrix = metrics.confusion_matrix(np.array([reference[i] for i in epochs]),
                                      np.array([predicted[i] for i in epochs]))
    summary = metrics.summarize(matrix)
    per_class = [{"name": name, **values}
                 for name, values in zip(CLASS_NAMES, summary.pop("per_class"), strict=True)]
    return MetricsPublic(job_id=job.id, scope=scope, n_epochs=job.recording.n_epochs,
                         class_names=list(CLASS_NAMES), per_class=per_class, **summary)


@router.get("/{job_id}/export", responses={200: {"content": {"text/csv": {}}}})
def export_job(job_id: uuid.UUID, session: SessionDep, releases: ReleasesDep,
               format: Literal["csv", "json"] = "csv"):
    """Download every epoch's stage and probabilities as CSV, or as JSON with provenance."""
    job = get_finished_job(session, job_id)
    reference = reference_stages(session, job.recording_id)
    # Uploaded names are untrusted: keep the header ASCII and free of quotes or line breaks.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(job.recording.filename).stem).strip("._")
    stem = stem or "recording"
    name = f"{stem[:80]}_{job.model_id}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{name}"'}
    if format == "csv":
        return Response(exports.to_csv(job, reference), media_type="text/csv", headers=headers)
    content = exports.to_json(job, job.recording, releases.get(job.model_id), reference)
    return Response(json.dumps(content, indent=2), media_type="application/json", headers=headers)
