"""Inference worker: ``python -m app.worker``.

PostgreSQL is the queue. One row is claimed with ``FOR UPDATE SKIP LOCKED``, so the job
state lives in exactly one place and survives restarts. Run a single worker: at startup it
treats every ``running`` job as interrupted.
"""

import logging
import time
import uuid
from datetime import timedelta

import numpy as np
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session

from .core.config import Settings, get_settings
from .core.db import get_sessionmaker
from .ml.registry import CheckpointError, load_releases
from .models import Job, JobStatus, Prediction, Recording, utcnow
from .services import storage

logger = logging.getLogger("worker")
PROGRESS_INTERVAL_SECONDS = 0.5
MAINTENANCE_INTERVAL_SECONDS = 600
STALE_UPLOAD_SECONDS = 3600


class JobFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def fail_interrupted_jobs(session: Session) -> int:
    """Called once at startup: a job left ``running`` belonged to a worker that died."""
    result = session.execute(
        update(Job).where(Job.status == JobStatus.RUNNING).values(
            status=JobStatus.FAILED, stage=None, finished_at=utcnow(),
            error_code="INTERRUPTED",
            error_message="The worker stopped before this job finished. Run it again.")
    )
    session.commit()
    return result.rowcount


def claim_next_job(session: Session) -> uuid.UUID | None:
    job = session.scalars(
        select(Job).where(Job.status == JobStatus.QUEUED).order_by(Job.created_at)
        .limit(1).with_for_update(skip_locked=True)
    ).first()
    if job is None:
        session.rollback()
        return None
    now = utcnow()
    job.status, job.stage, job.progress = JobStatus.RUNNING, "preprocessing", 0.0
    job.started_at = job.heartbeat_at = now
    session.commit()
    return job.id


class Worker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.sessions = get_sessionmaker()
        self.releases = load_releases(self.settings.CHECKPOINTS_DIR)
        self.ensembles: dict[str, list] = {}

    def report(self, job_id: uuid.UUID, **values) -> None:
        with self.sessions() as session:
            session.execute(update(Job).where(Job.id == job_id)
                            .values(heartbeat_at=utcnow(), **values))
            session.commit()

    def ensemble(self, model_id: str) -> list:
        from .ml.inference import load_ensemble  # Deferred: importing torch is slow.

        release = self.releases.get(model_id)
        if release is None:
            raise JobFailure("MODEL_NOT_FOUND", f"Model {model_id} is not installed.")
        if model_id not in self.ensembles:
            try:
                self.ensembles[model_id] = load_ensemble(release)
            except CheckpointError as error:
                raise JobFailure(error.code, str(error)) from error
        return self.ensembles[model_id]

    def predict(self, job_id: uuid.UUID) -> np.ndarray:
        from .ml.inference import predict_proba
        from .ml.preprocessing import SignalError, preprocess_recording

        with self.sessions() as session:
            job = session.get(Job, job_id)
            recording_id, model_id = job.recording_id, job.model_id
            expected_epochs = job.recording.n_epochs
        models = self.ensemble(model_id)
        try:
            prepared = preprocess_recording(storage.source_path(recording_id))
        except SignalError as error:
            raise JobFailure(error.code, str(error)) from error
        if prepared.n_epochs != expected_epochs:
            raise JobFailure("EPOCH_COUNT_MISMATCH",
                             f"Header promised {expected_epochs} epochs, signal has "
                             f"{prepared.n_epochs}.")

        waveform = storage.waveform_path(recording_id)
        if not waveform.is_file():
            partial = waveform.with_suffix(".partial.npy")
            np.save(partial, prepared.display_waveform())
            partial.replace(waveform)
        self.report(job_id, stage="inference", progress=0.1)
        with self.sessions() as session:
            session.execute(update(Recording).where(Recording.id == recording_id)
                            .values(waveform_ready=True))
            session.commit()

        last_report = time.monotonic()

        def on_progress(done: int, total: int) -> None:
            nonlocal last_report
            if time.monotonic() - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = time.monotonic()
                self.report(job_id, progress=0.1 + 0.85 * done / total)

        return predict_proba(models, prepared.epochs, batch_size=self.settings.BATCH_SIZE,
                             on_progress=on_progress)

    def publish(self, job_id: uuid.UUID, probabilities: np.ndarray) -> None:
        """Predictions and the ``succeeded`` status commit together or not at all."""
        stages = probabilities.argmax(axis=1)
        with self.sessions() as session:
            session.execute(insert(Prediction), [
                {"job_id": job_id, "epoch_index": index, "stage": int(stages[index]),
                 "p_w": float(row[0]), "p_n1": float(row[1]), "p_n2": float(row[2]),
                 "p_n3": float(row[3]), "p_rem": float(row[4])}
                for index, row in enumerate(probabilities)
            ])
            now = utcnow()
            session.execute(update(Job).where(Job.id == job_id).values(
                status=JobStatus.SUCCEEDED, stage=None, progress=1.0,
                finished_at=now, heartbeat_at=now))
            session.commit()

    def run(self, job_id: uuid.UUID) -> None:
        started = time.perf_counter()
        try:
            probabilities = self.predict(job_id)
            self.report(job_id, stage="saving", progress=0.95)
            self.publish(job_id, probabilities)
            logger.info("job %s succeeded in %.1f s", job_id, time.perf_counter() - started)
        except Exception as error:
            code = error.code if isinstance(error, JobFailure) else "INTERNAL_ERROR"
            if not isinstance(error, JobFailure):
                logger.exception("job %s crashed", job_id)
            message = str(error) if isinstance(error, JobFailure) else "Unexpected worker error."
            self.report(job_id, status=JobStatus.FAILED, stage=None, finished_at=utcnow(),
                        error_code=code, error_message=message)
            logger.warning("job %s failed: %s", job_id, code)

    def run_pending(self) -> int:
        """Run queued jobs until the queue is empty; return how many ran. Used by tests."""
        count = 0
        while True:
            with self.sessions() as session:
                job_id = claim_next_job(session)
            if job_id is None:
                return count
            self.run(job_id)
            count += 1

    def purge_expired(self) -> None:
        """Uploads are open to everyone, so they do not live forever."""
        if self.settings.RETENTION_DAYS > 0:
            cutoff = utcnow() - timedelta(days=self.settings.RETENTION_DAYS)
            with self.sessions() as session:
                expired = session.scalars(
                    select(Recording).where(~Recording.is_sample, Recording.created_at < cutoff)
                ).all()
                for recording in expired:
                    session.delete(recording)
                    session.commit()
                    storage.delete_recording_files(recording.id)
                    logger.info("purged expired recording %s", recording.id)
        for leftover in storage.temp_dir().glob("*.edf"):
            if time.time() - leftover.stat().st_mtime > STALE_UPLOAD_SECONDS:
                leftover.unlink(missing_ok=True)

    def serve_forever(self) -> None:
        import torch

        torch.set_num_threads(self.settings.TORCH_THREADS)
        with self.sessions() as session:
            interrupted = fail_interrupted_jobs(session)
        logger.info("worker ready: models=%s, interrupted jobs=%d",
                    sorted(self.releases), interrupted)
        next_maintenance = 0.0
        while True:
            if time.monotonic() >= next_maintenance:
                self.purge_expired()
                next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
            if not self.run_pending():
                time.sleep(self.settings.WORKER_POLL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    Worker().serve_forever()
