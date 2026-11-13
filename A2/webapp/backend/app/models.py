"""SQLAlchemy tables. ML architectures live in ``app/ml/networks``, not here."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .core.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    duration_seconds: Mapped[float] = mapped_column(Float)
    n_epochs: Mapped[int] = mapped_column(Integer)
    dropped_seconds: Mapped[float] = mapped_column(Float)
    channels: Mapped[list] = mapped_column(JSON)
    has_hypnogram: Mapped[bool] = mapped_column(Boolean, default=False)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    waveform_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list["Job"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan", order_by="Job.created_at")
    reference_stages: Mapped[list["ReferenceStage"]] = relationship(
        cascade="all, delete-orphan", order_by="ReferenceStage.epoch_index")


class ReferenceStage(Base):
    """One scored epoch of an uploaded hypnogram. Unscored and movement epochs have no row."""

    __tablename__ = "reference_stages"

    recording_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("recordings.id", ondelete="CASCADE"), primary_key=True)
    epoch_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    stage: Mapped[int] = mapped_column(SmallInteger)
    # True inside the A2 benchmark window: 30 minutes either side of scored sleep.
    in_benchmark: Mapped[bool] = mapped_column(Boolean)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        # One live or finished job per recording and model; a failed job may be retried.
        Index("uq_jobs_recording_model_not_failed", "recording_id", "model_id", unique=True,
              postgresql_where=text("status <> 'failed'")),
        Index("ix_jobs_status_created_at", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    recording_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("recordings.id", ondelete="CASCADE"))
    model_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED)
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    recording: Mapped[Recording] = relationship(back_populates="jobs")
    predictions: Mapped[list["Prediction"]] = relationship(
        cascade="all, delete-orphan", order_by="Prediction.epoch_index")


class Prediction(Base):
    __tablename__ = "predictions"

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    epoch_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    stage: Mapped[int] = mapped_column(SmallInteger)
    p_w: Mapped[float] = mapped_column(Float)
    p_n1: Mapped[float] = mapped_column(Float)
    p_n2: Mapped[float] = mapped_column(Float)
    p_n3: Mapped[float] = mapped_column(Float)
    p_rem: Mapped[float] = mapped_column(Float)

    @property
    def probabilities(self) -> list[float]:
        return [self.p_w, self.p_n1, self.p_n2, self.p_n3, self.p_rem]
