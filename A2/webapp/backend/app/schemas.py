"""Pydantic request and response models."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelPublic(BaseModel):
    id: str
    name: str
    network: str
    input_channels: list[str]
    preprocessing_version: str
    available: bool
    synthetic: bool
    published_metrics: dict | None


class ChannelPublic(BaseModel):
    label: str
    sampling_rate: float
    unit: str


class JobCreate(BaseModel):
    recording_id: uuid.UUID
    model_id: str


class JobPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    recording_id: uuid.UUID
    model_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str | None
    progress: float
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RecordingPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    sha256: str
    size_bytes: int
    duration_seconds: float
    n_epochs: int
    dropped_seconds: float
    channels: list[ChannelPublic]
    has_hypnogram: bool
    is_sample: bool
    waveform_ready: bool
    created_at: datetime
    expires_at: datetime | None
    jobs: list[JobPublic]


class RecordingsPublic(BaseModel):
    data: list[RecordingPublic]
    count: int


class EpochChannel(BaseModel):
    label: str
    samples: list[float]


class EpochPublic(BaseModel):
    recording_id: uuid.UUID
    epoch_index: int
    onset_seconds: float
    sampling_rate: float
    unit: str
    channels: list[EpochChannel]
    reference_stage: int | None


class PredictionsPublic(BaseModel):
    job_id: uuid.UUID
    recording_id: uuid.UUID
    model_id: str
    class_names: list[str]
    epoch_seconds: int
    stages: list[int]
    probabilities: list[list[float]]
    reference_stages: list[int | None] | None


class ClassMetrics(BaseModel):
    name: str
    precision: float | None
    recall: float | None
    f1: float | None
    support: int


class MetricsPublic(BaseModel):
    job_id: uuid.UUID
    scope: Literal["all", "benchmark"]
    n_epochs: int
    n_scored_epochs: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    kappa: float
    class_names: list[str]
    per_class: list[ClassMetrics]
    confusion_matrix: list[list[int]]


class HealthPublic(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    models_available: int
