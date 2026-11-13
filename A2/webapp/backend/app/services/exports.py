import csv
import io

from ..ml.registry import CLASS_NAMES, ModelRelease
from ..models import Job, Recording

EPOCH_SECONDS = 30


def _rows(job: Job, reference: dict[int, int]):
    for prediction in job.predictions:
        yield prediction, reference.get(prediction.epoch_index)


def to_csv(job: Job, reference: dict[int, int]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["epoch_index", "onset_seconds", "stage",
                     *(f"prob_{name}" for name in CLASS_NAMES), "reference_stage"])
    for prediction, stage in _rows(job, reference):
        writer.writerow([
            prediction.epoch_index, prediction.epoch_index * EPOCH_SECONDS,
            CLASS_NAMES[prediction.stage],
            *(f"{value:.6f}" for value in prediction.probabilities),
            "" if stage is None else CLASS_NAMES[stage],
        ])
    return buffer.getvalue()


def to_json(job: Job, recording: Recording, release: ModelRelease | None,
            reference: dict[int, int]) -> dict:
    return {
        "recording": {
            "id": str(recording.id), "filename": recording.filename, "sha256": recording.sha256,
            "duration_seconds": recording.duration_seconds, "n_epochs": recording.n_epochs,
            "dropped_seconds": recording.dropped_seconds,
        },
        "model": None if release is None else {
            "id": release.id, "network": release.network, "synthetic": release.synthetic,
            "fold_sha256": [fold["sha256"] for fold in release.folds],
            "preprocessing": release.preprocessing,
        },
        "job": {"id": str(job.id), "finished_at": job.finished_at.isoformat()},
        "class_names": list(CLASS_NAMES),
        "epoch_seconds": EPOCH_SECONDS,
        "epochs": [
            {"epoch_index": prediction.epoch_index,
             "onset_seconds": prediction.epoch_index * EPOCH_SECONDS,
             "stage": CLASS_NAMES[prediction.stage],
             "probabilities": [round(value, 6) for value in prediction.probabilities],
             "reference_stage": None if stage is None else CLASS_NAMES[stage]}
            for prediction, stage in _rows(job, reference)
        ],
    }
