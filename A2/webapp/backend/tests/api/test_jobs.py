import csv
import io
import re
from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import func, select, update

from app.core.db import get_sessionmaker
from app.models import Job, Prediction, Recording, utcnow
from app.services import storage
from app.worker import Worker, claim_next_job, fail_interrupted_jobs
from tests.conftest import ATTNSLEEP, SAMPLE_PSG, TRANSFORMER, edf_variant, upload


def error(response) -> str:
    return response.json()["detail"]["code"]


def job_of(recording: dict, model_id: str) -> dict:
    return next(job for job in recording["jobs"] if job["model_id"] == model_id)


@pytest.fixture
def finished(client, worker):
    """A recording with a hypnogram and both models run to completion."""
    recording = upload(client, hypnogram=True, model_ids=(TRANSFORMER, ATTNSLEEP)).json()
    assert worker.run_pending() == 2
    return client.get(f"/api/v1/recordings/{recording['id']}").json()


def test_job_runs_from_queued_to_succeeded(client, worker):
    recording = upload(client).json()
    response = client.post("/api/v1/jobs", json={"recording_id": recording["id"],
                                                 "model_id": TRANSFORMER})
    assert response.status_code == 202
    job = response.json()
    assert (job["status"], job["progress"], job["started_at"]) == ("queued", 0.0, None)

    assert worker.run_pending() == 1
    job = client.get(f"/api/v1/jobs/{job['id']}").json()
    assert (job["status"], job["stage"], job["progress"]) == ("succeeded", None, 1.0)
    assert job["error_code"] is None and job["finished_at"] >= job["started_at"]
    assert client.get(f"/api/v1/recordings/{recording['id']}").json()["waveform_ready"] is True


def test_predictions_cover_every_epoch_in_order(client, finished):
    for model_id in (TRANSFORMER, ATTNSLEEP):
        job = job_of(finished, model_id)
        body = client.get(f"/api/v1/jobs/{job['id']}/predictions").json()
        probabilities = np.array(body["probabilities"])
        assert body["class_names"] == ["W", "N1", "N2", "N3", "REM"]
        assert probabilities.shape == (120, 5) and len(body["stages"]) == 120
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
        assert body["stages"] == probabilities.argmax(axis=1).tolist()
        assert len(body["reference_stages"]) == 120
        assert set(body["reference_stages"]) == {0, 1, 2, 3, 4}


def test_served_predictions_equal_the_inference_engine(client, worker, finished):
    from app.ml.inference import predict_proba
    from app.ml.preprocessing import preprocess_recording

    job = job_of(finished, TRANSFORMER)
    expected = predict_proba(worker.ensemble(TRANSFORMER), preprocess_recording(SAMPLE_PSG).epochs)
    served = client.get(f"/api/v1/jobs/{job['id']}/predictions").json()["probabilities"]
    np.testing.assert_allclose(served, expected, atol=1e-5)


def test_reference_labels_never_change_predictions(client, worker):
    unlabeled = upload(client, edf_variant("no labels"), model_ids=(TRANSFORMER,)).json()
    labeled = upload(client, edf_variant("labels"), hypnogram=True,
                     model_ids=(TRANSFORMER,)).json()
    worker.run_pending()

    def predictions(recording):
        job_id = recording["jobs"][0]["id"]
        return client.get(f"/api/v1/jobs/{job_id}/predictions").json()

    assert predictions(unlabeled)["probabilities"] == predictions(labeled)["probabilities"]
    assert predictions(unlabeled)["reference_stages"] is None


def test_metrics_follow_the_requested_scope(client, finished):
    job = job_of(finished, TRANSFORMER)
    metrics = client.get(f"/api/v1/jobs/{job['id']}/metrics").json()
    assert metrics["scope"] == "all" and metrics["n_scored_epochs"] == metrics["n_epochs"] == 120
    matrix = np.array(metrics["confusion_matrix"])
    assert matrix.sum() == 120
    assert matrix.sum(axis=1).tolist() == [row["support"] for row in metrics["per_class"]]
    assert metrics["accuracy"] == pytest.approx(np.trace(matrix) / 120)
    for name in ("balanced_accuracy", "macro_f1"):
        assert 0 <= metrics[name] <= 1

    predicted = client.get(f"/api/v1/jobs/{job['id']}/predictions").json()
    agree = np.mean(np.array(predicted["stages"]) == np.array(predicted["reference_stages"]))
    assert metrics["accuracy"] == pytest.approx(agree)

    benchmark = client.get(f"/api/v1/jobs/{job['id']}/metrics", params={"scope": "benchmark"})
    assert benchmark.json()["scope"] == "benchmark"
    assert benchmark.json()["n_scored_epochs"] == 120  # the excerpt lies inside the sleep period
    assert client.get(f"/api/v1/jobs/{job['id']}/metrics",
                      params={"scope": "everything"}).status_code == 422


def test_metrics_need_a_hypnogram(client, worker):
    recording = upload(client, model_ids=(ATTNSLEEP,)).json()
    worker.run_pending()
    response = client.get(f"/api/v1/jobs/{recording['jobs'][0]['id']}/metrics")
    assert response.status_code == 409 and error(response) == "NO_HYPNOGRAM"


def test_export_csv_and_json(client, finished):
    job = job_of(finished, ATTNSLEEP)
    response = client.get(f"/api/v1/jobs/{job['id']}/export")
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="night-PSG_attnsleep-a2-v1.csv"' in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 120
    assert list(rows[0]) == ["epoch_index", "onset_seconds", "stage", "prob_W", "prob_N1",
                             "prob_N2", "prob_N3", "prob_REM", "reference_stage"]
    assert (rows[119]["epoch_index"], rows[119]["onset_seconds"]) == ("119", "3570")
    assert rows[0]["reference_stage"] == "W"

    exported = client.get(f"/api/v1/jobs/{job['id']}/export", params={"format": "json"}).json()
    assert exported["recording"]["sha256"] == finished["sha256"]
    assert exported["model"]["id"] == ATTNSLEEP and len(exported["model"]["fold_sha256"]) == 5
    assert exported["model"]["preprocessing"]["version"] == "a2-waveform-v1"
    assert len(exported["epochs"]) == 120
    assert exported["epochs"][5]["stage"] == rows[5]["stage"]


def test_epoch_waveform_is_the_filtered_signal(client, worker, finished):
    from app.ml.preprocessing import preprocess_recording

    response = client.get(f"/api/v1/recordings/{finished['id']}/epochs/7")
    epoch = response.json()
    assert (epoch["onset_seconds"], epoch["sampling_rate"], epoch["unit"]) == (210, 100, "uV")
    assert [channel["label"] for channel in epoch["channels"]] == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    assert epoch["reference_stage"] == 2
    expected = preprocess_recording(SAMPLE_PSG).display_waveform()[:, 21000:24000]
    served = np.array([channel["samples"] for channel in epoch["channels"]])
    np.testing.assert_allclose(served, expected, atol=0.006)

    response = client.get(f"/api/v1/recordings/{finished['id']}/epochs/120")
    assert response.status_code == 404 and error(response) == "EPOCH_NOT_FOUND"


def test_waveform_is_unavailable_until_a_job_has_preprocessed_it(client):
    recording = upload(client).json()
    response = client.get(f"/api/v1/recordings/{recording['id']}/epochs/0")
    assert response.status_code == 409 and error(response) == "WAVEFORM_NOT_READY"


def test_results_are_not_served_before_the_job_succeeds(client):
    recording = upload(client, hypnogram=True, model_ids=(TRANSFORMER,)).json()
    job_id = recording["jobs"][0]["id"]
    for path in ("predictions", "metrics", "export"):
        response = client.get(f"/api/v1/jobs/{job_id}/{path}")
        assert response.status_code == 409 and error(response) == "JOB_NOT_FINISHED"


def test_one_job_per_recording_and_model(client, worker):
    recording = upload(client, model_ids=(TRANSFORMER,)).json()
    body = {"recording_id": recording["id"], "model_id": TRANSFORMER}
    response = client.post("/api/v1/jobs", json=body)
    assert response.status_code == 409 and error(response) == "JOB_EXISTS"
    assert response.json()["detail"]["job_id"] == recording["jobs"][0]["id"]
    worker.run_pending()
    assert error(client.post("/api/v1/jobs", json=body)) == "JOB_EXISTS"  # result is reused
    assert client.post("/api/v1/jobs", json={**body, "model_id": ATTNSLEEP}).status_code == 202


def test_job_requests_for_unknown_things_are_rejected(client):
    recording = upload(client).json()
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.post("/api/v1/jobs", json={"recording_id": missing, "model_id": TRANSFORMER})
    assert response.status_code == 404 and error(response) == "RECORDING_NOT_FOUND"
    response = client.post("/api/v1/jobs", json={"recording_id": recording["id"],
                                                 "model_id": "nope"})
    assert response.status_code == 404 and error(response) == "MODEL_NOT_FOUND"
    assert client.get(f"/api/v1/jobs/{missing}").status_code == 404


def test_queue_length_is_capped(client, settings, monkeypatch):
    monkeypatch.setenv("MAX_QUEUED_JOBS", "1")
    settings()
    recording = upload(client, model_ids=(TRANSFORMER,)).json()
    response = client.post("/api/v1/jobs", json={"recording_id": recording["id"],
                                                 "model_id": ATTNSLEEP})
    assert response.status_code == 429 and error(response) == "QUEUE_FULL"
    response = upload(client, edf_variant("other"), model_ids=(TRANSFORMER,))
    assert response.status_code == 429
    assert client.get("/api/v1/recordings").json()["count"] == 1  # the upload was rolled back


def test_jobs_are_claimed_oldest_first_and_only_once(client):
    first = upload(client, edf_variant("first"), model_ids=(TRANSFORMER,)).json()
    second = upload(client, edf_variant("second"), model_ids=(TRANSFORMER,)).json()
    sessions = get_sessionmaker()
    with sessions() as one, sessions() as two:
        claimed = [claim_next_job(one), claim_next_job(two), claim_next_job(one)]
    assert [str(job_id) for job_id in claimed[:2]] == [first["jobs"][0]["id"],
                                                      second["jobs"][0]["id"]]
    assert claimed[2] is None


def test_interrupted_job_fails_cleanly_and_can_be_retried(client, worker):
    recording = upload(client, model_ids=(TRANSFORMER,)).json()
    job_id = recording["jobs"][0]["id"]
    with get_sessionmaker()() as session:
        claim_next_job(session)  # a worker took the job, then died
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"

    with get_sessionmaker()() as session:
        assert fail_interrupted_jobs(session) == 1  # what the next worker does on startup
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert (job["status"], job["error_code"]) == ("failed", "INTERRUPTED")
    assert error(client.get(f"/api/v1/jobs/{job_id}/predictions")) == "JOB_NOT_FINISHED"

    retry = client.post("/api/v1/jobs", json={"recording_id": recording["id"],
                                              "model_id": TRANSFORMER})
    assert retry.status_code == 202 and retry.json()["id"] != job_id
    worker.run_pending()
    assert client.get(f"/api/v1/jobs/{retry.json()['id']}").json()["status"] == "succeeded"


def test_failed_job_reports_a_code_and_publishes_nothing(client, worker):
    recording = upload(client, model_ids=(TRANSFORMER,)).json()
    source = storage.source_path(recording["id"])
    content = bytearray(source.read_bytes())
    content[256:272] = b"EEG C3-A2".ljust(16)  # the file changed on disk after validation
    source.write_bytes(content)
    worker.run_pending()
    job = client.get(f"/api/v1/jobs/{recording['jobs'][0]['id']}").json()
    assert (job["status"], job["error_code"]) == ("failed", "EDF_MISSING_CHANNEL")
    with get_sessionmaker()() as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 0


def test_crash_while_publishing_leaves_no_partial_predictions(client, worker, monkeypatch):
    recording = upload(client, model_ids=(TRANSFORMER,)).json()
    with get_sessionmaker()() as session:
        job_id = claim_next_job(session)
    monkeypatch.setattr("app.worker.utcnow", lambda: 1 / 0)  # fails after the rows are inserted
    with pytest.raises(ZeroDivisionError):
        worker.publish(job_id, np.full((120, 5), 0.2, dtype=np.float32))
    monkeypatch.undo()
    with get_sessionmaker()() as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 0
        assert session.get(Job, job_id).status == "running"
    assert recording["jobs"][0]["id"] == str(job_id)


def test_expired_uploads_are_purged_but_samples_stay(client, settings, monkeypatch):
    old = upload(client, edf_variant("old")).json()
    fresh = upload(client, edf_variant("fresh")).json()
    sample = upload(client, edf_variant("sample")).json()
    long_ago = utcnow() - timedelta(days=31)
    with get_sessionmaker()() as session:
        session.execute(update(Recording).where(Recording.id.in_([old["id"], sample["id"]]))
                        .values(created_at=long_ago))
        session.execute(update(Recording).where(Recording.id == sample["id"])
                        .values(is_sample=True))
        session.commit()

    Worker(settings()).purge_expired()
    remaining = {item["id"] for item in client.get("/api/v1/recordings").json()["data"]}
    assert remaining == {fresh["id"], sample["id"]}
    assert not storage.recording_dir(old["id"]).exists()


@pytest.mark.parametrize("export_format", ["csv", "json"])
def test_export_filename_is_safe_for_any_uploaded_name(client, worker, export_format):
    recording = upload(client, model_ids=(TRANSFORMER,),
                       filename='gi\u1ea5c ng\u1ee7 "1"\r\nX-Evil: 1.edf').json()
    worker.run_pending()
    response = client.get(f"/api/v1/jobs/{recording['jobs'][0]['id']}/export",
                          params={"format": export_format})
    assert response.status_code == 200
    # Multipart clients may percent-encode quotes and CR/LF before the API sees them.
    # The download header must be safe regardless of that transport representation.
    assert re.fullmatch(
        rf'attachment; filename="[A-Za-z0-9._-]+_{TRANSFORMER}\.{export_format}"',
        response.headers["content-disposition"],
    )
    assert "x-evil" not in response.headers


@pytest.mark.parametrize("export_format", ["csv", "json"])
def test_export_sanitizes_raw_stored_filename(client, finished, export_format):
    # Set the stored value directly so multipart escaping cannot hide unsafe characters.
    with get_sessionmaker()() as session:
        session.execute(update(Recording).where(Recording.id == finished["id"]).values(
            filename='gi\u1ea5c ng\u1ee7 "1"\r\nX-Evil: 1.edf'))
        session.commit()
    job = job_of(finished, TRANSFORMER)
    response = client.get(f"/api/v1/jobs/{job['id']}/export",
                          params={"format": export_format})
    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        f'attachment; filename="gi_c_ng_1_X-Evil_1_{TRANSFORMER}.{export_format}"')
    assert "x-evil" not in response.headers
