from app.services import storage
from tests.conftest import (
    ATTNSLEEP,
    SAMPLE_HYPNOGRAM,
    SAMPLE_PSG,
    TRANSFORMER,
    edf_variant,
    upload,
)


def error(response) -> str:
    return response.json()["detail"]["code"]


def test_health_reports_database_and_models(client):
    assert client.get("/api/health").json() == {"status": "ok", "database": "ok",
                                                "models_available": 2}


def test_models_lists_both_releases(client):
    models = {model["id"]: model for model in client.get("/api/v1/models").json()}
    assert list(models) == [TRANSFORMER, ATTNSLEEP]  # the default model comes first
    assert models[TRANSFORMER]["input_channels"] == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    assert models[ATTNSLEEP]["input_channels"] == ["EEG Fpz-Cz"]
    assert all(model["available"] and model["synthetic"] for model in models.values())


def test_upload_stores_the_file_and_its_metadata(client):
    response = upload(client)
    assert response.status_code == 201
    recording = response.json()
    assert recording["filename"] == "night-PSG.edf"
    assert (recording["duration_seconds"], recording["n_epochs"]) == (3600, 120)
    assert recording["dropped_seconds"] == 0
    assert [channel["label"] for channel in recording["channels"]] == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    assert recording["has_hypnogram"] is False and recording["jobs"] == []
    assert recording["expires_at"] is not None
    stored = storage.source_path(recording["id"])
    assert stored.read_bytes() == SAMPLE_PSG.read_bytes()

    assert client.get(f"/api/v1/recordings/{recording['id']}").json() == recording
    listing = client.get("/api/v1/recordings").json()
    assert listing["count"] == 1 and listing["data"][0]["id"] == recording["id"]


def test_upload_with_models_queues_one_job_each(client):
    recording = upload(client, model_ids=(TRANSFORMER, ATTNSLEEP, TRANSFORMER)).json()
    assert sorted(job["model_id"] for job in recording["jobs"]) == [ATTNSLEEP, TRANSFORMER]
    assert {job["status"] for job in recording["jobs"]} == {"queued"}


def test_uploading_the_same_file_again_reuses_the_recording(client):
    first = upload(client, model_ids=(TRANSFORMER,))
    second = upload(client, hypnogram=True, model_ids=(TRANSFORMER, ATTNSLEEP),
                    filename="renamed.edf")
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["filename"] == "night-PSG.edf"
    assert second.json()["has_hypnogram"] is True  # the missing hypnogram was attached
    assert len(second.json()["jobs"]) == 2  # only the model that was not queued yet is added
    assert client.get("/api/v1/recordings").json()["count"] == 1


def test_upload_with_hypnogram_keeps_the_annotation_file(client):
    recording = upload(client, hypnogram=True).json()
    assert recording["has_hypnogram"] is True
    assert storage.hypnogram_path(recording["id"]).read_bytes() == SAMPLE_HYPNOGRAM.read_bytes()


def test_invalid_uploads_are_rejected_with_a_code_and_leave_nothing_behind(client, settings):
    assert error(upload(client, b"definitely not an EDF" * 50)) == "EDF_INVALID"

    wrong_channel = bytearray(SAMPLE_PSG.read_bytes())
    wrong_channel[256:272] = b"EEG C3-A2".ljust(16)
    response = upload(client, bytes(wrong_channel))
    assert response.status_code == 422 and error(response) == "EDF_MISSING_CHANNEL"
    assert "EEG C3-A2" in response.json()["detail"]["message"]

    files = {"edf": ("a.edf", SAMPLE_PSG.read_bytes()), "hypnogram": ("h.edf", b"junk" * 100)}
    response = client.post("/api/v1/recordings", files=files)
    assert response.status_code == 422 and error(response) == "HYPNOGRAM_INVALID"

    assert client.get("/api/v1/recordings").json()["count"] == 0
    data_dir = settings().DATA_DIR
    assert not list((data_dir / "tmp").glob("*"))
    assert not list((data_dir / "recordings").glob("*"))


def test_unknown_model_is_rejected_before_anything_is_stored(client):
    response = upload(client, model_ids=("resnet-9000",))
    assert response.status_code == 404 and error(response) == "MODEL_NOT_FOUND"
    assert client.get("/api/v1/recordings").json()["count"] == 0


def test_upload_size_and_recording_count_are_capped(client, settings, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "100000")
    settings()
    response = upload(client)
    assert response.status_code == 413 and error(response) == "UPLOAD_TOO_LARGE"

    monkeypatch.delenv("MAX_UPLOAD_BYTES")
    monkeypatch.setenv("MAX_RECORDINGS", "1")
    settings()
    assert upload(client).status_code == 201
    response = upload(client, edf_variant("second"))
    assert response.status_code == 507 and error(response) == "STORAGE_FULL"
    assert upload(client).status_code == 200  # an already stored file is still served


def test_delete_removes_rows_and_files(client):
    recording = upload(client, hypnogram=True, model_ids=(TRANSFORMER,)).json()
    directory = storage.recording_dir(recording["id"])
    assert directory.is_dir()
    assert client.delete(f"/api/v1/recordings/{recording['id']}").status_code == 204
    assert not directory.exists()
    response = client.get(f"/api/v1/recordings/{recording['id']}")
    assert response.status_code == 404 and error(response) == "RECORDING_NOT_FOUND"
    job_id = recording["jobs"][0]["id"]
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404


def test_bundled_sample_is_seeded_once_and_is_read_only(client, settings):
    from app.core.db import get_sessionmaker
    from app.services.recordings import seed_samples

    for _ in range(2):
        with get_sessionmaker()() as session:
            seed_samples(session, settings())
    listing = client.get("/api/v1/recordings").json()
    assert listing["count"] == 1
    sample = listing["data"][0]
    assert sample["is_sample"] and sample["has_hypnogram"] and sample["expires_at"] is None
    response = client.delete(f"/api/v1/recordings/{sample['id']}")
    assert response.status_code == 403 and error(response) == "SAMPLE_READ_ONLY"
    # A visitor uploading the very same file is pointed at the sample instead of a copy.
    assert upload(client).json()["id"] == sample["id"]
