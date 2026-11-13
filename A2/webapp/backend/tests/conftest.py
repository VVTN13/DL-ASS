"""Shared fixtures. Tests run against a real PostgreSQL database and synthetic checkpoints.

Synthetic checkpoints (random weights, real architectures) keep the suite independent of
the trained ``.pt`` files, which are not in Git. Only ``-m published`` needs the real ones.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
WORKDIR = Path(tempfile.mkdtemp(prefix="sleep-webapp-tests-"))
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://sleep:sleep@localhost:5432/sleep_test")
os.environ["DATA_DIR"] = str(WORKDIR / "data")
os.environ["CHECKPOINTS_DIR"] = str(WORKDIR / "checkpoints")
os.environ["SEED_SAMPLE"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, make_url, text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.db import Base, get_engine  # noqa: E402

SAMPLE_PSG = BACKEND / "samples" / "SC4031-excerpt-PSG.edf"
SAMPLE_HYPNOGRAM = BACKEND / "samples" / "SC4031-excerpt-Hypnogram.edf"
TRANSFORMER = "transformer-a2-v1"
ATTNSLEEP = "attnsleep-a2-v1"


def edf_variant(tag: str) -> bytes:
    """The sample with a different recording-identification field, hence a new SHA-256."""
    content = bytearray(SAMPLE_PSG.read_bytes())
    content[88:168] = f"Startdate X X X {tag}".ljust(80).encode("ascii")
    return bytes(content)


def upload(client: TestClient, content: bytes | None = None, *, hypnogram: bool = False,
           model_ids: tuple[str, ...] = (), filename: str = "night-PSG.edf"):
    files = {"edf": (filename, content if content is not None else SAMPLE_PSG.read_bytes())}
    if hypnogram:
        files["hypnogram"] = ("night-Hypnogram.edf", SAMPLE_HYPNOGRAM.read_bytes())
    return client.post("/api/v1/recordings", files=files, data={"model_ids": list(model_ids)})


def ensure_database() -> None:
    """Create the test database on first use, so a fresh PostgreSQL needs no manual setup."""
    target = make_url(os.environ["DATABASE_URL"])
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        exists = connection.scalar(text("SELECT 1 FROM pg_database WHERE datname = :name"),
                                   {"name": target.database})
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{target.database}"'))
    admin.dispose()


@pytest.fixture(scope="session", autouse=True)
def environment():
    ensure_database()
    subprocess.run([sys.executable, "scripts/checkpoints.py", "synthetic",
                    "--out", os.environ["CHECKPOINTS_DIR"]], cwd=BACKEND, check=True)
    from app import models  # noqa: F401

    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())
    yield
    Base.metadata.drop_all(get_engine())


@pytest.fixture(autouse=True)
def clean_state():
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE recordings CASCADE"))
    shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(monkeypatch):
    """Override a setting with ``monkeypatch.setenv`` and call this to re-read it."""
    def reload():
        get_settings.cache_clear()
        return get_settings()

    return reload


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def worker(environment):
    from app.worker import Worker

    return Worker()
