from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://sleep:sleep@localhost:5432/sleep"
    DATA_DIR: Path = BACKEND_DIR.parent / "data"
    CHECKPOINTS_DIR: Path = BACKEND_DIR / "checkpoints"
    SAMPLES_DIR: Path = BACKEND_DIR / "samples"
    SEED_SAMPLE: bool = True

    # Uploads are open to every visitor, so every resource they consume is capped.
    MAX_UPLOAD_BYTES: int = 64 * MIB
    MAX_HYPNOGRAM_BYTES: int = 1 * MIB
    MAX_RECORDINGS: int = 100
    MAX_QUEUED_JOBS: int = 20
    RETENTION_DAYS: int = 30  # 0 keeps uploads forever

    TORCH_THREADS: int = 4
    BATCH_SIZE: int = 64
    WORKER_POLL_SECONDS: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
