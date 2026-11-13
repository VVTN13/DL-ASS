import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute

from .api.main import api_router
from .api.routes import health
from .core.config import get_settings
from .core.db import get_sessionmaker
from .services.recordings import seed_samples

logger = logging.getLogger("uvicorn.error")


def operation_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.SEED_SAMPLE:
        try:
            with get_sessionmaker()() as session:
                seed_samples(session, settings)
        except Exception:  # A missing sample must not keep the API from starting.
            logger.exception("Could not seed the sample recording")
    yield


app = FastAPI(
    title="A2 Sleep Staging",
    summary="Upload a Sleep-EDF recording, run the A2 models and inspect the hypnogram.",
    description="Research prototype. Not a medical device and not for clinical use.",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/docs",
    redoc_url=None,
    generate_unique_id_function=operation_id,
    lifespan=lifespan,
)
app.include_router(api_router, prefix="/api/v1")
app.include_router(health.router, prefix="/api")
