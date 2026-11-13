from fastapi import APIRouter

from .routes import jobs, models, recordings

api_router = APIRouter()
api_router.include_router(models.router)
api_router.include_router(recordings.router)
api_router.include_router(jobs.router)
