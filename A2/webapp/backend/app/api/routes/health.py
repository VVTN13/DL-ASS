from fastapi import APIRouter
from sqlalchemy import text

from ...schemas import HealthPublic
from ..deps import ReleasesDep, SessionDep

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthPublic)
def health_check(session: SessionDep, releases: ReleasesDep):
    """Ready when the database answers. Reports how many model releases have checkpoints."""
    session.execute(text("SELECT 1"))
    return HealthPublic(status="ok", database="ok",
                        models_available=sum(r.available for r in releases.values()))
