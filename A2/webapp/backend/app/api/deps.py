from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..core.db import get_db
from ..ml.registry import ModelRelease, load_releases


class ApiError(HTTPException):
    """Domain error rendered as ``{"detail": {"code": ..., "message": ..., ...}}``."""

    def __init__(self, status_code: int, code: str, message: str, **extra):
        super().__init__(status_code, detail={"code": code, "message": message, **extra})


@lru_cache
def get_releases() -> dict[str, ModelRelease]:
    return load_releases(get_settings().CHECKPOINTS_DIR)


SessionDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ReleasesDep = Annotated[dict[str, ModelRelease], Depends(get_releases)]
