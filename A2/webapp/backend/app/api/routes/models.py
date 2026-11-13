from fastapi import APIRouter

from ...schemas import ModelPublic
from ..deps import ReleasesDep

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelPublic])
def list_models(releases: ReleasesDep):
    """Model releases that can be run, with the metrics published in the A2 report."""
    return [
        ModelPublic(
            id=release.id, name=release.name, network=release.network,
            input_channels=list(release.input_channels),
            preprocessing_version=release.preprocessing["version"],
            available=release.available, synthetic=release.synthetic,
            published_metrics=release.published_metrics or None,
        )
        for release in releases.values()
    ]
