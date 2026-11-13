"""Immutable model releases: a ``manifest.json`` plus five fold checkpoints per directory.

Manifests are versioned in Git; the ``.pt`` files are not. ``scripts/checkpoints.py``
packages them from ``A2/runs`` or downloads them, and every load re-checks SHA-256.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")


class CheckpointError(RuntimeError):
    code = "CHECKPOINT_UNAVAILABLE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelRelease:
    id: str
    name: str
    order: int  # display order; the first release is the default
    network: str
    network_kwargs: dict
    input_channels: tuple[str, ...]
    folds: tuple[dict, ...]  # {"file": "fold_1.pt", "sha256": "..."}
    preprocessing: dict
    published_metrics: dict
    source: dict
    synthetic: bool
    directory: Path

    @property
    def fold_paths(self) -> list[Path]:
        return [self.directory / fold["file"] for fold in self.folds]

    @property
    def available(self) -> bool:
        return all(path.is_file() for path in self.fold_paths)

    def verify(self) -> None:
        for fold, path in zip(self.folds, self.fold_paths, strict=True):
            if not path.is_file():
                raise CheckpointError(f"{self.id}: missing checkpoint {path.name}.")
            if sha256(path) != fold["sha256"]:
                raise CheckpointError(f"{self.id}: SHA-256 mismatch for {path.name}.")


def load_release(directory: Path) -> ModelRelease:
    manifest = json.loads((directory / "manifest.json").read_text())
    if tuple(manifest["class_names"]) != CLASS_NAMES:
        raise CheckpointError(f"{manifest['id']}: unexpected class order.")
    return ModelRelease(
        id=manifest["id"],
        name=manifest["name"],
        order=int(manifest.get("order", 0)),
        network=manifest["network"],
        network_kwargs=manifest["network_kwargs"],
        input_channels=tuple(manifest["input_channels"]),
        folds=tuple(manifest["folds"]),
        preprocessing=manifest["preprocessing"],
        published_metrics=manifest.get("published_metrics", {}),
        source=manifest.get("source", {}),
        synthetic=bool(manifest.get("synthetic", False)),
        directory=directory,
    )


def load_releases(checkpoints_dir: str | Path) -> dict[str, ModelRelease]:
    root = Path(checkpoints_dir)
    releases = [load_release(path.parent) for path in root.glob("*/manifest.json")]
    releases.sort(key=lambda release: (release.order, release.id))
    return {release.id: release for release in releases}
