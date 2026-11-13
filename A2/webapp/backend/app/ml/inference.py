"""Five-fold soft-voting inference on CPU. Imported by the worker and the CLI only."""

from collections.abc import Callable

import numpy as np
import torch

from .networks import NETWORKS
from .preprocessing import PREPROCESSING_VERSION
from .registry import CheckpointError, ModelRelease


def load_ensemble(release: ModelRelease) -> list[torch.nn.Module]:
    """Verify hashes, then load every fold with ``strict=True`` in eval mode."""
    if release.preprocessing.get("version") != PREPROCESSING_VERSION:
        raise CheckpointError(f"{release.id}: trained with a different preprocessing version.")
    release.verify()
    models = []
    for path in release.fold_paths:
        model = NETWORKS[release.network](**release.network_kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        models.append(model.eval())
    return models


@torch.inference_mode()
def predict_proba(models: list[torch.nn.Module], epochs: np.ndarray, *, batch_size: int = 64,
                  on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Return ``(n_epochs, 5)`` float32: the mean of each fold's softmax probabilities."""
    total = np.zeros((len(epochs), 5), dtype=np.float64)
    steps = len(models) * -(-len(epochs) // batch_size)
    done = 0
    for model in models:
        for start in range(0, len(epochs), batch_size):
            batch = torch.from_numpy(epochs[start:start + batch_size])
            total[start:start + batch_size] += torch.softmax(model(batch).double(), dim=1).numpy()
            done += 1
            if on_progress is not None:
                on_progress(done, steps)
    return (total / len(models)).astype(np.float32)
