"""The serving code must reproduce the A2 research code it was ported from.

``A2/model_lib.py`` and ``A2/preprocessing.py`` are imported directly as the reference.
Architecture parity does not depend on trained weights, so it runs with synthetic ones.
"""

import csv
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from app.ml.inference import load_ensemble, predict_proba
from app.ml.networks import NETWORKS
from app.ml.preprocessing import CHANNELS, prepare_waveforms, preprocess_recording
from app.ml.registry import CLASS_NAMES, load_releases
from tests.conftest import ATTNSLEEP, BACKEND, SAMPLE_PSG, TRANSFORMER

A2 = BACKEND.parents[1]
sys.path.insert(0, str(A2))
research_models = pytest.importorskip("model_lib")
research_preprocessing = pytest.importorskip("preprocessing")


@pytest.mark.parametrize("model_id", [TRANSFORMER, ATTNSLEEP])
def test_network_matches_research_implementation(model_id):
    release = load_releases(os.environ["CHECKPOINTS_DIR"])[model_id]
    state = torch.load(release.fold_paths[0], map_location="cpu", weights_only=True)
    served = NETWORKS[release.network](**release.network_kwargs)
    reference = getattr(research_models, release.network)(**release.network_kwargs)
    assert set(served.state_dict()) == set(reference.state_dict())
    served.load_state_dict(state, strict=True)
    reference.load_state_dict(state, strict=True)

    batch = torch.from_numpy(
        np.random.default_rng(0).normal(0, 0.3, (16, 2, 30, 200)).astype(np.float32))
    with torch.inference_mode():
        np.testing.assert_allclose(served.eval()(batch).numpy(),
                                   reference.eval()(batch).numpy(), atol=1e-6)


def test_preprocessing_matches_research_implementation():
    def read():
        return mne.io.read_raw_edf(SAMPLE_PSG, include=list(CHANNELS), preload=False,
                                   verbose="ERROR")

    np.testing.assert_array_equal(prepare_waveforms(read()),
                                  research_preprocessing.prepare_waveforms(read()))


def test_serving_epochs_match_research_patches_at_the_same_onsets():
    prepared = preprocess_recording(SAMPLE_PSG)
    onsets = [(0.0, 0), (30.0, 0), (1770.0, 0), (3570.0, 0)]
    patches = research_preprocessing.make_patches(prepared.signal, onsets).numpy()
    np.testing.assert_array_equal(prepared.epochs[[0, 1, 59, 119]], patches)


# --- Against the published A2 run: needs the trained checkpoints and the Sleep-EDF night ---

RECORDING = "SC4031"  # held-out test participant SC403
REAL_CHECKPOINTS = BACKEND / "checkpoints"
RUNS = A2 / "runs"


@pytest.fixture(scope="module")
def published():
    edf_dir = os.environ.get("SLEEP_EDF_DIR")
    if not edf_dir:
        pytest.skip("set SLEEP_EDF_DIR to the Sleep-EDF sleep-cassette directory")
    releases = load_releases(REAL_CHECKPOINTS)
    if not all(release.available for release in releases.values()) or len(releases) < 2:
        pytest.skip("trained checkpoints are not installed (scripts/checkpoints.py package)")
    path = next(Path(edf_dir).glob(f"{RECORDING}*-PSG.edf"))
    prepared = preprocess_recording(path)
    # The AttnSleep export is the only one that records onsets; both share source_index.
    rows = [row for row in csv.DictReader(
        (RUNS / "AttnSleep/sleep_waveform_final/test_predictions.csv").open())
        if row["recording"] == RECORDING]
    return releases, prepared, rows


@pytest.mark.published
def test_attnsleep_probabilities_match_published_run(published):
    releases, prepared, rows = published
    probabilities = predict_proba(load_ensemble(releases[ATTNSLEEP]), prepared.epochs)
    index = [int(float(row["onset"]) // 30) for row in rows]
    reference = np.array([[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows])
    # Measured 2026-11-13 on CPU against the CUDA run: max 4.8e-4, mean 5e-5.
    np.testing.assert_allclose(probabilities[index], reference, atol=2e-3)
    assert (probabilities[index].argmax(1) == [int(row["prediction"]) for row in rows]).all()


@pytest.mark.published
def test_transformer_labels_match_published_run(published):
    releases, prepared, rows = published
    onset = {row["source_index"]: float(row["onset"]) for row in rows}
    reference = [row for row in csv.DictReader(
        (RUNS / "Transformer/sleep_waveform_final/test_predictions.csv").open())
        if row["source_index"] in onset]
    assert len(reference) == len(rows) == 952
    probabilities = predict_proba(load_ensemble(releases[TRANSFORMER]), prepared.epochs)
    disagreements = []
    for row in reference:
        epoch = probabilities[int(onset[row["source_index"]] // 30)]
        if int(epoch.argmax()) != int(row["prediction"]):
            top_two = np.sort(epoch)[-2:]
            disagreements.append(float(top_two[1] - top_two[0]))
    # Measured: 1 of 952, with a top-two margin of 2e-4. Only near-ties may flip.
    assert len(disagreements) <= 5
    assert all(margin < 0.01 for margin in disagreements)
