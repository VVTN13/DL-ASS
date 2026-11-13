import json
import os
import shutil
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from app.ml.inference import load_ensemble, predict_proba
from app.ml.preprocessing import (
    CHANNELS,
    SignalError,
    make_epochs,
    prepare_waveforms,
    preprocess_recording,
)
from app.ml.registry import CLASS_NAMES, CheckpointError, load_releases
from tests.conftest import ATTNSLEEP, SAMPLE_PSG, TRANSFORMER


def raw_array(seconds: float, *, channels=CHANNELS) -> mne.io.RawArray:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 20e-6, (len(channels), int(seconds * 100)))
    return mne.io.RawArray(data, mne.create_info(list(channels), 100, "eeg"), verbose="ERROR")


@pytest.fixture(scope="module")
def releases():
    return load_releases(os.environ["CHECKPOINTS_DIR"])


def test_epochs_are_consecutive_windows_and_the_remainder_is_dropped():
    signal = np.arange(2 * 15000, dtype=np.float32).reshape(2, 15000)  # 75 s at 200 Hz
    epochs = make_epochs(signal)
    assert epochs.shape == (2, 2, 30, 200)
    assert epochs[1, 0, 0, 0] == 6000 and epochs[1, 1, 29, 199] == 15000 + 11999


def test_channels_are_reordered_before_processing():
    swapped = raw_array(60, channels=CHANNELS[::-1])
    expected = swapped.get_data()[::-1]
    signal = prepare_waveforms(swapped)
    assert swapped.ch_names == list(CHANNELS)
    assert np.corrcoef(signal[0, ::2], expected[0])[0, 1] > 0.5


def test_output_is_float32_microvolts_over_100_at_200_hz():
    raw = raw_array(90)
    signal = prepare_waveforms(raw)
    assert signal.dtype == np.float32 and signal.shape == (2, 18000)
    assert 0.05 < signal.std() < 0.5  # 20 uV noise, band-limited, divided by 100


def test_nonfinite_signal_is_rejected():
    raw = raw_array(60)
    raw._data[1, 1234] = np.nan
    with pytest.raises(SignalError) as caught:
        prepare_waveforms(raw)
    assert caught.value.code == "SIGNAL_NONFINITE"


def test_sample_recording_preprocesses_to_120_epochs():
    prepared = preprocess_recording(SAMPLE_PSG)
    assert prepared.epochs.shape == (120, 2, 30, 200)
    assert prepared.dropped_seconds == 0
    display = prepared.display_waveform()
    assert display.shape == (2, 360000)
    np.testing.assert_allclose(display[:, :3000], prepared.epochs[0].reshape(2, 6000)[:, ::2] * 100)


@pytest.mark.parametrize("model_id", [TRANSFORMER, ATTNSLEEP])
def test_ensemble_is_the_mean_of_five_softmax_outputs(releases, model_id):
    models = load_ensemble(releases[model_id])
    assert len(models) == 5 and not any(model.training for model in models)
    epochs = np.random.default_rng(1).normal(0, 0.2, (70, 2, 30, 200)).astype(np.float32)
    calls = []
    probabilities = predict_proba(models, epochs, batch_size=32,
                                  on_progress=lambda done, total: calls.append((done, total)))

    with torch.inference_mode():
        expected = torch.stack(
            [torch.softmax(model(torch.from_numpy(epochs)), dim=1) for model in models]).mean(0)
    assert probabilities.shape == (70, len(CLASS_NAMES)) and probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities, expected.numpy(), atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert calls[-1] == (15, 15)  # 5 folds x 3 batches


def test_batch_size_does_not_change_predictions(releases):
    models = load_ensemble(releases[TRANSFORMER])
    epochs = np.random.default_rng(2).normal(0, 0.2, (9, 2, 30, 200)).astype(np.float32)
    np.testing.assert_allclose(predict_proba(models, epochs, batch_size=4),
                               predict_proba(models, epochs, batch_size=64), atol=1e-6)


def test_tampered_or_missing_checkpoint_is_refused(releases, tmp_path):
    source = releases[TRANSFORMER].directory
    copy = Path(shutil.copytree(source, tmp_path / source.name))
    with (copy / "fold_3.pt").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        load_ensemble(load_releases(tmp_path)[TRANSFORMER])
    (copy / "fold_3.pt").unlink()
    release = load_releases(tmp_path)[TRANSFORMER]
    assert not release.available
    with pytest.raises(CheckpointError, match="missing"):
        load_ensemble(release)


def test_release_with_another_class_order_is_refused(releases, tmp_path):
    source = releases[TRANSFORMER].directory
    copy = Path(shutil.copytree(source, tmp_path / source.name))
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["class_names"] = ["W", "REM", "N1", "N2", "N3"]
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match="class order"):
        load_releases(tmp_path)
