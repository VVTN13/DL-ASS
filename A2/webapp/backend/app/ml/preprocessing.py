"""Label-free preprocessing for serving: one continuous EDF in, 30-second epochs out.

The signal chain reproduces ``A2/preprocessing.prepare_waveforms`` exactly (parity-tested).
Unlike the research export, epochs are consecutive windows from the start of the recording:
no hypnogram is read here, so reference labels can never influence a prediction.
"""

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

PREPROCESSING_VERSION = "a2-waveform-v1"
CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
SOURCE_SFREQ = 100
TARGET_SFREQ = 200
EPOCH_SECONDS = 30
EPOCH_SAMPLES = EPOCH_SECONDS * TARGET_SFREQ

CONFIG = {
    "version": PREPROCESSING_VERSION,
    "channels": list(CHANNELS),
    "source_sfreq": SOURCE_SFREQ,
    "target_sfreq": TARGET_SFREQ,
    "epoch_seconds": EPOCH_SECONDS,
    "filter": {"low_hz": 0.3, "high_hz": 35.0, "method": "fir", "phase": "zero",
               "fir_window": "hamming", "fir_design": "firwin",
               "l_trans_bandwidth": 0.3, "h_trans_bandwidth": 8.75, "pad": "reflect_limited"},
    "resample": {"method": "scipy.signal.resample_poly", "up": 2, "down": 1,
                 "window": ["kaiser", 5.0], "padtype": "line"},
    "amplitude": "microvolts / 100",
    "epoch_grid": "consecutive 30 s windows from recording start; trailing remainder dropped",
}


class SignalError(ValueError):
    """The recording cannot be processed; ``code`` is a stable, user-facing identifier."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Preprocessed:
    epochs: np.ndarray  # (n_epochs, 2, 30, 200) float32, microvolts / 100
    signal: np.ndarray  # (2, n_samples) float32 at 200 Hz, microvolts / 100
    dropped_seconds: float

    @property
    def n_epochs(self) -> int:
        return self.epochs.shape[0]

    def display_waveform(self) -> np.ndarray:
        """Filtered signal at 100 Hz in microvolts, for plotting. Shape (2, n_samples)."""
        return np.ascontiguousarray(self.signal[:, ::2] * np.float32(100.0))


def read_raw(path: str | Path) -> mne.io.BaseRaw:
    try:
        raw = mne.io.read_raw_edf(path, include=list(CHANNELS), preload=False, verbose="ERROR")
    except Exception as error:  # MNE raises several unrelated types for corrupt files.
        raise SignalError("EDF_UNREADABLE", f"Could not read EDF: {error}") from error
    if set(raw.ch_names) != set(CHANNELS):
        raise SignalError("EDF_MISSING_CHANNEL",
                          f"Expected channels {list(CHANNELS)}, found {raw.ch_names}.")
    if raw.info["sfreq"] != SOURCE_SFREQ:
        raise SignalError("EDF_WRONG_SAMPLING_RATE",
                          f"Expected {SOURCE_SFREQ} Hz EEG, got {raw.info['sfreq']:g} Hz.")
    return raw


def prepare_waveforms(raw: mne.io.BaseRaw) -> np.ndarray:
    """Filter and resample one continuous recording; return float32 EEG in microvolts / 100."""
    raw.reorder_channels(list(CHANNELS))
    raw.load_data(verbose="ERROR")
    # Fail instead of contaminating the whole recording by filtering NaN/Inf samples.
    for start in range(0, raw.n_times, 60000):
        chunk = raw.get_data(start=start, stop=min(start + 60000, raw.n_times))
        if not np.isfinite(chunk).all():
            raise SignalError("SIGNAL_NONFINITE", "The EEG signal contains NaN or Inf samples.")
    raw.filter(
        0.3, 35.0, method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
        filter_length="auto", l_trans_bandwidth=0.3, h_trans_bandwidth=8.75,
        pad="reflect_limited", skip_by_annotation=(), n_jobs=1, verbose="ERROR",
    )
    signal = resample_poly(raw.get_data(), up=2, down=1, axis=-1,
                           window=("kaiser", 5.0), padtype="line")
    signal *= 10000.0  # MNE returns volts: volts * 1e6 / 100.
    signal = signal.astype(np.float32)
    if not np.isfinite(signal).all():
        raise SignalError("SIGNAL_NONFINITE", "Nonfinite signal after filtering and resampling.")
    return signal


def make_epochs(signal: np.ndarray) -> np.ndarray:
    """Cut consecutive 30 s windows; epoch ``i`` covers seconds ``[30 i, 30 i + 30)``."""
    n_epochs = signal.shape[1] // EPOCH_SAMPLES
    used = signal[:, :n_epochs * EPOCH_SAMPLES].reshape(2, n_epochs, EPOCH_SECONDS, TARGET_SFREQ)
    return np.ascontiguousarray(used.transpose(1, 0, 2, 3))


def preprocess_recording(path: str | Path) -> Preprocessed:
    raw = read_raw(path)
    try:
        signal = prepare_waveforms(raw)
    finally:
        raw.close()
    epochs = make_epochs(signal)
    if not len(epochs):
        raise SignalError("RECORDING_TOO_SHORT", "The recording is shorter than one 30 s epoch.")
    dropped = (signal.shape[1] - len(epochs) * EPOCH_SAMPLES) / TARGET_SFREQ
    return Preprocessed(epochs=epochs, signal=signal, dropped_seconds=dropped)
