"""Export Sleep-EDF as one shared two-channel log-PSD tensor dataset."""

from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import spectrogram


STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]


def preprocess(data_dir, output_file):
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    hypnograms = {p.name[:6]: p for p in data_dir.glob("*-Hypnogram.edf")}
    patches, labels, groups, recordings, onsets = [], [], [], [], []

    for psg_path in sorted(data_dir.glob("*-PSG.edf")):
        print(f"Processing {psg_path.name}", flush=True)
        recording = psg_path.name[:6]
        annotations = mne.read_annotations(hypnograms[recording])
        sleep = [a for a in annotations
                 if a["description"] in STAGE_MAP and a["description"] != "Sleep stage W"]
        first = max(0, min(a["onset"] for a in sleep) - 30 * 60)
        last = max(a["onset"] + a["duration"] for a in sleep) + 30 * 60

        raw = mne.io.read_raw_edf(psg_path, include=CHANNELS, preload=False, verbose="ERROR")
        raw.reorder_channels(CHANNELS)
        sfreq = raw.info["sfreq"]
        last = min(last, raw.n_times / sfreq)

        for annotation in annotations:
            description = annotation["description"]
            if description not in STAGE_MAP:
                continue
            end = annotation["onset"] + annotation["duration"]
            for onset in np.arange(annotation["onset"], end, 30):
                if onset < first or onset + 30 > min(end, last):
                    continue
                start = round(onset * sfreq)
                stop = start + round(30 * sfreq)
                eeg = raw.get_data(start=start, stop=stop) * 1e6
                frequencies, _, power = spectrogram(
                    eeg, fs=sfreq, window="hann", nperseg=200, noverlap=150,
                    detrend=False, scaling="density", mode="psd", axis=-1)
                keep = (frequencies >= 0.5) & (frequencies <= 30)
                patch = np.log10(np.maximum(power[:, keep, :], 1e-12)).astype(np.float32)
                patches.append(torch.from_numpy(patch))
                labels.append(STAGE_MAP[description])
                groups.append(recording[:5])
                recordings.append(recording)
                onsets.append(float(onset))

        raw.close()

    if not patches:
        print(f"No epochs found in {data_dir}")
        return

    dataset = {
        "patches": torch.stack(patches),
        "labels": torch.tensor(labels, dtype=torch.long),
        "groups": groups,
        "recordings": recordings,
        "onsets": torch.tensor(onsets, dtype=torch.float64),
        "class_map": {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4},
        "channels": CHANNELS,
        "sfreq": sfreq,
        "epoch_seconds": 30,
        "wake_margin_seconds": 30 * 60,
        "spectrogram": {"window": "hann", "nperseg": 200, "noverlap": 150,
                        "detrend": False, "scaling": "density", "mode": "psd",
                        "frequency_range": [0.5, 30], "log_floor": 1e-12,
                        "units": "log10(µV²/Hz)"},
    }
    del patches
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_file)
    print(f"Saved {output_file}: {tuple(dataset['patches'].shape)}")


if __name__ == "__main__":
    base_directory = Path(__file__).resolve().parent
    data_directory = base_directory / "sleep-cassette"
    output_file = base_directory / "dataset/sleep_edf.pt"

    preprocess(data_directory, output_file)

# Training example (select training participants using sleep_exploration/cv_folds.csv):
# from model_lib import ResCBAM_EEG_Net
# import torch.nn as nn
# data = torch.load(output_file, weights_only=True)
# model = ResCBAM_EEG_Net(in_channels=2, num_classes=5)
# logits = model(data["patches"][:8])
# labels = data["labels"][:8]
# criterion = nn.CrossEntropyLoss()
# loss = criterion(logits, labels)
