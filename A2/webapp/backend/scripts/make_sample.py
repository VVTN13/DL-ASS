"""Cut the bundled one-hour sample from a Sleep-EDF night (development-only; needs edfio).

    python scripts/make_sample.py --sleep-edf-dir /path/to/sleep-cassette

SC4031 belongs to participant SC403, one of the A2 held-out test participants, so neither
model saw it during training. The two EEG channels keep their original digital samples.
"""

import argparse
from pathlib import Path

from edfio import Edf, EdfAnnotation, read_edf

RECORDING = "SC4031"
START_SECONDS = 29400  # epoch 980: the hour that follows contains all five stages
DURATION_SECONDS = 3600
CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sleep-edf-dir", type=Path, required=True)
    args = parser.parse_args()
    stop = START_SECONDS + DURATION_SECONDS

    psg = read_edf(next(args.sleep_edf_dir.glob(f"{RECORDING}*-PSG.edf")))
    psg.drop_signals([signal.label for signal in psg.signals if signal.label not in CHANNELS])
    psg.slice_between_seconds(START_SECONDS, stop)
    psg.write(SAMPLES / f"{RECORDING}-excerpt-PSG.edf")

    hypnogram = read_edf(next(args.sleep_edf_dir.glob(f"{RECORDING}*-Hypnogram.edf")))
    annotations = []
    for annotation in hypnogram.annotations:
        onset = max(annotation.onset, START_SECONDS)
        end = min(annotation.onset + annotation.duration, stop)
        if end > onset:
            annotations.append(EdfAnnotation(onset - START_SECONDS, end - onset, annotation.text))
    Edf([], annotations=annotations).write(
        SAMPLES / f"{RECORDING}-excerpt-Hypnogram.edf")
    print(f"wrote {len(annotations)} annotations and {DURATION_SECONDS} s of EEG to {SAMPLES}")


if __name__ == "__main__":
    main()
