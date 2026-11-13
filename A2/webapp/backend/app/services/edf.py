"""EDF header validation and hypnogram parsing for uploads.

The header parser is deliberately dependency-free: it runs inside the API process on
every upload and turns each way a file can be unusable into a stable error code.
"""

from dataclasses import dataclass
from pathlib import Path

CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
SAMPLING_RATE = 100.0
EPOCH_SECONDS = 30
SUPPORTED_UNITS = {"uV", "µV", "mV", "V"}
STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
BENCHMARK_MARGIN_SECONDS = 1800


class EdfError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EdfHeader:
    labels: list[str]
    units: list[str]
    sampling_rates: list[float]
    n_records: int
    record_seconds: float
    discontinuous: bool

    @property
    def duration_seconds(self) -> float:
        return self.n_records * self.record_seconds

    @property
    def n_epochs(self) -> int:
        return int(self.duration_seconds // EPOCH_SECONDS)

    @property
    def dropped_seconds(self) -> float:
        return self.duration_seconds - self.n_epochs * EPOCH_SECONDS

    def channels(self) -> list[dict]:
        return [{"label": label, "sampling_rate": rate, "unit": unit}
                for label, rate, unit in zip(self.labels, self.sampling_rates, self.units,
                                             strict=True)]


def read_header(path: Path) -> EdfHeader:
    size = path.stat().st_size
    with path.open("rb") as handle:
        fixed = handle.read(256)
        try:
            if len(fixed) < 256 or fixed[:8].decode("ascii").strip() != "0":
                raise ValueError("missing EDF version field")
            header_bytes = int(fixed[184:192])
            reserved = fixed[192:236].decode("ascii")
            n_records = int(fixed[236:244])
            record_seconds = float(fixed[244:252])
            n_signals = int(fixed[252:256])
            if not 0 < n_signals <= 512 or header_bytes != 256 * (n_signals + 1):
                raise ValueError("inconsistent signal count")
            if record_seconds <= 0:
                raise ValueError("data record duration must be positive")
            block = handle.read(256 * n_signals)
            if len(block) < 256 * n_signals:
                raise ValueError("header is truncated")

            def column(offset: int, width: int) -> list[str]:
                start = offset * n_signals
                return [block[start + i * width:start + (i + 1) * width].decode("latin-1").strip()
                        for i in range(n_signals)]

            labels = column(0, 16)
            units = column(16 + 80, 8)
            samples_per_record = [int(value) for value in column(16 + 80 + 8 * 5 + 80, 8)]
        except (ValueError, UnicodeDecodeError) as error:
            raise EdfError("EDF_INVALID", f"Not a valid EDF file: {error}.") from error

    record_bytes = 2 * sum(samples_per_record)
    if n_records < 0:  # Recorder never finalised the header; trust the file size.
        n_records = (size - header_bytes) // record_bytes
    if size < header_bytes + n_records * record_bytes:
        raise EdfError("EDF_TRUNCATED", "The file is shorter than its header declares.")
    return EdfHeader(
        labels=labels, units=units,
        sampling_rates=[count / record_seconds for count in samples_per_record],
        n_records=n_records, record_seconds=record_seconds,
        discontinuous=reserved.startswith("EDF+D"),
    )


def validate_recording(header: EdfHeader) -> None:
    """Accept only what the A2 models were trained on: both EEG derivations at 100 Hz."""
    if header.discontinuous:
        raise EdfError("EDF_DISCONTINUOUS", "Discontinuous EDF+D recordings are not supported.")
    missing = [channel for channel in CHANNELS if channel not in header.labels]
    if missing:
        raise EdfError("EDF_MISSING_CHANNEL", f"Missing channel(s) {', '.join(missing)}. "
                       f"Found: {', '.join(header.labels)}.")
    for channel in CHANNELS:
        index = header.labels.index(channel)
        if header.sampling_rates[index] != SAMPLING_RATE:
            raise EdfError("EDF_WRONG_SAMPLING_RATE",
                           f"{channel} is sampled at {header.sampling_rates[index]:g} Hz; "
                           f"{SAMPLING_RATE:g} Hz is required.")
        if header.units[index] not in SUPPORTED_UNITS:
            raise EdfError("EDF_UNSUPPORTED_UNIT",
                           f"{channel} has unit {header.units[index]!r}; expected microvolts.")
    if header.n_epochs < 1:
        raise EdfError("RECORDING_TOO_SHORT", "The recording is shorter than one 30 s epoch.")


def parse_hypnogram(path: Path, n_epochs: int) -> dict[int, tuple[int, bool]]:
    """Map epoch index to ``(stage, in_benchmark)`` for every scored 30 s epoch.

    Sleep stages 3 and 4 merge into N3. Movement time and unscored spans are skipped.
    ``in_benchmark`` reproduces the A2 evaluation window (see ``A2/preprocessing.py``):
    from 30 minutes before the first scored sleep to 30 minutes after the last.
    """
    import mne  # Imported lazily: only uploads that include a hypnogram pay for it.

    try:
        annotations = mne.read_annotations(path)
    except Exception as error:
        raise EdfError("HYPNOGRAM_INVALID", f"Could not read hypnogram: {error}") from error
    scored = [(float(onset), float(duration), STAGE_MAP[description])
              for onset, duration, description in
              zip(annotations.onset, annotations.duration, annotations.description, strict=True)
              if description in STAGE_MAP]
    if not scored:
        raise EdfError("HYPNOGRAM_INVALID", "The hypnogram contains no sleep-stage annotations.")
    if any(onset % EPOCH_SECONDS or duration % EPOCH_SECONDS for onset, duration, _ in scored):
        raise EdfError("HYPNOGRAM_MISALIGNED",
                       "Hypnogram annotations do not fall on the 30 s epoch grid.")

    sleep = [(onset, duration) for onset, duration, stage in scored if stage != 0]
    first = min((onset for onset, _ in sleep), default=0.0) - BENCHMARK_MARGIN_SECONDS
    last = max((onset + duration for onset, duration in sleep), default=0.0)
    last += BENCHMARK_MARGIN_SECONDS
    stages: dict[int, tuple[int, bool]] = {}
    for onset, duration, stage in scored:
        for index in range(int(onset // EPOCH_SECONDS), int((onset + duration) // EPOCH_SECONDS)):
            if index >= n_epochs:
                break
            start = index * EPOCH_SECONDS
            in_benchmark = bool(sleep) and start >= first and start + EPOCH_SECONDS <= last
            stages[index] = (stage, in_benchmark)
    if not stages:
        raise EdfError("HYPNOGRAM_INVALID", "No scored epoch overlaps the recording.")
    return stages
