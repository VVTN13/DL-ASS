import struct

import pytest

from app.services.edf import EdfError, parse_hypnogram, read_header, validate_recording
from tests.conftest import SAMPLE_HYPNOGRAM, SAMPLE_PSG


def write_edf(path, labels=("EEG Fpz-Cz", "EEG Pz-Oz"), samples_per_record=(3000, 3000),
              units=("uV", "uV"), n_records=2, record_seconds=30, reserved="", body=True):
    count = len(labels)

    def column(values, width):
        return "".join(str(value).ljust(width) for value in values)

    header = ("0".ljust(8) + "X".ljust(80) + "X".ljust(80) + "01.01.00" + "00.00.00"
              + str(256 * (count + 1)).ljust(8) + reserved.ljust(44) + str(n_records).ljust(8)
              + str(record_seconds).ljust(8) + str(count).ljust(4)
              + column(labels, 16) + column([""] * count, 80) + column(units, 8)
              + column([-100] * count, 8) + column([100] * count, 8)
              + column([-2048] * count, 8) + column([2047] * count, 8)
              + column([""] * count, 80) + column(samples_per_record, 8)
              + column([""] * count, 32))
    data = struct.pack("<h", 0) * sum(samples_per_record) * n_records if body else b""
    path.write_bytes(header.encode("ascii") + data)
    return path


def error_code(path) -> str:
    with pytest.raises(EdfError) as caught:
        validate_recording(read_header(path))
    return caught.value.code


def test_sample_header():
    header = read_header(SAMPLE_PSG)
    assert header.labels == ["EEG Fpz-Cz", "EEG Pz-Oz"]
    assert header.sampling_rates == [100.0, 100.0]
    assert header.units == ["uV", "uV"]
    assert (header.duration_seconds, header.n_epochs, header.dropped_seconds) == (3600, 120, 0)
    validate_recording(header)


def test_trailing_partial_epoch_is_reported(tmp_path):
    header = read_header(write_edf(tmp_path / "a.edf", samples_per_record=(100, 100),
                                   record_seconds=1, n_records=75))
    assert (header.n_epochs, header.dropped_seconds) == (2, 15)


def test_unknown_record_count_uses_file_size(tmp_path):
    path = write_edf(tmp_path / "a.edf", n_records=3)
    content = bytearray(path.read_bytes())
    content[236:244] = b"-1      "
    path.write_bytes(content)
    assert read_header(path).n_records == 3


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"labels": ("EEG Fpz-Cz", "EOG horizontal")}, "EDF_MISSING_CHANNEL"),
    ({"samples_per_record": (3000, 7680)}, "EDF_WRONG_SAMPLING_RATE"),
    ({"units": ("uV", "DegC")}, "EDF_UNSUPPORTED_UNIT"),
    ({"reserved": "EDF+D"}, "EDF_DISCONTINUOUS"),
    ({"body": False}, "EDF_TRUNCATED"),
    ({"samples_per_record": (100, 100), "record_seconds": 1, "n_records": 29},
     "RECORDING_TOO_SHORT"),
])
def test_unusable_recordings_are_rejected(tmp_path, kwargs, code):
    assert error_code(write_edf(tmp_path / "a.edf", **kwargs)) == code


@pytest.mark.parametrize("content", [b"", b"not an edf file" * 40, b"%PDF-1.7" + b" " * 600])
def test_non_edf_content_is_rejected(tmp_path, content):
    path = tmp_path / "a.edf"
    path.write_bytes(content)
    assert error_code(path) == "EDF_INVALID"


def test_hypnogram_stages_on_the_epoch_grid():
    stages = parse_hypnogram(SAMPLE_HYPNOGRAM, n_epochs=120)
    assert len(stages) == 120  # the excerpt is fully scored
    assert [stages[i][0] for i in (0, 2, 3, 6, 7)] == [0, 0, 1, 1, 2]  # W W | N1 ... | N2
    assert {stage for stage, _ in stages.values()} == {0, 1, 2, 3, 4}
    assert all(in_benchmark for _, in_benchmark in stages.values())


def test_hypnogram_is_clipped_to_the_recording():
    assert max(parse_hypnogram(SAMPLE_HYPNOGRAM, n_epochs=50)) == 49


def test_benchmark_window_is_thirty_minutes_around_sleep(tmp_path):
    from edfio import Edf, EdfAnnotation

    # 2 h wake, 30 min N2, movement, 2 h wake: only wake within 30 min of sleep is in scope.
    path = tmp_path / "h.edf"
    Edf([], annotations=[
        EdfAnnotation(0, 7200, "Sleep stage W"), EdfAnnotation(7200, 1800, "Sleep stage 2"),
        EdfAnnotation(9000, 30, "Movement time"), EdfAnnotation(9030, 7200, "Sleep stage W"),
        EdfAnnotation(16230, 60, "Sleep stage ?"),
    ]).write(path)
    stages = parse_hypnogram(path, n_epochs=600)
    assert 300 not in stages and 541 not in stages  # movement and unscored epochs
    in_benchmark = sorted(index for index, (_, flag) in stages.items() if flag)
    assert (in_benchmark[0], in_benchmark[-1]) == (180, 359)  # 5400 s .. 10800 s
    assert stages[0] == (0, False) and stages[240] == (2, True)


def test_misaligned_hypnogram_is_rejected(tmp_path):
    from edfio import Edf, EdfAnnotation

    path = tmp_path / "h.edf"
    Edf([], annotations=[EdfAnnotation(10, 30, "Sleep stage 2")]).write(path)
    with pytest.raises(EdfError) as caught:
        parse_hypnogram(path, n_epochs=10)
    assert caught.value.code == "HYPNOGRAM_MISALIGNED"


def test_file_without_stage_annotations_is_not_a_hypnogram():
    with pytest.raises(EdfError) as caught:
        parse_hypnogram(SAMPLE_PSG, n_epochs=120)
    assert caught.value.code == "HYPNOGRAM_INVALID"
