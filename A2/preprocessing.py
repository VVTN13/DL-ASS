"""Prepare Sleep-EDF waveforms for CBraMod, independently of CNN spectrograms.

Run alongside this script: ``python preprocessing.py``.
Input and output paths are relative to this script's directory.
Selects up to 83 epochs per participant directly from EDF annotations.
Writes dataset/sleep_waveform_small.pt and a matching selection CSV.
Each sample is (2 EEG channels, 30 one-second patches, 200 samples).
"""

import argparse
import csv
import hashlib
import json
import platform
import tempfile
import time
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import scipy
import torch
from scipy.signal import resample_poly


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sleep-cassette"
OUTPUT_DIR = BASE_DIR / "dataset" / "sleep_cbramod"
OUTPUT_FILE = BASE_DIR / "dataset" / "sleep_waveform_small.pt"
FOLDS_PATH = BASE_DIR / "sleep_exploration" / "cv_folds.csv"
REFERENCE_PATH = BASE_DIR / "dataset" / "sleep_edf.pt"
CHECKSUMS_PATH = BASE_DIR / "SHA256SUMS.txt"
SAMPLES_PER_PARTICIPANT = 83
SEED = 42
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
CLASS_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
CONFIG = {
    "representation": "waveform", "channels": CHANNELS, "class_map": CLASS_MAP,
    "source_sfreq": 100, "target_sfreq": 200, "epoch_seconds": 30,
    "wake_margin_seconds": 1800, "sample_shape": [2, 30, 200],
    "filter": {"low_hz": 0.3, "high_hz": 35.0, "method": "fir",
               "phase": "zero", "fir_window": "hamming", "fir_design": "firwin",
               "filter_length": "auto", "l_trans_bandwidth": 0.3,
               "h_trans_bandwidth": 8.75, "pad": "reflect_limited"},
    "resample": {"method": "scipy.signal.resample_poly", "up": 2, "down": 1,
                 "window": ["kaiser", 5.0], "padtype": "line"},
    "amplitude": "microvolts / 100", "volts_multiplier": 10000.0,
    "population_normalization": None, "notch_filter": None,
    "filter_scope": "continuous recording before epoch extraction",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_folds(path):
    """Read saved participant assignments; never split or stratify again here."""
    folds = {}
    with Path(path).open(newline="") as source:
        for row in csv.DictReader(source):
            person, split, fold = row["participant"], row["split"], int(row["validation_fold"])
            valid = (split == "development" and 1 <= fold <= 5) or (split == "test" and fold == -1)
            if person in folds or not valid:
                raise ValueError(f"Invalid/duplicate participant assignment: {row}")
            folds[person] = {"split": split, "validation_fold": fold}
    if not folds:
        raise ValueError("The participant fold manifest is empty.")
    return folds


def select_epochs(annotations, recording_seconds):
    """Use the CNN export's annotation grid, stage merge, and 30-minute margins."""
    sleep = [a for a in annotations if a["description"] in STAGE_MAP
             and a["description"] != "Sleep stage W"]
    if not sleep:
        raise ValueError("No scored sleep intervals in this hypnogram.")
    first = max(0.0, min(a["onset"] for a in sleep) - 1800)
    last = min(recording_seconds, max(a["onset"] + a["duration"] for a in sleep) + 1800)
    epochs = []
    for annotation in annotations:
        description = annotation["description"]
        if description not in STAGE_MAP:
            continue  # Movement and unscored annotations are not classification samples.
        onset, duration = float(annotation["onset"]), float(annotation["duration"])
        if not np.isfinite([onset, duration]).all() or duration < 0:
            raise ValueError("Invalid annotation onset/duration.")
        end = onset + duration
        for start in np.arange(onset, end, 30.0):
            if start >= first and start + 30 <= min(end, last):
                if not np.isclose(start * 100, round(start * 100), atol=1e-6, rtol=0):
                    raise ValueError(f"Epoch is not aligned with a source sample: {start}")
                epochs.append((float(start), STAGE_MAP[description]))
    epochs.sort()
    if not epochs:
        raise ValueError("No complete 30-second epochs remain.")
    if any(b[0] < a[0] + 30 - 1e-8 for a, b in zip(epochs, epochs[1:])):
        raise ValueError("Duplicate or overlapping scored epochs.")
    return epochs


def sample_participant_epochs(candidates):
    """Share each person's 83-epoch quota across their available nights/stages."""
    rng = np.random.default_rng(SEED)
    selected = {recording: [] for recording in sorted(candidates)}
    for person in sorted({recording[:5] for recording in candidates}):
        nights = sorted(recording for recording in candidates if recording[:5] == person)
        total = sum(len(candidates[night]) for night in nights)
        if total <= SAMPLES_PER_PARTICIPANT:
            for night in nights:
                selected[night] = list(candidates[night])
            continue
        strata = [(night, [epoch for epoch in candidates[night] if epoch[1] == stage])
                  for night in nights for stage in CLASS_MAP.values()]
        strata = [(night, epochs) for night, epochs in strata if epochs]
        counts = np.array([len(epochs) for _, epochs in strata])
        # Keep every available night/stage; allocate the rest proportionally.
        extra = (SAMPLES_PER_PARTICIPANT - len(strata)) * (counts - 1) / (counts - 1).sum()
        quotas = 1 + np.floor(extra).astype(int)
        remainder = SAMPLES_PER_PARTICIPANT - int(quotas.sum())
        order = np.argsort(-(extra - np.floor(extra)), kind="stable")
        quotas[order[:remainder]] += 1
        for (night, epochs), count in zip(strata, quotas):
            indices = rng.choice(len(epochs), size=int(count), replace=False)
            selected[night].extend(epochs[i] for i in indices)
    return {night: sorted(epochs) for night, epochs in selected.items()}


def load_reference(path):
    """Read only the CNN export's metadata; its spectrograms are never model input."""
    if path is None:
        return None
    reference = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if reference["class_map"] != CLASS_MAP or reference["channels"] != CHANNELS:
        raise ValueError("CNN reference has a different stage map or channel order.")
    fields = [reference[k] for k in ("groups", "recordings", "onsets", "labels")]
    if len({len(field) for field in fields}) != 1:
        raise ValueError("CNN reference metadata lengths differ.")
    by_recording = {}
    for person, recording, onset, label in zip(*fields):
        if person != recording[:5]:
            raise ValueError(f"CNN reference participant mismatch: {recording}")
        by_recording.setdefault(recording, []).append((float(onset), int(label)))
    for recording, epochs in by_recording.items():
        epochs.sort()
        if len({onset for onset, _ in epochs}) != len(epochs):
            raise ValueError(f"Duplicate CNN reference epochs: {recording}")
    return by_recording


def check_reference(recording, epochs, reference):
    if reference is not None and reference.get(recording) != epochs:
        raise ValueError(
            f"{recording}: waveform epoch onsets/labels differ from the CNN reference. "
            "Resolve alignment/QC differences before comparing models."
        )


def prepare_waveforms(raw):
    """Filter/resample one continuous recording; return float32 EEG in microvolts/100."""
    if set(raw.ch_names) != set(CHANNELS) or raw.first_samp != 0:
        raise ValueError("Expected the two EEG channels and an uncropped recording.")
    if raw.info["sfreq"] != 100:
        raise ValueError(f"Expected Sleep Cassette at 100 Hz, got {raw.info['sfreq']}.")
    raw.reorder_channels(CHANNELS)
    raw.load_data(verbose="ERROR")
    # Fail instead of contaminating an entire recording by filtering NaN/Inf samples.
    for start in range(0, raw.n_times, 60000):
        if not np.isfinite(raw.get_data(start=start, stop=min(start + 60000, raw.n_times))).all():
            raise ValueError("Nonfinite source signal; resolve QC before exporting either model's data.")
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
        raise ValueError("Nonfinite signal after filtering/resampling/scaling.")
    return signal


def make_patches(signal, epochs):
    """Keep channel/time ordering and extract exactly 6000 samples per epoch."""
    patches = np.empty((len(epochs), 2, 30, 200), dtype=np.float32)
    for index, (onset, _) in enumerate(epochs):
        start = round(onset * 200)
        epoch = signal[:, start:start + 6000]
        if epoch.shape != (2, 6000):
            raise ValueError(f"Epoch at {onset}s extends outside the waveform.")
        patches[index] = epoch.reshape(2, 30, 200)
    return torch.from_numpy(patches)


def preprocess_shards(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, *, recordings=None,
                      folds_path=FOLDS_PATH, reference_path=None, checksums_path=None,
                      overwrite=False):
    """Export waveform shards. Optional reference/checksums enable stricter validation.

    ``recordings`` is a list such as ["SC4001"] for a pilot. The CLI automatically
    uses the existing CNN metadata and archive checksums when present.
    """
    started = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    psgs = {p.name[:6]: p for p in sorted(data_dir.glob("SC*-PSG.edf"))}
    if not psgs:
        raise FileNotFoundError(f"No Sleep Cassette PSG files in {data_dir}")
    selected = sorted(psgs if recordings is None else set(recordings))
    if not selected or set(selected) - set(psgs):
        raise ValueError(f"Unknown or empty recording selection: {selected}")
    folds = load_folds(folds_path)
    reference = load_reference(reference_path)
    checksums = {}
    if checksums_path is not None:
        for line in Path(checksums_path).read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            checksums[Path(name.lstrip('*')).name] = digest
    if recordings is None:
        missing_people = set(folds) - {recording[:5] for recording in selected}
        assert not missing_people, f"Missing participant EDFs: {sorted(missing_people)}"
        expected = {name for name in checksums if name.startswith("SC") and name.endswith("-PSG.edf")}
        missing_nights = expected - {path.name for path in psgs.values()}
        assert not missing_nights, f"Missing PSG files: {sorted(missing_nights)}"
    targets = [output_dir / f"{recording}.pt" for recording in selected]
    targets += [output_dir / "index.csv", output_dir / "summary.json"]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"Output already exists in {output_dir}; choose a new directory or use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # A failed replacement must not leave an old 'complete' summary/index in place.
    for name in ("index.csv", "summary.json"):
        (output_dir / name).unlink(missing_ok=True)

    # First pass reads headers/annotations only; no waveform tensors are created.
    candidates, details = {}, {}
    print(f"Reading EDF annotations: {len(selected)} recordings", flush=True)
    for recording in selected:
        psg = psgs[recording]
        hypotheses = sorted(data_dir.glob(recording + "*-Hypnogram.edf"))
        if len(hypotheses) != 1:
            raise ValueError(f"{recording}: expected exactly one matching hypnogram.")
        hypnogram = hypotheses[0]
        if recording[:5] not in folds:
            raise ValueError(f"{recording[:5]} is absent from the saved fold manifest.")
        hashes = {path.name: sha256(path) for path in (psg, hypnogram)}
        if checksums_path is not None:
            for name, digest in hashes.items():
                if checksums.get(name) != digest:
                    raise ValueError(f"Archive checksum missing or mismatched: {name}")
        with psg.open("rb") as first, hypnogram.open("rb") as second:
            if first.read(184)[168:184] != second.read(184)[168:184]:
                raise ValueError(f"{recording}: PSG/hypnogram start date/time differ.")
        raw = mne.io.read_raw_edf(psg, include=CHANNELS, preload=False, verbose="ERROR")
        try:
            annotations = mne.read_annotations(hypnogram)
            if annotations.orig_time is not None and annotations.orig_time != raw.info["meas_date"]:
                raise ValueError(f"{recording}: annotation origin differs from PSG start.")
            candidates[recording] = select_epochs(annotations, raw.n_times / raw.info["sfreq"])
            # Compare all eligible epochs with the optional CNN reference before sampling.
            check_reference(recording, candidates[recording], reference)
        finally:
            raw.close()
        details[recording] = {"source_sha256": hashes,
                              "annotation_interval_counts": dict(Counter(annotations.description))}
    sampled = sample_participant_epochs(candidates)
    print(f"Selected {sum(map(len, sampled.values())):,} epochs: up to {SAMPLES_PER_PARTICIPANT} per participant (seed {SEED})", flush=True)
    summary = {
        "status": "in_progress", "config": CONFIG, "recordings": [],
        "sampling": {"samples_per_participant": SAMPLES_PER_PARTICIPANT, "seed": SEED,
                     "source": "EDF annotations before waveform patch extraction",
                     "method": "proportional night/stage strata; minimum one; without replacement",
                     "eligible_epochs": sum(map(len, candidates.values()))},
        "fold_manifest_sha256": sha256(folds_path), "source_code_sha256": sha256(__file__),
        "cnn_reference": str(Path(reference_path).resolve()) if reference_path else None,
        "cnn_reference_checked": reference is not None,
        "archive_checksums_checked": checksums_path is not None,
        "versions": {"python": platform.python_version(), "mne": mne.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
    }
    counts = Counter()
    fields = ["shard", "row", "participant", "recording", "onset", "label", "stage", "split", "validation_fold"]
    index_tmp = output_dir / "index.csv.tmp"
    print(f"CBraMod waveform export: {len(selected)} recordings -> {output_dir}", flush=True)
    print(f"CNN epoch comparison: {reference is not None}; archive checksums: {checksums_path is not None}", flush=True)
    with index_tmp.open("w", newline="") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=fields)
        writer.writeheader()
        for number, recording in enumerate(selected, start=1):
            tick = time.perf_counter()
            psg = psgs[recording]
            person = recording[:5]
            hashes = details[recording]["source_sha256"]
            epochs = sampled[recording]
            raw = mne.io.read_raw_edf(psg, include=CHANNELS, preload=False, verbose="ERROR")
            try:
                print(f"[{number}/{len(selected)}] {recording}: filtering continuous EEG, exporting {len(epochs)} selected epochs", flush=True)
                signal = prepare_waveforms(raw)
            finally:
                raw.close()
            patches = make_patches(signal, epochs)
            del signal, raw
            labels = torch.tensor([label for _, label in epochs], dtype=torch.long)
            shard = {
                "patches": patches, "labels": labels,
                "groups": [person] * len(epochs), "recordings": [recording] * len(epochs),
                "onsets": torch.tensor([onset for onset, _ in epochs], dtype=torch.float64),
                "participant": person, "recording": recording, **folds[person],
                "class_map": CLASS_MAP, "channels": CHANNELS, "sfreq": 200,
                "preprocessing": CONFIG, "source_sha256": hashes,
            }
            shard_path = output_dir / f"{recording}.pt"
            temporary = shard_path.with_suffix(".pt.tmp")
            torch.save(shard, temporary)
            temporary.replace(shard_path)
            per_class = Counter(label for _, label in epochs)
            counts.update(per_class)
            for row, (onset, label) in enumerate(epochs):
                writer.writerow({"shard": shard_path.name, "row": row, "participant": person,
                                 "recording": recording, "onset": onset, "label": label,
                                 "stage": list(CLASS_MAP)[label], **folds[person]})
            elapsed = time.perf_counter() - tick
            summary["recordings"].append({
                "recording": recording, "participant": person, "epochs": len(epochs),
                "eligible_epochs": len(candidates[recording]),
                "class_counts": {stage: per_class[label] for stage, label in CLASS_MAP.items()},
                "source_sha256": hashes, "shard_sha256": sha256(shard_path), "seconds": elapsed,
                "annotation_interval_counts": details[recording]["annotation_interval_counts"], **folds[person],
            })
            print(f"[{number}/{len(selected)}] saved {shard_path.name} {tuple(patches.shape)} ({elapsed:.1f}s)", flush=True)
            del shard, patches, labels
    summary.update({"status": "complete", "epochs": sum(counts.values()),
                    "participants": len({recording[:5] for recording in selected}),
                    "class_counts": {stage: counts[label] for stage, label in CLASS_MAP.items()},
                    "seconds": time.perf_counter() - started})
    index_tmp.replace(output_dir / "index.csv")
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output_dir / "summary.json")
    print(f"Done: {summary['epochs']:,} waveform epochs. Index: {output_dir / 'index.csv'}", flush=True)
    return summary


def merge_shards(shard_dir, output_file=OUTPUT_FILE, *, overwrite=False):
    """Bundle a completed export without reading EDFs or changing its samples.

    Uses a temporary disk-backed array rather than retaining all shards and then
    torch.cat-ing a second full copy in RAM. Metadata/splits are embedded in .pt.
    """
    shard_dir, output_file = Path(shard_dir), Path(output_file)
    if output_file.suffix != ".pt":
        raise ValueError("The combined output file must have a .pt extension.")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists; choose another --output-file or use --overwrite.")
    summary = json.loads((shard_dir / "summary.json").read_text())
    records = summary["recordings"]
    total = summary["epochs"]
    if summary["status"] != "complete" or not records or total <= 0:
        raise ValueError("Only a completed, nonempty waveform export can be merged.")
    names = [record["recording"] for record in records]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("Duplicate or invalid recording names in the export summary.")
    if sum(record["epochs"] for record in records) != total:
        raise ValueError("Recording sample counts do not match the export total.")
    if any(output_file.resolve() == (shard_dir / f"{name}.pt").resolve() for name in names):
        raise ValueError("The combined output must not overwrite an input shard.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Combining {len(records)} recordings / {total:,} epochs -> {output_file}", flush=True)
    # TemporaryDirectory removes intermediates even if validation or saving fails.
    # Existing final output is replaced only after the new file is fully written.
    with tempfile.TemporaryDirectory(prefix=".sleep_waveform-merge-", dir=output_file.parent) as tmp:
        mapped = np.memmap(Path(tmp) / "patches.bin", mode="w+", dtype=np.float32,
                           shape=(total, 2, 30, 200))
        dataset = {
            "patches": torch.from_numpy(mapped),
            "labels": torch.empty(total, dtype=torch.long),
            "onsets": torch.empty(total, dtype=torch.float64),
            "groups": [], "recordings": [], "splits": [],
            "validation_folds": torch.empty(total, dtype=torch.long),
            "class_map": CLASS_MAP, "channels": CHANNELS, "sfreq": 200,
            "preprocessing": summary["config"], "summary": summary,
            "sampling": summary.get("sampling"),
            "format_version": 1,
        }
        counts = Counter()
        participant_folds = {}
        offset = 0
        try:
            for number, record in enumerate(records, start=1):
                name, count = record["recording"], record["epochs"]
                path = shard_dir / f"{name}.pt"
                if sha256(path) != record["shard_sha256"]:
                    raise ValueError(f"Shard checksum mismatch: {path}")
                shard = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if (tuple(shard["patches"].shape) != (count, 2, 30, 200)
                        or shard["patches"].dtype != torch.float32
                        or shard["labels"].dtype != torch.int64
                        or shard["preprocessing"] != summary["config"]
                        or shard["class_map"] != CLASS_MAP or shard["channels"] != CHANNELS
                        or shard["sfreq"] != 200):
                    raise ValueError(f"Incompatible waveform shard: {name}")
                if any(len(shard[field]) != count for field in ("labels", "onsets", "groups", "recordings")):
                    raise ValueError(f"Metadata length mismatch: {name}")
                person, split, fold = shard["participant"], shard["split"], shard["validation_fold"]
                if (shard["recording"] != name or person != name[:5]
                        or set(shard["groups"]) != {person} or set(shard["recordings"]) != {name}
                        or (person, split, fold) != (record["participant"], record["split"], record["validation_fold"])):
                    raise ValueError(f"Participant/recording metadata mismatch: {name}")
                valid_split = (split == "development" and 1 <= fold <= 5) or (split == "test" and fold == -1)
                if not valid_split or participant_folds.setdefault(person, (split, fold)) != (split, fold):
                    raise ValueError(f"Inconsistent participant split: {person}")
                if (not torch.isfinite(shard["patches"]).all()
                        or not torch.isfinite(shard["onsets"]).all()
                        or torch.any(shard["onsets"] < 0)
                        or torch.any(shard["onsets"][1:] - shard["onsets"][:-1] < 30 - 1e-8)
                        or torch.any((shard["labels"] < 0) | (shard["labels"] > 4))):
                    raise ValueError(f"Invalid signal/epoch metadata: {name}")
                end = offset + count
                mapped[offset:end] = shard["patches"].numpy()
                dataset["labels"][offset:end] = shard["labels"]
                dataset["onsets"][offset:end] = shard["onsets"]
                dataset["validation_folds"][offset:end] = fold
                dataset["groups"].extend(shard["groups"])
                dataset["recordings"].extend(shard["recordings"])
                dataset["splits"].extend([split] * count)
                counts.update(shard["labels"].tolist())
                mapped.flush()
                offset = end
                del shard
                print(f"[{number}/{len(records)}] combined {name}: {offset:,}/{total:,} epochs", flush=True)
            if ({stage: counts[label] for stage, label in CLASS_MAP.items()} != summary["class_counts"]
                    or len(participant_folds) != summary["participants"]):
                raise ValueError("Combined labels/participants do not match the export summary.")
            temporary = Path(tmp) / "sleep_waveform.pt"
            torch.save(dataset, temporary)
            temporary.replace(output_file)
        finally:
            del dataset
            del mapped
    print(f"Saved one file: {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)", flush=True)
    with (shard_dir / "index.csv").open(newline="") as source, output_file.with_suffix(".csv").open("w", newline="") as target:
        fields = ["participant", "recording", "onset", "label", "stage", "split", "validation_fold"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in csv.DictReader(source))
    print(f"Selection CSV: {output_file.with_suffix('.csv')}", flush=True)
    return summary


def preprocess(data_dir=DATA_DIR, output_file=OUTPUT_FILE, *, recordings=None,
               folds_path=FOLDS_PATH, reference_path=None, checksums_path=None,
               overwrite=False):
    """Process EDFs recording-by-recording, then export only one self-contained .pt."""
    output_file = Path(output_file)
    if output_file.suffix != ".pt":
        raise ValueError("The combined output file must have a .pt extension.")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists; choose another --output-file or use --overwrite.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sleep_waveform-preprocess-", dir=output_file.parent) as tmp:
        preprocess_shards(data_dir, tmp, recordings=recordings, folds_path=folds_path,
                          reference_path=reference_path, checksums_path=checksums_path)
        return merge_shards(tmp, output_file, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--merge-from", type=Path,
                        help="Combine an existing completed shard directory without reprocessing EDFs.")
    parser.add_argument("--folds", type=Path, default=FOLDS_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH if REFERENCE_PATH.is_file() else None,
                        help="Optional CNN tensor file: compare metadata only, never use spectrograms as EEG.")
    parser.add_argument("--checksums", type=Path, default=CHECKSUMS_PATH if CHECKSUMS_PATH.is_file() else None)
    parser.add_argument("--recording", action="append", help="Process only this six-character night ID; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.merge_from:
        if args.recording:
            parser.error("--recording selects EDFs and cannot be combined with --merge-from.")
        merge_shards(args.merge_from, args.output_file, overwrite=args.overwrite)
    else:
        preprocess(args.data_dir, args.output_file, recordings=args.recording, folds_path=args.folds,
                   reference_path=args.reference, checksums_path=args.checksums, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
