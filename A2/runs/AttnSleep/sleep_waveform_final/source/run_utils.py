"""Small file-writing helpers for the notebook; no training CLI or experiment framework."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def start_run(run_dir, config, dataset, device):
    """Create a NEW run and record the actual input, environment, code and splits."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)  # A new name prevents mixing old folds.
    (run_dir / "checkpoints").mkdir()
    save_json(run_dir / "config.json", config)
    data_path = Path(config["data_path"])
    print("Recording dataset SHA-256 (one sequential read; no tensor copy)...", flush=True)
    data_hash = sha256(data_path)
    raw = torch.load(data_path, map_location="cpu", weights_only=True, mmap=True)
    pd.DataFrame({
        "source_index": np.arange(len(raw["labels"])),
        "participant": raw["groups"], "recording": raw["recordings"],
        "onset": raw["onsets"].numpy(), "label": raw["labels"].numpy(),
        "split": raw["splits"], "validation_fold": raw["validation_folds"].numpy(),
    }).to_csv(run_dir / "samples.csv", index=False)
    save_json(run_dir / "preprocessing.json", {
        "config": raw["preprocessing"], "summary": raw.get("summary"),
        "sampling": raw.get("sampling"),
    })
    project = Path(__file__).resolve().parent
    source_dir = run_dir / "source"
    source_dir.mkdir()
    source_hashes = {}
    for name in ["training.ipynb", "run_utils.py", "model_lib.py", "attnsleep.py",
                 "cbramod.py", "criss_cross_transformer.py", "preprocessing.py",
                 "requirements.txt", "requirements-eeg.txt",
                 "sleep_exploration/cv_folds.csv", "sleep_exploration/cv_metadata.json"]:
        source = project / name
        if source.is_file():
            target = source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[name] = sha256(source)
    if config.get("config_file"):
        selected_config = Path(config["config_file"])
        target = source_dir / "configs" / selected_config.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(selected_config, target)
        source_hashes[f"configs/{selected_config.name}"] = sha256(selected_config)
    packages = sorted({f"{d.metadata['Name']}=={d.version}"
                       for d in importlib.metadata.distributions() if d.metadata['Name']})
    (run_dir / "environment.txt").write_text("\n".join(packages) + "\n")
    git = shutil.which("git")
    commit, status = None, None
    if git:
        head = subprocess.run([git, "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode == 0:
            commit = head.stdout.strip()
            status = subprocess.run([git, "-C", str(project), "status", "--short"], capture_output=True, text=True).stdout
    (run_dir / "git_status.txt").write_text(status if status is not None else "Git checkout unavailable\n")
    pretrained = config["model_kwargs"].get("checkpoint_path")
    metadata = {
        "layout_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "training",
        "completed_folds": [], "git_commit": commit,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "source_sha256": source_hashes,
        "fold_manifest": "sleep_exploration/cv_folds.csv",
        "fold_manifest_sha256": source_hashes.get("sleep_exploration/cv_folds.csv"),
        "dataset": {"path": str(data_path.resolve()), "bytes": data_path.stat().st_size,
                    "sha256": data_hash, "shape": list(raw["patches"].shape),
                    "development_samples": len(dataset),
                    "test_samples": raw["splits"].count("test"),
                    "channels": raw["channels"], "sfreq": raw["sfreq"],
                    "class_map": raw["class_map"]},
        "pretrained_sha256": sha256(pretrained) if pretrained else None,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark},
    }
    save_json(run_dir / "run.json", metadata)
    print("Run directory:", run_dir.resolve())
    return metadata


def save_predictions(path, dataset, indices, targets, probabilities, folds=None):
    """Keep source identities and all five probabilities for later error analysis."""
    probabilities = probabilities.detach().float().cpu().numpy()
    rows = pd.DataFrame({
        "source_index": dataset.indices[indices],
        "participant": dataset.groups[indices],
        "recording": dataset.recordings[indices], "onset": dataset.onsets[indices],
        "target": np.asarray(targets), "prediction": probabilities.argmax(axis=1),
    })
    if folds is not None:
        rows["validation_fold"] = folds
    for stage, label in dataset.class_to_idx.items():
        rows[f"prob_{stage}"] = probabilities[:, label]
    rows.to_csv(path, index=False)
