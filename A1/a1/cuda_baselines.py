"""Train, evaluate and verify the submitted CUDA Linear/MLP baselines.

Run from A1: python -m a1.cuda_baselines
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from .data import build_dataloaders
from .evaluate import evaluate_checkpoint
from .train import _config_hash, load_checkpoint, run_epoch, run_experiment


def verify(model_name, run_dir):
    summary = json.loads((run_dir / "summary.json").read_text())
    test = json.loads((run_dir / "test_summary.json").read_text())
    config = summary["config"]
    assert config == json.loads(Path("configs/m1.json").read_text())
    assert config["device"] == summary["device"] == test["hardware"]["device"] == "cuda"
    assert summary["hardware"]["gpu"] == test["hardware"]["gpu"] == "NVIDIA GeForce GTX 1660 SUPER"
    train, val, loader = build_dataloaders(
        mode="flatten", batch_size=config["batch_size"], seed=config["seed"],
        val_ratio=config["val_ratio"], data_dir=config["data_dir"],
        split_cache_path=config["split_cache_path"], num_workers=config["num_workers"],
    )
    assert (len(train.dataset), len(val.dataset), len(loader.dataset)) == (54000, 6000, 10000)
    model, checkpoint = load_checkpoint(run_dir / "best.pt", "cuda")
    assert checkpoint["config"] == config
    assert checkpoint["config_sha256"] == summary["config_sha256"] == _config_hash(config, model_name)
    assert checkpoint["source_sha256"] == summary["source_sha256"] == test["source_sha256"] == val.source_sha256
    assert checkpoint["split_sha256"] == summary["split_sha256"] == test["split_sha256"] == val.split_sha256
    assert checkpoint["train_mean"] == summary["train_mean"] == val.dataset.mean
    assert checkpoint["train_std"] == summary["train_std"] == val.dataset.std
    measured = run_epoch(model, val, nn.CrossEntropyLoss(), torch.device("cuda"))
    for key in ("loss", "accuracy", "macro_f1"):
        assert abs(measured[key] - summary[f"best_val_{key}"]) < 1e-5
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == config["epochs"] == 10
    assert int(min(history, key=lambda r: float(r["val_loss"]))["epoch"]) == summary["best_epoch"] == checkpoint["epoch"] == test["checkpoint_epoch"]
    assert hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest() == test["checkpoint_sha256"]
    with (run_dir / "test_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["test_index"]) for r in rows] == list(range(10000))
    labels = np.array([int(r["true_label"]) for r in rows])
    predicted = np.array([int(r["predicted_label"]) for r in rows])
    np.testing.assert_array_equal(labels, loader.dataset.base.targets.numpy())
    predictions, confidence = [], []
    loss = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.cuda())
            loss += nn.functional.cross_entropy(logits, targets.cuda(), reduction="sum").item()
            predictions.extend(logits.argmax(1).cpu().tolist())
            confidence.extend(logits.softmax(1).max(1).values.cpu().tolist())
    np.testing.assert_array_equal(predicted, predictions)
    np.testing.assert_allclose([float(r["confidence"]) for r in rows], confidence, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal([int(r["correct"]) for r in rows], (labels == predicted).astype(int))
    assert abs(loss / len(rows) - test["test_loss"]) < 1e-5
    assert accuracy_score(labels, predicted) == test["test_accuracy"]
    assert abs(f1_score(labels, predicted, average="macro") - test["test_macro_f1"]) < 1e-12
    saved = np.loadtxt(run_dir / "confusion_matrix.csv", delimiter=",", skiprows=1, usecols=range(1, 11), dtype=int)
    np.testing.assert_array_equal(saved, confusion_matrix(labels, predicted))
    cpu, _ = load_checkpoint(run_dir / "best.pt", "cpu")
    images, _ = next(iter(loader))
    with torch.inference_mode():
        gpu_logits, cpu_logits = model(images.cuda()).cpu(), cpu(images)
    torch.testing.assert_close(cpu_logits, gpu_logits, rtol=1e-4, atol=1e-4)
    assert torch.equal(cpu_logits.argmax(1), gpu_logits.argmax(1))
    for name in ("learning_curves", "confusion_matrix", "correct_examples", "incorrect_examples"):
        with Image.open(run_dir / f"{name}.png") as picture:
            picture.verify()
    result = {"status": "passed", "model": model_name, "device": "cuda",
              "test_prediction_rows": len(rows), "test_predictions_match_checkpoint": True,
              "test_metrics_and_confusion_recomputed": True, "split_matches_shared_cache": True,
              "checkpoint_selection_checked": True, "cpu_device_predictions_match": True,
              "cpu_device_max_logit_difference": (cpu_logits - gpu_logits).abs().max().item(),
              "png_files_valid": 4}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the submitted baseline pipeline")
    for name in ("linear", "mlp"):
        summary = run_experiment(name, "configs/m1.json", reuse_existing=True)
        run_dir = Path(summary["checkpoint"]).parent
        evaluate_checkpoint(run_dir / "best.pt", "cuda")
        verify(name, run_dir)


if __name__ == "__main__":
    main()
