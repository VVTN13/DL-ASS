"""Verify a saved run against its checkpoint, data and predictions."""

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from a1.data import build_dataloaders
from a1.models import input_mode, count_parameters
from a1.train import _config_hash, _device, load_checkpoint, run_epoch


def main(model_name="cnn", *, write_result=True):
    baseline = model_name in {"linear", "mlp"}
    run_dir = Path("runs/cuda_baselines" if baseline else "runs/m2") / model_name
    summary = json.loads((run_dir / "summary.json").read_text())
    test = json.loads((run_dir / "test_summary.json").read_text())
    config = summary["config"]
    train, val, test_loader = build_dataloaders(
        mode=input_mode(model_name), batch_size=config["batch_size"], seed=config["seed"],
        val_ratio=config["val_ratio"], data_dir=config["data_dir"],
        split_cache_path=config["split_cache_path"], num_workers=0,
    )
    assert (len(train.dataset), len(val.dataset), len(test_loader.dataset)) == (54000, 6000, 10000)
    assert summary["split_sha256"] == val.split_sha256 == test["split_sha256"]
    for name in ("linear", "mlp"):
        baseline_summary = json.loads((Path("runs/cuda_baselines") / name / "summary.json").read_text())
        assert baseline_summary["split_sha256"] == val.split_sha256
        assert baseline_summary["source_sha256"] == val.source_sha256
    assert not np.intersect1d(train.dataset.indices, val.dataset.indices).size
    assert (train.dataset.mean, train.dataset.std) == (val.dataset.mean, val.dataset.std)

    device = _device("auto")
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    assert checkpoint["model_name"] == summary["model"] == test["model"] == model_name
    expected_mode = input_mode(model_name)
    assert checkpoint["input_mode"] == summary["input_mode"] == expected_mode
    if baseline:
        assert config == json.loads(Path("configs/m1.json").read_text())
        assert config["device"] == summary["device"] == test["hardware"]["device"] == "cuda"
    assert test["input_mode"] == expected_mode
    assert checkpoint["config_sha256"] == summary["config_sha256"] == _config_hash(config, model_name)
    assert checkpoint["source_sha256"] == summary["source_sha256"] == test["source_sha256"] == val.source_sha256
    assert checkpoint["split_sha256"] == val.split_sha256
    assert checkpoint["train_mean"] == summary["train_mean"] == val.dataset.mean
    assert checkpoint["train_std"] == summary["train_std"] == val.dataset.std
    assert count_parameters(model) == summary["parameters"]
    if model_name == "lstm":
        sequence, _ = train.dataset[0]
        index = int(train.dataset.indices[0])
        raw, _ = train.dataset.base[index]
        torch.testing.assert_close(sequence, ((raw - train.dataset.mean) / train.dataset.std)[0])
        assert tuple(sequence.shape) == (28, 28)
    if model_name == "transformer":
        assert config == json.loads(Path("configs/transformer.json").read_text())
        patches, _ = train.dataset[0]
        index = int(train.dataset.indices[0])
        raw, _ = train.dataset.base[index]
        normalized = (raw - train.dataset.mean) / train.dataset.std
        assert tuple(patches.shape) == (49, 16)
        for row in range(7):
            for col in range(7):
                torch.testing.assert_close(
                    patches[7 * row + col].reshape(4, 4),
                    normalized[0, 4 * row:4 * row + 4, 4 * col:4 * col + 4],
                    rtol=0, atol=0,
                )
    measured = run_epoch(model, val, nn.CrossEntropyLoss(), device)
    for metric in ("loss", "accuracy", "macro_f1"):
        assert abs(measured[metric] - summary[f"best_val_{metric}"]) < 1e-5
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    best = min(history, key=lambda row: float(row["val_loss"]))
    assert checkpoint["epoch"] == int(best["epoch"]) == summary["best_epoch"]
    assert len(history) == config["epochs"]

    # Check portability of the submitted CUDA checkpoints.
    torch.set_num_threads(4)
    cpu_model, _ = load_checkpoint(run_dir / "best.pt", "cpu")
    images, _ = next(iter(test_loader))
    with torch.inference_mode():
        gpu_logits = model(images.to(device)).cpu()
        cpu_logits = cpu_model(images)
    torch.testing.assert_close(cpu_logits, gpu_logits, rtol=1e-4, atol=1e-4)
    assert torch.equal(cpu_logits.argmax(1), gpu_logits.argmax(1))

    with (run_dir / "test_predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert [int(row["test_index"]) for row in predictions] == list(range(10000))
    labels = np.array([int(row["true_label"]) for row in predictions])
    predicted = np.array([int(row["predicted_label"]) for row in predictions])
    np.testing.assert_array_equal(labels, test_loader.dataset.base.targets.numpy())
    # Re-run all test images: the CSV must come from this checkpoint, not just
    # agree internally with the JSON and confusion matrix.
    reloaded_predictions, reloaded_confidence = [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for images_batch, targets in test_loader:
            logits = model(images_batch.to(device))
            loss_sum += nn.functional.cross_entropy(logits, targets.to(device), reduction="sum").item()
            reloaded_predictions.extend(logits.argmax(1).cpu().tolist())
            reloaded_confidence.extend(logits.softmax(1).max(1).values.cpu().tolist())
    np.testing.assert_array_equal(predicted, reloaded_predictions)
    np.testing.assert_allclose([float(row["confidence"]) for row in predictions],
                               reloaded_confidence, rtol=1e-5, atol=1e-6)
    assert [int(row["correct"]) for row in predictions] == (labels == predicted).astype(int).tolist()
    assert abs(loss_sum / len(labels) - test["test_loss"]) < 1e-5
    digest = hashlib.sha256()
    digest.update(test_loader.dataset.base.data.numpy().tobytes())
    digest.update(test_loader.dataset.base.targets.numpy().tobytes())
    assert digest.hexdigest() == test["test_source_sha256"]
    assert abs(accuracy_score(labels, predicted) - test["test_accuracy"]) < 1e-12
    assert abs(f1_score(labels, predicted, average="macro") - test["test_macro_f1"]) < 1e-12
    with (run_dir / "confusion_matrix.csv").open(newline="") as stream:
        saved_matrix = np.array([list(map(int, row[1:])) for row in list(csv.reader(stream))[1:]])
    np.testing.assert_array_equal(saved_matrix, confusion_matrix(labels, predicted))
    assert hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest() == test["checkpoint_sha256"]
    for name in ("learning_curves", "confusion_matrix", "correct_examples", "incorrect_examples"):
        with Image.open(run_dir / f"{name}.png") as picture:
            picture.verify()
    verification = {
        "status": "passed", "model": model_name, "input_mode": input_mode(model_name),
        "split_matches_m1": True, "test_predictions_match_checkpoint": True,
        "test_dataset_sha256_matches": True,
        "checkpoint_validation_metrics": measured,
        "checkpoint_epoch": checkpoint["epoch"], "test_prediction_rows": len(predictions),
        "test_metrics_and_confusion_recomputed": True,
        "cpu_reload_batch_samples": len(images),
        "cpu_device_max_logit_difference": (cpu_logits - gpu_logits).abs().max().item(),
        "cpu_device_predictions_match": True, "png_files_valid": 4,
    }
    if model_name == "transformer":
        verification["patch_order_checked"] = True
        verification["encoder_layout"] = "sequence_first"
    if write_result:
        (run_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("linear", "mlp", "cnn", "lstm", "transformer"), default="cnn")
    parser.add_argument("--no-write", action="store_true", help="Check artifacts without replacing verification.json")
    args = parser.parse_args()
    main(args.model, write_result=not args.no_write)
