"""Check the saved CNN run against its checkpoint, data and predictions.

Run from A1: python tests/verify_cnn_run.py
"""

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
from a1.train import _device, load_checkpoint, run_epoch


def main():
    run_dir = Path("runs/m2/cnn")
    summary = json.loads((run_dir / "summary.json").read_text())
    test = json.loads((run_dir / "test_summary.json").read_text())
    config = summary["config"]
    train, val, test_loader = build_dataloaders(
        mode="image", batch_size=config["batch_size"], seed=config["seed"],
        val_ratio=config["val_ratio"], data_dir=config["data_dir"],
        split_cache_path=config["split_cache_path"], num_workers=0,
    )
    assert (len(train.dataset), len(val.dataset), len(test_loader.dataset)) == (54000, 6000, 10000)
    assert summary["split_sha256"] == val.split_sha256 == test["split_sha256"]
    for name in ("linear", "mlp"):
        baseline = json.loads((Path("runs/m1") / name / "summary.json").read_text())
        assert baseline["split_sha256"] == val.split_sha256
        assert baseline["source_sha256"] == val.source_sha256
    assert not np.intersect1d(train.dataset.indices, val.dataset.indices).size
    assert (train.dataset.mean, train.dataset.std) == (val.dataset.mean, val.dataset.std)

    device = _device("auto")
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    measured = run_epoch(model, val, nn.CrossEntropyLoss(), device)
    for metric in ("loss", "accuracy", "macro_f1"):
        assert abs(measured[metric] - summary[f"best_val_{metric}"]) < 1e-5
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    best = min(history, key=lambda row: float(row["val_loss"]))
    assert checkpoint["epoch"] == int(best["epoch"]) == summary["best_epoch"]
    assert len(history) == config["epochs"]

    # A checkpoint trained on CUDA must also load and predict on CPU.
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
        "status": "passed", "split_matches_m1": True,
        "checkpoint_validation_metrics": measured,
        "checkpoint_epoch": checkpoint["epoch"], "test_prediction_rows": len(predictions),
        "test_metrics_and_confusion_recomputed": True,
        "cpu_reload_batch_samples": len(images),
        "cpu_device_max_logit_difference": (cpu_logits - gpu_logits).abs().max().item(),
        "cpu_device_predictions_match": True, "png_files_valid": 4,
    }
    (run_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
