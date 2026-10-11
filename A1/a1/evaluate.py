"""Evaluate a saved checkpoint without using test results for model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from .data import build_dataloaders
from .models import input_mode
from .train import _device, _hardware, _synchronize, _versions, load_checkpoint


def plot_curves(history_path, output_path):
    with history_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes, ("loss", "accuracy", "macro_f1"),
                                ("Cross-entropy loss", "Accuracy", "Macro-F1")):
        for split in ("train", "val"):
            ax.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows], label=split)
        ax.set(xlabel="Epoch", ylabel=title, title=title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(matrix, class_names, output_path, model_name):
    fig, ax = plt.subplots(figsize=(9, 8))
    picture = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(picture, ax=ax, label="Images")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=class_names,
           yticklabels=class_names, xlabel="Predicted class", ylabel="True class",
           title=f"{model_name.upper()} test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_examples(dataset, labels, predictions, confidence, correct, output_path, seed):
    candidates = np.flatnonzero((labels == predictions) == correct)
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=min(12, len(candidates)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    names = dataset.base.classes
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, selected):
        image = dataset.base.data[int(dataset.indices[index])]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"#{index} True: {names[labels[index]]}\n"
                     f"Pred: {names[predictions[index]]} ({confidence[index]:.1%})", fontsize=9)
    fig.suptitle("Correct test predictions" if correct else "Incorrect test predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_checkpoint(checkpoint_path, device_choice="auto"):
    checkpoint_path = Path(checkpoint_path)
    device = _device(device_choice)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    _, _, loader = build_dataloaders(
        mode=input_mode(checkpoint["model_name"]), batch_size=config["batch_size"],
        val_ratio=config["val_ratio"], seed=config["seed"], data_dir=config["data_dir"],
        num_workers=config["num_workers"], split_cache_path=config["split_cache_path"],
    )
    if (loader.source_sha256 != checkpoint["source_sha256"]
            or loader.split_sha256 != checkpoint["split_sha256"]
            or loader.dataset.mean != checkpoint["train_mean"]
            or loader.dataset.std != checkpoint["train_std"]):
        raise ValueError("Evaluation data or normalization does not match the checkpoint")

    # Warm up before timing. Forward timing uses one fixed, full batch on the device.
    batch = next(iter(loader))[0].to(device)
    labels, predictions, confidence = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for _ in range(10):
            model(batch)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(100):
            model(batch)
        _synchronize(device)
        forward_seconds = (time.perf_counter() - start) / 100

        # This second timer includes loading, transfers, forward pass, loss and predictions.
        start = time.perf_counter()
        for images, targets in loader:
            logits = model(images.to(device))
            total_loss += criterion(logits, targets.to(device)).item()
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            confidence.extend(logits.softmax(dim=1).max(dim=1).values.cpu().tolist())
        _synchronize(device)
        evaluation_seconds = time.perf_counter() - start

    labels, predictions, confidence = map(np.asarray, (labels, predictions, confidence))
    names = loader.dataset.base.classes
    matrix = confusion_matrix(labels, predictions, labels=list(range(10)))
    digest = hashlib.sha256()
    digest.update(loader.dataset.base.data.numpy().tobytes())
    digest.update(loader.dataset.base.targets.numpy().tobytes())
    result = {
        "model": checkpoint["model_name"], "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"], "selection_rule": "lowest validation loss",
        "test_samples": len(labels), "test_loss": total_loss / len(labels),
        "test_accuracy": accuracy_score(labels, predictions),
        "test_macro_f1": f1_score(labels, predictions, average="macro"),
        "test_source_sha256": digest.hexdigest(),
        "source_sha256": loader.source_sha256, "split_sha256": loader.split_sha256,
        "seed": config["seed"], "input_mode": input_mode(checkpoint["model_name"]),
        "inference": {
            "batch_size": len(batch), "warmup_batches": 10, "timed_batches": 100,
            "forward_batch_ms": forward_seconds * 1000,
            "forward_ms_per_image": forward_seconds * 1000 / len(batch),
            "forward_scope": "device-resident fixed batch; excludes loading and host transfers",
            "evaluation_seconds": evaluation_seconds,
            "evaluation_scope": "entire test pass including loading, transfers, loss and predictions; excludes plots",
        },
        "class_report": classification_report(labels, predictions, labels=list(range(10)),
                                               target_names=names, output_dict=True, zero_division=0),
        "hardware": _hardware(device), "versions": _versions(),
    }
    output_dir = checkpoint_path.parent
    (output_dir / "test_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "true_label", "predicted_label", "confidence", "correct"])
        for i, (target, prediction, probability) in enumerate(zip(labels, predictions, confidence)):
            writer.writerow([i, target, prediction, probability, int(target == prediction)])
    with (output_dir / "confusion_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *names])
        writer.writerows([name, *row] for name, row in zip(names, matrix))
    plot_confusion(matrix, names, output_dir / "confusion_matrix.png", checkpoint["model_name"])
    plot_curves(output_dir / "history.csv", output_dir / "learning_curves.png")
    for correct in (True, False):
        plot_examples(loader.dataset, labels, predictions, confidence, correct,
                      output_dir / ("correct_examples.png" if correct else "incorrect_examples.png"),
                      config["seed"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected Fashion-MNIST checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.device)
    print(json.dumps({key: result[key] for key in ("model", "test_samples", "test_loss",
                                                 "test_accuracy", "test_macro_f1", "inference")}, indent=2))


if __name__ == "__main__":
    main()
