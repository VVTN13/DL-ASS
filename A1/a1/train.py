"""Train and validate the linear and MLP Fashion-MNIST models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .data import build_dataloaders, set_seed
from .models import build_model, count_parameters


def _device(choice: str):
    if choice == "auto":
        if torch.backends.mps.is_available():
            choice = "mps"
        elif torch.cuda.is_available():
            choice = "cuda"
        else:
            choice = "cpu"
    if choice == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if choice not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device {choice!r}")
    return torch.device(choice)


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or validation pass over the loader."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    true_labels = []
    predicted_labels = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, labels=list(range(10)), average="macro", zero_division=0),
        "samples": total_samples,
    }


def _read_config(config_path):
    config = json.loads(Path(config_path).read_text())
    required = {
        "seed", "data_dir", "split_cache_path", "output_dir", "val_ratio",
        "batch_size", "epochs", "learning_rate", "augment_train", "num_workers", "device",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if config["batch_size"] <= 0 or config["epochs"] <= 0 or config["learning_rate"] <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    return config


def _config_hash(config, model_name):
    payload = json.dumps({"config": config, "model": model_name}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _versions():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def _save_history(path, history):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def load_checkpoint(path, device="cpu"):
    """Load a trained M1 model and its run metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def run_experiment(model_name, config_path="configs/m1.json", *, reuse_existing=False):
    config = _read_config(config_path)
    device = _device(config["device"])
    set_seed(config["seed"])
    train_loader, val_loader, _ = build_dataloaders(
        mode="flatten", batch_size=config["batch_size"], val_ratio=config["val_ratio"],
        seed=config["seed"], augment_train=config["augment_train"],
        data_dir=config["data_dir"], num_workers=config["num_workers"],
        split_cache_path=config["split_cache_path"],
    )
    run_dir = Path(config["output_dir"]) / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    config_hash = _config_hash(config, model_name)
    saved_files_exist = all(path.exists() for path in (summary_path, checkpoint_path, history_path))
    if reuse_existing and saved_files_exist:
        summary = json.loads(summary_path.read_text())
        same_data_and_config = (
            summary.get("config_sha256") == config_hash
            and summary.get("source_sha256") == train_loader.source_sha256
            and summary.get("split_sha256") == train_loader.split_sha256
        )
        if same_data_and_config:
            model, checkpoint = load_checkpoint(checkpoint_path, device)
            measured = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
            same_metrics = (
                checkpoint["config_sha256"] == config_hash
                and abs(measured["loss"] - summary["best_val_loss"]) < 1e-5
                and abs(measured["accuracy"] - summary["best_val_accuracy"]) < 1e-8
                and abs(measured["macro_f1"] - summary["best_val_macro_f1"]) < 1e-8
            )
            if same_metrics:
                return summary

    # The same seed is applied just before each model is initialized.
    set_seed(config["seed"])
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    best_loss = float("inf")
    best_epoch = 0
    _synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        print(f"{model_name} epoch {epoch:02d}/{config['epochs']}: "
              f"train loss {train_metrics['loss']:.4f}, val loss {val_metrics['loss']:.4f}, "
              f"val acc {val_metrics['accuracy']:.4f}, val macro-F1 {val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            checkpoint = {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "config_sha256": config_hash,
                "source_sha256": train_loader.source_sha256,
                "split_sha256": train_loader.split_sha256,
                "train_mean": train_loader.dataset.mean,
                "train_std": train_loader.dataset.std,
            }
            torch.save(checkpoint, checkpoint_path)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    best_model, _ = load_checkpoint(checkpoint_path, device)
    best_metrics = run_epoch(best_model, val_loader, criterion, device)
    _save_history(history_path, history)
    summary = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_metrics["loss"],
        "best_val_accuracy": best_metrics["accuracy"],
        "best_val_macro_f1": best_metrics["macro_f1"],
        "parameters": count_parameters(model),
        "training_seconds": elapsed,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "device": str(device),
        "seed": config["seed"],
        "config": config,
        "config_sha256": config_hash,
        "source_sha256": train_loader.source_sha256,
        "split_sha256": train_loader.split_sha256,
        "train_mean": train_loader.dataset.mean,
        "train_std": train_loader.dataset.std,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "versions": _versions(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train an Assignment 1 M1 model")
    parser.add_argument("--model", choices=("linear", "mlp"), required=True)
    parser.add_argument("--config", default="configs/m1.json")
    parser.add_argument("--reuse-existing", action="store_true", help="Validate and reuse a matching completed run")
    args = parser.parse_args()
    result = run_experiment(args.model, args.config, reuse_existing=args.reuse_existing)
    print(json.dumps({key: value for key, value in result.items() if key in {
        "model", "best_epoch", "best_val_loss", "best_val_accuracy", "best_val_macro_f1",
        "parameters", "training_seconds", "device", "checkpoint", "history",
    }}, indent=2))


if __name__ == "__main__":
    main()
