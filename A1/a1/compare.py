"""Generate the five-model comparison from saved experiments, without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

RUNS = {"linear": "cuda_baselines", "mlp": "cuda_baselines", "cnn": "m2", "lstm": "m2", "transformer": "m2"}
DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "runs"
NOTES = (
    "All classification scores and timings use the submitted CUDA runs on GTX 1660 SUPER, "
    "with one seed and one split. Linear/MLP artifacts are in runs/cuda_baselines/. "
    "Forward timing uses a device-resident batch and is amortized per image; "
    "the full test pass includes loading, transfers, loss and predictions. "
    "Runs were measured in separate sessions, without repeated-session uncertainty estimates. "
    "This exporter checks summary consistency and checkpoint hashes; "
    "use tests/verify_run.py to recompute predictions and metrics."
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def collect_results(run_root=DEFAULT_RUNS):
    """Read submitted runs and reject mixed datasets, splits or checkpoints."""
    run_root = Path(run_root)
    rows, sources = [], []
    reference = None
    for model, stage in RUNS.items():
        run_dir = run_root / stage / model
        summary = _read_json(run_dir / "summary.json")
        test = _read_json(run_dir / "test_summary.json")
        _require(summary["model"] == test["model"] == model, f"{model}: model mismatch")
        _require(summary["best_epoch"] == test["checkpoint_epoch"], f"{model}: checkpoint epoch mismatch")
        _require(test["selection_rule"] == "lowest validation loss", f"{model}: selection rule mismatch")
        _require(test["test_samples"] == 10000, f"{model}: expected 10,000 test images")
        for key in ("source_sha256", "split_sha256", "seed"):
            _require(summary[key] == test[key], f"{model}: {key} mismatch")
        protocol = (
            summary["source_sha256"], summary["split_sha256"], test["test_source_sha256"],
            summary["seed"], summary["train_mean"], summary["train_std"],
            summary["train_samples"], summary["val_samples"],
            *(summary["config"][key] for key in (
                "epochs", "batch_size", "learning_rate", "augment_train", "val_ratio",
            )),
        )
        if reference is None:
            reference = protocol
        _require(protocol == reference, f"{model}: data or training protocol differs across models")
        checkpoint_hash = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
        _require(checkpoint_hash == test["checkpoint_sha256"], f"{model}: checkpoint hash mismatch")
        inference = test["inference"]
        row = {
            "model": model, "parameters": summary["parameters"],
            "selected_epoch": summary["best_epoch"], "test_samples": test["test_samples"],
            "test_loss": test["test_loss"], "test_accuracy": test["test_accuracy"],
            "test_macro_f1": test["test_macro_f1"],
            "training_seconds": summary["training_seconds"], "training_device": summary["device"],
            "evaluation_device": test["hardware"]["device"],
            "evaluation_gpu": test["hardware"].get("gpu", ""),
            "forward_batch_size": inference["batch_size"],
            "forward_ms_per_image": inference["forward_ms_per_image"],
            "test_pass_seconds": inference["evaluation_seconds"],
        }
        for key in ("test_loss", "test_accuracy", "test_macro_f1", "training_seconds",
                    "forward_ms_per_image", "test_pass_seconds"):
            _require(math.isfinite(row[key]) and row[key] >= 0, f"{model}: invalid {key}")
        for key in ("test_accuracy", "test_macro_f1"):
            _require(row[key] <= 1, f"{model}: invalid {key}")
        rows.append(row)
        for name in ("summary.json", "test_summary.json"):
            path = run_dir / name
            sources.append({"path": path.relative_to(run_root).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"results": rows, "notes": NOTES, "sources_relative_to_runs": sources}


def write_comparison(comparison, output_dir):
    """Export machine-readable results, a Markdown table and a comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison["results"]
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Five-model Fashion-MNIST comparison", "",
        "Generated by `python -m a1.compare` from the submitted CUDA runs. No training or evaluation is run.", "",
        "| Model | Parameters | Epoch | Test loss | Test accuracy | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['parameters']:,} | {row['selected_epoch']} | "
                     f"{row['test_loss']:.6f} | {row['test_accuracy']:.2%} | {row['test_macro_f1']:.6f} |")
    lines += ["", comparison["notes"], "",
              "The CSV and JSON retain the original timing values, devices, batch sizes and source hashes.", ""]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title in zip(axes, ("test_accuracy", "test_macro_f1"), ("Test accuracy (%)", "Test macro-F1")):
        values = [row[key] * (100 if key == "test_accuracy" else 1) for row in rows]
        bars = ax.bar([row["model"] for row in rows], values, color="#26728e")
        ax.bar_label(bars, fmt="%.2f" if key == "test_accuracy" else "%.4f", padding=3)
        ax.set(title=title, ylim=(0, 105 if key == "test_accuracy" else 1.05))
        ax.tick_params(axis="x", rotation=20)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Fashion-MNIST: 10,000 test images, one seed")
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, help="Default: <runs-dir>/m2/comparison")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.runs_dir / "m2/comparison"
    comparison = collect_results(args.runs_dir)
    write_comparison(comparison, output_dir)
    print((output_dir / "comparison.md").read_text(encoding="utf-8"))
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
