"""Recompute report metrics from immutable saved predictions; no model execution."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
A1 = HERE.parents[1]
ROOT = A1.parent
NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODELS = ["linear", "mlp", "cnn", "lstm", "transformer"]
DISPLAY = ["Linear", "MLP", "CNN", "LSTM", "Transformer"]
COLORS = ["#707070", "#4c80a5", "#195e55", "#b27937", "#816b97"]
FIG = HERE / "figures"
GEN = HERE / "generated"
QA = HERE / "qa"
for folder in [FIG, GEN, QA, HERE / "sections"]:
    folder.mkdir(exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def readjson(path):
    return json.loads(path.read_text(encoding="utf-8"))

def readcsv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def texfile(name, lines):
    (GEN / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")

def savefig(name):
    plt.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.savefig(FIG / (name + ".png"), dpi=190, bbox_inches="tight")
    plt.close()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titleweight": "bold"})
raw = A1 / "data/FashionMNIST/raw"
images = np.frombuffer((raw / "t10k-images-idx3-ubyte").read_bytes(), np.uint8, offset=16).reshape(-1, 28, 28)
labels = np.frombuffer((raw / "t10k-labels-idx1-ubyte").read_bytes(), np.uint8, offset=8)
test_hash = hashlib.sha256(images.tobytes() + labels.astype(np.int64).tobytes()).hexdigest()
eda = readjson(A1 / "report/data/eda_metrics.json")
for entry in eda["dataset_files"]:
    assert digest(raw / entry["file"]) == entry["sha256"]
for path, expected in eda["generated_figures_sha256"].items():
    assert digest(A1 / "report" / path.replace("\\", "/")) == expected
assert all(eda["checks"].values())
records = {}
histories = {}
preds = {}
confidences = {}
matrices = {}
checks = {}
for model in MODELS:
    run = A1 / "runs" / ("cuda_baselines" if model in MODELS[:2] else "m2") / model
    summary, test, previous = [readjson(run / name) for name in
                               ["summary.json", "test_summary.json", "verification.json"]]
    rows = readcsv(run / "test_predictions.csv")
    assert len(rows) == 10000
    assert [int(r["test_index"]) for r in rows] == list(range(10000))
    y = np.array([int(r["true_label"]) for r in rows])
    p = np.array([int(r["predicted_label"]) for r in rows])
    c = np.array([float(r["confidence"]) for r in rows])
    assert np.array_equal(y, labels) and np.all((p >= 0) & (p < 10))
    assert np.all((c >= 0) & (c <= 1))
    assert np.array_equal([int(r["correct"]) for r in rows], (y == p).astype(int))
    matrix = np.bincount(y.astype(int) * 10 + p, minlength=100).reshape(10, 10)
    stored = np.loadtxt(run / "confusion_matrix.csv", delimiter=",", skiprows=1,
                        usecols=range(1, 11), dtype=int)
    assert np.array_equal(matrix, stored)
    assert np.all(matrix.sum(axis=1) == 1000)
    precision = np.diag(matrix) / matrix.sum(axis=0)
    recall = np.diag(matrix) / matrix.sum(axis=1)
    f1 = 2 * np.diag(matrix) / (matrix.sum(axis=0) + matrix.sum(axis=1))
    assert abs(float(np.mean(p == y)) - test["test_accuracy"]) < 1e-12
    assert abs(float(f1.mean()) - test["test_macro_f1"]) < 1e-12
    for k, name in enumerate(NAMES):
        for key, value in zip(["precision", "recall", "f1-score"], [precision[k], recall[k], f1[k]]):
            assert abs(value - test["class_report"][name][key]) < 1e-12
    history = readcsv(run / "history.csv")
    assert len(history) == 10
    best = min(history, key=lambda r: float(r["val_loss"]))
    assert int(best["epoch"]) == summary["best_epoch"] == test["checkpoint_epoch"]
    assert abs(float(best["val_loss"]) - summary["best_val_loss"]) < 1e-12
    assert digest(run / "best.pt") == test["checkpoint_sha256"]
    assert test["test_source_sha256"] == test_hash
    assert summary["split_sha256"] == test["split_sha256"] == eda["split_sha256"]
    assert summary["source_sha256"] == test["source_sha256"] == eda["source_sha256"]
    config = summary["config"]
    for key, value in {"seed": 42, "batch_size": 128, "epochs": 10,
                       "learning_rate": .001, "val_ratio": .1, "augment_train": False}.items():
        assert config[key] == value
    assert abs(summary["train_mean"] - eda["training"]["mean_0_1"]) < 1e-12
    assert abs(summary["train_std"] - eda["training"]["std_0_1"]) < 1e-12
    assert previous["status"] == "passed"
    assert test["hardware"]["gpu"] == "NVIDIA GeForce GTX 1660 SUPER"
    records[model] = {"training": summary, "test": test,
                      "errors": int((p != y).sum()), "precision": precision.tolist(),
                      "recall": recall.tolist(), "f1": f1.tolist()}
    preds[model], confidences[model], matrices[model], histories[model] = p, c, matrix, history
    checks[model] = {"prediction_rows": len(rows), "metrics_recomputed": True,
                     "confusion_recomputed": True, "labels_match_local_test": True,
                     "checkpoint_hash_matches": True, "selection_matches_history": True,
                     "previous_checkpoint_reload_verification": previous["status"],
                     "source_files": {str(p.relative_to(A1)): digest(p) for p in run.iterdir()
                                      if p.suffix in {".json", ".csv", ".pt"}}}

texfile("results_rows", [
    f"{name} & {records[m]['training']['best_epoch']} & "
    f"{100*records[m]['training']['best_val_accuracy']:.2f} & "
    f"{100*records[m]['test']['test_accuracy']:.2f} & "
    f"{records[m]['test']['test_macro_f1']:.4f} & "
    f"{records[m]['test']['test_loss']:.4f} & {records[m]['errors']:,}" + r"\\"
    for m, name in zip(MODELS, DISPLAY)])
# Classification and cost tables use the same submitted CUDA checkpoints.
costs = {}
for model in MODELS:
    run = A1 / "runs" / ("cuda_baselines" if model in MODELS[:2] else "m2") / model
    training, test, verification = [readjson(run / name) for name in
                                  ["summary.json", "test_summary.json", "verification.json"]]
    assert training["device"] == test["hardware"]["device"] == "cuda"
    assert training["hardware"]["gpu"] == test["hardware"]["gpu"] == "NVIDIA GeForce GTX 1660 SUPER"
    assert verification["status"] == "passed"
    assert training["parameters"] == records[model]["training"]["parameters"]
    assert digest(run / "best.pt") == test["checkpoint_sha256"]
    assert training["split_sha256"] == test["split_sha256"] == eda["split_sha256"]
    for key in ("seed", "batch_size", "epochs", "learning_rate", "val_ratio", "augment_train", "num_workers"):
        assert training["config"][key] == records[model]["training"]["config"][key]
    assert (test["inference"]["batch_size"], test["inference"]["warmup_batches"], test["inference"]["timed_batches"]) == (128, 10, 100)
    costs[model] = {"training": training, "test": test,
                    "source_directory": str(run.relative_to(A1)),
                    "source_files": {p.name: digest(p) for p in run.iterdir()
                                     if p.suffix in {".json", ".csv", ".pt"}}}
texfile("runtime_rows", [
    f"{name} & {records[m]['training']['parameters']:,} & "
    f"{costs[m]['training']['training_seconds']:.2f} & "
    f"{costs[m]['training']['device'].upper()} & "
    f"{costs[m]['test']['inference']['forward_batch_ms']:.3f} & "
    f"{costs[m]['test']['inference']['evaluation_seconds']:.3f}" + r"\\"
    for m, name in zip(MODELS, DISPLAY)])
texfile("f1_rows", [
    name + " & " + " & ".join(f"{100*records[m]['f1'][i]:.2f}" for m in MODELS) + r"\\"
    for i, name in enumerate(NAMES)])
texfile("cnn_class_rows", [
    f"{name} & {100*records['cnn']['precision'][i]:.2f} & "
    f"{100*records['cnn']['recall'][i]:.2f} & {100*records['cnn']['f1'][i]:.2f} & "
    f"{1000-matrices['cnn'][i,i]}" + r"\\"
    for i, name in enumerate(NAMES)])
top = sorted(((int(matrices["cnn"][i, j] + matrices["cnn"][j, i]), i, j)
              for i in range(10) for j in range(i+1, 10)), reverse=True)[:6]
texfile("confusion_rows", [
    f"{NAMES[i]} & {NAMES[j]} & {matrices['cnn'][i,j]} & {matrices['cnn'][j,i]}" + r"\\"
    for n, i, j in top])
training_stats = eda["training"]
texfile("eda_rows", [
    f"{r['class']} & {r['mean_intensity_0_255']:.2f} & "
    f"{r['sd_image_mean_intensity']:.2f} & {r['mean_nonzero_fraction']*100:.2f}" + r"\\"
    for r in eda["class_statistics"]])
texfile("history_rows", [
    f"{name} & {float(histories[m][0]['val_loss']):.4f} & "
    f"{float(histories[m][-1]['val_loss']):.4f} & "
    f"{100*float(histories[m][-1]['train_accuracy']):.2f} & "
    f"{100*float(histories[m][-1]['val_accuracy']):.2f}" + r"\\"
    for m, name in zip(MODELS, DISPLAY)])

# Consistent comparison graphics, generated directly from verified data.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
for ax, key, title in zip(axes, ["test_accuracy", "test_macro_f1"], ["Test accuracy (%)", "Test macro-F1 (%)"]):
    values = [records[m]["test"][key] * 100 for m in MODELS]
    ax.barh(DISPLAY, values, color=COLORS)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel=title)
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=.15)
savefig("test_comparison")

fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
for ax, m, name in zip(axes.flat, MODELS, DISPLAY):
    h = histories[m]
    ax.plot(range(1, 11), [float(r["train_loss"]) for r in h], color="#707070", label="Training")
    ax.plot(range(1, 11), [float(r["val_loss"]) for r in h], color="#195e55", label="Validation")
    ax.axvline(records[m]["training"]["best_epoch"], ls=":", color="#b27937", label="Selected epoch")
    ax.set(title=name, xlabel="Epoch", ylabel="Cross-entropy", ylim=(0, 1.05),
           xticks=[1, 3, 5, 7, 10])
    ax.grid(alpha=.2)
axes.flat[-1].axis("off")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(handles, legend_labels, loc="center", frameon=False, fontsize=12)
savefig("learning_curves")

fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
values = np.array([records[m]["f1"] for m in MODELS]).T * 100
ax.imshow(values, cmap="YlGnBu", vmin=50, vmax=100, aspect="auto")
ax.set(xticks=range(5), xticklabels=DISPLAY, yticks=range(10), yticklabels=NAMES)
for i in range(10):
    for j in range(5):
        ax.text(j, i, f"{values[i,j]:.1f}", va="center", ha="center",
                color="white" if values[i,j] > 85 else "black")
ax.set_title("Per-class test F1 (%)")
savefig("class_f1")

def draw_matrix(ax, m):
    matrix = matrices[m]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1000)
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted label",
           ylabel="True label", title=DISPLAY[MODELS.index(m)])
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i,j]), ha="center", va="center", fontsize=9,
                    color="white" if matrix[i,j] > 500 else "#1b1b1b")
fig, ax = plt.subplots(figsize=(6, 5.7), layout="constrained")
draw_matrix(ax, "cnn")
savefig("cnn_confusion")
for group, models in [("baseline_confusions", ["linear", "mlp"]),
                      ("sequence_confusions", ["lstm", "transformer"])]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, model in zip(axes, models):
        draw_matrix(ax, model)
    savefig(group)

# Reproducible samples: independently sample four correct and four incorrect.
rng = np.random.default_rng(42)
sample_indices = []
fig, axes = plt.subplots(2, 4, figsize=(10, 5.4), layout="constrained")
for row, correct in enumerate([True, False]):
    selected = rng.choice(np.flatnonzero((preds["cnn"] == labels) == correct), 4, replace=False)
    sample_indices.extend(map(int, selected))
    for ax, index in zip(axes[row], selected):
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"#{index} | {'Correct' if correct else 'Incorrect'}\n"
                     f"True: {NAMES[labels[index]]}\n"
                     f"Pred: {NAMES[preds['cnn'][index]]} ({confidences['cnn'][index]:.1%})",
                     fontsize=10, color="#195e55" if correct else "#833c32")
savefig("cnn_examples")

shared_wrong = np.logical_and.reduce([preds[m] != labels for m in MODELS])
upper = [0, 2, 4, 6]
upper_errors = sum(matrices["cnn"][i,j] for i in upper for j in upper if i != j)
paired = {}
for model in ["linear", "mlp", "lstm", "transformer"]:
    paired[model] = {"cnn_only_correct": int(((preds["cnn"] == labels) & (preds[model] != labels)).sum()),
                     "other_only_correct": int(((preds["cnn"] != labels) & (preds[model] == labels)).sum()),
                     "both_wrong": int(((preds["cnn"] != labels) & (preds[model] != labels)).sum())}
texfile("paired_rows", [
    f"{DISPLAY[MODELS.index(m)]} & {v['cnn_only_correct']} & "
    f"{v['other_only_correct']} & {v['both_wrong']}" + r"\\"
    for m, v in paired.items()])
mlp_only = int(((preds["mlp"] == labels) & (preds["lstm"] != labels)).sum())
lstm_only = int(((preds["mlp"] != labels) & (preds["lstm"] == labels)).sum())
derived = {"upper_body_cnn_errors": int(upper_errors),
           "upper_body_error_percent": float(upper_errors / records["cnn"]["errors"] * 100),
           "all_five_wrong": int(shared_wrong.sum()),
           "mlp_only_correct_vs_lstm": mlp_only, "lstm_only_correct_vs_mlp": lstm_only,
           "paired_vs_cnn": paired, "cnn_sample_test_indices": sample_indices,
           "cnn_examples": [{"index": i, "true": NAMES[labels[i]], "pred": NAMES[preds["cnn"][i]],
                              "confidence": float(confidences["cnn"][i])} for i in sample_indices],
           "cnn_relative_error_reduction_vs_mlp": (records["mlp"]["errors"]-records["cnn"]["errors"])/records["mlp"]["errors"]*100}
texfile("derived_values", [
    rf"\newcommand{{\UpperErrors}}{{{upper_errors}}}",
    rf"\newcommand{{\UpperErrorPercent}}{{{derived['upper_body_error_percent']:.1f}}}",
    rf"\newcommand{{\SharedErrors}}{{{derived['all_five_wrong']}}}",
    rf"\newcommand{{\MlpOnly}}{{{mlp_only}}}",
    rf"\newcommand{{\LstmOnly}}{{{lstm_only}}}",
])

for stem in ["02_training_examples", "03_pixel_distribution",
             "04_brightness_coverage", "05_spatial_statistics",
             "07_preprocessing_representations"]:
    for suffix in [".pdf", ".png"]:
        shutil.copy2(A1 / "report/figures" / (stem + suffix), FIG / (stem + suffix))
shutil.copy2(A1 / "report/figures/hcmut-logo.png", FIG / "hcmut-logo.png")
method = (A1 / "report-part3/fresh-methodology/sections/03_methodology.tex").read_text(encoding="utf-8")
if not (HERE / "sections/03_methodology.tex").exists():
    (HERE / "sections/03_methodology.tex").write_text(method, encoding="utf-8")
evidence = {"status": "passed", "scope": "Saved predictions and artifacts checked without training or inference",
            "models": records, "checks": checks, "derived": derived,
            "cuda_costs": costs,
            "eda_file": str((A1 / "report/data/eda_metrics.json").relative_to(ROOT)),
            "eda_sha256": digest(A1 / "report/data/eda_metrics.json"),
            "figures": {str(p.relative_to(HERE)): digest(p) for p in FIG.iterdir()}}
(QA / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "passed", "checked_predictions": 50000, "derived": derived}, indent=2))
