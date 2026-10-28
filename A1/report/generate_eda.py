r"""Rebuild the report's training-set EDA on the local CUDA device.

From A1: .\.venv\Scripts\python.exe report\generate_eda.py --device cuda
Only report/ is written. Existing splits, checkpoints and disclosures are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import sklearn
import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

HERE = Path(__file__).resolve().parent
A1 = HERE.parent
sys.path.insert(0, str(A1))
from a1.data import FashionMNISTView, stratified_train_val_split, to_representation

NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
         "Shirt", "Sneaker", "Bag", "Ankle boot"]
COLORS = ["#24658c", "#c77b39", "#507d55"]
EXPECTED_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_figure(fig, name, figures):
    for ext in ("pdf", "png"):
        path = HERE / "figures" / f"{name}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        figures.append(path)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_hashes(images, labels):
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    counts = Counter(hashes)
    labels_by_hash = defaultdict(set)
    for key, label in zip(hashes, labels):
        labels_by_hash[key].add(int(label))
    return hashes, {
        "images": len(images), "unique_images": len(counts),
        "duplicate_excess": len(images) - len(counts),
        "duplicate_groups": sum(v > 1 for v in counts.values()),
        "conflicting_label_groups": sum(len(v) > 1 for v in labels_by_hash.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested; refusing to silently use CPU.")
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for name in ("figures", "data", "generated"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    protected_paths = [A1.parent / "AI_USAGE.md", A1 / "split_cache.json"]
    protected_before = {str(p.relative_to(A1.parent)): digest(p) for p in protected_paths}

    print("Reading local Fashion-MNIST and validating the cached split...", flush=True)
    official_train = FashionMNIST(str(A1 / "data"), train=True, download=False,
                                 transform=ToTensor())
    official_test = FashionMNIST(str(A1 / "data"), train=False, download=False,
                                transform=ToTensor())
    raw = official_train.data.numpy()
    labels = official_train.targets.numpy()
    test_raw = official_test.data.numpy()
    test_labels = official_test.targets.numpy()
    cache = json.loads((A1 / "split_cache.json").read_text())
    train_idx, val_idx = stratified_train_val_split(labels, val_ratio=0.1, seed=42)
    assert np.array_equal(train_idx, cache["train_idx"])
    assert np.array_equal(val_idx, cache["val_idx"])
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.array_equal(np.sort(np.r_[train_idx, val_idx]), np.arange(60000))
    source = hashlib.sha256(raw.tobytes() + labels.tobytes()).hexdigest()
    split_hash = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes()).hexdigest()
    assert source == cache["source_sha256"]
    assert split_hash == cache["split_sha256"]
    assert raw.shape == (60000, 28, 28) and test_raw.shape == (10000, 28, 28)
    assert raw.dtype == test_raw.dtype == np.uint8
    assert set(labels) == set(test_labels) == set(range(10))

    files = []
    for path in sorted((A1 / "data" / "FashionMNIST" / "raw").iterdir()):
        if not path.is_file():
            continue
        content = path.read_bytes()
        row = {"file": path.name, "bytes": len(content),
               "sha256": hashlib.sha256(content).hexdigest()}
        if path.name in EXPECTED_MD5:
            row["md5"] = hashlib.md5(content).hexdigest()
            row["published_md5_match"] = row["md5"] == EXPECTED_MD5[path.name]
            assert row["published_md5_match"], path.name
        files.append(row)

    train_raw = raw[train_idx]
    train_y = labels[train_idx]
    counts = []
    for label, name in enumerate(NAMES):
        counts.append({"label": label, "class": name,
                       "train": int((train_y == label).sum()),
                       "validation": int((labels[val_idx] == label).sum()),
                       "test": int((test_labels == label).sum())})
    assert all((r["train"], r["validation"], r["test"]) == (5400, 600, 1000)
               for r in counts)
    write_csv(HERE / "data" / "split_counts.csv", counts)

    print(f"Computing all 42,336,000 training-pixel statistics on {device}...", flush=True)
    # The project's ToTensor runs on CPU. Preserve its exact quantization before
    # CUDA reductions: GPU division by 255 can differ by one float32 ULP.
    pixels = torch.from_numpy(train_raw).float().div_(255).to(device)
    y = torch.as_tensor(train_y, device=device)
    # Match ToTensor's float32 quantization and the existing loader's moment rule.
    mean = pixels.sum(dtype=torch.float64).item() / pixels.numel()
    second = pixels.square().sum(dtype=torch.float64).item() / pixels.numel()
    std = (second - mean * mean) ** 0.5
    assert abs(mean - cache["mean"]) < 1e-10
    assert abs(std - cache["std"]) < 1e-10
    per_brightness = pixels.mean(dim=(1, 2)).cpu().numpy() * 255
    per_coverage = pixels.gt(0).float().mean(dim=(1, 2)).cpu().numpy()
    hist = torch.bincount(torch.as_tensor(train_raw, device=device).flatten().long(),
                          minlength=256).cpu().numpy()
    cpu_hist = np.bincount(train_raw.reshape(-1), minlength=256)
    assert np.array_equal(hist, cpu_hist)
    assert abs(mean - float(train_raw.mean() / 255)) < 2e-8
    assert abs(std - float(train_raw.std() / 255)) < 2e-8
    means, stdevs, class_rows = [], [], []
    for label, name in enumerate(NAMES):
        subset = pixels[y == label]
        means.append(subset.mean(dim=0).cpu().numpy())
        stdevs.append(subset.std(dim=0, correction=0).cpu().numpy())
        brightness = per_brightness[train_y == label]
        coverage = per_coverage[train_y == label]
        class_rows.append({"label": label, "class": name, "n": int(len(brightness)),
                           "mean_intensity_0_255": float(brightness.mean()),
                           "sd_image_mean_intensity": float(brightness.std()),
                           "median_intensity_0_255": float(np.median(brightness)),
                           "mean_nonzero_fraction": float(coverage.mean()),
                           "median_nonzero_fraction": float(np.median(coverage))})
    means, stdevs = np.stack(means), np.stack(stdevs)
    proto = torch.as_tensor(means, device=device).flatten(1)
    unit = torch.nn.functional.normalize(proto, dim=1)
    similarity = (unit @ unit.T).cpu().numpy()
    sim_offdiag = similarity.copy()
    np.fill_diagonal(sim_offdiag, -np.inf)
    a, b = np.unravel_index(sim_offdiag.argmax(), sim_offdiag.shape)
    write_csv(HERE / "data" / "class_statistics.csv", class_rows)
    write_csv(HERE / "data" / "pixel_histogram.csv",
              [{"intensity": i, "count": int(v)} for i, v in enumerate(hist)])
    np.savez_compressed(HERE / "data" / "spatial_statistics.npz",
                        class_mean=means, class_std=stdevs, cosine_similarity=similarity)

    print("Auditing exact image duplicates (all partitions; no data are removed)...", flush=True)
    train_hashes, train_dupes = audit_hashes(train_raw, train_y)
    val_hashes, val_dupes = audit_hashes(raw[val_idx], labels[val_idx])
    test_hashes, test_dupes = audit_hashes(test_raw, test_labels)
    duplicate_audit = {"training": train_dupes, "validation": val_dupes, "test": test_dupes}
    for left, right, lhash, rhash in [
        ("train", "validation", train_hashes, val_hashes),
        ("train", "test", train_hashes, test_hashes),
        ("validation", "test", val_hashes, test_hashes),
    ]:
        shared = set(lhash) & set(rhash)
        duplicate_audit[f"{left}_{right}"] = {
            "shared_unique_hashes": len(shared),
            "left_images_in_shared_groups": sum(h in shared for h in lhash),
            "right_images_in_shared_groups": sum(h in shared for h in rhash),
        }

    # Verify actual representation values, including full inverse patch ordering.
    sample_index = int(train_idx[0])
    view = FashionMNISTView(official_train, train_idx, mean, std, mode="image")
    image, label = view[0]
    reps = {mode: to_representation(image, mode) for mode in
            ("image", "flatten", "row_seq", "patch_seq")}
    patches = reps["patch_seq"]
    reconstruction = patches.reshape(7, 7, 4, 4).permute(0, 2, 1, 3).reshape(28, 28)
    assert torch.equal(reps["row_seq"], image[0])
    assert torch.equal(reps["flatten"], image.flatten())
    assert torch.equal(reconstruction, image[0])
    for mode, value in reps.items():
        assert torch.equal(to_representation(image.to(device), mode).cpu(), value)
    normalized = (pixels - mean) / std
    normalized_mean = normalized.mean(dtype=torch.float64).item()
    normalized_std = normalized.double().std(correction=0).item()
    assert abs(normalized_mean) < 1e-6 and abs(normalized_std - 1) < 1e-6
    blank_count = int((train_raw.reshape(len(train_raw), -1).sum(axis=1) == 0).sum())

    print("Drawing nine report figures as both vector PDF and 220-dpi PNG...", flush=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "axes.labelcolor": "#253441",
                         "text.color": "#253441", "axes.titlepad": 10})
    figures = []

    # Dots remain visible when every count is zero; no artificial bar heights.
    audit_panels = [
        ("Within each split", ["Training", "Validation", "Test"],
         [duplicate_audit[key]["duplicate_excess"]
          for key in ("training", "validation", "test")],
         "Duplicate excess (images)", COLORS[0]),
        ("Between splits", ["Training / validation", "Training / test", "Validation / test"],
         [duplicate_audit[key]["shared_unique_hashes"]
          for key in ("train_validation", "train_test", "validation_test")],
         "Distinct shared hashes", COLORS[2]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.5), layout="constrained")
    limit = max(1, max(max(panel[2]) for panel in audit_panels))
    for ax, (title, labels_, values, xlabel, color) in zip(axes, audit_panels):
        positions = np.arange(len(values))
        ax.axvline(0, color="#b6c3cc", linewidth=1, zorder=1)
        ax.hlines(positions, 0, values, color=color, linewidth=3, zorder=2)
        ax.scatter(values, positions, color=color, s=85, zorder=3)
        for value, position in zip(values, positions):
            ax.annotate(str(value), (value, position), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=14,
                        fontweight="bold", color=color)
        ax.set_yticks(positions, labels_, fontsize=12)
        ax.set(title=title, xlabel=xlabel, xlim=(-.06 * limit, 1.25 * limit),
               ylim=(2.6, -.6))
        ax.set_xticks(np.arange(0, limit + 1, max(1, int(np.ceil(limit / 4)))))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e7edf1", linewidth=.8)
        ax.set_axisbelow(True)
    save_figure(fig, "09_duplicate_audit", figures)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), layout="constrained")
    for ax, key, total, color in zip(axes, ("train", "validation", "test"),
                                    (54000, 6000, 10000), COLORS):
        values = [r[key] for r in counts]
        bars = ax.bar(range(10), values, color=color, width=.72)
        ax.set(title=f"{key.title()} (n = {total:,})", xlabel="Class label", ylabel="Images")
        ax.set_xticks(range(10))
        ax.set_ylim(0, max(values) * 1.23)
        ax.text(.5, .91, f"{values[0]:,} per class | 10% each", transform=ax.transAxes,
                ha="center", fontsize=13)
        ax.yaxis.grid(alpha=.15)
        ax.set_axisbelow(True)
    save_figure(fig, "01_class_balance", figures)

    fig, axes = plt.subplots(2, 5, figsize=(12, 3.1), layout="constrained")
    rng = np.random.default_rng(42)
    selected = {}
    for k, ax in enumerate(axes.flat):
        ids = rng.choice(train_idx[train_y == k], size=3, replace=False)
        selected[NAMES[k]] = ids.tolist()
        strip = np.concatenate([raw[i] for i in ids], axis=1)
        ax.imshow(strip, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{k}: {NAMES[k]}", fontsize=14)
        ax.set_xticks([13.5, 41.5, 69.5], [f"#{i}" for i in ids], fontsize=11)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    save_figure(fig, "02_training_examples", figures)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7), layout="constrained")
    axes[0].bar(range(256), hist / hist.sum() * 100, width=1, color=COLORS[0])
    axes[0].set(title="All training pixels", ylabel="Percentage of all pixels (%)",
                xlabel="Raw uint8 intensity (0-255)", xlim=(-3, 257))
    axes[0].text(.4, .82, f"Exactly zero: {hist[0] / hist.sum():.2%}\n"
                 f"Mean: {train_raw.mean():.2f}\nStd: {train_raw.std():.2f}",
                 transform=axes[0].transAxes, fontsize=14)
    axes[1].bar(range(1, 256), hist[1:] / hist[1:].sum() * 100, width=1, color=COLORS[1])
    axes[1].set(title="Conditional distribution: nonzero pixels only",
                ylabel="Percentage of nonzero pixels (%)", xlabel="Raw intensity (1-255)",
                xlim=(0, 257))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    save_figure(fig, "03_pixel_distribution", figures)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), layout="constrained")
    for ax, values, title, xlabel, color in [
        (axes[0], per_brightness, "Brightness per image", "Mean raw intensity (0-255)", COLORS[0]),
        (axes[1], per_coverage * 100, "Nonzero coverage per image", "Pixels greater than zero (%)", COLORS[1]),
    ]:
        boxes = ax.boxplot([values[train_y == k] for k in range(10)],
                           orientation="horizontal", tick_labels=NAMES, showfliers=False,
                           patch_artist=True, widths=.6, whis=1.5,
                           medianprops={"color": "#111111", "linewidth": 1.2})
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(.65)
        ax.invert_yaxis()
        ax.set(title=title, xlabel=xlabel)
        ax.grid(axis="x", alpha=.15)
    save_figure(fig, "04_brightness_coverage", figures)

    fig, axes = plt.subplots(4, 5, figsize=(11.6, 8.6), layout="constrained")
    for k in range(10):
        ax = axes[k // 5, k % 5]
        mappable_mean = ax.imshow(means[k], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{NAMES[k]} | mean", fontsize=14)
        ax.axis("off")
        ax = axes[2 + k // 5, k % 5]
        mappable_std = ax.imshow(stdevs[k], cmap="magma", vmin=0, vmax=.5)
        ax.set_title(f"{NAMES[k]} | std", fontsize=14)
        ax.axis("off")
    fig.colorbar(mappable_mean, ax=axes[:2].flatten().tolist(), shrink=.7,
                 label="Mean intensity (0-1)", fraction=.025, pad=.02)
    fig.colorbar(mappable_std, ax=axes[2:].flatten().tolist(), shrink=.7,
                 label="Population std (0-1 scale)", fraction=.025, pad=.02)
    save_figure(fig, "05_spatial_statistics", figures)

    fig, ax = plt.subplots(figsize=(8.5, 7.3), layout="constrained")
    im = ax.imshow(similarity, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(10), NAMES, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(10), NAMES, fontsize=12)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if similarity[i, j] > .65 else "#253441")
    ax.set_title("Cosine similarity of training class-average images", fontsize=15)
    fig.colorbar(im, ax=ax, shrink=.75, label="Cosine similarity (0-1)")
    save_figure(fig, "06_prototype_similarity", figures)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.3), layout="constrained")
    ax = axes[0, 0]
    ax.imshow(raw[sample_index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"Raw: {NAMES[label]} (index {sample_index})", fontsize=14)
    ax.set(xlabel="Column", ylabel="Row")
    ax = axes[0, 1]
    im = ax.imshow(image[0], cmap="coolwarm", vmin=-mean/std, vmax=(1-mean)/std)
    ax.set_title("CNN: (1, 28, 28) image", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=.8, label="Z")
    ax = axes[0, 2]
    ax.plot(reps["flatten"].numpy(), linewidth=.55, color=COLORS[0])
    ax.set(title="Linear / MLP: (784,)", xlabel="Flattened position (row-major)",
           ylabel="Standardized value", xlim=(0, 783))
    ax = axes[1, 0]
    ax.imshow(reps["row_seq"], cmap="gray", vmin=-mean/std, vmax=(1-mean)/std)
    ax.set_title("LSTM: (28 steps, 28 features)", fontsize=14)
    ax.set(xlabel="Pixel feature within row", ylabel="Row step")
    ax = axes[1, 1]
    ax.imshow(patches, aspect="auto", cmap="gray", vmin=-mean/std, vmax=(1-mean)/std)
    ax.set_title("Transformer: (49 patches, 16 pixels)", fontsize=14)
    ax.set(xlabel="Pixel within 4 x 4 patch", ylabel="Patch index")
    ax.set_xticks([0, 3, 7, 11, 15])
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(.02, .9, "Same 784 pixels\nin every representation\n\n"
            "1. uint8 -> float32 / 255\n2. Subtract training mean\n3. Divide by training std\n"
            "4. Change layout for the model\n\n"
            "Row / flat / patch inverses: exact\nCPU / CUDA layout: exact\n"
            "Augmentation: disabled", transform=ax.transAxes,
            fontsize=13, linespacing=1.4, va="top")
    save_figure(fig, "07_preprocessing_representations", figures)

    fig, ax = plt.subplots(figsize=(11.8, 4.4), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 4.7))
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08",
                                   facecolor=color, edgecolor="#577080", linewidth=1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=14,
                linespacing=1.4)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     color="#577080", linewidth=1.4))

    box(.15, 2.7, 2.8, 1.1, "Official training source\n60,000 labeled images", "#e6eff4")
    box(4.1, 3.15, 3.15, 1.1, "Training: 54,000\nFit normalization + model weights", "#e5efe7")
    box(4.1, 1.6, 3.15, 1.1, "Validation: 6,000\nSelect lowest-loss checkpoint", "#f5ecdf")
    box(.15, .15, 2.8, 1.1, "Official test source\n10,000 labeled images", "#e6eff4")
    box(8.4, 1.6, 3.15, 1.1, "Fixed selected checkpoint\nFinal test evaluation", "#e8e5ef")
    arrow((3.05, 3.4), (4, 3.7))
    arrow((3.05, 3), (4, 2.15))
    arrow((7.35, 3.7), (8.3, 2.65))
    arrow((7.35, 2.15), (8.3, 2.15))
    arrow((3.05, .7), (9.9, .7))
    arrow((9.9, .7), (9.9, 1.5))
    ax.text(3.35, 4.5, "Stratified 90/10 split | seed 42 | cached indices", fontsize=15)
    ax.text(4, .18, "Training mean/std are reused unchanged for validation and test.", fontsize=13)
    save_figure(fig, "08_data_protocol", figures)

    # TeX values are generated directly from measured data, not retyped in prose.
    macros = {
        "TrainMean": f"{mean:.6f}", "TrainStd": f"{std:.6f}",
        "RawMean": f"{train_raw.mean():.2f}", "RawStd": f"{train_raw.std():.2f}",
        "ZeroPercent": f"{hist[0] / hist.sum() * 100:.2f}",
        "BlankImages": str(blank_count),
        "NormalizedMinimum": f"{-mean / std:.3f}",
        "NormalizedMaximum": f"{(1 - mean) / std:.3f}",
        "TrainDuplicateExcess": str(train_dupes["duplicate_excess"]),
        "ValDuplicateExcess": str(val_dupes["duplicate_excess"]),
        "TestDuplicateExcess": str(test_dupes["duplicate_excess"]),
        "SharedTrainVal": str(duplicate_audit["train_validation"]["shared_unique_hashes"]),
        "SharedTrainTest": str(duplicate_audit["train_test"]["shared_unique_hashes"]),
        "SharedValTest": str(duplicate_audit["validation_test"]["shared_unique_hashes"]),
        "MostSimilarClasses": f"{NAMES[a]} and {NAMES[b]}",
        "MostSimilarCosine": f"{similarity[a,b]:.4f}",
        "LowestBrightness": f"{min(class_rows, key=lambda r:r['mean_intensity_0_255'])['mean_intensity_0_255']:.2f}",
        "HighestBrightness": f"{max(class_rows, key=lambda r:r['mean_intensity_0_255'])['mean_intensity_0_255']:.2f}",
    }
    (HERE / "generated" / "eda_values.tex").write_text(
        "% Generated by report/generate_eda.py. Do not edit manually.\n" +
        "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items()),
        encoding="utf-8")
    rows = [f"{r['label']} & {r['class']} & {r['mean_intensity_0_255']:.2f} & "
            f"{r['sd_image_mean_intensity']:.2f} & {100*r['mean_nonzero_fraction']:.2f} \\\\"
            for r in class_rows]
    (HERE / "generated" / "class_statistics_rows.tex").write_text(
        "\\newcommand{\\ClassStatisticsRows}{%\n" + "\n".join(rows) + "\n}\n")

    try:
        smi = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
        ], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        smi = None
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "numpy": np.__version__,
                   "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                   "device": str(device), "cuda_runtime": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "nvidia_smi": smi}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        environment.update({"gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
                            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    protected_after = {str(p.relative_to(A1.parent)): digest(p) for p in protected_paths}
    assert protected_before == protected_after
    metrics = {
        "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "scope": "training-only descriptive EDA; validation/test structural and duplicate audit only",
        "seed": 42, "source_sha256": source, "split_sha256": split_hash,
        "environment": environment, "dataset_files": files,
        "training": {"images": len(train_raw), "pixels": int(pixels.numel()),
                     "dtype": str(train_raw.dtype), "min": int(train_raw.min()),
                     "max": int(train_raw.max()), "mean_0_1": mean, "std_0_1": std,
                     "mean_raw": float(train_raw.mean()), "std_raw": float(train_raw.std()),
                     "zero_fraction": float(hist[0] / hist.sum()), "blank_images": blank_count,
                     "normalized_mean": normalized_mean, "normalized_std": normalized_std},
        "split_counts": counts, "class_statistics": class_rows,
        "duplicate_audit": duplicate_audit,
        "prototype_similarity": {"metric": "cosine of flattened uncentered class means",
                                 "most_similar_pair": [NAMES[a], NAMES[b]],
                                 "value": float(similarity[a, b])},
        "selected_training_examples": selected,
        "representation_check": {"source_index": sample_index,
                                 "shapes": {k: list(v.shape) for k, v in reps.items()},
                                 "exact_reconstruction": True, "cpu_device_agreement": True},
        "checks": {"cache_matches_seed42_split": True, "source_and_split_hashes_match": True,
                   "cuda_or_cpu_histogram_matches_independent_numpy": True,
                   "normalization_matches_cache": True, "raw_numpy_moments_agree": True,
                   "all_four_published_gzip_md5s_match": True,
                   "normalized_training_mean_zero_std_one": True,
                   "protected_files_unchanged": protected_before == protected_after},
        "protected_files_sha256": protected_after,
        "generated_figures_sha256": {str(p.relative_to(HERE)): digest(p) for p in figures},
        "code_sha256": digest(__file__),
    }
    (HERE / "data" / "eda_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps({"training": metrics["training"], "duplicates": duplicate_audit,
                      "environment": environment, "checks": metrics["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
