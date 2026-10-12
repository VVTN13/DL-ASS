"""Export a training-image illustration of the LSTM row-sequence layout.

Run from A1: python -m a1.plot_sequence
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .data import build_dataloaders


def main():
    train, _, _ = build_dataloaders(mode="row_seq")
    sequence, label = train.dataset[0]
    index = int(train.dataset.indices[0])
    raw = train.dataset.base.data[index].numpy()
    row = 12  # Zero-based image row 12 becomes sequence step 13.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    fig.suptitle("One image becomes 28 ordered rows, each with 28 pixel features", fontsize=15)

    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=255)
    axes[0].set(title=f"Training image #{index}: {train.dataset.base.classes[label]}",
                xlabel="Pixel column (0 to 27)", ylabel="Pixel row (0 to 27)",
                xticks=[0, 7, 14, 21, 27], yticks=[0, 7, 14, 21, 27])
    axes[0].add_patch(Rectangle((-0.5, row - 0.5), 28, 1, fill=False,
                                edgecolor="#f97316", linewidth=2))
    axes[0].text(0.5, -0.22, "Orange box: image row 12", transform=axes[0].transAxes,
                 ha="center", color="#b45309")

    picture = axes[1].imshow(sequence.numpy(), cmap="viridis", aspect="equal")
    axes[1].set(title="Sequence S: shape (28 steps, 28 features)",
                xlabel="Feature = pixel column (0 to 27)",
                ylabel="Step = image row + 1 (top to bottom)",
                xticks=[0, 7, 14, 21, 27], yticks=[0, 6, 12, 20, 27],
                yticklabels=[1, 7, 13, 21, 28])
    axes[1].add_patch(Rectangle((-0.5, row - 0.5), 28, 1, fill=False,
                                edgecolor="#f97316", linewidth=2))
    axes[1].text(0.5, -0.22, "Same row = step 13; all 784 pixels retained",
                 transform=axes[1].transAxes, ha="center", color="#b45309")
    fig.colorbar(picture, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Normalized pixel: (raw / 255 - mean) / std")

    axes[2].axis("off")
    axes[2].set_title("Batch and classifier shapes")
    boxes = [(0.82, "B images\nS: (B, 28 steps, 28 features)"),
             (0.57, "LSTM reads steps 1 to 28\n128 hidden + 128 cell values per image"),
             (0.32, "Final hidden state: (B, 128)\nDropout(0.2), then Linear(128, 10)"),
             (0.07, "10 logits per image: (B, 10)\nArgmax selects the clothing class")]
    for y, text in boxes:
        axes[2].text(0.5, y, text, transform=axes[2].transAxes, ha="center", va="center",
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.65", facecolor="#eff6ff",
                                           edgecolor="#2563eb"))
    for y in (0.70, 0.45, 0.20):
        axes[2].annotate("", xy=(0.5, y - 0.04), xytext=(0.5, y + 0.04),
                         xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#2563eb"))
    output = Path("figures/lstm_row_sequence.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
