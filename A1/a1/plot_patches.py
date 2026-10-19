"""Illustrate the Transformer patch order using one training image.

Run from A1: python -m a1.plot_patches
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .data import build_dataloaders


def main():
    train, _, _ = build_dataloaders(mode="patch_seq")
    patches, label = train.dataset[0]
    index = int(train.dataset.indices[0])
    raw = train.dataset.base.data[index].numpy()
    patch_row, patch_col = 3, 2
    token = 7 * patch_row + patch_col
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), layout="constrained")
    fig.suptitle("Transformer: 49 image patches become 49 tokens", fontsize=16)
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=255)
    axes[0].set(title=f"Training image #{index}: {train.dataset.base.classes[label]}",
                xlabel="Pixel column", ylabel="Pixel row",
                xticks=[0, 7, 14, 21, 27], yticks=[0, 7, 14, 21, 27])
    for edge in range(4, 28, 4):
        axes[0].axhline(edge - 0.5, color="#93c5fd", linewidth=0.7)
        axes[0].axvline(edge - 0.5, color="#93c5fd", linewidth=0.7)
    for row in range(7):
        for col in range(7):
            axes[0].text(4 * col + 1.5, 4 * row + 1.5, str(7 * row + col),
                         ha="center", va="center", fontsize=8, color="#f97316",
                         bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=0.4))
    axes[0].add_patch(Rectangle((4 * patch_col - 0.5, 4 * patch_row - 0.5), 4, 4,
                                fill=False, edgecolor="#f97316", linewidth=2.5))
    picture = axes[1].imshow(patches.numpy(), cmap="viridis", aspect="auto")
    axes[1].set(title="Patch sequence: (49 patches, 16 pixels)",
                xlabel="Pixel within a patch (row-major, 0 to 15)",
                ylabel="Patch index (raster order, 0 to 48)",
                xticks=[0, 3, 7, 11, 15], yticks=[0, 7, 14, token, 35, 48])
    axes[1].add_patch(Rectangle((-0.5, token - 0.5), 16, 1, fill=False,
                                edgecolor="#f97316", linewidth=2))
    axes[1].text(0.5, -0.15, f"Orange patch: index {token} = 7 x {patch_row} + {patch_col}",
                 transform=axes[1].transAxes, ha="center", color="#b45309", fontsize=10)
    fig.colorbar(picture, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized pixel")
    axes[2].axis("off")
    axes[2].set_title("Batch and classifier shapes")
    boxes = [
        (0.88, "B images -> (B, 49, 16)\n49 non-overlapping 4 x 4 patches"),
        (0.66, "Linear(16, 64) + learned positions\nDropout(0.2) -> (B, 49, 64)"),
        (0.44, "2 encoder blocks: 4 attention heads\nGELU feed-forward: 64 -> 128 -> 64"),
        (0.22, "LayerNorm, mean over 49 tokens\n(B, 64), then Dropout(0.2)"),
        (0.00, "Linear(64, 10) -> (B, 10) logits\nArgmax selects the clothing class"),
    ]
    for y, text in boxes:
        axes[2].text(0.5, y, text, transform=axes[2].transAxes, ha="center", va="center",
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.6", facecolor="#eff6ff",
                                           edgecolor="#2563eb"))
    for y in (0.77, 0.55, 0.33, 0.11):
        axes[2].annotate("", xy=(0.5, y - 0.035), xytext=(0.5, y + 0.035),
                         xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#2563eb"))
    output = Path("figures/transformer_patch_sequence.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
