# Transformer classifier

[Overall workflow](overall.md) · [Model code](../a1/models.py) · [Config](../configs/transformer.json)

## Main idea

The Transformer divides an image into small patches. Self-attention lets each
patch combine information from the other patches before the model classifies
the whole image. It is trained from scratch and uses no pretrained weights.

## Input and layers

The `patch_seq` conversion divides the normalized 28 x 28 image into a 7 x 7
grid of non-overlapping 4 x 4 patches. Patches go left to right, then top to
bottom. Pixels inside each patch use the same order. All 784 pixels are kept:
49 patches with 16 values each.

`B` means the number of images in a batch.

| Stage | Output shape |
| --- | --- |
| Patch input | `(B, 49, 16)` |
| Linear(16, 64) patch embedding | `(B, 49, 64)` |
| Add learned positions, Dropout(0.2) | `(B, 49, 64)` |
| Transpose for encoder | `(49, B, 64)` |
| Two Transformer encoder blocks | `(49, B, 64)` |
| Transpose back, LayerNorm(64) | `(B, 49, 64)` |
| Mean over patches, Dropout(0.2) | `(B, 64)` |
| Linear(64, 10) | `(B, 10)` |

`TransformerClassifier` has **71,946 trainable parameters**, including biases
and learned positions.

Each encoder block uses four attention heads and a feed-forward network
`64 -> 128 -> 64` with GELU activation. Layer normalization comes before each
of these operations, and residual connections add their inputs back to their
outputs. Dropout is 0.2. The two blocks are initialized separately.

Learned position vectors identify where each patch belongs. Every patch can
attend to all patches in its own image; there is no causal or padding mask.
The internal sequence-first layout follows the attention path selected in the
code for CPU/CUDA numerical agreement. It does not mix different images.

The model averages the final patch features rather than using a separate class
token. Its last layer returns raw class scores for cross-entropy loss.
Evaluation disables dropout.

## Why include it?

It explores relationships between patches through attention. The architecture
has fewer parameters than the CNN, but attention has its own computation cost.
A result from this small model and ten-epoch budget does not establish how all
Transformers compare with other model families.

## Run it

Run from `A1/` in PowerShell after the [shared setup](overall.md#setup).

```powershell
.\.venv\Scripts\python.exe -m a1.train --model transformer --config configs/transformer.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m2/transformer/best.pt
.\.venv\Scripts\python.exe -m a1.plot_patches
```

Run files go to `runs/m2/transformer/`. The last command creates
`figures/transformer_patch_sequence.png`. See [overall.md](overall.md#saved-files)
for the shared output files and settings. Training again can replace run files.
