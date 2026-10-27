# Linear classifier

[Overall workflow](overall.md) · [Model code](../a1/models.py) · [Config](../configs/m1.json)

## Main idea

The simplest baseline: each pixel contributes directly to a score for each
clothing class. There are no hidden layers or nonlinear activations.

## Input and layers

`B` means the number of images in a batch.

```text
Normalized image -> flatten to 784 pixels
Input: (B, 784)
Linear(784, 10)
Output: (B, 10) class scores
```

`LinearClassifier` also flattens its input in `forward()`, so it can accept an
image tensor directly. The normal data pipeline already supplies flat inputs.
The model has **7,850 trainable parameters**, including biases.

For each class, the layer multiplies the pixels by learned weights and adds a
bias. The class with the largest score becomes the prediction. Training passes
the raw scores, called logits, directly to cross-entropy loss.

## Why include it?

It gives a small, easy-to-understand baseline. More complex models should show
whether hidden layers, local image features, or sequences improve the result.
This model cannot build multiple levels of features or explicitly model
nonlinear combinations of pixels.

## Run it

Run from `A1/` in PowerShell after the [shared setup](overall.md#setup).

```powershell
.\.venv\Scripts\python.exe -m a1.train --model linear --config configs/m1.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/cuda_baselines/linear/best.pt --device cuda
```

The config requires CUDA and saves files in `runs/cuda_baselines/linear/`.
See [overall.md](overall.md#saved-files) for the output files and shared training
settings. Training again can replace the files in that directory.
