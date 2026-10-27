# MLP classifier

[Overall workflow](overall.md) · [Model code](../a1/models.py) · [Config](../configs/m1.json)

## Main idea

The multilayer perceptron (MLP) reads all 784 pixels together. Two hidden layers
with ReLU activations let it learn more complex patterns than the Linear model.

## Input and layers

`B` means the number of images in a batch.

```text
Normalized image -> flatten to 784 pixels
Input: (B, 784)
Flatten
Linear(784, 256) -> ReLU -> Dropout(0.2)
Linear(256, 128) -> ReLU -> Dropout(0.2)
Linear(128, 10)
Output: (B, 10) class scores
```

`MLPClassifier` has **235,146 trainable parameters**, including biases.
Each linear layer combines its input features. ReLU replaces negative values
with zero, introducing nonlinearity. During training, dropout randomly zeros
20% of its input values to reduce reliance on particular features. Evaluation
disables dropout.

The final layer returns raw class scores for cross-entropy loss. The class with
the largest score becomes the prediction.

## Why include it?

It tests whether nonlinear combinations of pixels improve on a single linear
layer. Flattening keeps every pixel, but the architecture does not explicitly
use nearby pixels as image neighborhoods. It also has many more parameters than
the Linear model.

## Run it

Run from `A1/` in PowerShell after the [shared setup](overall.md#setup).

```powershell
.\.venv\Scripts\python.exe -m a1.train --model mlp --config configs/m1.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/cuda_baselines/mlp/best.pt --device cuda
```

The config requires CUDA and saves files in `runs/cuda_baselines/mlp/`.
See [overall.md](overall.md#saved-files) for the output files and shared training
settings. Training again can replace the files in that directory.
