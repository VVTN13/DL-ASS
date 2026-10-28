# CNN classifier

[Overall workflow](overall.md) · [Model code](../a1/models.py) · [Config](../configs/cnn.json)

## Main idea

The convolutional neural network (CNN) keeps the image layout. Small filters
learn local patterns, and pooling reduces the image size before classification.

## Input and layers

`B` means the number of images in a batch. Image dimensions below are
`(batch, channels, height, width)`.

| Layer | Output shape |
| --- | --- |
| Normalized grayscale image | `(B, 1, 28, 28)` |
| Conv2d(1, 32, kernel=3, padding=1), ReLU | `(B, 32, 28, 28)` |
| MaxPool2d(2) | `(B, 32, 14, 14)` |
| Conv2d(32, 64, kernel=3, padding=1), ReLU | `(B, 64, 14, 14)` |
| MaxPool2d(2) | `(B, 64, 7, 7)` |
| Flatten | `(B, 3136)` |
| Linear(3136, 128), ReLU, Dropout(0.2) | `(B, 128)` |
| Linear(128, 10) | `(B, 10)` |

`CNNClassifier` has **421,642 trainable parameters**, including biases.

Each convolution reuses the same learned filters across the image. Padding
keeps the spatial size unchanged during convolution. Each pooling layer keeps
the maximum value in each 2 x 2 region, halving the height and width.

The final layers combine the learned features into raw class scores.
Cross-entropy uses those scores directly; the largest score gives the predicted
class. Dropout is active during training and disabled during evaluation.

## Why include it?

Clothing images contain useful local shapes and textures. Convolution gives
the model a direct way to learn from neighboring pixels. Pooling compresses
spatial detail, and this model has the largest parameter count of the five.

## Run it

Run from `A1/` in PowerShell after the [shared setup](overall.md#setup).

```powershell
.\.venv\Scripts\python.exe -m a1.train --model cnn --config configs/cnn.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m2/cnn/best.pt
```

Files are saved in `runs/m2/cnn/`. See [overall.md](overall.md#saved-files)
for the output files and shared training settings. Training again can replace
the files in that directory.
