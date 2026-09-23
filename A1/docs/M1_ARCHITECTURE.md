# M1 pipeline and models

## Data pipeline

We use Fashion-MNIST from `torchvision`. It contains 60,000 training images and
10,000 test images. For M1, the official training set is split into 54,000
training images and 6,000 validation images using stratification and seed 42.
The official test set is not used yet.

The processing steps are:

```text
Fashion-MNIST image
    -> convert pixels to a tensor
    -> normalize using training-set statistics
    -> flatten to 784 values
    -> model
    -> 10 class logits
```

The normalization mean is `0.28620901297676965` and the standard deviation is
`0.3531597492723218`. Both were calculated using only the 54,000 training
images. Training batches are shuffled, while validation batches are not.

## Models

The linear model contains one `Linear(784, 10)` layer. It has 7,850 trainable
parameters and is used as the baseline.

The MLP contains:

```text
Flatten
Linear(784, 256) -> ReLU -> Dropout(0.2)
Linear(256, 128) -> ReLU -> Dropout(0.2)
Linear(128, 10)
```

It has 235,146 trainable parameters. Both models return raw logits because
`CrossEntropyLoss` applies the required softmax calculation internally.

## Training setup

- optimizer: Adam
- learning rate: 0.001
- batch size: 128
- epochs: 10
- seed: 42
- loss: cross-entropy
- checkpoint: epoch with the lowest validation loss
- metrics: loss, accuracy, and macro-F1

The same split and settings are used for both models. No data augmentation is
enabled for these M1 runs.

## Results

| Model | Best epoch | Validation loss | Accuracy | Macro-F1 | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 9 | 0.393237 | 0.864833 | 0.864135 | 7,850 |
| MLP | 8 | 0.282546 | 0.898667 | 0.898466 | 235,146 |

The MLP performs better on this validation split, but it also has many more
parameters. These results come from one seed, so they are only preliminary.
The held-out test set and the full five-model comparison are M2 work.

## Reproducing the runs

```sh
.venv/bin/python -m a1.train --model linear --config configs/m1.json
.venv/bin/python -m a1.train --model mlp --config configs/m1.json
```

Each run saves `best.pt`, `history.csv`, and `summary.json` under
`runs/m1/<model>/`. The notebook contains the EDA and plots.
