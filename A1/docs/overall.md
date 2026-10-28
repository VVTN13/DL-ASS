# Fashion-MNIST: overall workflow

This project trains five models to classify 28 x 28 grayscale clothing images
into ten classes. Each model has its own short guide:

| Model | Guide | Input per image | Parameters |
| --- | --- | --- | ---: |
| Linear | [LINEAR.md](LINEAR.md) | 784 pixel values | 7,850 |
| MLP | [MLP.md](MLP.md) | 784 pixel values | 235,146 |
| CNN | [CNN.md](CNN.md) | 1 x 28 x 28 image | 421,642 |
| LSTM | [LSTM.md](LSTM.md) | 28 rows x 28 values | 82,186 |
| Transformer | [TRANSFORMER.md](TRANSFORMER.md) | 49 patches x 16 values | 71,946 |

## Setup

Run commands from `A1/` in PowerShell. To create the Windows environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

`requirements-windows-lock.txt` records the tested dependency versions.
The Linear and MLP configs explicitly require CUDA. The other model configs
use `auto`, which chooses MPS when available, then CUDA, then CPU.

## Where the code lives

| File or folder | Purpose |
| --- | --- |
| [a1/data.py](../a1/data.py) | Download data, split it, normalize pixels, and build batches |
| [a1/models.py](../a1/models.py) | Define all five models and select their input layouts |
| [a1/train.py](../a1/train.py) | Run training and validation; save the best checkpoint |
| [a1/evaluate.py](../a1/evaluate.py) | Evaluate a checkpoint on the test set and create plots |
| [a1/compare.py](../a1/compare.py) | Compare the five saved runs |
| [a1/cuda_baselines.py](../a1/cuda_baselines.py) | Train or reuse, evaluate, and verify both CUDA baselines |
| [configs/](../configs/) | JSON settings for each experiment |
| [tests/](../tests/) | Model tests and saved-run verification |
| [dl1-261.ipynb](../dl1-261.ipynb) | Explore data and display saved results |
| [report/](../report/) | Report sources, figures, build scripts, and PDFs |

## Data preparation

1. Download Fashion-MNIST through `torchvision` into `data/`.
2. Split the official 60,000 training images into **54,000 training** and
   **6,000 validation** images. Stratification keeps class proportions, using
   seed 42. Keep the official **10,000 test** images separate.
3. Convert pixel values from 0–255 to floating-point values in 0–1.
4. Calculate the mean and standard deviation using only the training subset.
   Normalize every split with `(image - mean) / std`. The current cache gives
   approximately **0.286209** mean and **0.353160** standard deviation.
5. Rearrange the normalized image into the model's input layout from the table
   above. Every layout retains all 784 input pixels.
6. Build data loaders: shuffle training batches; keep validation and test
   batches in fixed order.

`split_cache.json` stores the indices, normalization values, and hashes.
The loader checks the cache against the dataset, seed, and split before reusing
it. Augmentation is supported for training, but all current configs disable it.
The optional `col_seq` layout is also supported by the data code; none of the
five models selects it.

## Shared training settings

| Setting | Value in the current configs |
| --- | --- |
| Seed | 42 |
| Batch size | 128 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Cross-entropy |
| Augmentation | Off |
| Data-loader workers | 0 |
| Checkpoint selection | Lowest validation loss |

There is no learning-rate scheduler or early stopping in the training code.
All models run the configured number of epochs.

## The training loop

`run_experiment()` loads the config and data, creates the model, and creates
the optimizer. It then repeats this loop:

```text
For each epoch:
    Training pass, for each batch:
        Move inputs and labels to the selected device
        Clear old gradients
        Run the model to get 10 class scores per image
        Calculate cross-entropy loss
        Backpropagate the loss
        Update weights with Adam

    Validation pass:
        Disable dropout and gradient calculation
        Predict validation batches without updating weights
        Calculate validation loss, accuracy, and macro-F1

    Record training and validation metrics
    If validation loss is lower than every earlier epoch:
        Save best.pt
```

`run_epoch()` handles both passes: supplying an optimizer enables training;
omitting it enables validation. Loss is averaged over images, and predictions
use the class with the largest score. Cross-entropy receives raw scores, so
the models do not apply softmax first.

After the loop, the code reloads the best checkpoint, checks its validation
metrics, and writes the history and training summary. The test loader is
created during data preparation, but its batches are not used by this loop.

## Evaluation and metrics

`a1.evaluate` reloads `best.pt`, checks that the dataset, split, and normalization
match its metadata, and predicts all test images with gradients and dropout
disabled. It saves:

- **Loss:** average cross-entropy; lower is better.
- **Accuracy:** fraction of images classified correctly; higher is better.
- **Macro-F1:** average F1 across the ten classes, giving each class equal weight.
- **Confusion matrix:** true classes in rows and predicted classes in columns.
- **Example grids:** sampled correct and incorrect predictions. Softmax supplies
  displayed confidence, which is not calibrated.

Forward timing uses one device-resident batch, ten warmup passes, and 100 timed
passes. Full-test timing includes loading, transfers, loss, and predictions;
it excludes plot creation. These timings cover different work.

## Saved files

Paths are relative to `A1/`:

| Models | Config | Run directory |
| --- | --- | --- |
| Linear, MLP | `configs/m1.json` | `runs/cuda_baselines/<model>/` |
| CNN | `configs/cnn.json` | `runs/m2/cnn/` |
| LSTM | `configs/lstm.json` | `runs/m2/lstm/` |
| Transformer | `configs/transformer.json` | `runs/m2/transformer/` |

`configs/m1_cuda.json` currently has the same settings as `configs/m1.json`.

| File | Created by | Contents |
| --- | --- | --- |
| `best.pt` | Training | Selected weights, epoch, config, normalization, and hashes |
| `history.csv` | Training | Training and validation metrics for every epoch |
| `summary.json` | Training | Best validation metrics, parameter count, time, and environment |
| `test_summary.json` | Evaluation | Test metrics, per-class scores, timing, and hashes |
| `test_predictions.csv` | Evaluation | One prediction and confidence per test image |
| `confusion_matrix.csv` | Evaluation | Counts for each true/predicted class pair |
| Four PNGs | Evaluation | Learning curves, confusion matrix, correct and incorrect examples |
| `verification.json` | Verification | Saved-run check results, when writing is enabled |

Training and evaluation write into the run directory. `--reuse-existing` on
the training command checks and reuses a matching completed training run;
if it is missing or fails the matching checks, training runs again.

## Checks and comparison

Run model tests without training:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Once both baseline runs and the target model's artifacts exist, check a saved
run without rewriting its verification record. Replace `cnn` as needed:

```powershell
.\.venv\Scripts\python.exe tests/verify_run.py --model cnn --no-write
```

The verifier reloads weights, checks hashes and checkpoint selection, recomputes
test predictions and metrics, checks plots, and compares CPU/device predictions
on one batch. Model-specific verifier wrappers also exist for CNN, LSTM, and
Transformer.

To run the complete CUDA Linear/MLP pipeline, then export a comparison once all
five runs exist:

```powershell
.\.venv\Scripts\python.exe -m a1.cuda_baselines
.\.venv\Scripts\python.exe -m a1.compare
```

The comparison checks shared settings and checkpoint hashes, then writes JSON,
CSV, Markdown, and a PNG to `runs/m2/comparison/`. It reads saved artifacts and
does not train or evaluate models. Use each run's summaries as the source of
scores; one seed and split do not establish a general ranking of model families.

In this checkout, `runs/cuda_baselines/` is currently absent. The baseline
pipeline must create those runs before the full comparison and shared saved-run
verifier can work. Older files under `runs/m1/` are not read by these commands.

## Diagrams and report

`python -m a1.plot_sequence` creates `figures/lstm_row_sequence.png`.
`python -m a1.plot_patches` creates `figures/transformer_patch_sequence.png`.
Both show how the shared preprocessing forms sequence inputs.

The notebook contains data exploration and saved-result views. The HTML
[assignment page](../../assignment1.html) presents the project, and
[report/](../report/) contains the separate report generation materials.
These presentation files do not replace the training or evaluation code.
