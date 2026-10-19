# CO3133 Assignment 1

The M1 Linear/MLP baseline is documented below. The custom CNN is now implemented
for M2; see [CNN experiment notes](docs/CNN.md) for its architecture, evaluation,
and measured results. The LSTM is also implemented and evaluated; see
[LSTM experiment notes](docs/LSTM.md) for the image-to-sequence mapping, diagram,
architecture and measured results.
The patch-based Transformer is documented in [Transformer experiment notes](docs/TRANSFORMER.md),
including token ordering, attention blocks and reproduction commands.

## Local Windows setup (CNN)

From `A1/`, use PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python -m a1.train --model cnn --config configs/cnn.json
python -m a1.evaluate --checkpoint runs/m2/cnn/best.pt
python tests/verify_cnn_run.py
```

Activation is optional: the full `.\.venv\Scripts\python.exe` path works for every
command. The Windows requirements use CUDA 12.8 PyTorch wheels; `device: auto`
uses CUDA when available and CPU otherwise. The original `requirements.txt`
preserves the environment recorded for the M1 runs.
`requirements-windows-lock.txt` records every package installed in the tested
local environment; use it instead of `requirements-windows.txt` to reproduce
the full set of dependency versions.

CNN outputs are saved under `runs/m2/cnn/`. To validate and reuse an existing
training run, add `--reuse-existing` to the training command. Test results are
reported after validation-based checkpoint selection; they must not be used to
tune the model. The evaluator also accepts `--device cpu` for a reload check.

## LSTM reproduction

Use the same local `.venv`, then run from `A1/`:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m a1.train --model lstm --config configs/lstm.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m2/lstm/best.pt
.\.venv\Scripts\python.exe tests/verify_lstm_run.py
.\.venv\Scripts\python.exe -m a1.plot_sequence
```

Outputs are in `runs/m2/lstm/`. As with CNN, `--reuse-existing` verifies and
reuses a matching completed training run. Each image becomes 28 top-to-bottom
row steps with 28 pixel features per step, giving `(batch, 28, 28)` inputs.
The final 128-value hidden state feeds dropout and a 10-class linear head.
Normalization, split and training budget match the CNN. The detailed notes
explain the recurrent memory, parameter count and limitations.

## Transformer reproduction

Use the same local `.venv`, then run from `A1/`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m a1.train --model transformer --config configs/transformer.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m2/transformer/best.pt
.\.venv\Scripts\python.exe tests/verify_transformer_run.py
.\.venv\Scripts\python.exe -m a1.plot_patches
```

The model reads 49 non-overlapping 4x4 patches in raster order. Each patch's
16 pixels project to 64 features, receive a learned position vector, and pass
through two encoder blocks with four attention heads. Mean pooling feeds the
10-class head. It has 71,946 parameters and uses the shared split, normalization
and 10-epoch budget. Outputs are in `runs/m2/transformer/`.
Add `--reuse-existing` to validate and reuse a completed training run.

## M1 draft

This repository contains our M1 work on Fashion-MNIST. For this milestone we
implemented the data pipeline, exploratory data analysis, a linear classifier,
and an MLP. Their validation-selected checkpoints have now been evaluated on
the official test set for M2; see [baseline test notes](docs/BASELINE_TEST.md).

## Setup

We used Python 3.14. From the `A1/` folder, run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Fashion-MNIST is downloaded automatically on the first run.

## Training

```sh
.venv/bin/python -m a1.train --model linear --config configs/m1.json
.venv/bin/python -m a1.train --model mlp --config configs/m1.json
```

Both models use the same seed and train/validation split. The main settings are:

- seed: 42
- split: 54,000 training and 6,000 validation images
- batch size: 128
- optimizer: Adam
- learning rate: 0.001
- epochs: 10
- checkpoint rule: lowest validation loss

The program chooses MPS, CUDA, or CPU depending on the machine. To force CPU,
change `device` to `cpu` in `configs/m1.json`.

## Notebook

Open `dl1-261.ipynb` and run all cells. It contains the EDA, data checks,
training results, and learning curves. Existing matching runs can be reused,
so the models do not have to be trained again every time the notebook opens.

## Files

- `a1/data.py`: dataset loading, train/validation split, preprocessing, and DataLoaders
- `a1/models.py`: linear, MLP, custom CNN, row-sequence LSTM, and patch Transformer models
- `a1/train.py`: training and validation loops
- `a1/evaluate.py`: checkpoint-based test evaluation, timing, and plots
- `configs/m1.json`: experiment settings
- `configs/cnn.json`: CNN experiment settings
- `configs/lstm.json`: LSTM experiment settings
- `configs/transformer.json`: Transformer experiment settings
- `a1/plot_sequence.py`: reproducible image-to-row-sequence diagram
- `a1/plot_patches.py`: reproducible image-to-patch-sequence diagram
- `tests/verify_run.py`: shared CNN/LSTM/Transformer artifact and checkpoint verification
- `runs/m1/`: checkpoints, histories, and result summaries
- `docs/M1_ARCHITECTURE.md`: short explanation of the M1 pipeline
- `figures/`: plots exported from the notebook (used on the GitHub Pages site)
- `../AI_USAGE.md`: AI-use disclosure (repository root)

## Current results

| Model | Best epoch | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: | ---: |
| Linear | 9 | 0.8648 | 0.8641 |
| MLP | 8 | 0.8987 | 0.8985 |

These are validation results from one split and one seed. Final baseline test
evaluation on 19 October 2026 used the existing checkpoints without retraining:

| Model | Selected epoch | Test loss | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Linear | 9 | 0.445443 | 84.26% | 0.841694 |
| MLP | 8 | 0.325145 | 88.50% | 0.884585 |

Each model was evaluated on all 10,000 official test images. Test summaries,
per-class metrics, prediction CSVs, confusion matrices, correct/incorrect example
plots and verification records are in `runs/m1/linear/` and `runs/m1/mlp/`.
To reproduce from `A1/` using the local Windows environment:

```powershell
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m1/linear/best.pt
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m1/mlp/best.pt
.\.venv\Scripts\python.exe tests/verify_run.py --model linear
.\.venv\Scripts\python.exe tests/verify_run.py --model mlp
```

Both saved-run checks passed, including all predictions, recomputed metrics and
confusion counts, and CPU/CUDA agreement on a 128-image batch.
CNN training and test evaluation are documented in `docs/CNN.md`.
The LSTM results are documented in `docs/LSTM.md`, and Transformer results in
`docs/TRANSFORMER.md`. The complete five-model comparison remains M2 work.
Original M1 training times were measured on MPS, while these test evaluations
used a GTX 1660 SUPER. A fair speed comparison requires shared hardware.

The Transformer selected epoch 10 by validation loss and reached **85.55% test
accuracy** and **0.852387 test macro-F1** on 10,000 images. Its saved-run
verification passed, including all predictions and CPU/CUDA agreement.
