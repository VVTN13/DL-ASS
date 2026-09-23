# CO3133 Assignment 1 - M1 Draft

This repository contains our M1 work on Fashion-MNIST. For this milestone we
implemented the data pipeline, exploratory data analysis, a linear classifier,
and an MLP. The official test set is saved for the final milestone.

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
- `a1/models.py`: linear and MLP models
- `a1/train.py`: training and validation loops
- `configs/m1.json`: experiment settings
- `runs/m1/`: checkpoints, histories, and result summaries
- `docs/M1_ARCHITECTURE.md`: short explanation of the M1 pipeline
- `figures/`: plots exported from the notebook (used on the GitHub Pages site)
- `../AI_USAGE.md`: AI-use disclosure (repository root)

## Current results

| Model | Best epoch | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: | ---: |
| Linear | 9 | 0.8648 | 0.8641 |
| MLP | 8 | 0.8987 | 0.8985 |

These are validation results from one split and one seed. They are not final
test results. CNN, LSTM/GRU, Transformer, test evaluation, and the complete
comparison will be added for M2.
