# CO3133 Assignment 1: Fashion-MNIST

Five classifiers trained from scratch: Linear, MLP, CNN, row-sequence LSTM,
and patch Transformer. All use seed 42, the same stratified 54,000/6,000
training/validation split, training-only normalization, Adam at 0.001, batch
size 128, ten epochs and no augmentation. Checkpoints are selected by lowest
validation loss; the official 10,000-image test set is used after selection.

## Setup

Run these PowerShell commands from `A1/` using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

The Windows requirements pin PyTorch with the CUDA 12.8 runtime. Training and
verification select CUDA when available and otherwise CPU (MPS on supported
Macs). `requirements-windows-lock.txt` records the complete tested Windows
environment. `requirements.txt` includes the same Windows submission dependencies.
Fashion-MNIST downloads to ignored `data/` on first use. `split_cache.json`
records the shared split, source hashes and normalization statistics.

## Check the submitted runs without retraining

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
foreach ($model in @('linear', 'mlp', 'cnn', 'lstm', 'transformer')) {
    .\.venv\Scripts\python.exe tests/verify_run.py --model $model --no-write
    if ($LASTEXITCODE -ne 0) { throw "Verification failed: $model" }
}
.\.venv\Scripts\python.exe -m a1.compare
```

The verifier reloads each checkpoint, checks validation-based selection,
recomputes all 10,000 test predictions, losses, metrics and confusion counts,
checks dataset/split/checkpoint hashes, compares CPU and selected-device
predictions on one batch, and validates the saved PNGs. `--no-write` preserves
the original verification records. On CPU-only systems the portability check
compares two CPU loads.

The comparison command reads the five original `summary.json` and
`test_summary.json` files, checks shared provenance and checkpoint hashes, then
writes CSV, JSON, Markdown and a plot to `runs/m2/comparison/`. It never trains
or evaluates a model. It also works from the repository root:

```powershell
A1\.venv\Scripts\python.exe -m A1.a1.compare
```

## Saved test results

| Model | Parameters | Epoch | Test accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Linear | 7,850 | 9 | 84.26% | 0.841694 |
| MLP | 235,146 | 9 | 88.16% | 0.880775 |
| CNN | 421,642 | 10 | 91.94% | 0.918886 |
| LSTM | 82,186 | 9 | 88.48% | 0.883496 |
| Transformer | 71,946 | 10 | 85.55% | 0.852387 |

These results are from one seed and split. CNN has the highest measured
accuracy and macro-F1. LSTM exceeds MLP by 0.32 percentage points (32 test
images); repeated seeds are needed to assess whether this difference is stable.
The [generated comparison](runs/m2/comparison/comparison.md) is the reproducible
table source. Every model was trained and evaluated on CUDA using the GTX
1660 SUPER. Each row uses the same run for classification scores and timings.
Forward time is amortized over 128 device-resident images; full-test timing
includes loading, transfers, loss and predictions. Sessions were recorded
separately, without repeated-session uncertainty estimates.

The submitted Linear/MLP runs are in `runs/cuda_baselines/` and use
`configs/m1.json`, which explicitly requires CUDA. Run the complete baseline
pipeline with `python -m a1.cuda_baselines`; this trains or validates completed
matching runs, evaluates them on CUDA and verifies all saved predictions.
The former MPS runs are excluded from Git tracking and ignored locally.

## Reproduce training and evaluation

Run from `A1/` in the environment above. These commands can replace files in
the corresponding run directory; archive any run you need to retain first.

| Model | Config | Run directory | Notes |
| --- | --- | --- | --- |
| Linear | `configs/m1.json` | `runs/cuda_baselines/linear/` | [Linear guide](docs/LINEAR.md) |
| MLP | `configs/m1.json` | `runs/cuda_baselines/mlp/` | [MLP guide](docs/MLP.md) |
| CNN | `configs/cnn.json` | `runs/m2/cnn/` | [CNN notes](docs/CNN.md) |
| LSTM | `configs/lstm.json` | `runs/m2/lstm/` | [LSTM notes](docs/LSTM.md) |
| Transformer | `configs/transformer.json` | `runs/m2/transformer/` | [Transformer notes](docs/TRANSFORMER.md) |

For example, to train and evaluate CNN:

```powershell
.\.venv\Scripts\python.exe -m a1.train --model cnn --config configs/cnn.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m2/cnn/best.pt --device cuda
.\.venv\Scripts\python.exe tests/verify_run.py --model cnn
```

Use the matching model/config/run path from the table for the other models.
`--reuse-existing` validates and reuses matching completed training results,
but trains if they are absent or fail its matching checks. For a check that
never trains, use the verifier instead. The evaluator accepts `--device cpu`
and writes results beside the checkpoint.

## Notebook, diagrams and report

Open `dl1-261.ipynb` from `A1/`. It contains EDA, preprocessing checks,
saved validation/test results for all five models and the final comparison.
The result cells read saved artifacts; training is an explicit CLI step.
Recreate the sequence diagrams with `python -m a1.plot_sequence` and
`python -m a1.plot_patches`.

- `a1/`: data pipeline, models, training, evaluation, comparison and diagrams.
- `configs/`: reproducible experiment settings.
- `tests/`: model tests and independent saved-run verification.
- `figures/`: EDA and representation figures used by the assignment page.
- `docs/`: one guide per model and the [overall workflow](docs/overall.md).
- `report-part3/complete-report/`: [complete PDF](report-part3/complete-report/report.pdf), editable LaTeX, figures and build instructions.
- `report/` and the other `report-part3/` files: retained earlier report stages and references.
- `../AI_USAGE.md`: existing disclosure, preserved during this cleanup.

See the [assignment page](../assignment1.html) for the full results discussion.
