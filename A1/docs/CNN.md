# Custom CNN experiment

This is a small CNN written directly with PyTorch layers. It uses no pretrained
weights. Fashion-MNIST images stay as `1 x 28 x 28` tensors so the convolutions can
use the neighboring pixels instead of treating every pixel independently.

## Architecture

| Layer | Output per image |
| --- | --- |
| Input, normalized with training-only statistics | 1 x 28 x 28 |
| Conv2d(1, 32, 3, padding=1), ReLU | 32 x 28 x 28 |
| MaxPool2d(2) | 32 x 14 x 14 |
| Conv2d(32, 64, 3, padding=1), ReLU | 64 x 14 x 14 |
| MaxPool2d(2) | 64 x 7 x 7 |
| Flatten | 3,136 |
| Linear(3136, 128), ReLU, Dropout(0.2) | 128 |
| Linear(128, 10) | 10 logits |

The model has **421,642 trainable parameters**. A 3 x 3 convolution learns filters
over local neighborhoods; the same filter is shared across image locations.
Each output channel is a feature map showing where that learned pattern appears.
ReLU introduces nonlinearity. Pooling retains the largest activation in each
2 x 2 region, reduces computation, and gives some tolerance to small shifts.
The classifier combines these spatial features. Dropout is active only during
training and reduces reliance on individual hidden units.

The output is raw logits. `CrossEntropyLoss` handles the required log-softmax;
softmax is used only to display prediction confidence during evaluation.

## Protocol

- Same cached stratified split as M1: 54,000 train, 6,000 validation, 10,000 test.
- Seed 42; train-only mean 0.28620901297676965 and std 0.3531597492723218.
- Adam, learning rate 0.001, batch size 128, 10 epochs, no augmentation.
- Training and inference use float32, with no mixed precision.
- Choose the checkpoint with the lowest validation loss, then evaluate test data.
- cuDNN benchmarking is disabled and deterministic convolution algorithms are
  requested. Numerical identity across different hardware/software is not promised.

Ten epochs were chosen to keep the M1 budget. This is one fixed run, not a
test-guided architecture or hyperparameter search. Test metrics describe this
checkpoint only. Compare final test results only once the other models have their
own test evaluations under the same protocol.

## Reproduction

Run the commands in `A1/README.md`. Training and evaluation are separate commands.
The full downloaded dataset and local virtual environment are excluded from Git.

`runs/m2/cnn/` contains:

- `best.pt`: validation-selected weights and data/config metadata.
- `summary.json`, `history.csv`: training settings, validation metrics and curves.
- `test_summary.json`: test metrics, class metrics, timing, checkpoint checksum,
  dataset checksums and evaluation hardware/software.
- `test_predictions.csv`: all test predictions, labels and confidence values.
- `confusion_matrix.csv`, `confusion_matrix.png`: error counts by class.
- `learning_curves.png`: training/validation loss, accuracy and macro-F1.
- `correct_examples.png`, `incorrect_examples.png`: 12 reproducibly sampled examples
  of each kind, identified by their official test index.
- `verification.json`: checks of the saved split, checkpoint, metrics, predictions,
  confusion counts, PNG files, and CPU/device agreement.

Inference timing warms up for 10 batches and averages 100 forward passes of a
fixed device-resident batch of 128 images. It excludes data loading and host
transfers. Dividing batch latency by 128 reports an amortized cost per image,
not the latency of a batch of one. A separate full-test timer includes loading,
transfers, loss and predictions. Training time includes validation and checkpoint
writes, and excludes downloading/preparing the data and final test evaluation.

The saved M1 times are from Apple Silicon/MPS. They cannot support a speed ranking
against this Windows/CUDA run. Rerun all models on one machine for that comparison.

## Results from 11 October 2026

The run used Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128, torchvision
0.24.1+cu128 and a GTX 1660 SUPER (6 GiB). Full environment versions are recorded
in `requirements-windows-lock.txt`; run-specific metadata is in the JSON files.

| Measurement | Result |
| --- | ---: |
| Selected epoch | 10 |
| Validation loss | 0.200061 |
| Validation accuracy | 93.2833% |
| Validation macro-F1 | 0.932262 |
| Test loss | 0.236603 |
| Test accuracy | 91.94% (9,194 / 10,000) |
| Test macro-F1 | 0.918886 |
| Parameters | 421,642 |
| Training + validation time, 10 epochs | 105.03 s |
| Forward time, batch of 128 | 1.074 ms |
| Amortized forward time per image | 0.00839 ms |
| Complete test pass | 1.382 s |

The validation loss falls from 0.3124 to 0.2001, with a small increase at epoch 8.
Training loss continues falling to 0.1289. The growing train/validation gap
suggests some overfitting, although validation loss still improves at epoch 10.
Test accuracy is about 1.34 percentage points below validation accuracy.

CNN validation accuracy is about 3.42 percentage points above the saved MLP
result on the same split. Local filters and weight sharing give the CNN a useful
spatial bias, but this comparison also changes the architecture and parameter
count, so it does not isolate the effect of spatial structure. The MLP has
235,146 parameters; the CNN has about 1.79 times as many. The earlier MLP result
is validation-only, so it must not be compared directly with CNN test accuracy.

## Error analysis

The model makes 806 test errors. Shirt is the hardest class: recall is 71.8% and
F1 is 0.7518. Of the 1,000 true shirts, 103 are predicted as Coat, 95 as
T-shirt/top, and 47 as Pullover. There are also 94 true T-shirt/top images
predicted as Shirt. These measured errors support the EDA observation that
upper-body garments have similar silhouettes at this resolution.

Trouser recall is 98.8% and Bag recall is 98.4%; these classes are easier for
this checkpoint. The sampled error figure includes a Sandal predicted as Ankle
boot with 96.7% confidence (test index 902). High softmax confidence can therefore
accompany a wrong answer; these scores have not been calibrated.

This is one seed, one architecture and one 10-epoch budget. There is no estimate
of run-to-run variability, no robustness study, and no full five-model test
comparison yet. The test results were evaluated after the validation checkpoint
was fixed and did not guide changes to the model.

## Checks actually run

- `python -m pip check`: no broken dependencies.
- `python -m unittest discover -s tests -v`: four tests passed. They check image
  batches/raw logits, optimization reaching convolution weights and reducing
  loss, checkpoint reload/evaluation dropout, and existing Linear/MLP inputs.
- Full 10-epoch CUDA training on all 54,000 training examples, followed by
  evaluation of the selected checkpoint on all 10,000 test examples.
- `python tests/verify_cnn_run.py`: passed. Reloaded checkpoint metrics match the
  saved validation results; its epoch has the minimum validation loss. The data
  source and split hashes match both M1 baselines. All 10,000 saved labels match
  the official test labels; accuracy, F1 and confusion counts are recomputed.
- The same 128-image batch has identical CPU/CUDA predicted labels. Maximum
  absolute logit difference is 0.00001073, within the numerical tolerance.
- All four figures pass PNG integrity checks and were visually inspected for
  readable labels, layout and consistency with the reported results.

The check results are saved in `runs/m2/cnn/verification.json`.
