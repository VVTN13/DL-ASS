# LSTM classifier

[Overall workflow](overall.md) · [Model code](../a1/models.py) · [Config](../configs/lstm.json)

## Main idea

The long short-term memory network (LSTM) reads the image one row at a time,
from top to bottom. Its hidden state carries information from earlier rows.

## Input and layers

The `row_seq` conversion removes the single channel dimension. Each image
becomes 28 steps, with 28 pixel values per step. All 784 pixels are retained:

```text
sequence[row, column] = normalized_image[0, row, column]
```

`B` means the number of images in a batch.

```text
Input: (B, 28 rows, 28 pixel values)
LSTM(input_size=28, hidden_size=128, batch_first=True)
Take the final hidden state: (B, 128)
Dropout(0.2)
Linear(128, 10)
Output: (B, 10) class scores
```

`LSTMClassifier` has **82,186 trainable parameters**, including biases. It uses
one recurrent layer and reads in one direction. Each forward call starts with
zero hidden and cell states; information is not carried between images or
batches.

Inside the LSTM, gates control what information to keep, update, and expose.
The code classifies `hidden[-1]`, the hidden state after the last row. Dropout
is applied to this readout during training and disabled during evaluation.
The final layer returns raw class scores for cross-entropy loss.

## Why include it?

It explores image classification as sequence processing. The model can combine
information from different rows, but it must process 28 dependent steps and
compress the image into one final state. This imposes a top-to-bottom order on
a two-dimensional image; fewer parameters do not guarantee faster execution.

## Run it

Run from `A1/` in PowerShell after the [shared setup](overall.md#setup).

```powershell
.\.venv\Scripts\python.exe -m a1.train --model lstm --config configs/lstm.json
.\.venv\Scripts\python.exe -m a1.evaluate --checkpoint runs/m2/lstm/best.pt
.\.venv\Scripts\python.exe -m a1.plot_sequence
```

Run files go to `runs/m2/lstm/`. The last command creates
`figures/lstm_row_sequence.png`. See [overall.md](overall.md#saved-files) for
the shared output files and settings. Training again can replace run files.
