r"""Build the offline LSTM/Transformer study document from this checkout's evidence.

Run from A1: .\.venv\Scripts\python.exe docs\study-guide\build_guide.py
Only writes this directory. Does not retrain models or rewrite run artifacts.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import inspect
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from a1.data import build_dataloaders, to_representation
from a1.models import LSTMClassifier, TransformerClassifier, count_parameters, input_mode
from a1.train import load_checkpoint, run_epoch, _config_hash

torch.set_num_threads(4)
E = html.escape


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_data(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th scope="col">{x}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def code(text, language="Python"):
    return f'<div class="code-label">{language}</div><pre><code>{E(text.strip())}</code></pre>'


def note(title, text, kind="note"):
    return f'<aside class="{kind}"><strong>{title}</strong><p>{text}</p></aside>'


def eq(text):
    return f'<div class="equation">{text}</div>'


chapters = []


def chapter(slug, title, subtitle, body):
    chapters.append((slug, title, f'<section id="{slug}" class="chapter"><div class="eyebrow">Chapter {len(chapters)+1:02d}</div><h2>{title}</h2><p class="deck">{subtitle}</p>{body}</section>'))


def source(obj):
    lines, start = inspect.getsourcelines(obj)
    rel = Path(inspect.getfile(obj)).relative_to(ROOT).as_posix()
    numbered = '\n'.join(f'{start+i:3d}  {line.rstrip()}' for i, line in enumerate(lines))
    return f'<p class="source">Snapshot: <a href="../../{rel}">{rel}</a>, starting at line {start}.</p>' + code(numbered)


records = {}
audit = {"date": "2026-10-28", "scope": "Current checkout, direct checkpoint verification; no retraining", "models": {}}
sample_base = None
for name in ("lstm", "transformer"):
    p = ROOT / "runs" / "m2" / name
    s, t, v = (read_json(p / f) for f in ("summary.json", "test_summary.json", "verification.json"))
    history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader((p / "history.csv").open())]
    config = s["config"]
    tr, val, test = build_dataloaders(mode=input_mode(name), batch_size=128, seed=config['seed'], val_ratio=config['val_ratio'], data_dir=config['data_dir'], split_cache_path=config['split_cache_path'], num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(p / "best.pt", device)
    assert (len(tr.dataset), len(val.dataset), len(test.dataset)) == (54000, 6000, 10000)
    assert not np.intersect1d(tr.dataset.indices, val.dataset.indices).size
    assert ck['source_sha256'] == s['source_sha256'] == t['source_sha256'] == test.source_sha256
    assert ck['split_sha256'] == s['split_sha256'] == t['split_sha256'] == test.split_sha256
    assert ck['config_sha256'] == s['config_sha256'] == _config_hash(config, name)
    assert config == read_json(ROOT / "configs" / f"{name}.json")
    assert ck['train_mean'] == s['train_mean'] == test.dataset.mean
    assert ck['train_std'] == s['train_std'] == test.dataset.std
    assert count_parameters(model) == s['parameters']
    assert ck['epoch'] == s['best_epoch'] == t['checkpoint_epoch'] == int(min(history, key=lambda r: r['val_loss'])['epoch'])
    assert hashlib.sha256((p / 'best.pt').read_bytes()).hexdigest() == t['checkpoint_sha256']
    measured = run_epoch(model, val, nn.CrossEntropyLoss(), torch.device(device))
    for metric in ('loss', 'accuracy', 'macro_f1'):
        assert abs(measured[metric] - s[f'best_val_{metric}']) < 1e-5
    rows = list(csv.DictReader((p / 'test_predictions.csv').open()))
    expected = np.array([int(r['predicted_label']) for r in rows])
    labels = np.array([int(r['true_label']) for r in rows])
    assert [int(r['test_index']) for r in rows] == list(range(10000))
    np.testing.assert_array_equal(labels, test.dataset.base.targets.numpy())
    predictions, confidences, total_loss = [], [], 0.
    with torch.inference_mode():
        for x, y in test:
            logits = model(x.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            confidences.extend(logits.softmax(1).amax(1).cpu().tolist())
            total_loss += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
    np.testing.assert_array_equal(expected, predictions)
    np.testing.assert_allclose([float(r['confidence']) for r in rows], confidences, rtol=1e-5, atol=1e-6)
    assert [int(r['correct']) for r in rows] == (expected == labels).astype(int).tolist()
    assert abs(accuracy_score(labels, expected) - t['test_accuracy']) < 1e-12
    assert abs(f1_score(labels, expected, average='macro') - t['test_macro_f1']) < 1e-12
    assert abs(total_loss / len(rows) - t['test_loss']) < 1e-5
    with (p / 'confusion_matrix.csv').open() as f:
        matrix = np.array([list(map(int, row[1:])) for row in list(csv.reader(f))[1:]])
    np.testing.assert_array_equal(matrix, confusion_matrix(labels, expected))
    digest = hashlib.sha256()
    digest.update(test.dataset.base.data.numpy().tobytes())
    digest.update(test.dataset.base.targets.numpy().tobytes())
    assert digest.hexdigest() == t['test_source_sha256']
    for stem in ('learning_curves', 'confusion_matrix', 'correct_examples', 'incorrect_examples'):
        with Image.open(p / f'{stem}.png') as picture:
            picture.verify()
    cpu, _ = load_checkpoint(p / 'best.pt', 'cpu')
    bx, _ = next(iter(test))
    with torch.inference_mode():
        a, b = cpu(bx), model(bx.to(device)).cpu()
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    assert torch.equal(a.argmax(1), b.argmax(1))
    # A few real held-out examples, with fixed checkpoint probabilities for the HTML.
    indices = [0, 1, 2, 3, 4, 6, 12]
    sample_x = torch.stack([test.dataset[i][0] for i in indices])
    with torch.inference_mode():
        probabilities = cpu(sample_x).softmax(1).tolist()
    if sample_base is None:
        sample_base = [{"index": i, "label": int(test.dataset.base.targets[i]), "pixels": test.dataset.base.data[i].flatten().tolist(), "probabilities": {}} for i in indices]
    for item, probs in zip(sample_base, probabilities):
        item['probabilities'][name] = probs
    audit['models'][name] = {'status': 'passed', 'device': device, 'test_images_recomputed': len(rows), 'validation_recomputed': measured, 'checkpoint_sha256': t['checkpoint_sha256'], 'parameters': count_parameters(cpu), 'cpu_device_max_logit_difference': float((a-b).abs().max()), 'checks': ['source/split/config/checkpoint hashes', 'normalization', 'all test labels, predictions, confidences and loss', 'accuracy, macro-F1 and confusion matrix', 'checkpoint selection from history', 'CPU/device labels on 128 images', 'four PNGs']}
    records[name] = {'summary': s, 'test': t, 'verification': v, 'history': history, 'matrix': matrix.tolist()}
    print(f'{name}: checkpoint, validation and all 10,000 test predictions verified', flush=True)

# Confirm the gate equations reproduce nn.LSTM on real normalized data.
lm, _ = load_checkpoint(ROOT / 'runs/m2/lstm/best.pt', 'cpu')
px = torch.tensor(sample_base[0]['pixels'], dtype=torch.float32).reshape(1,28,28) / 255
px = (px - records['lstm']['summary']['train_mean']) / records['lstm']['summary']['train_std']
h = torch.zeros(1,128); c = torch.zeros_like(h)
with torch.inference_mode():
    for row in px.unbind(1):
        gates = nn.functional.linear(row, lm.lstm.weight_ih_l0, lm.lstm.bias_ih_l0) + nn.functional.linear(h, lm.lstm.weight_hh_l0, lm.lstm.bias_hh_l0)
        i,f,g,o = gates.chunk(4,dim=-1)
        c = f.sigmoid()*c + i.sigmoid()*g.tanh()
        h = o.sigmoid()*c.tanh()
    _, (hn, cn) = lm.lstm(px)
    torch.testing.assert_close(h, hn[-1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(c, cn[-1], rtol=1e-4, atol=1e-5)
audit['manual_lstm_equations_match'] = True

chapter('start', 'Know what you are presenting', 'A learning guide tied to this repository, from first principles to an oral defense.', '''
<p>This project classifies a 28 × 28 grayscale clothing image into one of ten Fashion-MNIST classes. The LSTM reads the image as a sequence of rows. The Transformer reads it as a sequence of small patches. Both end with ten class scores and use the same supervised training pipeline.</p>
<div class="two"><div class="card"><span class="badge teal">LSTM</span><h3>Read, remember, classify</h3><p>28 steps × 28 pixels → a 128-feature final hidden state → 10 scores.</p><p><strong>82,186 trainable parameters · 88.48% test accuracy</strong></p></div><div class="card"><span class="badge purple">Transformer</span><h3>Compare patches, combine, classify</h3><p>49 patches × 16 pixels → 64-feature tokens → 2 encoder blocks → mean pooling → 10 scores.</p><p><strong>71,946 trainable parameters · 85.55% test accuracy</strong></p></div></div>
<h3>How to study this document</h3><ol><li><strong>First pass:</strong> chapters 1–4, then each model's overview and shape tables. Explain the forward pass without equations.</li><li><strong>Second pass:</strong> derive the LSTM gates and attention calculation by hand. Use the sliders to check your intuition.</li><li><strong>Third pass:</strong> read the exact code, parameter counts, training loop, and saved results. Point to evidence for every project-specific claim.</li><li><strong>Rehearsal:</strong> answer the oral questions before opening the answers; then do the reconstruction exercises.</li></ol>
<p>A productive way to demonstrate your understanding is to trace unfamiliar inputs, explain design choices, and make a small change yourself. Use your own words in a presentation and describe any tools or assistance accurately. This guide teaches the implementation; it does not establish who originally authored it.</p>
<p><strong>Scope:</strong> these two image classifiers and the theory needed to understand them deeply. Language-model generation, decoder training, advanced LSTM variants, and large-scale Transformer pretraining are discussed only where they clarify this code.</p>
<div class="note"><strong>Offline document</strong><p>All illustrations, saved plots, code snapshots, numerical demos, and sample predictions are embedded. Internet access is needed only if you choose to open an external reference. JavaScript powers the demos; the main text remains readable without it.</p></div>
''')

chapter('map', 'The repository and the full pipeline', 'Understand how an image reaches the model and how a saved result is produced.', table(['File or directory, relative to A1','Responsibility'], [
('<code>a1/data.py</code>','Download/read data; split; compute training statistics; normalize; turn images into rows or patches; create mini-batches.'),
('<code>a1/models.py</code>','Define both classifiers, choose an input representation, build a model, count parameters.'),
('<code>configs/lstm.json</code>, <code>configs/transformer.json</code>','Training settings; model widths and depths are hard-coded in models.py.'),
('<code>a1/train.py</code>','Set seeds; select device; run training/validation; save the checkpoint with lowest validation loss.'),
('<code>a1/evaluate.py</code>','Reload that checkpoint; evaluate the official test set; save predictions, metrics, and plots.'),
('<code>tests/test_lstm.py</code>, <code>tests/test_transformer.py</code>','Focused tests for representations, shapes, gradients, readout, dropout, and reloads.'),
('<code>runs/m2/lstm/</code>, <code>runs/m2/transformer/</code>','Saved evidence: best.pt, summary.json, history.csv, test_summary.json, predictions, confusion matrix, plots, verification.json.')]) + '''
<div class="flow"><span>Raw image + label</span><b>→</b><span>Normalize</span><b>→</b><span>Rows / patches</span><b>→</b><span>Model</span><b>→</b><span>10 logits</span><b>→</b><span>Loss or prediction</span></div>
<p>During training, the true label enters the loss calculation, then gradients update parameters. The label is never an input feature. During inference, the model needs only the image and the same preprocessing used in training.</p>
<h3>Dispatch matters</h3>''' + code('''input_mode("lstm")         # "row_seq"
input_mode("transformer")  # "patch_seq"
build_model("lstm")        # LSTMClassifier()
build_model("transformer") # TransformerClassifier()''') + '''
<p>The general data code also supports <code>flatten</code>, <code>image</code>, and <code>col_seq</code>. The trained LSTM uses <code>row_seq</code>; the trained Transformer uses <code>patch_seq</code>. Do not describe a supported alternative as if it were the saved experiment.</p>
''')

chapter('foundations', 'The minimum mathematics, explained', 'Tensors, learned features, activations, and gradients are the vocabulary of both models.', '''
<h3>A tensor is an array with named roles for its axes</h3><p>A scalar has one value. A vector is a list of values. A matrix has rows and columns. A tensor generalizes these to more axes. The tuple <code>(B, 49, 16)</code> means B independent images, 49 patches per image, and 16 scalar pixel features per patch. An axis length does not tell you its meaning: always name the axes.</p>
<p><strong>B</strong> is batch size, <strong>T</strong> sequence length, <strong>D</strong> feature width, <strong>H</strong> LSTM hidden width, <strong>C</strong> class count. One batch is a group of independent examples processed together. It is not one long sequence formed by concatenating images.</p>
<h3>A linear layer learns weighted combinations</h3>''' + eq('y = Wx + b &nbsp; (column-vector notation)<br>For row-vector batches: Y = XW<sup>T</sup> + b') + '''
<p><code>nn.Linear(in_features, out_features)</code> stores a weight of shape <code>(out_features, in_features)</code> and a bias of shape <code>(out_features,)</code>. It transforms the last axis and keeps earlier axes. Thus <code>Linear(16,64)</code> maps <code>(B,49,16)</code> directly to <code>(B,49,64)</code> using the same weights for every patch.</p>
<p>A parameter is a learned number, such as a weight or bias. A hyperparameter is a chosen setting, such as learning rate or hidden size. An activation is an intermediate value produced for a particular input. The hidden state is an activation, not a persistent trainable table for individual images.</p>
''' + table(['Function','Formula / behavior','Role here'],[
('Sigmoid σ(a)','1 / (1 + exp(−a)); values between 0 and 1','LSTM gates act as soft multiplicative controls.'),
('tanh(a)','Values between −1 and 1; derivative 1 − tanh²(a)','Signed LSTM candidate and bounded hidden readout.'),
('GELU(a)','a Φ(a), with Φ the standard normal cumulative distribution','Smooth nonlinear activation in the Transformer feed-forward network; negative values are softly suppressed.'),
('Softmax(z)<sub>k</sub>','exp(z<sub>k</sub>) / Σ<sub>j</sub> exp(z<sub>j</sub>)','Turn scores into a normalized distribution across classes or attention keys.'),
('Elementwise product ⊙','Multiply matching coordinates','LSTM gates control each hidden coordinate separately.'),
('Dot product q · k','Σ q<sub>j</sub>k<sub>j</sub>','An attention compatibility score. It can be negative and is not a probability.')]) + '''
<h3>What learning actually does</h3><p>A forward pass computes scores and a loss. Backpropagation uses the chain rule to compute how each parameter affects that loss. The optimizer uses those derivatives to update parameters. Repeating this over many batches can improve predictions on unseen images. Generalization must be measured on held-out data; a low training loss alone cannot establish it.</p>
<p>For a composition L = f(g(w)), the derivative is dL/dw = f′(g(w))g′(w). Deep networks combine many such factors. Very small products can make learning early computations difficult; very large products can make it unstable. This motivates gated memory, residual paths, and normalization.</p>
<h3>Logits are not probabilities</h3><p>A logit is an unconstrained real-valued class score. The output shape is <code>(B,10)</code>, not ten predicted labels per image. <code>argmax(dim=1)</code> chooses one class per image. Applying softmax preserves the winning class because it preserves score ordering.</p>
''')

chapter('data', 'Data, normalization, and leakage', 'Both models receive the same information, arranged in different ways.', '''
<p>Fashion-MNIST provides 60,000 official training images and 10,000 official test images. Each is one channel, 28 pixels high and 28 pixels wide. The code stratifies the 60,000 training labels with seed 42 to create 54,000 training and 6,000 validation examples. With balanced classes, this gives 5,400 training, 600 validation, and 1,000 test examples per class.</p>
''' + table(['ID','Class','ID','Class'],[(0,'T-shirt/top',5,'Sandal'),(1,'Trouser',6,'Shirt'),(2,'Pullover',7,'Sneaker'),(3,'Dress',8,'Bag'),(4,'Coat',9,'Ankle boot')]) + '''
<h3>The exact preprocessing order</h3><ol><li><code>ToTensor()</code> converts byte pixels into float32 values in [0,1] and creates shape <code>(1,28,28)</code>.</li><li>The training-only mean and standard deviation are computed across all pixels of the 54,000 training images. The code accumulates totals in float64; statistics are population-style, using E[x²] − E[x]².</li><li>Every split uses these same training statistics. No per-image normalization is performed.</li><li>The normalized image is converted to its model representation. DataLoader then stacks examples along a new batch axis.</li></ol>
''' + eq('Z[0,r,c] = (I[r,c] / 255 − μ<sub>train</sub>) / σ<sub>train</sub><br>μ = 0.28620901297676965 &nbsp; σ = 0.3531597492723218') + '''
<p>A black pixel becomes about −0.8104. A white pixel becomes about 2.0212. Negative inputs are expected after normalization. This global transform does not force every individual image to have mean zero and standard deviation one.</p>
<h3>Why use the training split only?</h3><p>Preprocessing is part of what the model learns from data. Computing statistics from validation or test images would let held-out distribution information influence training. Labels need not be exposed for leakage to occur. The test set should remain a final assessment, not a tool for choosing architecture or checkpoint.</p>
<p>The split cache stores the indices, seed, ratio, dataset hash, split hash, mean, and standard deviation. The code recomputes expected indices and checks cache compatibility before reusing it. A hash detects a mismatch; it does not prove the dataset or experiment is scientifically correct.</p>
<p><strong>No augmentation in these runs:</strong> both configs set <code>augment_train=false</code>. The data wrapper can perform a horizontal flip and small rotation when enabled, but those transforms were not used for the saved LSTM or Transformer.</p>
<h3>Interactive image-to-sequence explorer</h3><p>Select a real held-out example and switch between row and patch views. The probabilities below were computed by the saved checkpoints during guide generation; the browser is displaying those stored outputs, not running PyTorch.</p>
<div class="lab" id="image-lab"><div class="controls"><label>Test example <select id="sample-select"></select></label><label>Representation <select id="representation"><option value="row">LSTM rows</option><option value="patch">Transformer patches</option></select></label></div><div class="two"><div><canvas id="image-canvas" width="336" height="336" role="img" aria-label="Fashion-MNIST image with selected row or patch highlighted"></canvas><label for="token-index">Row / patch index (zero-based)</label><input id="token-index" type="range" min="0" max="27" value="12"><p id="token-location" class="mono"></p></div><div><h4 id="sample-title"></h4><p id="predictions"></p><p id="token-shape"></p><pre id="token-values"></pre><p class="small">The image shows original byte brightness. Token values are normalized float features. This selection is a teaching subset, not a balanced evaluation sample.</p></div></div></div>
''')

chapter('lstm-overview', 'LSTM: read an image as 28 rows', 'The sequence is spatial. It is not a video, sentence, or stream of 784 time steps.', '''
<p>LSTM means <strong>Long Short-Term Memory</strong>. It is a recurrent neural network designed to maintain and update information along a sequence. In this project, step t is image row t, read from top to bottom. Each step receives 28 pixel features, ordered left to right.</p>
''' + eq('S[r,c] = Z[0,r,c]<br>(1,28,28) → squeeze channel → (28 steps,28 features)<br>After batching: X ∈ ℝ<sup>B×28×28</sup>') + '''
<p>No pixel is discarded: 28 × 28 = 784 scalar values. The layout change does not itself learn anything. At each step, learned input weights combine all 28 horizontal pixel positions; recurrence lets the model combine that row with previous rows.</p>
<div class="flow"><span>row 0<br>28 pixels</span><b>→</b><span>LSTM state</span><b>→</b><span>row 1 … row 27</span><b>→</b><span>final h<br>128 values</span><b>→</b><span>dropout → Linear → 10</span></div>
<h3>The architectural choices actually used</h3><ul><li>One recurrent layer, one direction, input width 28, hidden width 128.</li><li>Zero initial hidden and cell states for every image on every forward call.</li><li>No recurrent state carried between images or mini-batches.</li><li>Final hidden state only; no averaging of all row outputs and no attention readout.</li><li>Dropout probability 0.2 on the final hidden vector, followed by <code>Linear(128,10)</code>.</li></ul>
<p>A single recurrent layer is reused at all 28 steps. There are not 28 separately parameterized LSTMs. Increasing the sequence length changes computation and activation storage, but does not automatically increase this layer's parameter count.</p>
''' + note('What the final state means','The final state is computed when the last row is read, but depends recursively on all preceding rows. Saying “the model uses only the bottom row” confuses the final input with the accumulated state.') + '''
<h3>Why an LSTM for an image?</h3><p>It provides a way to test sequential feature aggregation on a vision task and to compare it with dense and convolutional models. It imposes a top-to-bottom processing order. That order is a modeling choice, not a natural time axis of clothing. Rotating or reversing an image changes what arrives early and late.</p>
''')

chapter('lstm-math', 'Inside the LSTM cell', 'One cell has a memory state, an exposed state, three gates, and a candidate update.', '''
<p>At step t, let x<sub>t</sub> be the current row (28 values), h<sub>t−1</sub> the previous hidden state (128 values), and c<sub>t−1</sub> the previous cell state (128 values). Each gate below is a 128-dimensional vector. The equations use column-vector notation.</p>
''' + eq('i<sub>t</sub> = σ(W<sub>ii</sub>x<sub>t</sub> + b<sub>ii</sub> + W<sub>hi</sub>h<sub>t−1</sub> + b<sub>hi</sub>)<br>f<sub>t</sub> = σ(W<sub>if</sub>x<sub>t</sub> + b<sub>if</sub> + W<sub>hf</sub>h<sub>t−1</sub> + b<sub>hf</sub>)<br>g<sub>t</sub> = tanh(W<sub>ig</sub>x<sub>t</sub> + b<sub>ig</sub> + W<sub>hg</sub>h<sub>t−1</sub> + b<sub>hg</sub>)<br>o<sub>t</sub> = σ(W<sub>io</sub>x<sub>t</sub> + b<sub>io</sub> + W<sub>ho</sub>h<sub>t−1</sub> + b<sub>ho</sub>)<br><br>c<sub>t</sub> = f<sub>t</sub> ⊙ c<sub>t−1</sub> + i<sub>t</sub> ⊙ g<sub>t</sub><br>h<sub>t</sub> = o<sub>t</sub> ⊙ tanh(c<sub>t</sub>)') + '''
''' + table(['Symbol','Plain meaning','Example intuition'],[
('f: forget gate','Scale the previous cell memory coordinate by a number between 0 and 1.','Keep much of a useful accumulated feature, or allow it to fade.'),
('i: input gate','Choose how much of the candidate to write.','Write a newly detected row pattern strongly or weakly.'),
('g: candidate','Propose signed content to add to memory.','A learned feature can increase or decrease a memory coordinate.'),
('o: output gate','Control how much transformed memory is exposed as hidden state.','The cell can retain information while exposing only some of it.'),
('c: cell state','The recurrent memory path, updated additively.','Carries accumulated features; can exceed the interval [−1,1].'),
('h: hidden state','The output sent to the next step and, at the end, the classifier.','Bounded by tanh and the output gate before the final dropout.')]) + '''
<p>The gates are learned functions of the current row and previous hidden state. They are not handwritten rules such as “forget background.” Human descriptions of memory coordinates are intuition, not evidence that individual learned dimensions have a specific semantic meaning. This implementation has no peephole connections from c directly into the gates.</p>
<h3>One numerical coordinate</h3><p>Suppose c<sub>t−1</sub> = 0.8, f = 0.9, i = 0.2, g = −0.5, and o = 0.7. Then c<sub>t</sub> = 0.9 × 0.8 + 0.2 × (−0.5) = 0.62. The new hidden value is 0.7 × tanh(0.62) ≈ 0.386. One part retained old information; another part wrote a negative update.</p>
<div class="lab"><h3>Memory gate laboratory</h3><p class="small">A single coordinate with manually chosen gate outputs, not the trained network. In the real model, affine layers plus sigmoid/tanh compute these values.</p><div class="slider-grid">
<label>Previous memory c<input type="range" id="gc" min="-2" max="2" step="0.01" value="0.8"></label><label>Forget f<input type="range" id="gf" min="0" max="1" step="0.01" value="0.9"></label><label>Input i<input type="range" id="gi" min="0" max="1" step="0.01" value="0.2"></label><label>Candidate g<input type="range" id="gg" min="-1" max="1" step="0.01" value="-0.5"></label><label>Output o<input type="range" id="go" min="0" max="1" step="0.01" value="0.7"></label></div><output id="gate-result" class="equation" aria-live="polite"></output><p>Try f=1 and i=0 to preserve c exactly; then set o=0. The exposed h becomes zero while the cell still stores its memory. Values 0 and 1 are idealized limiting cases for sigmoid gates.</p></div>
<h3>How PyTorch packs the four computations</h3><p><code>weight_ih_l0</code> has shape <code>(512,28)</code>; <code>weight_hh_l0</code> has shape <code>(512,128)</code>. The two bias vectors each have 512 values. The packed gate order is <strong>i, f, g, o</strong>. One pair of matrix multiplications computes four preactivations together; splitting the 512 features recovers four 128-feature vectors.</p>
<p>The current code leaves initialization to <code>nn.LSTM</code>; it does not explicitly set the forget-gate bias to 1. Be careful: many tutorials recommend a custom forget bias, but that is not a change present here. Parameter storage and default state behavior can be checked in the <a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html">PyTorch 2.9 LSTM API</a>.</p>
''')

chapter('lstm-learning', 'Why memory helps, and how it learns', 'Backpropagation through time connects the final classification loss to earlier rows.', '''
<h3>Start with a plain recurrent network</h3>''' + eq('h<sub>t</sub> = tanh(W<sub>x</sub>x<sub>t</sub> + W<sub>h</sub>h<sub>t−1</sub> + b)') + '''
<p>Backpropagating through many such transitions multiplies recurrent matrices and activation derivatives. Tanh derivatives are at most one and often much smaller. Depending on weights and states, gradients may shrink or grow across steps. This makes learning long dependencies difficult.</p>
<h3>The LSTM adds a controlled additive path</h3><p>Along the direct cell-memory path, holding gate values fixed, ∂c<sub>t</sub>/∂c<sub>t−1</sub> = diag(f<sub>t</sub>). If forget values remain close to one, this route can carry information and derivatives with less repeated contraction. Across many steps, its coordinate-wise factor is a product of forget values.</p>
''' + eq('Direct memory-path factor across 28 transitions:<br>0.5<sup>28</sup> ≈ 3.73 × 10<sup>−9</sup><br>0.99<sup>28</sup> ≈ 0.755') + '''
<p>This is a partial-path explanation, not the complete total derivative: gates also depend on earlier hidden states. LSTMs mitigate vanishing gradients; they do not guarantee perfect memory or eliminate exploding gradients. Gate saturation, output tanh saturation, optimization, and the task all matter.</p>
<h3>Backpropagation through time (BPTT)</h3><ol><li>Conceptually unroll the same cell across 28 row steps.</li><li>Compute the final hidden state and classification loss.</li><li>Apply the chain rule backward through each state transition.</li><li>Sum the contributions to shared weights from all steps.</li><li>Let Adam update the weights once for the mini-batch.</li></ol>
<p>The model's forward pass is sequential along rows. The framework still vectorizes work across B images and hidden features. No state is detached partway through the 28 steps, so this is full BPTT for each image, not truncated BPTT over subsequences. The code does not call gradient clipping.</p>
<h3>Why final hidden state instead of the cell?</h3><p>The hidden state is the LSTM's exposed representation: an output-gated tanh transformation of cell memory. The classifier is trained to use that representation. Using the final cell, averaging all hidden outputs, using a bidirectional LSTM, or adding attention would be different architectures requiring a new experiment.</p>
<h3>What this design gives up</h3><p>Earlier rows must influence the classifier through later recurrent transitions. The 128-dimensional final vector is a fixed-size summary of the whole image. A convolutional model has stronger explicit local two-dimensional structure; this LSTM instead learns row features and their ordered relationships. These observations motivate comparisons, but they do not establish the cause of a particular accuracy gap.</p>
''')

chapter('lstm-code', 'LSTM code, shapes, and parameter count', 'Be able to explain every executable line and recover the number of learned values.', source(LSTMClassifier) + '''
''' + table(['Expression','Shape','Meaning'],[
('x','(B,28,28)','Batch, row steps, pixel features.'),
('self.lstm(x)[0]','(B,28,128)','All hidden outputs; computed but ignored by this readout.'),
('hidden / h_n','(1,B,128)','Final hidden state for each layer/direction; only one of each.'),
('c_n','(1,B,128)','Final cell memory; ignored by the classifier.'),
('hidden[-1]','(B,128)','Select the last layer, preserving all images.'),
('dropout(hidden[-1])','(B,128)','Randomly mask features during training.'),
('classifier(...)','(B,10)','Raw class logits.')]) + '''
<p><code>super().__init__()</code> initializes PyTorch's module machinery. Assigning submodules as attributes registers their parameters. <code>forward</code> defines the computation called by <code>model(x)</code>. The explicit shape guard prevents accidentally passing flattened vectors or channel-first images.</p>
<p><code>_, (hidden, _)</code> unpacks the returned pair: all step outputs, then the final hidden/cell pair. The underscore is just a conventional Python variable for an unused result. <code>hidden[-1]</code> indexes the <em>layer/direction axis</em>, not the last image. In this one-layer unidirectional model, it equals the last time output <code>output[:, -1, :]</code>.</p>
<p><code>batch_first=True</code> changes the input and step-output layouts. It does not change h_n or c_n to batch-first. If B=128, the input happens to have two axes of size 28; a careless transpose can preserve its shape while changing its meaning. Tests of pixel order matter as well as shape tests.</p>
<h3>Count the parameters by hand</h3>
''' + table(['Part','Arithmetic','Count'],[
('Input-to-gates weights','4 × 128 × 28','14,336'),
('Hidden-to-gates weights','4 × 128 × 128','65,536'),
('Two packed gate biases','2 × 4 × 128','1,024'),
('LSTM subtotal','4H(I + H + 2), I=28, H=128','80,896'),
('Classifier weights + bias','128 × 10 + 10','1,290'),
('Total','80,896 + 1,290','<strong>82,186</strong>')]) + '''
<p>Dropout, zero initial states, and tensor indexing contribute no trainable parameters. The two stored biases per gate explain why a textbook count with a single combined bias differs by 512. Combining the two biases algebraically is possible; PyTorch still stores and learns both.</p>
<h3>Why dropout is outside the LSTM</h3><p>PyTorch's built-in LSTM dropout applies between stacked recurrent layers. This model has one layer, so its useful explicit dropout is on the final readout. With p=0.2, training drops each readout element with probability 0.2 and scales retained elements by 1/0.8. In evaluation it is an identity operation. The gate recurrence itself has no configured dropout.</p>
''')

chapter('transformer-overview', 'Transformer: the image becomes 49 tokens', 'This is a small encoder for image classification, trained from scratch.', '''
<p>A Transformer uses attention to combine information across a sequence. Here, each token begins as a non-overlapping 4 × 4 image patch. A 28 × 28 image has a 7 × 7 grid of such patches: 49 tokens, each containing 16 normalized pixel values. The word “token” does not imply text.</p>
''' + eq('P[7r+c, 4i+j] = Z[0, 4r+i, 4c+j]<br>r,c ∈ {0,…,6}; i,j ∈ {0,…,3}<br>(1,28,28) → (49,16) → batch → (B,49,16)') + '''
<p>Patches are visited left-to-right across the first patch row, then across the next patch row. Pixels inside each patch are flattened left-to-right, top-to-bottom. Patch 0 covers image rows 0–3 and columns 0–3; patch 1 covers rows 0–3 and columns 4–7; patch 7 begins rows 4–7 and columns 0–3.</p>
<p><strong>Patchification preserves all 784 pixels.</strong> It rearranges them. It is not mean pooling, random sampling, or a learned compression. The next learned linear layer produces features.</p>
<h3>Why unfold followed by contiguous and view?</h3>''' + code('''patches = image.unfold(1, 4, 4).unfold(2, 4, 4)
# shape: (1, 7, 7, 4, 4)
patches = patches.contiguous().view(-1, 16)
# shape: (49, 16)''') + '''
<p><code>unfold(dimension,size,step)</code> exposes windows along the selected axis. Using step=size makes them non-overlapping. The resulting tensor is a strided view; <code>contiguous()</code> puts the values into a suitable ordered memory layout. <code>view(-1,16)</code> merges the patch-grid axes and flattens each local patch. Here <code>-1</code> asks PyTorch to infer 49. It does not mean “last patch.”</p>
<h3>The complete architecture at a glance</h3><div class="flow"><span>49 × 16<br>pixel patches</span><b>→</b><span>Linear 16→64<br>+ positions</span><b>→</b><span>2 pre-norm<br>encoder blocks</span><b>→</b><span>LayerNorm<br>mean over 49</span><b>→</b><span>Dropout<br>Linear 64→10</span></div>
<p>Each block uses four attention heads and a 64→128→64 feed-forward network with GELU and dropout 0.2. There is no decoder, no text tokenizer, no causal generation, no pretrained weights, no class token, and no convolutional patch stem.</p>
<p>This is a compact Vision Transformer-style classifier. The original <a href="https://arxiv.org/abs/2010.11929">Vision Transformer paper</a> motivates patch sequences, but its training scale and standard configurations should not be attributed to this small experiment.</p>
''')

chapter('embeddings', 'Patch embeddings and positional information', 'An image patch supplies content; a learned position vector supplies its location.', '''
<h3>Patch projection</h3>''' + eq('e<sub>p</sub> = W<sub>embed</sub>P<sub>p</sub> + b<sub>embed</sub> ∈ ℝ<sup>64</sup><br>z<sub>p</sub><sup>(0)</sup> = Dropout(e<sub>p</sub> + E<sub>pos</sub>[p])') + '''
<p><code>Linear(16,64)</code> transforms each 16-pixel patch into a 64-feature token. Those features are learned combinations of local pixels. Expanding 16 to 64 provides a representation space; it does not create new measured image information. The same projection is shared across all patch positions.</p>
<h3>Learned absolute position embeddings</h3><p><code>position_embedding</code> is a trainable tensor of shape <code>(1,49,64)</code>. Its leading size-one axis broadcasts across B images, so each image receives the same trainable position table. Position p gets a distinct 64-vector that is added to its content embedding.</p>
<p><code>nn.Parameter</code> registers this tensor for optimization. It starts with a truncated normal initialization using <code>std=0.02</code>; then classification gradients update it. It is neither a sinusoidal formula nor an embedding of the class label. The 49 entries are absolute raster-grid positions learned for this fixed input resolution. There is no explicit relative-position bias or separate learned row/column table.</p>
<h3>Why attention alone cannot identify arrangement</h3><p>With no position information, self-attention is permutation equivariant: permuting the input tokens permutes the output tokens in the same way. A token-wise feed-forward network and LayerNorm preserve that property. Mean pooling then removes order, making the whole classifier permutation invariant in evaluation.</p>
<p>Concretely, a bag of identical patch contents arranged differently would give the same pooled result if the position embeddings were zero. With the position table active, moving patch content to a different fixed slot changes which position vector is added and can change the prediction.</p>
''' + note('A precise permutation statement','Permuting content tokens while keeping the position table fixed generally changes this model. Permuting the already position-tagged token vectors as a group would still leave mean pooling invariant. Position matters through content–location pairing, not because token index has a hidden special meaning inside attention.') + '''
<h3>What the tests establish</h3><p>The Transformer tests reverse patch order and check that outputs change. They then zero the positional table and check approximate permutation invariance in evaluation. These checks establish useful architectural behavior; they do not measure the accuracy loss from training an entirely new model without positions.</p>
''')

chapter('attention', 'Self-attention, from one score to four heads', 'Every patch can collect information from every other patch in the same image.', '''
<h3>Queries, keys, and values</h3><p>For one head, learned projections make three versions of each token: a <strong>query</strong> expresses what the receiving token matches; a <strong>key</strong> is used to score each source token; a <strong>value</strong> is the information that will be mixed. These are mathematical roles, not natural-language questions. In self-attention, Q, K, and V come from the same input token sequence, with different learned projections.</p>
''' + eq('Q = XW<sub>Q</sub> + b<sub>Q</sub>, &nbsp; K = XW<sub>K</sub> + b<sub>K</sub>, &nbsp; V = XW<sub>V</sub> + b<sub>V</sub><br>S = QK<sup>T</sup> / √d<sub>k</sub><br>A<sub>ij</sub> = exp(S<sub>ij</sub>) / Σ<sub>m=1</sub><sup>49</sup> exp(S<sub>im</sub>)<br>Y = AV') + '''
<p>This uses row-token notation. In each attention row i, the softmax runs over source-key index j. A<sub>ij</sub> says how much value j contributes to the updated token i, before attention dropout. It is not a probability that patch j belongs to a class. All 49 keys, including i itself, are eligible.</p>
<p>Without dropout, each attention row sums to one and Y<sub>i</sub> is a weighted average of the projected values. The score matrix need not be symmetric: Q and K have different learned projections, and row normalization is directional. Large attention weights alone are not a complete explanation of a model's decision.</p>
<h3>Why divide by the square root?</h3><p>If query/key components have roughly unit variance and are sufficiently independent, a sum of d<sub>k</sub> products has variance that grows with d<sub>k</sub>. Dividing by √d<sub>k</sub> moderates score scale and helps avoid excessively saturated softmax. Here d<sub>k</sub>=64/4=16, so the divisor is 4. It is not √64 and not √49.</p>
<h3>A hand-computable example</h3><p>Use one two-dimensional query q=(1,0), keys k₁=(1,0), k₂=(0,1), k₃=(1,1), and values v₁=(2,0), v₂=(0,2), v₃=(1,1). Raw dot products are (1,0,1). After division by √2, scores are about (0.707,0,0.707). Softmax gives (0.4011,0.1978,0.4011). The output is about (1.2033,0.7967).</p>
<div class="lab"><h3>Attention laboratory</h3><p class="small">Toy 3-token, 2-feature calculation. These vectors are illustrative and are not learned patch features from the checkpoint.</p><div class="controls"><label>Query x <input id="qx" type="range" min="-3" max="3" step="0.1" value="1"></label><label>Query y <input id="qy" type="range" min="-3" max="3" step="0.1" value="0"></label><label><input id="scale-attention" type="checkbox" checked> Divide by √2</label></div><div id="attention-bars"></div><output id="attention-result" class="equation" aria-live="polite"></output><p>Try q=(0,0): all three scores tie, so weights are 1/3. Try a large positive query and toggle scaling to see the distribution become more or less concentrated. The browser uses max-subtraction for numerically stable softmax.</p></div>
<h3>Four heads, then one output projection</h3><p>The 64-dimensional projections are split into four 16-dimensional heads. Each head computes its own 49×49 attention pattern. Concatenating the four outputs restores width 64, then a learned output projection mixes these channels. The heads are not four encoder layers or four separate classifiers.</p>
''' + table(['Tensor (conceptual batch-first view)','Shape'],[
('Input to attention','(B,49,64)'),
('Packed Q,K,V projection output','(B,49,192)'),
('Q, K, V split into four heads','Each (B,4,49,16)'),
('Scores and attention weights','(B,4,49,49)'),
('Per-head weighted values','(B,4,49,16)'),
('Concatenated heads and output projection','(B,49,64)')]) + '''
<p>The combined PyTorch input projection stores <code>in_proj_weight</code> of shape (192,64) and bias (192,). The output projection has weight (64,64) and bias (64,). The code does not impose human-assigned specializations on heads. Two heads can learn similar behavior.</p>
<h3>Why there is no mask</h3><p>All 49 real patches are present; no sequence padding is needed. An image classifier is allowed to inspect the whole image, so later raster positions are not forbidden. A causal mask is useful in next-token generation but would impose an unnecessary restriction here. Attention mixes tokens within each image, not across unrelated batch examples.</p>
<p>For the reference formulation see <a href="https://papers.neurips.cc/paper/7181-attention-is-all-you-need">Attention Is All You Need</a>; implementation-specific shapes and head splitting are documented in <a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html">PyTorch MultiheadAttention</a>.</p>
''')

chapter('encoder', 'What an encoder block actually computes', 'Attention mixes positions; the feed-forward network transforms features at each position.', '''
<h3>Pre-normalization and residual connections</h3><p>Let X be the input to one block. This implementation sets <code>norm_first=True</code>, so it normalizes the input to each sublayer before applying that sublayer. A simplified expression, with the correct operation order, is:</p>
''' + eq('U = X + Dropout₁(MHA(LN₁(X), LN₁(X), LN₁(X)))<br>Y = U + Dropout₂(Linear₂(Dropout(GELU(Linear₁(LN₂(U))))))') + '''
<p>The repeated LN₁ symbols indicate the same normalized input used as query, key, and value; this is not three independently learned norms. MHA also includes dropout on attention weights during training. Residual additions retain a direct X→U and U→Y path while the sublayers learn modifications. Their shapes must match, which is why both sublayers return width 64.</p>
<h3>Layer normalization</h3>''' + eq('LN(x)<sub>j</sub> = γ<sub>j</sub>(x<sub>j</sub> − μ<sub>x</sub>) / √(σ²<sub>x</sub> + ε) + β<sub>j</sub>') + '''
<p>For each token independently, mean and variance are computed across its 64 features. Learned γ and β each have 64 entries. The default ε is 10⁻⁵ and variance uses the population form. It does not compute statistics across images or normalize the attention probabilities. Unlike BatchNorm, it has no running batch statistics. It behaves the same way in train and eval modes apart from the values arriving from dropout upstream.</p>
<h3>The feed-forward network (FFN)</h3><p>Each token is transformed by <code>Linear(64,128) → GELU → Dropout → Linear(128,64)</code>, with shared weights across all 49 token positions. It temporarily expands features then maps back to width 64. It does not map the 49 token axis to 128 tokens; the sequence length stays 49.</p>
<p>Attention supplies communication among patches. The FFN supplies nonlinear processing of the resulting features at each patch. Combining them lets a block do more than merely average neighboring information.</p>
<h3>Two independent blocks</h3><p>The list comprehension constructs a new <code>TransformerEncoderLayer</code> twice. The blocks have the same architecture but distinct trainable tensors and independently drawn initial weights. Reusing the same object twice would share weights and define a different model. <code>ModuleList</code> registers both blocks for optimization; the explicit for loop determines execution order.</p>
<h3>Final norm and mean pooling</h3><p>After both blocks, tokens return to shape (B,49,64). The model applies a final <code>LayerNorm(64)</code> to each token, then <code>mean(dim=1)</code> averages the 49 tokens into a single 64-feature image vector. Dropout and <code>Linear(64,10)</code> produce the class scores.</p>
''' + eq('g<sub>b,d</sub> = (1/49) Σ<sub>p=0</sub><sup>48</sup> LN(T<sub>b,p,:</sub>)<sub>d</sub><br>logits = Linear<sub>64→10</sub>(Dropout(g))') + '''
<p>Mean pooling uses all contextualized output tokens. It is not averaging raw pixel patches, averaging probabilities, or selecting a special class token. Norm-before-mean is the actual order: LayerNorm and mean pooling generally do not commute.</p>
<p>Relevant APIs: <a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html">TransformerEncoderLayer</a> and <a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html">LayerNorm</a>. The local code determines which options are enabled.</p>
''')

chapter('transformer-code', 'Transformer code, shapes, and parameter count', 'Trace both transposes carefully; neither one changes which pixels belong to an image.', source(TransformerClassifier) + '''
''' + table(['Stage','Shape','Reason'],[
('patch_seq input','(B,49,16)','49 non-overlapping 4×4 patches.'),
('patch_embedding(x)','(B,49,64)','Project the last axis.'),
('+ position_embedding','(B,49,64)','Broadcast the leading size-one axis.'),
('outer input dropout','(B,49,64)','No shape change.'),
('transpose(0,1)','(49,B,64)','Sequence-first layout expected by these blocks.'),
('block 0; block 1','(49,B,64)','Each block preserves token count and width.'),
('transpose(0,1) back','(B,49,64)','Restore batch-first layout.'),
('final norm, then mean(dim=1)','(B,64)','Normalize features per token, average patch axis.'),
('dropout and classifier','(B,10)','One logit per class per image.')]) + '''
<h3>Why sequence-first here?</h3><p>The source comment records that fused batch-first inference on the project's CUDA stack drifted from the float32 CPU result. The blocks therefore use <code>batch_first=False</code> and explicit transposes. This is a backend-compatibility choice recorded by the implementation, not a mathematical requirement of Transformers. It does not mean the LSTM must also be sequence-first.</p>
<p>A Transformer can be implemented batch-first. Changing only the flag without adjusting the input would make it interpret the image batch as the sequence axis, mixing examples and breaking the model. The existing tests compare batched and individual inference and check CPU/CUDA agreement. This guide rechecked the current checkpoint on a CPU/device batch; it did not reproduce the historical fused-path issue.</p>
<h3>Parameter count, including all biases and norms</h3>
''' + table(['Part','Arithmetic','Count'],[
('Patch projection','16 × 64 + 64','1,088'),
('Learned position table','1 × 49 × 64','3,136'),
('Q,K,V weights + biases, per block','3 × 64 × 64 + 3 × 64','12,480'),
('Attention output projection, per block','64 × 64 + 64','4,160'),
('FFN first linear, per block','64 × 128 + 128','8,320'),
('FFN second linear, per block','128 × 64 + 64','8,256'),
('Two LayerNorms, per block','2 × (64 γ + 64 β)','256'),
('One encoder block','12,480 + 4,160 + 8,320 + 8,256 + 256','33,472'),
('Two encoder blocks','2 × 33,472','66,944'),
('Final LayerNorm','64 + 64','128'),
('Classifier','64 × 10 + 10','650'),
('Grand total','1,088 + 3,136 + 66,944 + 128 + 650','<strong>71,946</strong>')]) + '''
<p>Dropout, GELU, transposes, residual additions, softmax, and mean pooling add no trainable parameters. Multi-head attention has its own projection weights; it does not multiply the full 64×64 projection count by four again. Splitting the projected 64 features into heads is an organization of the same total width.</p>
<h3>Complexity without misleading speed claims</h3><p>For N tokens and width D, dense attention has a score/value-mixing term O(BN²D), plus projection terms O(BND²). The FFN costs O(BNDF), with F=128 here. The explicit attention weight layout has B × heads × N² entries, although fused kernels can avoid materializing some arrays. With N=49 and four heads, that is 9,604 score entries per image per block.</p>
<p>The LSTM costs roughly O(BTH(I+H)) with T=28, I=28, H=128, and four gates as a constant factor; it has a sequential dependency across T steps. Attention can operate on all patch positions within a block in parallel, while the two blocks still run in order. More parallelism does not guarantee faster end-to-end inference for a small model on a particular device.</p>
<p>If patch size changed from 4 to 2, tokens would increase from 49 to 196 and the attention score count would grow by 16×. If patch size were 7, there would be 16 tokens, each with 49 input features. Such changes also require matching input guards, projection dimensions, position tables, and retrained checkpoints.</p>
''')

chapter('training', 'How both models are trained', 'The architecture computes scores; the shared training procedure learns parameters.', '''
''' + table(['Setting','Actual value','Meaning'],[
('Seed','42','Seeds Python, NumPy, PyTorch and available CUDA devices; train loader has its own seeded generator.'),
('Split','54,000 / 6,000 / 10,000','Train / validation / official test; stratified train–validation split.'),
('Mini-batch','128','128 independent examples per usual optimizer step.'),
('Epochs','10','Ten complete passes through the training set; no early stopping.'),
('Optimizer','Adam, lr=0.001','Adaptive gradient updates; default betas=(0.9,0.999), epsilon=1e-8, weight decay=0.'),
('Loss','CrossEntropyLoss()','Unweighted class-index cross-entropy, default mean reduction, no label smoothing.'),
('Checkpoint rule','Strictly lower validation loss','Keep best.pt whenever val loss improves; equality keeps the earlier checkpoint.'),
('Augmentation / workers','false / 0','No train augmentation; loading runs in the main process.'),
('Device','auto; saved runs used CUDA','Auto checks MPS, then CUDA, then CPU.'),
('Precision / schedule','Ordinary float32 model training; fixed learning rate','No autocast/GradScaler, scheduler, warmup, gradient clipping, or explicit L2 penalty in this loop.')]) + '''
<h3>Cross-entropy from logits</h3>''' + eq('p<sub>k</sub> = softmax(z)<sub>k</sub><br>L(z,y) = −log p<sub>y</sub> = −z<sub>y</sub> + log Σ<sub>k=0</sub><sup>9</sup> exp(z<sub>k</sub>)<br>∂L/∂z<sub>k</sub> = p<sub>k</sub> − 1[k=y]') + '''
<p>The label y is an integer class index, not a floating-point image vector. The network returns raw logits. <code>CrossEntropyLoss</code> performs the stable log-softmax and negative log-likelihood calculation internally; putting a softmax layer before it would feed the wrong kind of scores to this API.</p>
<p>For p<sub>y</sub>=0.8 the loss is about 0.2231; for p<sub>y</sub>=0.1 it is about 2.3026; for p<sub>y</sub>=0.01 it is about 4.6052. Confident mistakes are heavily penalized. Accuracy only records the winning class, so validation loss and validation accuracy need not select the same epoch.</p>
<h3>One training mini-batch, line by line</h3>''' + code('''model.train()                       # enable training behavior, including dropout
optimizer.zero_grad(set_to_none=True) # discard gradients from the previous batch
logits = model(images)              # forward pass: (B,10)
loss = criterion(logits, labels)    # one scalar mean loss
loss.backward()                    # compute parameter gradients by chain rule
optimizer.step()                   # use Adam to update learned parameters''') + '''
<p>PyTorch accumulates gradients by default, which is why each ordinary batch begins with zeroing them. <code>backward()</code> computes gradients; it does not update weights. <code>optimizer.step()</code> updates weights; it does not rerun the forward pass. <code>set_to_none=True</code> clears gradient buffers to None until they are computed again.</p>
<h3>Adam in enough detail to explain it</h3>''' + eq('g<sub>t</sub> = ∇<sub>θ</sub>L<sub>t</sub><br>m<sub>t</sub> = β₁m<sub>t−1</sub> + (1−β₁)g<sub>t</sub><br>v<sub>t</sub> = β₂v<sub>t−1</sub> + (1−β₂)g<sub>t</sub>²<br>m̂<sub>t</sub> = m<sub>t</sub>/(1−β₁<sup>t</sup>), &nbsp; v̂<sub>t</sub> = v<sub>t</sub>/(1−β₂<sup>t</sup>)<br>θ<sub>t</sub> = θ<sub>t−1</sub> − α m̂<sub>t</sub>/(√v̂<sub>t</sub> + ε)') + '''
<p>Squares, division, and square roots act per parameter coordinate. m tracks smoothed gradients; v tracks smoothed squared gradients; bias correction compensates for their zero initialization. These are optimizer states, distinct from the LSTM hidden and cell states. Adam does not choose architecture or guarantee the best possible solution. See the <a href="https://arxiv.org/abs/1412.6980">original Adam paper</a> for the algorithm.</p>
<h3>Batch and epoch arithmetic</h3><p>54,000 = 421 × 128 + 112, so each epoch has 422 optimizer steps, with 112 images in the last batch. Ten epochs give 4,220 updates. Validation has 47 batches; the last has 112 images. Test has 79 batches; the last has 16 images. The loaders do not drop those last batches.</p>
<p>For reported epoch loss, the code multiplies each mean batch loss by the actual batch size, sums, then divides by total images. Averaging the batch means equally would overweight the shorter final batch. Epoch accuracy and macro-F1 are computed from all accumulated labels and predictions.</p>
<h3>Train versus validation versus test</h3><p>Training uses <code>model.train(True)</code>, enabled gradients, and Adam. Validation calls the same epoch function without an optimizer, causing <code>model.train(False)</code> and disabled gradient recording. Evaluation reloads the chosen checkpoint, calls <code>eval()</code>, and uses <code>inference_mode()</code>. <code>eval()</code> changes dropout behavior; by itself it does not disable autograd.</p>
<p>Training metrics are collected across batches while the weights are changing and dropout is active. Validation measures the end-of-epoch model with dropout disabled. A higher validation accuracy than training accuracy can therefore occur without data leakage. It still deserves inspection if unexpectedly large.</p>
<h3>Saving the right checkpoint</h3><p>The loop runs all ten epochs, saves weights whenever validation loss strictly improves, then reloads the best checkpoint. The LSTM selected epoch 9; the Transformer selected epoch 10. Selecting by validation is not early stopping: training does not stop at epoch 9. The checkpoint stores weights and run metadata but no Adam optimizer state, so it is sufficient for inference, not an exact mid-training resume.</p>
<p>Seed control, deterministic cuDNN settings, versions, data hashes, and split hashes aid reproducibility. They do not promise bitwise-identical training across arbitrary hardware or PyTorch versions. This guide's recomputed outputs verify these particular saved checkpoints in this checkout.</p>
''')

chapter('results', 'Read the measured results correctly', 'Saved artifacts establish these outcomes; broad architecture claims require more experiments.', '''
<p>The following table is generated from this copy's saved JSON summaries. The builder reloaded both checkpoints, recomputed the 6,000-image validation pass and all 10,000 test predictions, and checked accuracy, macro-F1, loss, confidence values, confusion matrices, checkpoint hashes, and history-based selection.</p>
''' + table(['Metric','LSTM','Transformer'],[
('Trainable parameters','82,186','71,946'),
('Selected epoch','9','10'),
('Best validation loss','0.283240','0.371269'),
('Selected validation accuracy','89.80%','86.43%'),
('Test accuracy','88.48% (8,848 / 10,000)','85.55% (8,555 / 10,000)'),
('Test errors','1,152','1,445'),
('Test macro-F1','0.883496','0.852387'),
('Test cross-entropy','0.325417','0.408519'),
('Saved training duration','122.12 s','139.45 s'),
('Saved device-resident forward batch, 128 images','1.1889 ms','2.0607 ms')]) + '''
<p>The LSTM was 2.93 percentage points higher in test accuracy here and made 293 fewer mistakes. The Transformer had 10,240 fewer parameters. Fewer parameters do not imply better accuracy or lower latency. These are single-seed runs using a shared split and training budget, not proof that LSTMs are generally better than Transformers for images.</p>
<p>Timing is historical evidence from the saved summaries on an NVIDIA GeForce GTX 1660 SUPER, with Python 3.12.10 and PyTorch 2.9.1+cu128. The forward timing used ten warm-up batches and 100 repetitions of one fixed batch already on the device. It excludes image loading and host-to-device transfer. Dividing a batch's duration by 128 is an amortized throughput measure, not measured single-image request latency. Training duration includes the loop's validation and checkpoint saves; it is not a universal speed benchmark. Timings were not rerun to build this guide.</p>
<h3>Learning curves and checkpoint selection</h3><div class="controls"><label>Inspect model <select id="curve-model"><option value="lstm">LSTM</option><option value="transformer">Transformer</option></select></label></div><div id="history-table"></div>
<p>Notice the LSTM's epoch 8 validation accuracy is about 89.92%, slightly higher than epoch 9's 89.80%, while epoch 9 has lower validation loss. Epoch 9 is correct under the implemented selection rule. Epoch 10 training loss improves but validation loss rises slightly; one such change is not proof of severe overfitting.</p>
<p>The Transformer reaches its lowest observed validation loss at the final allowed epoch. This does not establish that ten epochs are optimal or that longer training would improve test accuracy. Those would require a new validation-driven study.</p>
''' + ''.join(f'<details><summary>{name.upper()}: saved learning curves</summary><img loading="lazy" class="plot" src="{image_data(ROOT / "runs/m2" / name / "learning_curves.png")}" alt="Saved {name} training and validation loss, accuracy and macro-F1 over ten epochs"></details>' for name in ('lstm','transformer')) + '''
<h3>Accuracy, precision, recall, and F1</h3>''' + eq('Accuracy = number correct / total<br>Precision<sub>c</sub> = TP<sub>c</sub> / (TP<sub>c</sub> + FP<sub>c</sub>)<br>Recall<sub>c</sub> = TP<sub>c</sub> / (TP<sub>c</sub> + FN<sub>c</sub>)<br>F1<sub>c</sub> = 2 Precision<sub>c</sub> Recall<sub>c</sub> / (Precision<sub>c</sub> + Recall<sub>c</sub>)<br>Macro-F1 = (1/10) Σ<sub>c</sub> F1<sub>c</sub>') + '''
<p>Precision asks “of predictions named class c, how many are correct?” Recall asks “of actual class c images, how many were found?” Macro-F1 weights classes equally; it is not the harmonic mean of macro-precision and macro-recall. With equal test support of 1,000 per class, weighted-F1 equals macro-F1 in these results. Macro-recall equals accuracy for this balanced test set, but macro-F1 does not have to equal accuracy.</p>
<h3>Where mistakes happen</h3>
''' + table(['Class','LSTM recall','Transformer recall','LSTM F1','Transformer F1'],[[E(cls), f"{records['lstm']['test']['class_report'][cls]['recall']:.1%}",f"{records['transformer']['test']['class_report'][cls]['recall']:.1%}",f"{records['lstm']['test']['class_report'][cls]['f1-score']:.4f}",f"{records['transformer']['test']['class_report'][cls]['f1-score']:.4f}"] for cls in ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']]) + '''
<p>Shirt is the lowest-recall and lowest-F1 class for both. The LSTM correctly recognizes 618 of 1,000 Shirts; the Transformer recognizes 489. A high overall accuracy can therefore coexist with a difficult class. Similar grayscale silhouettes among upper-body garments are a plausible explanation for confusion, but that explanation is an interpretation, not a controlled causal experiment.</p>
''' + ''.join(f'<details><summary>{name.upper()}: saved test confusion matrix</summary><img loading="lazy" class="plot square" src="{image_data(ROOT / "runs/m2" / name / "confusion_matrix.png")}" alt="Saved {name} test confusion matrix, true class on rows and predicted class on columns"></details>' for name in ('lstm','transformer')) + '''
<p>Confusion-matrix rows are true classes and columns are predictions. The diagonal contains correct predictions. A large entry at true Shirt, predicted Coat is a missed Shirt and a false positive for Coat. Always check axes before interpreting a matrix.</p>
<h3>Reasonable hypotheses versus established facts</h3><p>Training from scratch with a small model, ten epochs, fixed Adam settings, no augmentation, and weaker built-in image-locality assumptions may help explain this Transformer's result. None of those is isolated by the current comparison. Changing one factor at a time, using multiple seeds, and selecting with validation would test those ideas. A larger pretrained Transformer can behave very differently.</p>
''')

chapter('comparison', 'Compare the two models without mixing concepts', 'Shared supervised task; different mechanisms for combining spatial information.', table(['Question','LSTM here','Transformer here'],[
('What is one sequence item?','One complete row: 28 pixels.','One 4×4 patch: 16 pixels.'),
('Sequence length / representation width','28 / 28 raw features.','49 / 16 raw features, projected to 64.'),
('How is order represented?','Top-to-bottom recurrence.','Learned absolute position vectors.'),
('How do locations interact?','Through hidden/cell transitions along the row sequence.','Global unmasked self-attention within each block.'),
('What is carried between steps?','Hidden state h and cell state c.','No recurrent step state; contextual token representations pass between blocks.'),
('What reaches the classifier?','Final hidden state: 128 features.','Mean of 49 normalized output tokens: 64 features.'),
('Main nonlinearities','Sigmoid and tanh.','Attention softmax and FFN GELU.'),
('Direct interactions','Earlier rows influence later states through recurrence.','Every patch can attend to all patches in one block.'),
('Parallelism','Batch and feature computations parallel; row dependency remains.','Token attention calculations parallel within a block; blocks are sequential.'),
('Persistent state across images?','No.','No.'),
('Class prediction','Linear → 10 logits → argmax.','Linear → 10 logits → argmax.'),
('Shared training conditions','Cross-entropy, Adam 0.001, batch 128, 10 epochs.','Same protocol and same train/validation/test split.')]) + '''
<h3>Relation to CNNs</h3><p>A CNN explicitly learns shared local filters across a two-dimensional grid. This LSTM sees each row's whole width through dense input weights, then learns vertical relationships recurrently. This Transformer has a local patch projection followed by global token interactions. These are different inductive biases: assumptions built into the architecture about which patterns are easy to learn.</p>
<p>The saved CNN in this checkout has 91.94% test accuracy, higher than either model here. It is a useful contextual result, not a matched parameter-count comparison. This guide focuses verification on the LSTM and Transformer.</p>
<h3>Related architectures you might be asked about</h3><p><strong>Bidirectional LSTM:</strong> reads in both directions and combines forward/backward features; not used here. <strong>GRU:</strong> a different gated recurrence with a single recurrent state; not used here. <strong>Transformer decoder:</strong> often uses causal self-attention and may use cross-attention; this classifier has neither a decoder nor cross-attention. <strong>CLS token:</strong> a learnable token whose final output can classify an image; this code instead uses mean pooling. <strong>Pretraining:</strong> learning weights on another large dataset before fine-tuning; these runs start from initialized weights.</p>
''')

chapter('defense', 'Present it clearly and answer follow-up questions', 'Practice explaining, calculating, and locating evidence rather than memorizing a speech.', '''
<h3>A five-minute presentation structure</h3><ol><li><strong>0:00–0:40 — Task and protocol.</strong> “The task is ten-class Fashion-MNIST classification. I use a fixed 54,000/6,000 split and the official 10,000 test images. Normalization uses training pixels only.”</li><li><strong>0:40–1:40 — LSTM.</strong> Draw 28 row vectors feeding one recurrent cell repeatedly. Explain h versus c and the forget/write/output operations. State the final 128-feature readout.</li><li><strong>1:40–2:50 — Transformer.</strong> Draw a 7×7 patch grid. Explain 16→64 projection, positions, four 16-feature attention heads, two pre-norm blocks, and mean pooling.</li><li><strong>2:50–3:40 — Learning.</strong> Describe logits, cross-entropy, backward, Adam, dropout modes, and selection by validation loss.</li><li><strong>3:40–4:30 — Results.</strong> Show 88.48% versus 85.55%, the selected epochs, and the difficult Shirt class. Explain what one seed cannot establish.</li><li><strong>4:30–5:00 — Evidence.</strong> Open the model definition, configuration, checkpoint metadata, prediction CSV, and one confusion matrix. Be ready to trace an image and derive a parameter count.</li></ol>
<p>Adapt the wording to what you actually did and understand. If you do not know why a past choice was made, say what the code does and what experiment would evaluate alternatives. Do not invent a tuning process or claim to have observed an unrun improvement.</p>
<h3>Oral examination questions</h3><p>Answer aloud first, then open the explanation. The question filter searches these cards only.</p><label for="qa-search">Filter questions <input id="qa-search" type="search" placeholder="Try: hidden, softmax, position, checkpoint"></label><p id="qa-count" class="small"></p><div id="qa-list">__QA__</div>
''')

chapter('practice', 'Exercises that demonstrate understanding', 'Work on paper first. Then check the answers and try a small implementation yourself.', '''
<details><summary>1. Trace a batch of 3 images through both models.</summary><p>LSTM: (3,28,28) → all outputs (3,28,128), h_n/c_n (1,3,128) → h_n[-1] (3,128) → logits (3,10). Transformer: (3,49,16) → (3,49,64) → transpose (49,3,64) → two shape-preserving blocks → (3,49,64) → final norm → mean patch axis (3,64) → logits (3,10).</p></details>
<details><summary>2. Find the patch index and within-patch feature for pixel (row 13, column 22).</summary><p>Patch row floor(13/4)=3; patch column floor(22/4)=5; patch index 7×3+5=26. Local row=1, local column=2, feature index 4×1+2=6. Therefore P[26,6]=Z[0,13,22]. Indices are zero-based.</p></details>
<details><summary>3. Compute c and h when old c=−0.4, f=0.5, i=0.8, g=0.25, o=0.9.</summary><p>c = 0.5×(−0.4)+0.8×0.25=0. h=0.9×tanh(0)=0. Forgetting and writing can cancel; a nonzero output gate does not guarantee a nonzero hidden value.</p></details>
<details><summary>4. Recount the LSTM if hidden size becomes 64.</summary><p>With input width 28: 4×64×(28+64+2)=24,064 recurrent parameters. Head: 64×10+10=650. Total 24,714. Update the classifier input width as well. This calculation is a proposed variant, not the saved model.</p></details>
<details><summary>5. What changes if the Transformer uses 8 heads but keeps width 64?</summary><p>Each head width becomes 8; divisor becomes √8. Q/K/V and output projection parameter counts stay the same for this implementation. Conceptual attention weights have twice as many head matrices. Whether accuracy or runtime improves requires measurement.</p></details>
<details><summary>6. Why is mean(dim=2) wrong for Transformer pooling here?</summary><p>At that line tokens are (B,49,64). dim=1 averages patches to (B,64). dim=2 averages features to (B,49), which does not fit Linear(64,10) and loses the intended feature vector.</p></details>
<details><summary>7. A classifier predicts the correct class with probability 0.25. What is its cross-entropy?</summary><p>−ln(0.25)=ln(4)≈1.3863. Its prediction can still be correct if all other individual class probabilities are smaller than 0.25. Accuracy and confidence-weighted loss measure different aspects.</p></details>
<details><summary>8. Explain a counterintuitive LSTM checkpoint choice using the history.</summary><p>Epoch 8 has higher validation accuracy than epoch 9, but epoch 9 has lower validation cross-entropy. The implementation selects on loss, so epoch 9 is expected. It can reward better-calibrated or less-confidently-wrong scores even when a few more argmax decisions are wrong.</p></details>
<h3>Reconstruct the key pieces without looking</h3><ol><li>Write an image-to-row function, then test a synthetic image whose pixel value is 28r+c.</li><li>Write a patchifier and an inverse that reconstructs the image exactly.</li><li>Implement one LSTM step from matrix multiplications and compare its h/c against PyTorch.</li><li>Implement the three-token attention example with stable softmax.</li><li>Rebuild each classifier from its shape table and recover 82,186 or 71,946 parameters.</li><li>Explain why each training-loop statement is needed; predict what changes in eval mode.</li></ol>
<h3>A practical code-reading demonstration</h3><p>Start from <code>input_mode</code>, follow one dataset item through <code>FashionMNISTView.__getitem__</code> and <code>to_representation</code>, then trace <code>forward</code>. Show the shape before and after each operation. Finally open <code>summary.json</code> and <code>test_summary.json</code> to connect architecture and protocol to the measured result.</p>
''' + code('''# Run from A1 in a Python session; this only loads and predicts.
import torch
from a1.train import load_checkpoint

model, ck = load_checkpoint("runs/m2/lstm/best.pt", "cpu")
print(ck["epoch"], ck["input_mode"])
print(model)  # inspect layers
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Synthetic shape probe, not a meaningful clothing prediction:
with torch.inference_mode():
    print(model(torch.zeros(3, 28, 28)).shape)  # torch.Size([3,10])''') + '''
<p>For a real image, reproduce ToTensor and the checkpoint's training mean/std, then apply the correct representation. A synthetic zero tensor is useful for tracing shapes but is not evidence of classification quality.</p>
''')

chapter('evidence', 'Verification, commands, and source snapshots', 'Know what was checked now, what is historical, and how to inspect it yourself.', '''
<h3>Checks performed to build this guide</h3><ul><li>All four existing LSTM unit tests and all five existing Transformer unit tests passed in the current environment.</li><li>Both saved checkpoints were reloaded. Their data/split/config hashes, normalization, parameter counts, selected epochs, and checkpoint file hashes matched the saved metadata.</li><li>Both 6,000-image validation passes and all 10,000 test predictions were recomputed. Saved predicted labels, confidence values, loss, accuracy, macro-F1, and confusion matrices matched within the stated numerical tolerances.</li><li>CPU/device predictions agreed on a batch of 128 images. Four saved PNGs per model passed image integrity checks.</li><li>A manual implementation of the packed i/f/g/o equations reproduced the LSTM's final h and c on a real test image within floating-point tolerance.</li></ul>
<p>The direct audit is saved next to this document as <a href="guide_verification.json">guide_verification.json</a>. It does not replace the original run verification files. Training was not repeated, and historical durations were not remeasured.</p>
''' + note('A repository-specific verifier limitation','The existing tests/verify_run.py currently expects runs/cuda_baselines/linear/summary.json and the corresponding MLP baseline. Those files are absent in this copy, so the stock verifier stops before checking either requested model. The guide builder independently checks the LSTM/Transformer against their available checkpoints and data without creating substitute baseline evidence. The missing cross-baseline comparison is outside that direct audit.') + '''
<h3>Useful PowerShell commands (from the repository root)</h3>''' + code('''Set-Location 'D:\\DL\\DL-ASS - Copy\\A1'

# Existing focused tests; no model training:
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_lstm.py -v
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_transformer.py -v

# Rebuild this guide and repeat its direct checkpoint audit:
.\\.venv\\Scripts\\python.exe docs\\study-guide\\build_guide.py

# Stock verifier, read-only. Currently blocked by the missing baseline files above:
.\\.venv\\Scripts\\python.exe tests\\verify_run.py --model lstm --no-write
.\\.venv\\Scripts\\python.exe tests\\verify_run.py --model transformer --no-write''', 'PowerShell') + '''
<p>The following are the project's original train/evaluate entry points. They are shown for understanding; running them with the saved output paths will replace run artifacts. For practice, copy a config, choose a separate output directory such as <code>runs/practice</code>, and train into that directory.</p>
''' + code('''# Original commands; training/evaluation write to their configured run paths.
.\\.venv\\Scripts\\python.exe -m a1.train --model lstm --config configs/lstm.json
.\\.venv\\Scripts\\python.exe -m a1.train --model transformer --config configs/transformer.json
.\\.venv\\Scripts\\python.exe -m a1.evaluate --checkpoint runs/m2/lstm/best.pt
.\\.venv\\Scripts\\python.exe -m a1.evaluate --checkpoint runs/m2/transformer/best.pt''', 'PowerShell') + '''
<h3>Exact image representation helper</h3>''' + source(to_representation) + '''
<h3>Exact epoch function</h3>''' + source(run_epoch) + '''
<h3>Evidence inventory</h3>''' + table(['Artifact','What it establishes','What it does not establish'],[
('best.pt','A saved model state and metadata that can be reloaded.','Identity of the human author; an exact training resume without optimizer state.'),
('summary.json + history.csv','Recorded protocol, best validation epoch, learning curves, parameter count, training duration.','That another hyperparameter configuration would be worse.'),
('test_summary.json','Measured test metrics and class report for the selected checkpoint.','General performance across datasets, seeds, or future distributions.'),
('test_predictions.csv','One predicted class and confidence per indexed test image.','That confidence is perfectly calibrated.'),
('confusion_matrix.csv / PNG','Which true/predicted class pairs account for errors.','A causal account of why those errors happened.'),
('tests + direct audit','Specified behavior and consistency between artifacts and reloaded checkpoints.','Exhaustive absence of all bugs or independent authorship.')]) + '''
<p>For a teacher, understanding is demonstrated by explaining the mechanisms, solving unseen examples, and modifying or reconstructing parts under observation. Metrics and file hashes support technical reproducibility; they are not authorship certificates.</p>
''')

chapter('reference', 'Final revision sheet and references', 'Return here before presenting; use earlier chapters to explain each bullet in depth.', '''
<div class="two"><div class="card"><h3>LSTM essentials</h3><ul><li>Input: (B,28,28), 28 row steps.</li><li>One direction, one layer, H=128.</li><li>i/f/o are sigmoid; candidate g is tanh.</li><li>c=f⊙c_old+i⊙g; h=o⊙tanh(c).</li><li>Zero initial h/c on every forward.</li><li>h_n: (1,B,128); hidden[-1]: (B,128).</li><li>Dropout 0.2 → Linear(128,10).</li><li>82,186 parameters; selected epoch 9.</li><li>Test: 88.48%, macro-F1 0.883496.</li></ul></div><div class="card"><h3>Transformer essentials</h3><ul><li>Input: (B,49,16), 4×4 patches.</li><li>Linear(16,64) + learned (1,49,64) positions.</li><li>2 pre-norm blocks; 4 heads; d_head=16.</li><li>Attention softmax(QKᵀ/4)V per head.</li><li>FFN: 64→128→64, GELU, dropout 0.2.</li><li>Blocks receive (49,B,64); no mask.</li><li>Final LayerNorm → mean over patches.</li><li>Dropout → Linear(64,10).</li><li>71,946 parameters; selected epoch 10.</li><li>Test: 85.55%, macro-F1 0.852387.</li></ul></div></div>
<p><strong>Shared:</strong> seed 42; 54k train, 6k validation, 10k test; train-only normalization; Adam 0.001; batch 128; ten epochs; no augmentation; raw logits to cross-entropy; checkpoint by minimum validation loss.</p>
<h3>Glossary you should be able to explain aloud</h3>
''' + table(['Term','Meaning in this project'],[
('Feature / embedding','A numeric representation used by a model; a learned embedding is produced or stored as trainable values.'),
('State / activation','Input-dependent intermediate values. LSTM h/c summarize a prefix of rows; token activations summarize patch content and context.'),
('Receptive field','Which input locations can affect a representation. A final LSTM state can depend on all rows; one unmasked attention block can mix all patches.'),
('Inductive bias','Structural assumptions that make some functions easier to learn, such as CNN locality or LSTM sequence order.'),
('Residual connection','Add a sublayer result to its input, retaining a direct computational path.'),
('Overfitting / underfitting','Learning training-specific behavior that fails to generalize / failing to fit useful patterns adequately. Diagnose with appropriate train/validation evidence.'),
('Epoch / step','One training-set pass / one mini-batch optimizer update.'),
('Checkpoint / state_dict','A saved run payload / the mapping from named model tensors to their stored values.'),
('Calibration','How well predicted probabilities correspond to actual correctness rates; high confidence alone does not establish it.'),
('Ablation','A controlled change that isolates a component, such as training with versus without positions under matched conditions.')]) + '''
<h3>Primary references and implementation authority</h3><p>Project-specific statements in this document come from the local source and saved artifacts, inspected on 28 October 2026. The exact classes and representation helper are embedded above. External documents below support the general algorithms and library conventions; they do not supply this project's experimental results.</p>
<ol><li><a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html">PyTorch 2.9 — LSTM</a>: API, state shapes, gate ordering, and parameter layout.</li><li><a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html">PyTorch 2.9 — TransformerEncoderLayer</a>: pre-norm and tensor-layout options.</li><li><a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html">PyTorch 2.9 — MultiheadAttention</a>: projected heads and attention interface.</li><li><a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html">PyTorch 2.9 — LayerNorm</a>: feature normalization and learned affine parameters.</li><li><a href="https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html">PyTorch 2.9 — CrossEntropyLoss</a>: class-index targets and logits.</li><li><a href="https://papers.neurips.cc/paper/7181-attention-is-all-you-need">Vaswani et al. (2017), Attention Is All You Need</a>: scaled dot-product and multi-head attention. This project uses a different pre-norm encoder configuration and an image-classification readout.</li><li><a href="https://arxiv.org/abs/2010.11929">Dosovitskiy et al., An Image is Worth 16×16 Words</a>: image-patch Transformer approach; the model here uses 4×4 patches and no CLS token.</li><li><a href="https://arxiv.org/abs/1412.6980">Kingma and Ba, Adam</a>: adaptive first/second-moment optimization.</li></ol>
<p class="small">This HTML is a study artifact for D:\\DL\\DL-ASS - Copy. It does not change model code, experiment results, the report, or AI_USAGE.md. Relative source links work when the document stays in A1/docs/study-guide; the embedded teaching content remains available if the HTML is copied elsewhere.</p>
''')

QA = [
('Why can an LSTM classify a still image?', 'It only needs an ordered sequence of feature vectors. Here each image row is one step; the order is spatial rather than temporal.'),
('Is the LSTM sequence length 784 or 28?', '28. Each step has 28 pixel features. A 784-step pixel-by-pixel LSTM would be a different representation.'),
('Are pixels lost when the image becomes rows or patches?', 'No. Rows retain 28×28=784 values; patches retain 49×16=784 values. The order changes for patches, not the total information in the representation.'),
('What is the difference between h and c?', 'c is the gated additive cell-memory path. h is the exposed state o⊙tanh(c), used for the next transition and the final classifier. Both have width 128 here.'),
('What are the four packed gate chunks?', 'Input i, forget f, candidate g, output o. Sigmoid for i/f/o, tanh for g.'),
('Does the forget gate choose individual old rows to erase?', 'Not explicitly. It scales each coordinate of an accumulated memory vector. Coordinates are learned features, not stored rows.'),
('Does the final hidden state see only the final row?', 'No. It receives the final row and a state that recursively depends on earlier rows. Forgetting may attenuate information, but the dependency spans the sequence.'),
('Why hidden[-1] rather than hidden[:, -1]?', 'hidden has shape (layers×directions,B,H). hidden[-1] selects the last layer/direction and keeps the whole batch. hidden[:, -1] selects the last image along B.'),
('Does batch_first=True change h_n to (B,1,128)?', 'No. It changes only input and all-step output layouts. h_n and c_n remain (1,B,128) in this model.'),
('Does the LSTM remember the previous mini-batch?', 'No. No initial state is supplied, so the forward call initializes zero h/c for each new image. No recurrent state is stored for reuse.'),
('Why not set dropout=0.2 inside this one-layer LSTM?', 'Built-in LSTM dropout is between recurrent layers. There is only one layer. The explicit readout Dropout(0.2) is what regularizes this classifier.'),
('Does an LSTM completely solve vanishing gradients?', 'No. Its additive memory path can help when forget gates retain information, but saturation and other paths still affect learning.'),
('How does the loss reach the first row?', 'Backpropagation follows the recurrent graph from the final classifier through all 28 steps. Shared gate weights receive contributions from each step. Existing tests check nonzero early-row gradients.'),
('Why does the LSTM have two bias vectors?', 'PyTorch stores separate input-hidden and hidden-hidden biases for all four gates. Both are counted, even though their sum appears in the affine preactivation.'),
('What is one Transformer token here?', 'Initially a 4×4 patch flattened to 16 values; after projection and positions, a 64-feature vector associated with that patch location.'),
('Why exactly 49 patches?', '28/4=7 patches per image axis, so 7×7=49. They are non-overlapping and cover the image.'),
('Is patch_embedding a word lookup table?', 'No. It is a learned Linear(16,64) on continuous pixel features. The position tensor is a separate learnable table.'),
('Why add positions before attention?', 'Attention alone does not know absolute token locations. Adding a distinct trainable vector associates each patch content with its grid slot.'),
('Are positions sinusoidal or learned?', 'Learned absolute positions: nn.Parameter of shape (1,49,64), initialized with a truncated normal std=0.02.'),
('What do Q, K, and V mean?', 'Different learned projections of token features. A query is compared with keys to compute weights; values are combined using those weights.'),
('Why is the attention divisor 4?', 'Each head has width d_k=64/4=16. Scaled dot-product attention divides by √16=4.'),
('Does four heads mean four 64-feature attention outputs concatenated to 256?', 'No. Each head has 16 output features. Four heads concatenate to 64, then an output projection mixes them.'),
('Over which axis is attention softmax applied?', 'Over the 49 key/source tokens separately for each query and each head, within one image. Not over images and not over classes.'),
('Must attention be symmetric?', 'No. Q and K are projected differently, and row-wise normalization is directional. A_ij can differ from A_ji.'),
('Do attention weights always sum to one during training?', 'The softmax probabilities do before dropout. Attention dropout can change the sum of the effective weights for an individual stochastic training pass. In evaluation dropout is disabled.'),
('Why is there no causal mask?', 'The full image is available for classification. Every patch may inspect every other patch. Causal restrictions are needed for different tasks such as predicting future text without seeing it.'),
('Can a patch attend to patches in another image in the batch?', 'Not when the axes are correct. B identifies independent attention examples. The explicit transposes preserve that role; tests compare batched and separate inference.'),
('What is pre-norm?', 'LayerNorm is applied before attention and before the FFN. Each sublayer result is then added residually to its input. norm_first=True enables it here.'),
('What is the role of the FFN?', 'It applies a shared nonlinear 64→128→64 transformation at each token. Attention exchanges information between locations; the FFN transforms features at a location.'),
('What does LayerNorm normalize?', 'The 64 features of each token independently, with learned scale and bias. It does not normalize across the batch or maintain running batch statistics.'),
('Are the two encoder blocks sharing weights?', 'No. The list comprehension instantiates two separate modules with distinct parameter tensors. ModuleList registers them and the loop applies them sequentially.'),
('Why transpose before and after the encoder?', 'The blocks use sequence-first layout (49,B,64), while the surrounding model uses batch-first (B,49,64). Transpose swaps axes without changing token membership.'),
('Why does the classifier use mean pooling?', 'It creates one fixed-size image representation by averaging contextual output tokens. It is a simple readout without a special class token; its superiority to other readouts was not tested here.'),
('Does mean pooling make the full model position-blind?', 'No. Token representations already depend on content–position pairings and attention interactions. Without positions, self-attention plus mean pooling would be permutation invariant in eval mode.'),
('Why does the classifier not end in softmax?', 'CrossEntropyLoss expects raw logits and performs log-softmax internally. Evaluation applies softmax only when it needs reported confidence, while argmax can use logits directly.'),
('What changes between model.train() and model.eval()?', 'Dropout is random and scaled in training, disabled in evaluation. eval() alone does not turn off gradient recording; the code separately uses disabled gradients or inference_mode.'),
('What does loss.backward() do?', 'Computes gradients by reverse-mode automatic differentiation. It does not update weights. Adam performs the parameter update in optimizer.step().'),
('Why clear gradients before every mini-batch?', 'PyTorch accumulates gradients into parameter .grad fields. This training loop intends a fresh update from each batch, so it clears the previous gradients.'),
('How many optimizer steps are performed?', 'ceil(54,000/128)=422 per epoch, or 4,220 in ten epochs. The final training batch has 112 images.'),
('Why is the LSTM checkpoint epoch 9 rather than epoch 8?', 'The selection metric is validation loss, not validation accuracy. Epoch 9 has a lower validation loss even though epoch 8 accuracy is slightly higher.'),
('Is checkpoint selection the same as early stopping?', 'No. The loop always executes ten epochs. It saves the best validation-loss checkpoint and reloads it after training.'),
('Why not choose the epoch with highest test accuracy?', 'That would use final assessment data for model selection and make the reported test score optimistic. Use validation for choices, test for final assessment.'),
('Why use train-only normalization?', 'To prevent held-out data statistics from affecting training. All splits should receive the same training-derived transform.'),
('Does seed 42 guarantee identical results on every machine?', 'No. It controls many randomness sources, but kernels, devices, versions and nondeterministic operations can still produce differences. Saved artifacts and environment metadata improve reproducibility.'),
('Which model was better in this experiment?', 'The LSTM: 88.48% test accuracy versus 85.55%, and macro-F1 0.883496 versus 0.852387. This is a single-seed, fixed-budget comparison, not a universal ranking.'),
('Why did the Transformer underperform?', 'The comparison establishes the outcome, not its cause. Small-data training from scratch, the short budget, model settings, and inductive bias are hypotheses requiring controlled experiments.'),
('Which class was most difficult?', 'Shirt for both models: recall 61.8% for LSTM and 48.9% for Transformer; inspect the true-Shirt row in each confusion matrix.'),
('Do fewer parameters guarantee faster inference?', 'No. FLOPs, sequence length, kernel implementation, parallelism, memory movement and batch size also matter. This Transformer has fewer parameters but higher saved fixed-batch forward time.'),
('Can you load the saved checkpoint and resume training exactly?', 'You can reload weights for inference or begin further training, but the saved payload lacks Adam state and complete mid-run RNG/loader state. It is not an exact resume checkpoint.'),
('How would you test whether positional embeddings help?', 'Train matched models with and without positions under the same data split and protocol, preferably across several seeds, and compare validation/test results under a fixed selection procedure. Zeroing positions at inference is a different intervention.'),
('How can you demonstrate understanding beyond reciting numbers?', 'Trace unseen tensor shapes, derive gate/attention calculations and parameter counts, explain test failures, and implement or modify a small part under observation. Describe authorship and assistance accurately.'),
]
qa_html = ''.join(f'<details class="qa"><summary>{i+1:02d}. {E(q)}</summary><p>{E(a)}</p></details>' for i,(q,a) in enumerate(QA))

CSS = r'''
:root{--ink:#192e3c;--muted:#526675;--paper:#f6f4ee;--white:#fffefa;--line:#d8dfdd;--teal:#106a66;--purple:#6250a0;--code:#152b37;--accent:#d99c3b;--sidebar:268px}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:32px}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.76 system-ui,-apple-system,'Segoe UI',sans-serif}a{color:var(--teal);text-underline-offset:3px}a:hover{color:#253a8e}button,input,select{font:inherit}button{cursor:pointer;border:1px solid var(--line);border-radius:7px;padding:7px 12px;background:var(--white);color:var(--ink)}button:hover{background:#eaf2ee}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:4px}aside.nav{position:fixed;top:0;left:0;bottom:0;width:var(--sidebar);padding:26px 22px 24px;background:#17343e;color:#fff;overflow:auto;z-index:5}.brand{font:700 22px/1.3 Georgia,serif;letter-spacing:-.3px}.nav .small{color:#bcd1d2}.nav ol{list-style:none;margin:20px 0;padding:0;counter-reset:nav}.nav li{margin:0 0 3px}.nav a{display:block;color:#dbe8e7;text-decoration:none;font-size:13px;line-height:1.45;padding:7px 8px;border-radius:5px}.nav a::before{counter-increment:nav;content:counter(nav,decimal-leading-zero) ' ';color:#8fb9b7;margin-right:7px}.nav a.active{background:#31545c;color:#fff}.nav progress{width:100%;height:5px;accent-color:#80c9b9}.nav button{font-size:13px;background:transparent;color:#e2efec;border-color:#58777e}main{margin-left:var(--sidebar);max-width:1320px;padding:46px clamp(25px,5vw,86px) 80px}.hero{border-top:6px solid var(--teal);padding:36px 0 44px;border-bottom:1px solid var(--line)}.eyebrow{text-transform:uppercase;letter-spacing:2px;color:var(--teal);font-size:12px;font-weight:800}.hero h1{font:700 clamp(40px,5vw,66px)/1.08 Georgia,serif;letter-spacing:-1.8px;max-width:850px;margin:17px 0 24px}.hero p{max-width:760px;color:var(--muted);font-size:20px;line-height:1.6}.meta{font-size:13px;color:var(--muted)}.pills{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0}.pill,.badge{font-size:12px;font-weight:750;padding:5px 10px;border-radius:100px;display:inline-block;background:#e3efea;color:var(--teal)}.purple{background:#eee9f7;color:var(--purple)}.chapter{padding:48px 0 36px;border-bottom:1px solid var(--line);scroll-margin-top:20px}h2{font:700 clamp(29px,3vw,39px)/1.22 Georgia,serif;letter-spacing:-.6px;margin:10px 0 12px}h3{font-size:22px;line-height:1.4;margin:32px 0 12px}h4{font-size:18px;margin:8px 0 14px}.deck{font-size:19px;line-height:1.55;color:var(--muted);margin-bottom:27px}p{margin:14px 0}li{padding-left:3px;margin:8px 0}strong{font-weight:720}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:25px 0}.card{border:1px solid var(--line);background:var(--white);border-radius:12px;padding:24px}.card h3{margin:14px 0}.card p:last-child{margin-bottom:0}.table-wrap{overflow-x:auto;margin:24px 0;border:1px solid var(--line);border-radius:9px}table{width:100%;border-collapse:collapse;background:var(--white);font-size:14px;line-height:1.65;text-align:left}th{background:#e7eeea;color:#263e43;font-size:12px;text-transform:uppercase;letter-spacing:.6px}td,th{padding:13px 15px;vertical-align:top;border-bottom:1px solid var(--line)}tr:last-child td{border-bottom:0}td:first-child{font-weight:600}code,.mono{font-family:Consolas,'Cascadia Code',monospace;font-size:.9em}p code,li code,td code{background:#e8ede8;padding:2px 4px;border-radius:4px;overflow-wrap:anywhere}pre{background:var(--code);color:#dbeee9;padding:22px;border-radius:0 0 8px 8px;overflow:auto;line-height:1.65;font-size:13px;tab-size:4;margin:0 0 24px;max-height:700px}pre code{font-size:inherit}.code-label{background:#24434d;color:#a8cbc5;font-size:11px;text-transform:uppercase;letter-spacing:1px;padding:7px 16px;border-radius:8px 8px 0 0;margin-top:22px}.source{font-size:13px;color:var(--muted)}.equation{background:#ecf1ec;border:1px solid #d6e2db;border-left:4px solid var(--teal);padding:20px 24px;margin:23px 0;font:16px/2.1 Cambria,Georgia,serif;overflow:auto;border-radius:0 8px 8px 0}.note{border-left:4px solid var(--accent);background:#fff5de;padding:18px 22px;margin:24px 0;border-radius:0 8px 8px 0}.note p{margin:8px 0 0}.flow{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;margin:28px 0;padding:24px 14px;background:#e8efea;border-radius:10px}.flow span{border:1px solid #c0d6ce;background:var(--white);padding:12px 13px;border-radius:7px;text-align:center;font-size:13px;line-height:1.5}.flow b{color:var(--teal)}.lab{background:var(--white);padding:25px;border:1px solid #bfd2cb;border-radius:12px;margin:28px 0}.lab h3{margin-top:0}.lab pre{border-radius:7px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:none}.lab .two{align-items:start}.controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:18px 0}label{display:block;font-size:14px;font-weight:650}select,input[type=search]{display:block;max-width:100%;border:1px solid #b7c8c2;background:#fff;padding:7px 10px;color:var(--ink);border-radius:6px;margin:5px 0}input[type=range]{display:block;width:100%;accent-color:var(--teal);margin:12px 0}input[type=checkbox]{accent-color:var(--teal)}.slider-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px 28px}.small{font-size:13px;line-height:1.65;color:var(--muted)}canvas{display:block;background:#111;border-radius:8px;width:100%;max-width:336px;image-rendering:pixelated}.plot{width:100%;height:auto;display:block;background:white;border-radius:6px;margin:20px 0}.square{max-width:820px}details{border:1px solid var(--line);border-radius:7px;background:var(--white);margin:10px 0;padding:0 18px}summary{cursor:pointer;padding:16px 0;font-weight:650;line-height:1.55}details[open]{border-color:#95bcb0}details p{margin:0 0 20px}details .plot{margin-top:0}#qa-search{width:100%;max-width:500px}.bar-row{display:grid;grid-template-columns:100px 1fr 70px;gap:10px;align-items:center;font-size:13px;margin:12px 0}.bar-track{height:20px;background:#edf2ee;border-radius:5px;overflow:hidden}.bar-fill{height:100%;background:var(--teal);min-width:1px}.selected-row td{background:#e0eee6;font-weight:700}output{display:block}footer{padding:35px 0;font-size:13px;color:var(--muted)}.mobile-menu{display:none}.print-only{display:none}mark{background:#ffe8a3}noscript{display:block;padding:20px;background:#fff0c9}[hidden]{display:none!important}
@media(min-width:1600px){main{margin-left:calc(var(--sidebar) + (100vw - 1600px)/4)}}
@media(max-width:1050px){:root{--sidebar:230px}aside.nav{padding:22px 15px}main{padding-left:28px;padding-right:28px}.two{gap:15px}.lab .two{grid-template-columns:1fr}.slider-grid{gap:12px 18px}}
@media(max-width:760px){:root{--sidebar:0px}body{font-size:16px}aside.nav{position:relative;width:100%;max-height:none;padding:20px 22px}.nav ol{display:none;columns:2;margin-bottom:0}.nav.open ol{display:block}.nav progress{display:none}.nav .small{margin:6px 0}.mobile-menu{display:inline-block;float:right}.nav .toolbar{margin-top:12px}main{margin:0;padding:26px 21px 45px}.hero{padding-top:26px}.hero h1{font-size:44px}.hero p{font-size:18px}.chapter{padding-top:35px}.two{grid-template-columns:1fr}.slider-grid{grid-template-columns:1fr}.equation{font-size:14px;padding:16px 14px}.lab{padding:18px}pre{font-size:12px;padding:15px}td,th{padding:10px 12px}.bar-row{grid-template-columns:70px 1fr 58px}}
@media print{body{font-size:10pt;background:#fff;line-height:1.6}aside.nav,.controls,button,.mobile-menu,#qa-search,#qa-count,.lab{display:none}main{margin:0;padding:0;max-width:none}.hero{padding:10px 0 20px}.hero h1{font-size:32pt}.hero p{font-size:13pt}.chapter{padding:20px 0;break-before:page}h2{font-size:23pt}h3{font-size:14pt;break-after:avoid}.deck{font-size:12pt}p,li{orphans:3;widows:3}table{font-size:9pt}tr,figure,.equation{break-inside:avoid}.table-wrap{overflow:visible}pre{max-height:none;white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt;color:#152b37;background:#f0f0f0;border:1px solid #bbb}.code-label{color:#152b37;background:#ddd}.plot{max-height:220mm;object-fit:contain}details{border:0;padding:0}summary{font-size:10pt}.two{gap:14px}.print-only{display:block}a{color:inherit}.qa p{margin-bottom:14px}}
'''

JS = r'''
const $ = id => document.getElementById(id);
const classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'];
const data = JSON.parse($('study-data').textContent);
const fmt = (v,n=4)=>Number(v).toFixed(n);
const softmax = a => {const m=Math.max(...a), e=a.map(x=>Math.exp(x-m)),s=e.reduce((x,y)=>x+y,0);return e.map(x=>x/s)};
function drawImage(){
 const sample=data.samples[Number($('sample-select').value)||0];
 const patch=$('representation').value==='patch';
 $('token-index').max=patch?'48':'27';
 const t=Math.min(Number($('token-index').value),patch?48:27);$('token-index').value=t;
 const ctx=$('image-canvas').getContext('2d'); const cell=12;
 for(let r=0;r<28;r++) for(let c=0;c<28;c++){let v=sample.pixels[28*r+c];ctx.fillStyle=`rgb(${v},${v},${v})`;ctx.fillRect(c*cell,r*cell,cell,cell)}
 let values=[],r0=patch?4*Math.floor(t/7):t,c0=patch?4*(t%7):0,rn=patch?4:1,cn=patch?4:28;
 if(patch){ctx.strokeStyle='rgba(255,255,255,.23)';ctx.lineWidth=1;for(let n=4;n<28;n+=4){ctx.beginPath();ctx.moveTo(n*cell,0);ctx.lineTo(n*cell,336);ctx.stroke();ctx.beginPath();ctx.moveTo(0,n*cell);ctx.lineTo(336,n*cell);ctx.stroke()}}
 ctx.fillStyle='rgba(58,211,185,.24)';ctx.fillRect(c0*cell,r0*cell,cn*cell,rn*cell);ctx.strokeStyle='#45e2bc';ctx.lineWidth=3;ctx.strokeRect(c0*cell+1.5,r0*cell+1.5,cn*cell-3,rn*cell-3);
 for(let r=r0;r<r0+rn;r++)for(let c=c0;c<c0+cn;c++)values.push((sample.pixels[28*r+c]/255-data.mean)/data.std);
 $('sample-title').textContent=`Test #${sample.index} · true class: ${classes[sample.label]}`;
 $('predictions').textContent=['lstm','transformer'].map(name=>{const p=sample.probabilities[name],i=p.indexOf(Math.max(...p));return `${name.toUpperCase()}: ${classes[i]} (${fmt(100*p[i],2)}% confidence; ${i===sample.label?'correct':'incorrect'})`}).join(' | ');
 $('token-location').textContent=patch?`Patch ${t}: rows ${r0}–${r0+3}, columns ${c0}–${c0+3}`:`Step ${t}: image row ${t}, columns 0–27`;
 $('token-shape').textContent=patch?'One token: 16 features. Full image: (49,16).':'One step: 28 features. Full image: (28,28).';
 $('token-values').textContent=values.map((v,i)=>`${String(i).padStart(2)}: ${fmt(v)}`).join('   ');
}
data.samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`#${s.index} — ${classes[s.label]}`;$('sample-select').append(o)});
['sample-select','representation','token-index'].forEach(id=>$(id).addEventListener('input',drawImage));drawImage();
function gate(){const v=Object.fromEntries(['c','f','i','g','o'].map(x=>[x,Number($('g'+x).value)]));const retained=v.f*v.c,written=v.i*v.g,c=retained+written,h=v.o*Math.tanh(c);$('gate-result').textContent=`c_old=${fmt(v.c,2)}, f=${fmt(v.f,2)}, i=${fmt(v.i,2)}, g=${fmt(v.g,2)}, o=${fmt(v.o,2)}\nretained = ${fmt(retained)}; written = ${fmt(written)}\nc_new = ${fmt(c)}; h_new = ${fmt(h)}`;$('gate-result').style.whiteSpace='pre-wrap'}
['gc','gf','gi','gg','go'].forEach(id=>$(id).addEventListener('input',gate));gate();
function attention(){const q=[Number($('qx').value),Number($('qy').value)], keys=[[1,0],[0,1],[1,1]], vals=[[2,0],[0,2],[1,1]],div=$('scale-attention').checked?Math.sqrt(2):1,scores=keys.map(k=>(q[0]*k[0]+q[1]*k[1])/div),a=softmax(scores),out=[0,1].map(j=>a.reduce((s,w,i)=>s+w*vals[i][j],0));$('attention-bars').innerHTML=a.map((w,i)=>`<div class="bar-row"><span>Key ${i+1}</span><div class="bar-track"><div class="bar-fill" style="width:${w*100}%"></div></div><strong>${fmt(100*w,1)}%</strong></div>`).join('');$('attention-result').textContent=`q=(${q.map(x=>fmt(x,1)).join(', ')}) · scores=(${scores.map(x=>fmt(x)).join(', ')}) · output=(${out.map(x=>fmt(x)).join(', ')})`}
['qx','qy','scale-attention'].forEach(id=>$(id).addEventListener('input',attention));attention();
function history(){const name=$('curve-model').value, rows=data.records[name].history,best=data.records[name].summary.best_epoch;$('history-table').innerHTML='<div class="table-wrap"><table><thead><tr><th>Epoch</th><th>Train loss</th><th>Val loss</th><th>Train acc.</th><th>Val acc.</th></tr></thead><tbody>'+rows.map(r=>`<tr class="${r.epoch===best?'selected-row':''}"><td>${r.epoch}${r.epoch===best?' · selected':''}</td><td>${fmt(r.train_loss)}</td><td>${fmt(r.val_loss)}</td><td>${fmt(100*r.train_accuracy,2)}%</td><td>${fmt(100*r.val_accuracy,2)}%</td></tr>`).join('')+'</tbody></table></div>'}
$('curve-model').addEventListener('change',history);history();
function filterQA(){let n=0;const q=$('qa-search').value.trim().toLowerCase();document.querySelectorAll('.qa').forEach(el=>{el.hidden=!el.textContent.toLowerCase().includes(q);if(!el.hidden)n++});$('qa-count').textContent=`${n} of ${document.querySelectorAll('.qa').length} questions shown`;}
$('qa-search').addEventListener('input',filterQA);filterQA();
$('toggle-menu').addEventListener('click',()=>{const open=$('navigation').classList.toggle('open');$('toggle-menu').setAttribute('aria-expanded',String(open))});
$('expand-all').addEventListener('click',()=>document.querySelectorAll('details').forEach(el=>el.open=true));
$('collapse-all').addEventListener('click',()=>document.querySelectorAll('details').forEach(el=>el.open=false));
$('print-guide').addEventListener('click',()=>window.print());
let beforePrint=[];window.addEventListener('beforeprint',()=>{beforePrint=[...document.querySelectorAll('details')].map(el=>el.open);document.querySelectorAll('details').forEach(el=>el.open=true)});window.addEventListener('afterprint',()=>document.querySelectorAll('details').forEach((el,i)=>el.open=beforePrint[i]));
const sections=[...document.querySelectorAll('.chapter')];const links=[...document.querySelectorAll('.nav a')];
function scrollState(){const progress=window.scrollY/Math.max(1,document.documentElement.scrollHeight-window.innerHeight);$('reading-progress').value=Math.min(100,100*progress);let current=sections[0];for(const s of sections){if(s.getBoundingClientRect().top<180)current=s}links.forEach(a=>a.classList.toggle('active',a.hash==='#'+current.id))}
window.addEventListener('scroll',scrollState,{passive:true});scrollState();
'''

data = {'samples':sample_base,'mean':records['lstm']['summary']['train_mean'],'std':records['lstm']['summary']['train_std'],'records':records}
toc = ''.join(f'<li><a href="#{slug}">{E(title)}</a></li>' for slug,title,_ in chapters)
body = ''.join(x[2] for x in chapters).replace('__QA__',qa_html)
doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A comprehensive offline study guide to the exact Fashion-MNIST LSTM and Transformer in this repository, with equations, tensor shapes, code, results and oral-exam practice."><title>LSTM &amp; Transformer — Codebase Study Guide</title><style>{CSS}</style></head><body>
<aside class="nav" id="navigation" aria-label="Study guide contents"><button id="toggle-menu" class="mobile-menu" aria-expanded="false" aria-controls="contents">Contents</button><div class="brand">From pixels<br>to predictions.</div><p class="small">LSTM &amp; Transformer<br>Fashion-MNIST · A1</p><progress id="reading-progress" value="0" max="100" aria-label="Reading progress"></progress><ol id="contents">{toc}</ol><div class="toolbar"><button id="expand-all">Open answers</button> <button id="collapse-all">Close answers</button> <button id="print-guide">Print / PDF</button></div></aside>
<main><header class="hero"><div class="eyebrow">Implementation-grounded learning companion</div><h1>Understand the model.<br>Explain every step.</h1><p>The LSTM and Transformer in your codebase, from the first normalized pixel to the final class prediction.</p><div class="pills"><span class="pill">20 chapters</span><span class="pill purple">51 oral questions</span><span class="pill">3 interactive laboratories</span><span class="pill">Works offline</span></div><div class="meta">Inspected 28 October 2026 · D:\\DL\\DL-ASS - Copy · Theory + exact source + verified checkpoint evidence</div></header>
<noscript>JavaScript is disabled. The complete theory and code are available below, but the numerical laboratories and history selector require JavaScript. Saved plots and all oral answers remain readable.</noscript>{body}<footer>Built from this checkout's implementation and saved evidence. Read, calculate, reconstruct, then explain in your own words.</footer></main>
<script id="study-data" type="application/json">{json.dumps(data,separators=(',',':')).replace('<','\\u003c')}</script><script>{JS}</script></body></html>'''
assert len(chapters)==20, len(chapters)
assert len(QA)==51, len(QA)
HERE.mkdir(parents=True, exist_ok=True)
(HERE / 'lstm_transformer_study_guide.html').write_text(doc,encoding='utf-8')
(HERE / 'guide_verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(f'Wrote {len(doc):,} characters, {len(chapters)} chapters, {len(QA)} oral questions.')
