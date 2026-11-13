# Assignment 2: five-stage sleep classification from EEG

**Workflow:** `python preprocessing.py` → `training.ipynb` → `runs/<model>/<dataset>_final/`.
One notebook supports Custom CNN, temporal Transformer, AttnSleep and CBraMod through four JSON configurations. The current notebook selects `AttnSleep.json`.

The completed AttnSleep run is now in [runs/AttnSleep/sleep_waveform_final](runs/AttnSleep/sleep_waveform_final). It includes all five local checkpoints, validation/test probability exports, training history and verification. Its original run identifier is `sleep_waveform_20261110_134729_471355`; the original cloud files remain preserved locally.

## 1. Start from a fresh clone

```bash
git clone --depth 1 https://github.com/quocthescaredycat/DL-ASS.git
cd DL-ASS/A2
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt jupyterlab
python -m jupyter lab
```

Open `training.ipynb` with **A2 as the working directory**. On Colab or Vast.ai, upload the contents of A2 into the notebook working directory; no extra parent folder or Drive mounting is needed. If a notebook kernel is already available, install requirements using its interpreter:

```python
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path.cwd()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
```

Restart the kernel after changing imported packages. Use a CUDA-enabled PyTorch environment for GPU training; inspect `torch.cuda.is_available()` before starting. Requirements specify version ranges; each run's `environment.txt` records its installed versions.

The repository contains code, configurations, participant folds, the CBraMod initialization weights, and recorded results. **Raw EDF recordings, waveform tensors and fine-tuned fold checkpoints are not included in a fresh clone.** Rebuild the dataset and train to generate weights, or obtain those files separately to evaluate without training. The tracked `pretrained_weights/pretrained_weights.pth` initializes CBraMod; it is not a five-stage fine-tuned checkpoint.

Required project files:

```text
./
├── training.ipynb
├── preprocessing.py
├── run_utils.py
├── model_lib.py
├── attnsleep.py
├── cbramod.py
├── criss_cross_transformer.py
├── requirements.txt
├── requirements-eeg.txt
├── configs/
├── sleep_exploration/
│   ├── cv_folds.csv
│   └── cv_metadata.json
└── pretrained_weights/
    └── pretrained_weights.pth       # CBraMod initialization only
```

`attnsleep.py` is required because the model library imports it. Old experimental notebooks, `models.py`, `make_small_dataset.py`, the spectrogram dataset and a local virtual environment are not training inputs. If `dataset/sleep_waveform.pt` is already available, training does not need the raw EDF files.

## 2. Download the source recordings

Source: [Sleep-EDF Expanded, Sleep Cassette, version 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/). Use the 153 Sleep Cassette PSG/hypnogram pairs. The following notebook cell downloads the checksum manifest, skips valid files and replaces missing or invalid downloads:

```python
from pathlib import Path
from urllib.request import urlretrieve
from preprocessing import sha256

base_url = "https://physionet.org/files/sleep-edfx/1.0.0/"
manifest = Path("SHA256SUMS.txt")
if not manifest.exists():
    urlretrieve(base_url + "SHA256SUMS.txt", manifest)

entries = [line.split(maxsplit=1) for line in manifest.read_text().splitlines()]
edfs = [(digest, name) for digest, name in entries
        if name.startswith("sleep-cassette/SC") and name.endswith(".edf")]
assert len(edfs) == 306
for i, (digest, name) in enumerate(edfs, 1):
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256(path) != digest:
        print(f"[{i}/{len(edfs)}] Downloading {path.name}", flush=True)
        temporary = path.with_suffix(".edf.part")
        urlretrieve(base_url + name, temporary)
        assert sha256(temporary) == digest, path.name
        temporary.replace(path)
print("Ready: 153 PSG/hypnogram pairs.")
```

Allow approximately **40 GB of free disk space before downloading**, in addition to the Python environment and existing datasets. The EDF archive is approximately 7.6 GB; processing temporarily keeps per-recording shards and merge files alongside the approximately 9.4 GB full output. Retaining another full output requires additional space. Preprocessing runs on CPU, one recording at a time.

## 3. Rebuild the full waveform dataset

From the directory containing `preprocessing.py`:

```bash
python preprocessing.py
```

The default keeps all eligible epochs and writes `dataset/sleep_waveform.pt`, plus a sample-index CSV and a JSON summary. With `SHA256SUMS.txt` beside the script, it uses the same processing parameters as:

```python
from preprocessing import preprocess
summary = preprocess(samples_per_participant=None,
                     checksums_path="SHA256SUMS.txt", overwrite=False)
```

CLI default paths are relative to the script; explicitly supplied relative Python paths are relative to the working directory. The CLI automatically enables checksum verification when the manifest exists. To require it explicitly, run `python preprocessing.py --checksums SHA256SUMS.txt`.

In a notebook, avoid a Bash dependency:

```python
import sys, subprocess
subprocess.check_call([sys.executable, "preprocessing.py"])
```

An existing output is protected by default. Use `--overwrite` only when intentionally rebuilding it. The pipeline reads EDF signals and annotations; it does not reconstruct EEG from a spectrogram.

| Export | Tensor shape | Development epochs | Test epochs | Participants / recordings |
|---|---|---:|---:|---:|
| Full, used in the final comparison | `(195469, 2, 30, 200)` | 170,537 | 24,932 | 78 / 153 |
| Small, 83 epochs per participant | `(6474, 2, 30, 200)` | 5,478 | 996 | 78 / 153 |

The shared preparation uses Fpz-Cz and Pz-Oz EEG, continuous zero-phase FIR filtering at 0.3–35 Hz, resampling from 100 to 200 Hz, 30-second epochs, and amplitude units of **microvolts / 100**. Stages 3 and 4 become N3; movement/unscored intervals are excluded; wake is retained within 30 minutes around scored sleep. Upsampling matches the model interface and does not add measured high-frequency information.

The full export keeps every eligible epoch. The optional small export selects epochs without shortening or compressing individual waveforms:

```bash
python preprocessing.py --samples-per-participant 83 --output-file dataset/sleep_waveform_small.pt
```

The `.pt` file contains tensors, identities, labels, splits and preprocessing metadata. Keep its companion CSV/JSON for provenance. `sleep_edf.pt` belongs to an older spectrogram pipeline and is not used here. Participant fold assignments come from `sleep_exploration/cv_folds.csv`.

## 4. Train with one notebook and four configurations

Change the configuration filename in the path cell:

```python
CONFIG_PATH = PROJECT_DIR / "configs" / "AttnSleep.json"
```

| Configuration | Model | Initial LR | Batch size | Weight decay | Scheduler |
|---|---|---:|---:|---:|---|
| `StandardCNN2D.json` | Custom CNN | `1e-5` | 64 | `1e-4` | Cosine, T_max=80 |
| `Transformer.json` | Temporal Transformer, 30 time tokens | `3e-5` | 64 | `1e-4` | Cosine, T_max=80 |
| `AttnSleep.json` | AttnSleep, Fpz-Cz | `1e-3` | 128 | `1e-3` | MultiStep, epoch 10 × 0.1 |
| `CBraModClassifier.json` | Fully fine-tuned CBraMod | `3e-5` | 128 | `1e-4` | Cosine, T_max=80 |

All four configurations use `dataset/sleep_waveform.pt`. AttnSleep selects Fpz-Cz and adapts the band-limited input to 100 Hz internally. The other three models use both channels. Do not divide by 100 again or apply spectrogram normalization.

Run the eleven cells in order. Save the notebook before training so the source snapshot records the intended code. To change a setting, edit the JSON, rerun the path/dataset cells and then train. The visible training variables may also be overridden before `config = {...}`. The effective configuration is saved in the run. Adjust the cosine scheduler's `T_max` when changing its intended duration.

Shared training and evaluation choices:

- Saved participant folds, five development folds and a disjoint 12-participant test set.
- PyTorch and NumPy seed 42; no guarantee of bitwise-identical CUDA results.
- Adam with AMSGrad, weighted cross-entropy and class weights computed from training labels only.
- Best validation macro-F1 checkpoint; early stopping patience 10 and minimum improvement 0.001; at most 80 epochs.
- Correct epoch-loss aggregation using the sum of target class weights.
- Accuracy, balanced accuracy, macro-F1, Cohen's kappa, MGm, macro-precision, macro-recall and five per-stage F1 scores. MGm is the arithmetic mean of the five one-vs-rest geometric means. Macro-recall equals balanced accuracy here.
- Linux/CUDA defaults: workers 4, prefetch 2, persistent workers true and pinned memory true. For a macOS notebook/CPU check, use `0`, `null`, `false`, `false` in the JSON.
- `# if fold > 1: break` remains available for a two-fold pilot. Final soft-voting requires all five completed folds.

A second GPU can be selected with `torch.device("cuda:1")` when it exists. The default uses CUDA 0 when available, otherwise CPU.

## 5. Fixed output directories and overwrite behaviour

```text
runs/AttnSleep/sleep_waveform_final/
├── config.json
├── run.json
├── fold_1.pt                         # Local only; ignored by Git
├── fold_1.history.csv
├── fold_1.best.json
├── fold_1.setup.json
├── ... fold_2–fold_5
├── history.csv
├── cv_results.csv
├── cv_summary.csv
├── pooled_validation_metrics.csv
├── pooled_validation_predictions.csv
├── pooled_validation_confusion_matrix.csv
├── test_metrics.csv
├── test_predictions.csv
├── test_confusion_matrix.csv
├── samples.csv
├── preprocessing.json
├── environment.txt
└── source/
```

There is no timestamp to enter and no nested checkpoint folder. Small-data runs use `sleep_waveform_small_final`.

**Rerunning the training cell replaces the previous run for the same model and dataset from scratch.** Old folds and evaluation outputs are cleared to prevent mixing runs. Other model/dataset directories and `spec_model/` are untouched. Copy an existing run first if it must remain available for comparison. Do not train the same model/dataset concurrently into one directory.

Checkpoints contain model state dictionaries, not optimizer/scheduler states for exact interrupted-training resumption. Download the full run folder before terminating a cloud instance. Fine-tuned `.pt`/`.pth` files are ignored by Git and do not need to be published; configs, metrics, probabilities and provenance can be committed. See [CHECKPOINTS.md](CHECKPOINTS.md).

The three summaries answer different questions: fold mean ± population SD, pooled validation with one prediction per development epoch, and held-out test with equal-weight averaging of five models' probabilities. Validation is used for model selection and is not an independent test result.

## 6. Evaluate existing weights without training

Run the cells through path, device and dataset setup, then use this cell instead of training. Change only `RUN_DIR` for the desired model. It requires the dataset and all five local fold checkpoints.

```python
RUN_DIR = PROJECT_DIR / "runs" / "AttnSleep" / "sleep_waveform_final"
config = json.loads((RUN_DIR / "config.json").read_text())
run_metadata = json.loads((RUN_DIR / "run.json").read_text())
DATA_PATH = PROJECT_DIR / config["data_path"]

# Imported runs may include a separately verified local reconstruction.
verified = run_metadata.get("verified_local_reconstruction")
expected_data_hash = (verified["dataset_sha256"] if verified
                      else run_metadata["dataset"]["sha256"])
expected_sources = (verified["source_sha256"] if verified
                    else run_metadata["source_sha256"])
assert sha256(DATA_PATH) == expected_data_hash
for filename in ["model_lib.py", "attnsleep.py", "cbramod.py", "criss_cross_transformer.py"]:
    assert sha256(PROJECT_DIR / filename) == expected_sources[filename]

dataset = EEGDataset(pt_filepath=DATA_PATH)
kwargs = dict(config["model_kwargs"])
if config["model_name"] == "CBraModClassifier":
    CHECKPOINT = PROJECT_DIR / "pretrained_weights" / "pretrained_weights.pth"
    kwargs["checkpoint_path"] = str(CHECKPOINT)
    assert sha256(CHECKPOINT) == run_metadata["pretrained_sha256"]
model = getattr(model_lib, config["model_name"])(**kwargs).to(device)
batch_size, n_splits = config["batch_size"], config["n_splits"]
y, groups = dataset.labels.numpy(), np.asarray(dataset.groups)

if (RUN_DIR / "samples.csv").exists():
    saved = pd.read_csv(RUN_DIR / "samples.csv")
    saved_dev = saved.loc[saved["split"] == "development"]
    assert np.array_equal(saved_dev["source_index"].to_numpy(), dataset.indices)
    fold_ids = saved_dev["validation_fold"].to_numpy()
else:
    manifest_path = PROJECT_DIR / run_metadata["fold_manifest"]
    assert sha256(manifest_path) == run_metadata["fold_manifest_sha256"]
    manifest = pd.read_csv(manifest_path).set_index("participant")
    fold_ids = np.array([int(manifest.loc[p, "validation_fold"]) for p in groups])
assert np.array_equal(fold_ids, dataset.validation_folds)
fold_splits = [(np.flatnonzero(fold_ids != i), np.flatnonzero(fold_ids == i))
               for i in range(1, n_splits + 1)]
cv_results = pd.read_csv(RUN_DIR / "cv_results.csv", header=[0, 1], index_col=0)
stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
assert all((RUN_DIR / f"fold_{i + 1}.pt").is_file() for i in range(n_splits))
for item in run_metadata["completed_folds"]:
    assert sha256(RUN_DIR / item["checkpoint"]) == item["sha256"]
```

Skip the training cell and run the final pooled-validation and test soft-voting cells. They overwrite evaluation CSVs but preserve weights. A new serialization of the dataset can have a different file hash; verify sample identities, processing and source compatibility before accepting it, rather than removing hash checks blindly.

For the completed AttnSleep import, [verification.json](runs/AttnSleep/sleep_waveform_final/verification.json) records the original cloud dataset hash and the different local serialization hash. All 195,469 sample identities, labels, participant splits/folds and preprocessing settings match. The active AttnSleep class and author module are identical, although the source library's unrelated Transformer changed. A 128-example CPU float32 soft-voting check matches all archived predicted labels (maximum probability difference 0.000339 versus CUDA AMP). This verifies the tested reconstruction; it does not prove bitwise equality of every cloud/local waveform. Original cloud metadata remain in `source/cloud_run.json` and `source/cloud_config.json`.

## 7. Final recorded results and provenance

| Model | Validation macro-F1, mean ± SD | Pooled validation macro-F1 | Held-out test accuracy | Held-out test macro-F1 |
|---|---:|---:|---:|---:|
| Custom CNN | 0.6750 ± 0.0144 | 0.6756 | 0.7634 | 0.7063 |
| Temporal Transformer | 0.7498 ± 0.0039 | 0.7504 | 0.7989 | 0.7564 |
| AttnSleep, completed rerun | 0.7451 ± 0.0053 | 0.7456 | 0.7970 | 0.7506 |
| CBraMod | 0.7349 ± 0.0070 | 0.7362 | 0.7863 | 0.7422 |

AttnSleep's completed rerun replaces its older aggregate-only main result. It supplies five verified checkpoints, 170,537 pooled validation predictions, 24,932 test predictions and exact training histories. Its selected epochs are 16, 29, 12, 15 and 14; 129 epochs were completed across five folds. Recorded training/validation time is 4,466.29 seconds (1.241 hours) on an RTX 3090. Test balanced accuracy is 0.7610, kappa 0.7250 and MGm 0.8468. All 12 metrics agree with the saved confusion matrices and probability CSVs. Runtime is an observed run cost, not a controlled hardware-throughput comparison.

All four models now have five local checkpoints and test prediction records. Original CNN/Transformer/CBraMod files in `spec_model/` remain unchanged and ignored. Older AttnSleep aggregates are preserved locally under `releases/attnsleep-before-completed-run/`; the timestamped cloud run is also retained locally. Neither archive is required for a new training execution.

**Commit [run_records.json](run_records.json).** It inventories reported settings, timings, hardware and checkpoint hashes; the notebook does not read it. New runs write their own `run.json`, not this historical inventory. Some source-notebook and checkpoint paths describe locally inspected evidence that is not distributed in Git. The three earlier models have retrospective provenance with unknown cloud commit/time. AttnSleep has recorded environment/source snapshots, but its cloud checkout had no Git commit. Its completed notebook is retained under the run's `source/` directory; the runnable notebook stays clean.

[Run layout](runs/README.md) · [Final report](G09_A2_Report.pdf) · [Report source and evidence](DL-ASS-A2/README.md) · [Assignment page](../assignment2.html) · [AI disclosure](../AI_USAGE.md).

## 8. Verification and scope

- Source annotation identities were checked for all 195,469 full-data epochs. Two recordings were re-exported: 1,968 waveforms matched the existing local full dataset exactly. A small-export pilot also matched the corresponding waveforms. The complete 9.4 GB tensor was not regenerated during this audit.
- Nine preprocessing tests passed. A synthetic CPU five-fold train/save/evaluate/reload check reproduced identical probabilities. Repeating training into the fixed directory removed stale results and blocked an incomplete five-model ensemble.
- An isolated copy containing tracked files plus the documented support files imported successfully. All four configs returned finite `(2, 5)` logits for `(2, 2, 30, 200)` input. This used installed dependencies, not a fresh environment installation.
- AttnSleep's five checkpoint/source hashes, strict state loading, train-only class weights, best epochs, fold histories, full prediction identities and all reported metrics were verified. No full-data retraining was performed during report preparation.

The comparison uses one dataset protocol with model-specific input adapters. Five-fold validation and the separate test ensemble are not the author's 20-fold benchmark. The preparation-recipe sensitivity experiment changes several preprocessing choices together and has ten additional development Wake epochs; it does not isolate a single filtering effect. The report preserves these limits and distinguishes validation selection from held-out evaluation. Matched GPU inference latency and peak-memory measurements are not available.

## 9. Citations and source attribution

The requested BibTeX entries are in [references.bib](references.bib), with keys `wang2025cbramod` and `emadeldeen_attnSleep`.

- **CBraMod:** Wang et al., *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*, ICLR 2025. [Paper](https://openreview.net/forum?id=NPNUHgHF2w), [official implementation and pretrained model](https://github.com/wjq-learning/CBraMod). This project uses the released backbone and criss-cross attention with a five-stage classification head and fine-tuning workflow.
- **AttnSleep:** Eldele et al., *An Attention-based Deep Learning Approach for Sleep Stage Classification with Single-Channel EEG*, IEEE TNSRE 29, 809–818, 2021. [Paper / DOI](https://doi.org/10.1109/TNSRE.2021.3076234), [official implementation](https://github.com/emadeldeen24/AttnSleep). The author architecture and attribution/license notice are retained in `attnsleep.py`; the project adapter handles channel selection and sample rate. The custom temporal Transformer's convolution branches borrow a multi-resolution idea, not the complete AttnSleep architecture.

Using author source code does not imply reproducing every preprocessing choice, evaluation protocol or published score. The shared data preparation and training choices are described above.
