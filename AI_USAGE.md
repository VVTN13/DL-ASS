## Assignment 1

### Website Assistance and Figure Export

**Tool:** Claude (Anthropic), via Claude Code.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** M1 Draft, September 2026.\

**Task:** Building the Assignment 1 GitHub Pages page using the M1 notebook and experimental results.\

**Prompt summary:** Review the project layout and update the Assignment 1 page from the M1 notebook and run results.\

**AI contribution:** Filled in the Assignment 1 page sections, added CSS for figures and the pipeline diagram, and exported notebook figures to `A1/figures/` for use on the page.\

**Student verification:** Checked every number on the page against `A1/runs/m1/*/summary.json`, `history.csv`, and notebook outputs. Confirmed that figure paths existed in `A1/figures/`. Reviewed changes in PR #2 before merging.\

**Affected files/sections:** `assignment1.html`, `style.css`, `A1/figures/`.\

**Responsible members for final verification:** Nguyễn Vũ Long, Lê Phước Vũ.

### Setup and Architecture Documentation

**Tool:** Claude (Anthropic), via Claude Code.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** M1 Draft, September 2026.\

**Task:** Preparing setup, reproduction, and architecture documentation for Assignment 1.\

**Prompt summary:** The existing disclosure records documentation assistance but does not provide a separate prompt summary for this task.\

**AI contribution:** Wrote `A1/README.md` with setup and reproduction instructions and `A1/docs/M1_ARCHITECTURE.md` with architecture documentation.\

**Student verification:** Re-ran the training commands in `A1/README.md` for both models and checked that the results matched `A1/runs/m1/*/summary.json`.\

**Affected files/sections:** `A1/README.md`, `A1/docs/M1_ARCHITECTURE.md`.\

**Responsible members for final verification:** Nguyễn Vũ Long, Lê Phước Vũ.

### Repository Structure and Link Review

**Tool:** Claude (Anthropic), via Claude Code.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** M1 Draft, September 2026.\

**Task:** Reviewing the repository structure and correcting links and file paths.\

**Prompt summary:** Review the repository layout, fix broken links, and update the Assignment 1 GitHub Pages content.\

**AI contribution:** Reviewed the repository structure and corrected references from `DL-Assignment` to `DL-ASS`, the `AI_USAGE.md` link, and paths in `README.md` and `A1/README.md`.\

**Student verification:** Checked repository links against the git remote (`VVTN13/DL-ASS`). Reviewed changes in PR #2 before merging.\

**Affected files/sections:** Repository and navigation links in `assignment1.html`, `index.html`, `assignment2.html`, `assignment3.html`, `README.md`, and `A1/README.md`.\

**Responsible members for final verification:** Nguyễn Vũ Long, Lê Phước Vũ.

### Code and Environment Troubleshooting

**Tool:** Claude (Anthropic), via Claude Code.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** M1 Draft, September 2026.\

**Task:** Resolving code and environment issues affecting model training.\

**Prompt summary:** The existing disclosure records troubleshooting assistance but does not provide a separate prompt summary for this task.\

**AI contribution:** Helped fix code and environment issues, including MPS/CUDA/CPU device selection and library version errors.\

**Student verification:** Re-ran the training commands in `A1/README.md` for both models and checked that the results matched `A1/runs/m1/*/summary.json`.\

**Affected files/sections:** `A1/a1/` (device/environment fixes).\

**Responsible members for final verification:** Nguyễn Vũ Long, Lê Phước Vũ.

### Exploratory Data Analysis and Dataset Statistics

**Tool:** GPT-6 Sol.\

**Used by:** Lê Phước Vũ.\

**Development stage:** Data exploration and analysis (Milestone 1).\

**Task:** Reviewing the exploratory data analysis of the Fashion-MNIST dataset, including class distributions, image properties, pixel-intensity statistics, and representative samples.\

**Prompt summary:** Requested assistance with interpreting the dataset exploration results and identifying useful statistical and visual analyses for understanding the characteristics of Fashion-MNIST before model training.\

**AI contribution:** Potential assistance included explaining the purpose of class-distribution analysis, pixel-intensity distributions, representative image grids, and dataset statistics. The focus was on interpreting existing observations and determining how they related to subsequent preprocessing and classification experiments.\

**Student verification:** Confirm the observations against the original Fashion-MNIST data, the EDA notebook outputs, and the statistics calculated by the implementation.\

**Affected files/sections:** `A1/dl1-261.ipynb`, EDA sections and associated figures.\

**Responsible members for final verification:** Lê Phước Vũ.

### Data Preprocessing and Split Validation

**Tool:** GPT-6 Sol.\

**Used by:** Lê Phước Vũ.\

**Development stage:** Data preparation (Milestone 1).\

**Task:** Reviewing the dataset splitting, normalization, and preprocessing procedures to ensure consistency between training, validation, and test data.\

**Prompt summary:** Requested an explanation and technical review of the stratified training-validation split, training-derived normalization statistics, and the handling of the official Fashion-MNIST test set.\

**AI contribution:** Potential assistance included clarifying the importance of preventing data leakage, calculating normalization statistics exclusively from the training subset, preserving class proportions, and applying identical normalization parameters across dataset partitions.\

**Student verification:** Check that the implementation creates 54,000 training samples, 6,000 validation samples, and 10,000 test samples. Review the cached split indices and normalization statistics to confirm consistency with the intended preprocessing workflow.\

**Affected files/sections:** `A1/a1/data.py`, `A1/split_cache.json`, preprocessing sections of `A1/dl1-261.ipynb`.\

**Responsible members for final verification:** Lê Phước Vũ.

### Linear Classifier Implementation and Baseline Evaluation

**Tool:** GPT-6 Sol.\

**Used by:** Lê Phước Vũ.\

**Development stage:** Linear baseline implementation and evaluation (Milestones 1–2).\

**Task:** Reviewing the Linear classifier implementation and interpreting its evaluation results as a baseline for other architectures.\

**Prompt summary:** Provided the Linear model implementation and evaluation results and requested checks of the input dimensions, output logits, parameter count, accuracy, macro-F1 score, and confusion patterns.\

**AI contribution:** Potential assistance included explaining the flattening of 28×28 images into 784-dimensional inputs, the Linear classifier’s ten class scores, trainable parameters, and limitations of linear decision functions when interpreting classification errors.\

**Student verification:** Confirm the input and output dimensions and independently calculate the parameter count. Compare the accuracy and macro-F1 statements with saved Linear results, and inspect the confusion matrix and prediction examples before using the interpretations.\

**Affected files/sections:** `A1/a1/models.py`, `A1/docs/LINEAR.md`, `A1/dl1-261.ipynb`, `A1/runs/cuda_baselines/linear/`.\

**Responsible members for final verification:** Lê Phước Vũ.

### MLP Architecture and Layer Configuration Review

**Tool:** GPT-6 Sol.\

**Used by:** Nguyễn Hoàng Quốc.\

**Development stage:** MLP implementation (Milestone 1).\

**Task:** Reviewing the Multi-Layer Perceptron architecture, including its hidden layers, activation functions, and connections between successive layers.\

**Prompt summary:** Provided the MLP implementation and requested a technical review of its input and output dimensions, hidden-layer sizes, nonlinear activation functions, and trainable parameter calculations.\

**AI contribution:** Potential assistance included explaining the roles of fully connected layers, ReLU activations, and intermediate feature representations. The review could also cover the relationship between the MLP architecture and the simpler Linear classifier.\

**Student verification:** Check that the implementation uses the intended 784-dimensional input, 256- and 128-unit hidden layers, and ten output classes. Compare the trainable parameter count with the programmatically calculated value.\

**Affected files/sections:** `A1/a1/models.py`, `A1/docs/MLP.md`.\

**Responsible members for final verification:** Nguyễn Hoàng Quốc.

### CNN Feature Extraction and Tensor Dimension Verification

**Tool:** GPT-6 Sol.\

**Used by:** Nguyễn Hoàng Quốc.\

**Development stage:** CNN implementation (Milestone 2).\

**Task:** Reviewing the Convolutional Neural Network architecture and verifying the dimensional consistency of its convolutional, pooling, and classification layers.\

**Prompt summary:** Provided the CNN implementation and requested assistance checking how image dimensions and feature-map channels change through convolution and max-pooling operations. The review also covered the transition from extracted feature maps to the final fully connected classifier.\

**AI contribution:** Potential assistance included explaining convolutional feature extraction, spatial downsampling, channel expansion, and flattening operations. The dimensional review could help identify mismatches between the convolutional backbone and the classification head.\

**Student verification:** Check that the implemented model accepts images of shape `(batch, 1, 28, 28)` and produces ten class scores. Inspect intermediate feature-map dimensions and confirm the classifier input size using model tests.\

**Affected files/sections:** `A1/a1/models.py`, `A1/docs/CNN.md`, `A1/tests/test_cnn.py`.\

**Responsible members for final verification:** Nguyễn Hoàng Quốc.

### MLP and CNN Regularization, Training Behavior, and Performance Comparison

**Tool:** GPT-6 Sol.\

**Used by:** Nguyễn Hoàng Quốc.\

**Development stage:** Model refinement and comparative evaluation (Milestone 2).\

**Task:** Reviewing MLP and CNN regularization, training behavior, and comparative Fashion-MNIST results.\

**Prompt summary:** Requested explanations of dropout and training/evaluation modes, and suggestions for comparing saved MLP and CNN accuracy, macro-F1 scores, parameter counts, classification errors, and computational characteristics.\

**AI contribution:** Potential assistance included explaining dropout placement and overfitting indicators, the distinction between fully connected and convolutional processing, and ways to discuss training curves and class-specific errors without confusing measured results with possible architectural explanations.\

**Student verification:** Inspect the implemented dropout modules and probabilities. Cross-check reported metrics against the saved experimental summaries, training histories, and confusion matrices, and avoid unsupported causal claims about performance differences.\

**Affected files/sections:** `A1/a1/models.py`, `A1/a1/train.py`, `A1/a1/compare.py`, `A1/runs/cuda_baselines/mlp/`, `A1/runs/m2/cnn/`, MLP/CNN discussion sections.\

**Responsible members for final verification:** Nguyễn Hoàng Quốc.

### LSTM Sequence Representation and Hidden-State Verification

**Tool:** GPT-6 Sol.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** LSTM implementation (Milestone 2).\

**Task:** Reviewing the representation of Fashion-MNIST images as row sequences and verifying the LSTM architecture's input, hidden-state, and output dimensions.\

**Prompt summary:** Requested technical explanations of LSTM hidden states, cell states, recurrent gate operations, and the processing of 28 image rows containing 28 pixel values each. The review also covered how the final hidden representation is converted into class predictions.\

**AI contribution:** Potential assistance included clarifying the roles of input, forget, and output gates, explaining the difference between hidden and cell states, and illustrating how sequential processing produces a 128-dimensional final hidden representation used for classification.\

**Student verification:** Check that the model accepts input tensors of shape `(batch, 28, 28)`, uses a hidden size of 128, and produces ten classification logits. Inspect the corresponding LSTM tests and model implementation for dimensional consistency.\

**Affected files/sections:** `A1/a1/models.py`, `A1/docs/LSTM.md`, `A1/tests/test_lstm.py`.\

**Responsible members for final verification:** Nguyễn Vũ Long.

### Transformer Patch Embedding and Self-Attention Review

**Tool:** GPT-6 Sol.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** Transformer implementation (Milestone 2).\

**Task:** Reviewing the Transformer model's patch-based input representation, embedding layers, positional information, and self-attention operations.\

**Prompt summary:** Requested explanations of how 28×28 grayscale images are divided into 4×4 patches, how those patches are projected into embedding vectors, and how the Transformer processes spatial information through positional embeddings and multi-head self-attention.\

**AI contribution:** Potential assistance included explaining the construction of 49 patch tokens, the transformation of 16-dimensional patches into 64-dimensional embeddings, the role of positional embeddings, and the use of attention blocks and mean pooling for image classification.\

**Student verification:** Check the patch representation and confirm that the model receives tensors of shape `(batch, 49, 16)` and produces ten classification logits. Inspect the Transformer tests for output dimensions, positional behavior, and parameter consistency.\

**Affected files/sections:** `A1/a1/models.py`, `A1/docs/TRANSFORMER.md`, `A1/tests/test_transformer.py`.\

**Responsible members for final verification:** Nguyễn Vũ Long.

### Sequence-Model Testing and Implementation Debugging

**Tool:** GPT-6 Sol.\

**Used by:** Nguyễn Vũ Long.\

**Development stage:** LSTM and Transformer verification (Milestone 2).\

**Task:** Reviewing the testing procedures used to validate the behavior and implementation consistency of the LSTM and Transformer classifiers.\

**Prompt summary:** Requested assistance understanding how to test sequence-based model implementations, including tensor shapes, gradient propagation, output consistency, dropout behavior, and checkpoint restoration.\

**AI contribution:** Potential assistance included explaining useful unit-testing strategies for recurrent and attention-based models, identifying potential tensor-layout errors, and suggesting ways to validate model outputs under training and evaluation settings.\

**Student verification:** Execute the LSTM and Transformer unit tests, inspect detected failures, and confirm that corrected implementations satisfy the intended tensor dimensions and numerical checks.\

**Affected files/sections:** `A1/tests/test_lstm.py`, `A1/tests/test_transformer.py`, `A1/a1/models.py`.\

**Responsible members for final verification:** Nguyễn Vũ Long.

### Sources Used for Verification

The existing disclosure lists the following verification sources for Assignment 1:

- `A1/runs/m1/linear/summary.json` and `A1/runs/m1/mlp/summary.json`.

- `history.csv` files and `A1/dl1-261.ipynb` outputs.

- `A1/configs/m1.json`.

- [Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist).

- [torchvision FashionMNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).

### Stronger A1 model adaptation, 19 October 2026

| Field | Value |
|---|---|
| Tool name / version | OpenAI Codex |
| Used by | Repository requester; member name not recorded in this session |
| Time / stage | A1 M2 implementation, 19 October 2026 |
| Purpose | Adapt the supplied A2 `ResNet50_CBAM` and `ConvLSTMAttentionTransformer` for Fashion-MNIST; preserve exact class names and older baseline checkpoint compatibility. |
| Affected section(s) | `A1/a1/strong_models.py`, model/CLI selection, new configs and tests, artifact verifier, `A1/README.md`, `A1/docs/STRONG_MODELS.md` |
| Representative prompt or prompt-log link | Requested stronger CNN and LSTM-based models like those in `A2/model_lib.py`, using the exact class names. |
| How output was edited and verified | Adapted channels, output classes, small-image stem and patch/sequence dimensions; checked tensor shapes, gradients, attention, and training/checkpoint reuse with synthetic data. No full Fashion-MNIST training or improved benchmark score is claimed. |
| Member responsible for final verification | Pending group review and full training/evaluation |
| Sources used for verification | Supplied `A2/model_lib.py`, A1 data/training pipeline, existing checkpoints and automated tests. |

### Direct notebook CBAM workflow, 19 October 2026

| Field | Value |
|---|---|
| Tool name / version | OpenAI Codex |
| Used by | Repository requester; member name not recorded in this session |
| Time / stage | A1 M2 notebook setup, 19 October 2026 |
| Purpose | Add a direct import, batch check, training and evaluation workflow for `ResNet50_CBAM`, matching the teammate's notebook style. |
| Affected section(s) | Section 9 of `A1/dl1-261.ipynb`, `A1/README.md` |
| Representative prompt or prompt-log link | Requested simple `a1` imports and `run_experiment` calls like the existing LSTM cells instead of Colab upload/setup helpers. |
| How output was edited and verified | Preserved all original cells and stored outputs; added three direct CBAM code cells with no requirement for previously saved CBAM results. Checked syntax and exercised the cells locally with synthetic data. No live Colab/GPU run or new Fashion-MNIST benchmark is claimed. |
| Member responsible for final verification | Pending group execution in Colab |
| Sources used for verification | Local A1 data, model, training and evaluation code; the original LSTM notebook cells. |

## Assignment 2

### EEG sleep M2 planning and workspace audit, 27 October 2026

| Field | Value |
|---|---|
| Tool name / version | OpenAI Codex; exact model version not recorded |
| Used by | Repository requester; member attribution to be confirmed by the group |
| Time / stage | A2 M2 draft planning, 27 October 2026 |
| Purpose | Map EEG sleep work to handbook milestones/rubric; audit current dataset/code; plan baseline, pretrained comparison, controlled experiment, evidence and report. |
| Affected section(s) | `A2/EXECUTION_PLAN.md`, `A2/M2_REPORT_TEMPLATE.md`, approval-status update in `A2/SLEEP_DATASET_PROPOSAL.md`, this disclosure entry |
| Representative prompt or prompt-log link | Requested a complete EEG Sleep plan against the handbook for the approaching 25% Assignment 2 deadline, for later execution and reporting. |
| How output was edited and verified | Extracted the handbook and visually checked the milestone/rubric pages; inspected local proposal, code, notebook outputs and saved manifests; loaded tensor metadata with memory mapping, checked labels/fold counts and three tensor samples; ran synthetic CPU forward passes for three existing models; consulted official dataset/MNE/PyTorch/scikit-learn documentation. No training, full raw-signal audit, or performance benchmark was run. |
| Approval-status follow-up | User confirmed instructor approval without conditions on 27 October 2026; updated proposal, plan and report template. The instructor's original decision date was not supplied. |
| Member responsible for final verification | Pending group review and actual experiment execution |
| Sources used for verification | `handbook-ene-v2.pdf`, A2 source/data/manifests and the official links recorded in `A2/EXECUTION_PLAN.md` |

Other A2 AI use, if any, should be recorded below.

### CBraMod comparison planning, 27 October 2026

| Field | Value |
|---|---|
| Tool name / version | OpenAI Codex; exact model version not recorded |
| Used by | Repository requester; member attribution to be confirmed by the group |
| Time / stage | A2 M2 model selection and planning, 27 October 2026 |
| Purpose | Assess the user-supplied CBraMod backbone against a custom CNN; revise the experiment/report plan around EEG pretraining. |
| Affected section(s) | `A2/CBRAMOD_COMPARISON.md`, `A2/EXECUTION_PLAN.md`, `A2/M2_REPORT_TEMPLATE.md`, method/compute update in `A2/SLEEP_DATASET_PROPOSAL.md`, this entry |
| Representative prompt or prompt-log link | Proposed pretrained CBraMod, linked the upstream repository and local backbone files, and suggested comparison with a custom CNN from `model_lib.py`. |
| How output was edited and verified | Reviewed the official repository, paper, ISRUC preprocessing/loader/model and checkpoint listing. Checked local import failure, random-weight waveform forward pass, proposed classifier output/parameter counts and rejection of spectrogram input. The forward check temporarily supplied the upstream module namespace in memory; production Python files were not changed. Checkpoint download failed due to host resolution in the tool environment, so actual weights were not loaded. No training or performance benchmark was run. |
| Member responsible for final verification | Pending group review, checkpoint integration and actual experiments |
| Sources used for verification | User-supplied `A2/cbramod.py`, `A2/criss_cross_transformer.py`, `A2/model_lib.py`; official links in `A2/CBRAMOD_COMPARISON.md` |

| Field | Value |

|---|---|

| Tool name / version |  |

| Used by | |

| Time / stage | |

| Purpose | |

| Affected section(s) | |

| Representative prompt or prompt-log link | |

| How output was edited and verified | |

| Member responsible for final verification | |

| Sources used for verification | |

### CBraMod waveform preprocessing, 27 October 2026

| Field | Value |
|---|---|
| Tool name / version | OpenAI Codex; exact model version not recorded |
| Used by | Repository requester; member attribution to be confirmed by the group |
| Time / stage | A2 M2 waveform pipeline implementation, 27 October 2026 |
| Purpose | Implement a separate raw-EEG preprocessing pipeline for CBraMod, preserving the CNN dataset's epoch labels and participant assignments. |
| Affected section(s) | `A2/preprocessing.py`, sibling import in `A2/cbramod.py`, `A2/tests/test_preprocessing.py`, `A2/.gitignore`, `A2/CBRAMOD_PREPROCESSING.md`, related plan/proposal documentation and this entry |
| Representative prompt or prompt-log link | Requested `A2/preprocessing.py` tailored to CBraMod and clarified that CBraMod accepts signals rather than spectrograms. |
| How output was edited and verified | Implemented continuous 0.3–35 Hz FIR filtering, polyphase 100-to-200 Hz resampling, microvolt/100 scaling, 30-second waveform patches, five-stage annotation mapping, fold preservation, checksum/reference checks and per-recording shards. Seven automated tests passed. Exported real recording SC4001 to a temporary directory: 841 finite `(2,30,200)` samples, with class counts `[188,58,250,220,125]`, verified archive checksums and exact onset/label agreement with the CNN export. Reloaded the shard and passed actual waveform samples through the local random-weight CBraMod backbone. Both package and notebook-style imports were verified using the dependency-equipped Python environment. |
| Limits of verification | One real recording processed; full-cohort preprocessing, pretrained-weight loading, training and accuracy evaluation were not run. The A1 virtual environment lacks MNE, so users must select/install an environment with A2 dependencies. |
| Member responsible for final verification | Pending group execution of the full export and experiments |
| Sources used for verification | Local EDFs, hypnograms, `SHA256SUMS.txt`, `cv_folds.csv`, CNN tensor metadata; official MNE/SciPy APIs and the CBraMod ISRUC preprocessing/loader linked in `A2/CBRAMOD_PREPROCESSING.md` |

### Simplified CBraMod training notebook, 28 October 2026

| Field | Value |
|---|---|
| Tool name / version | OpenAI Codex; exact model version not recorded |
| Used by | Repository requester; member attribution to be confirmed by the group |
| Time / stage | A2 M2 training implementation, 28 October 2026 |
| Purpose | Adapt the user's earlier ASD training workflow to the combined Sleep-EDF waveform dataset, keeping the notebook simple without custom try/except or raise statements. |
| Affected section(s) | `A2/training.ipynb`, `A2/requirements.txt`, this entry |
| Representative prompt or prompt-log link | Requested reuse of `/Users/quocjacob/Desktop/ASD_EEG_thesis/training_2d-dropout-decay.ipynb` for CBraMod and `sleep_waveform.pt`, avoiding overengineering and exception wrappers. |
| How output was edited | Simplified the existing CBraMod notebook around EEGDataset, DataLoader, train/validation, EarlyStopping, plots and fold summaries. Retained Adam, dropout, cosine scheduling and configurable folds; changed binary BCE/sigmoid to five-class cross-entropy/softmax. Used validation macro-F1 for both checkpoint selection and early stopping. Preserved saved participant folds and waveform scale; removed repeated normalization, unused Optuna and redundant validation wrappers. The existing flattened CBraMod head remains 12000→800→200→5, with configurable dropout. |
| How output was verified | Inspected all 195,469 sample metadata entries through memory mapping; verified five development folds contain all five stages, cover each development sample exactly once as validation, and exclude all 12 held-out participants. Strictly loaded the local pretrained checkpoint (SHA-256 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`). Checked a real-data forward/backward pass, finite gradients in backbone/head and the fixed mask parameter. Executed the actual notebook training and summary cells for two CPU epochs using five training and five validation samples from separate development participants; checked best-checkpoint reload/tie handling, prediction-derived macro-F1, early stopping and all CSV/JSON/PNG artifacts. Notebook code cells compile and contain no try/except or raise statements. |
| Limits of verification | The small CPU run verifies execution, not model quality. Full-cohort training, CUDA AMP execution, final test evaluation and the frozen-versus-finetuned ablation were not performed. The default configuration launches five development folds; one-fold execution is explicitly configurable and correctly labelled. |
| Member responsible for final verification | Pending group execution and review of actual experiment results |
| Sources used for verification | User's ASD notebook (read-only), local waveform dataset and pretrained weights, `cbramod.py`, `criss_cross_transformer.py`, PyTorch FFT/AMP documentation |

### Follow-up: restore the original notebook cells, 28 October 2026

The requester explicitly superseded the simplified notebook structure above: restored the reference notebook's nine code cells in their original order, including the exact device/seed/dataset cell, `importlib.reload(model_lib)`, `if fold > 1: break`, `lr=3e-5`, four workers, prefetch factor two, persistent workers and cosine `T_max=100`. Restored `get_stats`/`Normalize`, validation-AUROC checkpoint selection, validation-loss early stopping, and the original function signatures/return values. Removed `seed_everything`, `folds_to_run`, added artifact-export helpers and the extra tqdm dependency. This version no longer automatically writes the earlier notebook's CSV/JSON/checkpoint artifacts; it restores the best weights in memory and displays results, as in the reference.

Added `CBraModClassifier` to `A2/model_lib.py` while retaining the other model definitions. The dataset exposes only the 66 development participants through an index map, so the requested `StratifiedGroupKFold` regenerates development folds without including held-out test participants. Training-only z-score normalization is an explicit requested addition on top of the stored waveform amplitude scaling. The user's subsequent correction selects `CrossEntropyLoss`: integer labels, five raw logits, softmax/argmax metrics and macro one-vs-rest AUROC. The source ASD notebook and pasted attachment were read but not modified.

Verification: original cell positions and verbatim device/seed cell checked; code compilation and absence of custom try/except/raise checked. Checked development-only grouping, waveform index alignment, train-subset normalization, strict pretrained loading, real-data gradients, a two-epoch CPU smoke run and restoration of the AUROC-selected model. Optional package installation, CUDA AMP and four-worker notebook execution were not exercised; full training and held-out test evaluation remain for the group. Member attribution and final experimental verification remain with the requesting group.

### Waveform pipeline review and Cohen's kappa, 28 October 2026

At the requester's instruction, added unweighted `cohen_kappa_score` to `A2/training.ipynb`: train/validation logs, the selected epoch's metrics, the optional objective's return unpacking, and per-fold/fold-summary reporting. Kept the nine original cells, device/seed cell, manual fold stopping, hyperparameters and existing model-selection rule. Preserved the user's separate AdamW edit in the optional objective; the main training loop still uses Adam.

Reviewed the loaded dataset schema and epoch/label/channel alignment. `EEGDataset` now reports waveform channel/rate/scale metadata and explicitly returns float32 waveforms and int64 labels. Reworded and implemented `get_stats` as a per-channel waveform diagnostic over `(B,C,6000)` samples. The previous reduction axes were mathematically valid, but automatic z-scoring was an extra transformation of the already scaled waveform. Under this review request, disabled that transformation in both the main fold loop and optional objective to retain the export's microvolts/100 amplitude, following the upstream ISRUC loader (`https://github.com/wjq-learning/CBraMod/blob/main/datasets/isruc_dataset.py`). This supersedes the preceding request to retain spectrogram-style normalization. The saved dataset and pretrained backbone files were not changed.

Verification used all development split metadata, checked all five newly generated grouped folds, verified waveform/label mapping and exclusion of 12 test participants, compared `get_stats` with direct channel statistics on synthetic and real data, and checked known kappa values. Executed the actual manual two-fold loop with five real development participants (one epoch per class per participant), two CPU training epochs per fold, and no extra normalization. Recomputed all six reported metrics from the restored model; tested optional-objective wiring separately without another training run. Optional dependency installation, CUDA AMP and four-worker notebook execution were not exercised. These are execution checks, not full-data model-quality results; group attribution and final verification remain pending.

### Cloud working directory without a parent project folder, 28 October 2026

At the requester's instruction, removed the automatic parent-folder lookup from the notebook import and path cells. `PROJECT_DIR` now uses only `os.getcwd()`, and imports use that same directory. Preserved all nine cells and the training implementation. Updated the preprocessing invocation in its docstring and added `A2/CLOUD_SETUP.md` with a two-file minimum upload (`preprocessing.py` and the saved fold CSV), direct official Sleep Cassette downloads, and the additional files needed for training.

Verified the official PhysioNet Sleep Cassette listing and checksum endpoint; the local cohort matches all 153 Sleep Cassette PSG entries in RECORDS. Checked notebook Python syntax and the absence of parent-folder path references. Ran the unchanged preprocessing logic on real recording SC4001 in a temporary flat directory with no CNN reference file, verifying source checksums. All 841 generated waveforms, labels, onsets and participant split/fold assignments exactly matched the corresponding entries in the existing combined dataset. Full cloud downloads, full-cohort regeneration and GPU training were not run. Group attribution and final experiment verification remain pending.

### Participant-covering small waveform dataset, 28 October 2026

Following the requester's confirmed limit of 6,500 epochs for the entire file and request for an executable extraction step, added `A2/make_small_dataset.py`. It memory-maps the existing waveform export, selects 83 epochs per participant with seed 42, and writes a separate small `.pt` plus a CSV selection manifest. Each available recording/stage stratum receives at least one epoch, with remaining quotas allocated proportionally and sampled without replacement. Waveforms, labels, onsets and participant split/fold assignments are copied without signal processing. The original export summary is stored as source provenance and the subset receives a new summary. No notebook cells or training hyperparameters were changed; cloud instructions show the single `%run` command and the dataset-path replacement.

Executed the script locally: 6,474 epochs, 78 participants, all 153 recordings, 5,478 development epochs and 996 held-out test epochs; output size 311,137,320 bytes. Verified every selected waveform and metadata entry exactly against the source, all CSV rows, participant quotas, repeatability, a short-participant edge case and subset summary totals. Loaded the file with the actual notebook EEGDataset and checked all five regenerated grouped folds for participant separation and presence of all classes. The source export was not overwritten. Full training/model-quality evaluation and execution on the user's cloud instance were not performed; group attribution and final experimental review remain pending.

### Direct EDF preprocessing with an 83-epoch participant quota, 28 October 2026

The requester superseded extraction from the existing waveform tensor and asked to regenerate the small dataset directly through `A2/preprocessing.py`. Added a metadata pass over EDF headers and hypnograms, followed by deterministic participant-level sampling across all available nights/stages. Each participant contributes up to 83 unique epochs in total; short participants retain their available epochs. The existing continuous filtering, 200 Hz resampling and microvolt/100 scaling remain unchanged, and only the selected epochs are turned into patches. Existing EDF checksum, clock, label/reference and fold checks were retained; added full-run coverage checks against the participant and source checksum manifests. The default output is now `dataset/sleep_waveform_small.pt`, accompanied by a selection CSV. The older waveform tensor is not a preprocessing input and is not overwritten.

Updated cloud instructions and only the dataset-path line in the nine-cell training notebook. Updated an older shard test to use the current shard API, added a two-night quota/rare-stage/reproducibility test, and checked the combined output and CSV. All eight preprocessing tests passed. Ran the complete raw-EDF pipeline locally across 153 recordings: 6,474 epochs, exactly 83 for each of 78 participants, 5,478 development epochs and 996 test epochs; output size 311,066,339 bytes. Verified every regenerated waveform, label, onset and split/fold assignment against the corresponding original full-export entry, all CSV rows, source-code provenance and the notebook's dataset loader/five grouped folds. Archive checksum and optional CNN-reference checks both passed. No model training or cloud execution was performed in this step; group attribution and experimental review remain pending.

### Prioritize validation Macro-F1 for model selection, 28 October 2026

Following the requester's confirmation that macro averaging was appropriate and that selection priority should change, updated the helper and fold-summary cells in `training (2).ipynb` and `A2/training.ipynb`. Checkpoint selection now maximizes validation Macro-F1. EarlyStopping also maximizes validation Macro-F1 with the existing patience 10 and minimum improvement 0.001. All six returned metrics describe the same selected epoch, and the optional Optuna objective now returns its Macro-F1. The return tuple order, architecture, data paths, split logic, hyperparameters and cell counts remain as before. Updated checkpoint/plot/fold-summary wording to name the selection metric.

Preserved the uploaded notebook's original AUROC-run outputs in `training (2) - AUROC results.ipynb`, and cleared the active notebooks' training-output cells so old results are not presented as results of the new rule. Verified both notebooks' Python syntax and that only the two intended cells changed. Exercised the actual training functions with controlled metrics where AUROC keeps rising but F1 peaks at epoch 2: both implementations restored epoch-2 weights, returned all six epoch-2 metrics and stopped at epoch 12. Checked the minimum-improvement/patience behavior and optional objective's F1 return. No full-data retraining or model-quality improvement claim was made. Group attribution and final experimental verification remain pending.

### Make StandardCNN2D compatible with five-stage waveform classification, 28 October 2026

The requester asked to substitute StandardCNN2D for CBraMod in the existing training pipeline. Inspected the original CNN and found its fixed `256*7*7` flattening and single output incompatible with `(B,2,30,200)` waveforms and five-class cross-entropy. Updated only the StandardCNN2D definition in `A2/model_lib.py`: defaults are two channels/five classes, an adaptive average pool feeds the existing 512-unit classifier, and the final layer emits configurable raw class logits. Other model definitions were verified unchanged by AST comparison. Updated two stale planning-document sentences that described this CNN adaptation as pending. The notebook continues to select CBraMod until the requester substitutes the documented CNN constructor.

Verified the CNN with five real development epochs covering all classes: `(5,2,30,200)` inputs produce `(5,5)` logits; cross-entropy, backward gradients and an Adam step are finite. Checked single-epoch inference and the existing spectrogram spatial shape. The configured CNN has 523,237 trainable parameters and starts from random weights; the CBraMod checkpoint is not used for this baseline. Full CNN training and comparative model quality were not evaluated. Group attribution and final experiment review remain pending.

### Adapt every classifier to the waveform input shape, 28 October 2026

The requester expanded the shape fix to all models in `A2/model_lib.py`. Standardized the remaining classifiers' defaults to two channels/five classes. Replaced the fixed-size head of CBAM_EEG_Net with adaptive pooling; retained the effective global-pooling ResCBAM_EEG_Net and removed its shadowed duplicate methods. Fixed the transformer channel embeddings' fixed flatten/17-position assumptions and corrected the constructors' positional-argument misuse. The hybrid now feeds eight CNN features per pooled temporal step to its LSTM, flattening the pooled waveform grid in row-major order instead of requiring a fixed image-height feature count. Added an explicit lstm_hidden_size argument and clamped channel-attention reduction to at least one hidden channel. Kept existing class names, the already adapted StandardCNN2D, the CBraMod classifier/backbone loading and both training notebooks unchanged. Updated related model/cloud documentation.

Added `A2/tests/test_model_shapes.py`. All six tests passed, exercising all seven classifiers with actual development waveforms covering the five stages: default `(5,2,30,200)` input to `(5,5)` logits, cross-entropy backward with finite gradients, optimizer step and single-epoch inference. Verified configurable channels/classes and rectangular grids for scratch models, both transformer attention paths, embedding token counts, the explicit LSTM hidden size, and attention/residual helpers at two channels. Refined the temporal-order check to observe the actual CNN/LSTM boundary with hooks, then reran that affected test successfully. Strict pretrained CBraMod loading passed; AST comparison confirmed its definition and StandardCNN2D are unchanged, and hashes confirmed neither notebook was edited. These execution checks do not measure training quality or CUDA AMP behavior. Older CNN/transformer checkpoint shapes may differ after head/embedding adaptations; full comparative training and group review remain pending.

### A2 M2 report baseline and evidence audit, 28 October 2026

At the requester's instruction, reviewed the supplied LaTeX template, the course handbook, the Sleep Cassette proposal, the exploration notebook, current waveform preprocessing/models, and five uploaded result notebooks. Created `A2/report_m2/main.tex`, bibliography, report notes, figures and extracted evidence tables. Kept the supplied cover/member details and documented Introduction, full-cohort and sampled-data EDA, direct EDF waveform preprocessing, StandardCNN2D, the custom channel-token Transformer, pretrained CBraMod/criss-cross attention, supplementary ResCBAM, preliminary results and remaining M3 work. The report does not use the obsolete post-hoc sampling workflow.

Read the handbook directly with PDFKit and visually checked its A2 rubric. Verified public dataset and upstream model citations. Recomputed hard-label metrics from 24 completed-fold confusion matrices and checked their supports against reconstructed grouped sampled-data splits. Retained the notebooks' AUROC checkpoint selection, loss-based stopping and population fold SD rather than relabelling historical runs as Macro-F1-selected. Distinguished four complete five-fold configurations from the incomplete CNN variant; reported no Transformer or held-out-test score. Recorded source hashes and current model parameter counts; acknowledged missing historical model/checkpoint provenance, GPU identity and matched inference measurements. Flagged that the current 4,316--4,399 training epochs per fold do not establish the handbook's default 5,000-training-sample threshold. Approval of the earlier proposal was not treated as an explicit exception for the later subset.

Generated and visually inspected five report figures. Reproduced the exploratory spectrogram preview from a checksum-verified raw PSG because the exploration notebook's saved preview cell had failed. Checked LaTeX environment pairs, figure paths, citation keys and cross-references; no TeX engine was available, so PDF compilation/layout verification remains for the authors. No model training, source/notebook mutation, publication or submission was performed; code/notebook hashes remained unchanged. Group review, actual contributions and missing external links remain the authors' responsibility.

### M2 clarification: accepted subset and open selection choices, 28 October 2026

The requester clarified that the 6,474-sample dataset and fold protocol have been accepted, and that the draft should discuss AUROC versus Macro-F1 as checkpoint-selection choices under investigation. Updated the M2 LaTeX report, report README and proposal status note to reflect that confirmation. Removed the report's unresolved sample-threshold requirement while preserving the actual 5,478-development/996-test counts and the distinction between preprocessing/sampling and StratifiedGroupKFold partitioning. This supersedes the approval uncertainty recorded in the preceding audit entry; the confirmation source is the requesting group member.

Added a concise comparison table identifying AUROC as the criterion for the existing results and Macro-F1 as an alternative without a completed comparison. Documented a validation-only comparison on a shared training trajectory, separating checkpoint choice from the stopping policy; removed wording that prematurely fixed the final choice to Macro-F1. Regenerated the Overleaf ZIP and checked LaTeX structure, references and archive contents. Existing result values, figures and training implementations were unchanged. No training, test-set evaluation, publication or PDF compilation was performed.

## Assignment 3

| Field | Value |

|---|---|

| Tool name / version | GPT-6 Sol |

| Used by | |

| Time / stage | |

| Purpose | |

| Affected section(s) | |

| Representative prompt or prompt-log link | |

| How output was edited and verified | |

| Member responsible for final verification | |

| Sources used for verification | |