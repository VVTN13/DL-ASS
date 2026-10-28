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

## Assignment 2

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