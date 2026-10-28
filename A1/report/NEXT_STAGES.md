# A1 report: continuation guide

## Current deliverable

Sections 2.4 and 2.5 have been expanded in the editable fragments
`sections/024_preprocessing.tex` and `sections/025_key_findings.tex`.
`02_dataset_eda.tex` includes both. The expanded preprocessing covers tensor
conversion, training-only standardization, leakage prevention, all five input
layouts, data retention/augmentation, and verification. Section 2.5 connects
the measured EDA findings to methodology and implementation.

The user's supplied `sections/Report_Style_Template (7).pdf` is preserved.
`template_continuation.tex` uses the same fragments to produce the new pages;
`build_template_continuation.py` combines them with that PDF, corrects three
existing EDA figure references, and updates pagination and contents. The
continued version is `../../output/pdf/assignment1_template_continued.pdf`.
That output folder also contains a paste-ready LaTeX fragment and a ZIP
containing the fragment plus its representation figure. No model training
or preprocessing implementation was changed by this revision.

To rebuild the supplied-template version after compiling
`template_continuation.tex`, run from `A1/report`:

```powershell
..\.venv\Scripts\python.exe build_template_continuation.py
```

Visual review and document checks are saved in `qa/template_continued/`.
`qa/preprocessing_live_check.json` records a separate 32-training-image
loader/layout check on CPU and the GTX 1660 SUPER, including exact patch
reconstruction. The all-training-pixel EDA checks remain in the original
saved EDA evidence and are also checked by `verify_report.py`.

`report.tex` is the LaTeX entry point. It includes the cover, a stage-specific
abstract, the table of contents, **Section 1: Introduction**, **Section 2: Dataset
and Exploratory Analysis**, and references. The abstract describes this stage's
evidence; rewrite it after the final results are verified.

The report uses the requested section names and subsection numbering. Sections
3–7 are the next stages described below; they are not represented as completed
experiments or filled with invented results. `AI_USAGE.md` must remain unchanged.

### Rebuild the completed stage

Run from `D:\DL\DL-ASS - Copy\A1` in PowerShell:

```powershell
.\.venv\Scripts\python.exe report\generate_eda.py --device cuda
.\report\build_latex.ps1
.\.venv\Scripts\python.exe report\verify_report.py
```

The generator uses this checkout's existing environment and local dataset.
CUDA is required by that command; it fails instead of silently falling back to
CPU. `--device cpu` is available only as an explicitly labelled alternative.
The numerical reductions run on CUDA, CPU preprocessing follows the existing
`ToTensor` protocol, NumPy checks the moments/histogram, and CPU hashing checks
file integrity and exact duplicates. Matplotlib draws the exported figures.

`build_latex.ps1` needs Tectonic or XeLaTeX. This session used the official
Tectonic 0.17.0 Windows executable downloaded to the task's temporary tools
directory. The build script recognizes that location on this PC. On another
machine, put a compiler on PATH or supply its full path:

```powershell
.\report\build_latex.ps1 -Compiler 'C:\path\to\tectonic.exe'
# Alternative with a TeX installation:
.\report\build_latex.ps1 -Compiler 'C:\path\to\xelatex.exe'
```

Tectonic may download its TeX packages on its first run. The document uses Times
New Roman, Arial, and Consolas, available on the current Windows PC. On Overleaf,
upload the whole `report` folder, select XeLaTeX, and replace those fonts with
available equivalents (for example TeX Gyre Termes, TeX Gyre Heros, and Latin
Modern Mono). Keep the generated values and figures beside the source.

PDF verification uses the optional `PyMuPDF==1.26.5` package installed in this
checkout's `.venv`. It is needed for rendering/QA, not model training:

```powershell
.\.venv\Scripts\python.exe -m pip install PyMuPDF==1.26.5
```

### Files and evidence

| Path under `report/` | Purpose |
| --- | --- |
| `report.tex` | Main document, cover, abstract, contents, section includes |
| `sections/01_introduction.tex` | All four Introduction subsections |
| `sections/02_dataset_eda.tex` | All five Dataset and Exploratory Analysis subsections |
| `references.tex` | Primary dataset and library references |
| `generate_eda.py` | Executable source for every EDA visualization and table |
| `figures/01_*.pdf` through `09_*.pdf` | Nine vector figures included by LaTeX; `09_duplicate_audit` replaces the duplicate table |
| `figures/*.png` | Matching 220-dpi figures; institutional logo copied locally |
| `generated/*.tex` | Measured macros and table rows, rebuilt by the generator |
| `data/eda_metrics.json` | CUDA environment, data fingerprints, duplicate audit, checks, figure hashes |
| `data/split_counts.csv` | Ten-class counts across the three partitions |
| `data/class_statistics.csv` | Brightness and coverage summaries |
| `data/pixel_histogram.csv` | All 256 exact intensity counts |
| `data/spatial_statistics.npz` | Class means, population SD maps, cosine similarities |
| `report.pdf` | Compiled LaTeX document |
| `qa/verification.json` | PDF and evidence checks from `verify_report.py` |
| `qa/page-*.png`, `qa/contact-*.png` | Rendered pages for visual review |
| `configs/shared_cuda.json` | Explicit CUDA configuration for the future matched five-model run |

Numerical highlights from this execution: 54,000 / 6,000 / 10,000 images;
mean 0.28620901297676954 and standard deviation 0.35315974927232197;
50.2142998866% zero-valued training pixels; zero blank training images; zero
exact-image duplicates within or across the three partitions. Coat/Shirt has
the largest off-diagonal class-prototype cosine similarity, approximately
0.9873. These similarities are descriptive, not model confusion probabilities.

## Common CUDA reference for all later experiments

The stage-1 evidence records the actual PC: NVIDIA GTX 1660 SUPER, 6 GiB VRAM;
Windows; Python 3.12.10; PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128; CUDA
runtime 12.8; cuDNN 91002; NVIDIA driver 591.86. Distinguish the installed
driver's supported CUDA version from `torch.version.cuda`.

Preserve seed 42, the cached split, training-only normalization, batch 128,
Adam learning rate 0.001, ten epochs, and disabled augmentation for the primary
comparison. Use float32, the same worker count, and the lowest-validation-loss
checkpoint rule. Record changes explicitly if a later experiment alters them.
Do not select settings using test scores. The test set was already evaluated
in earlier work; a later rerun cannot be described as a previously unseen test.

**Existing training provenance matters:** `runs/m1/linear/summary.json` and
`runs/m1/mlp/summary.json` identify MPS as their training device. CNN, LSTM, and
Transformer training summaries identify CUDA. Keep those artifacts intact.
Their historical training seconds must not be plotted as a matched five-model
CUDA speed comparison. Existing evaluation summaries have CUDA inference
measurements, which should also be rerun consistently for the final comparison.

## Stage 2 — Section 3: Methodology

Create `sections/03_methodology.tex` and include it after Section 2 in `report.tex`.
Use exactly these subsections:

| Required subsection | What to write and verify |
| --- | --- |
| **3.1 Overall System Architecture** | Data → fixed split → training-only normalization → model-specific representation → logits → loss/checkpoint selection → evaluation. Export a pipeline diagram from code under `report/`. |
| **3.2 Baseline Approach** | Linear softmax classifier and MLP, with layer widths, activation, dropout, parameter counts, and why they are useful reference points. |
| **3.3 Proposed Models / Algorithms** | CNN, row-sequence LSTM, and patch Transformer as implemented in `a1/models.py`. Document tensor dimensions after each layer; explain row ordering and patch ordering explicitly. State that these are assignment configurations of established architectures, not claims of architectural novelty. |
| **3.4 Mathematical Formulation** | Logits/softmax, cross-entropy, CNN operation, LSTM gates/state readout, patch embedding/positions, attention and pooling. Match equations to the actual code, not a larger standard model. Cite primary architecture papers. |
| **3.5 Training Strategy** | Seed, stratification, Adam settings, ten-epoch budget, dropout, augmentation status, and validation-loss checkpoint selection. State any architecture-specific differences. |
| **3.6 Evaluation Metrics** | Accuracy, per-class precision/recall/F1, macro-F1, confusion matrices, parameter count, latency, training time, and memory. Give denominators and timing boundaries. |

Inspect `a1/models.py`, `a1/data.py`, `a1/train.py`, `a1/evaluate.py`, all four
current configs, and the architecture notes in `docs/`. Recount model parameters
from instantiated models. Generate architecture diagrams from code, placing the
script and all figures in `report/`. Do not use a diagram to imply a layer that
the implementation does not contain.

Completion gate: equations, shape tables, and parameter counts agree with the
code; the LaTeX compiles with all citations resolved; diagrams are readable.

## Stage 3 — Section 4: Implementation and Experimental Setup

Create `sections/04_implementation.tex` with:

1. **4.1 Tools and Environment:** capture Python/package versions, OS, CPU, GPU,
   usable VRAM, driver, CUDA runtime and cuDNN from the actual runs.
2. **4.2 Implementation Details:** explain modules, loaders, random-state control,
   checkpoint contents, optimizer, inference mode, and device selection.
3. **4.3 Experimental Configurations:** tabulate the shared settings and the
   architecture-specific ones. Keep a single primary comparison protocol.
4. **4.4 Reproducibility:** list commands, configuration/data/split/checkpoint
   fingerprints, artifact paths, and the verification procedure.

The supplied `configs/shared_cuda.json` writes fresh runs to
`report/evidence/cuda_baseline/<model>/` and explicitly requests CUDA.
It does not change the project's historical run directories. These commands
are **future-stage work**, not experiments already executed for Sections 1–2:

```powershell
# Run from A1. Keep the existing .venv and local dataset.
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$modelNames = @('linear', 'mlp', 'cnn', 'lstm', 'transformer')
foreach ($modelName in $modelNames) {
    .\.venv\Scripts\python.exe -m a1.train --model $modelName --config report\configs\shared_cuda.json
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $modelName" }
    .\.venv\Scripts\python.exe -m a1.evaluate --checkpoint "report/evidence/cuda_baseline/$modelName/best.pt" --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $modelName" }
}
```

The training command overwrites its selected output directory if rerun without
reuse. Before a repetition, use a distinct configuration/output directory.
Do not change or remove the historical artifacts. Evaluate the validation-
selected checkpoint, rather than selecting the best test score across epochs.

For a fair efficiency study, add measurement code under `report/` that records:

- Total training wall time with CUDA synchronization and an explicit scope.
- Inference warm-up, repeat count, batch size, precision, mean/median/spread,
  and whether data loading and host transfers are included.
- Peak PyTorch allocated/reserved GPU memory after resetting the counters,
  clearly separated from driver/desktop memory.
- Parameter counts and checkpoint size. Do not infer speed from parameter count.

The current `tests/verify_run.py` accepts `--model` and constructs historical
`runs/m1` or `runs/m2` paths internally; it has no run-directory argument.
Before verifying the new report runs, extend the checker to accept a run path,
or add a report-specific verifier with the same substantive checks. Merely
rerunning the existing wrappers would verify historical directories. Save the
new verifier's exact command and result alongside the new evidence.

Completion gate: all five primary models train/evaluate on the GTX 1660 SUPER;
the new checkpoints, histories, predictions, metrics, and confusion matrices
are verified and have matching data/split hashes. If a run fails, report that
failure and fix it before completing the results table.

## Stage 4 — Section 5: Results and Analysis

Create `sections/05_results.tex`. Generate every new plot from saved data using
a script inside `report/`; do not type plot values manually.

| Required subsection | Evidence and visualization |
| --- | --- |
| **5.1 Quantitative Results** | A table of validation-selected epoch, test loss, accuracy, macro-F1, and per-class support. Plot test accuracy and macro-F1 with clear scales. |
| **5.2 Training and Validation Analysis** | Loss/accuracy curves for every model from `history.csv`; mark the selected epoch. Explain differences between validation-loss selection and maximum validation accuracy. |
| **5.3 Model Comparison** | Compare specific configurations under the shared protocol. Quantify gaps in percentage points and error counts; distinguish capacity from architecture effects. |
| **5.4 Error Analysis** | Confusion matrices (rows true, columns predicted), per-class F1/recall, and seed-fixed correct/incorrect image grids with source index and labels. Check the Coat/Shirt prototype hypothesis against actual errors. Confidence is not calibrated certainty. |
| **5.5 Efficiency and Computational Cost** | Matched CUDA training times, inference latency distributions, peak memory, parameter counts, and checkpoint size. Separate forward-only and end-to-end timing. |
| **5.6 Additional Experiments / Ablations** | Only experiments actually executed. Possible choices are augmentation on/off, longer training, LSTM row versus column order, Transformer patch size, or several seeds. Change one factor at a time and preserve the primary baseline. |

If no ablation has been performed, state that explicitly under 5.6. Do not
present suggestions as results. Single-seed accuracy is not a mean with a
standard deviation; obtain repeated runs before reporting training variability.
Avoid selecting ablations because they improved test performance.

Completion gate: tables agree with predictions and summaries; each confusion
matrix sums to 10,000; per-class metrics have support 1,000; selected epochs
minimize validation loss; all five primary timing rows use the same PC and scope.

## Stage 5 — Section 6: Discussion

Create `sections/06_discussion.tex` with:

- **6.1 Interpretation of Findings:** connect measured model errors and learning
  behavior to the EDA and the implemented input representations. Treat possible
  mechanisms as interpretations unless an ablation isolates them.
- **6.2 Trade-offs:** explain observed quality, model size, memory, and speed
  together. Identify the appropriate configuration for a stated resource budget
  using actual measurements.
- **6.3 Limitations:** include fixed seed/split, short budget, unequal parameter
  capacities, possible near duplicates, previously evaluated test data, no
  evidence for real catalogue generalization, and timing noise on a desktop GPU.

Completion gate: every conclusion cites a table/figure or is explicitly framed
as a limitation or hypothesis. Do not claim that one architecture family is
universally superior from this assignment.

## Stage 6 — Section 7, front matter, and References

Create `sections/07_conclusion.tex` with **7.1 Summary of Achievements** and
**7.2 Future Improvements**. State what was implemented and measured, then propose
specific improvements justified by the failures or limits actually observed.
Keep proposed experiments separate from completed achievements.

Rewrite the abstract after Section 5 is final. Retain the cover and update the
date only when appropriate. The final contents must list Sections 1–7 and all
subsections exactly as requested. Expand `references.tex` with checked primary
papers for any newly described algorithms and with software sources where useful.
Keep the report's narrative self-contained and the file paths/commands in the
reproducibility subsection or an appendix. Respect the instruction not to edit
`AI_USAGE.md` throughout.

Final gate: rebuild figures, compile the complete LaTeX, resolve all citations
and cross-references, inspect every rendered PDF page, verify numerical claims,
and record which experiments were actually executed. Remove the stage-1 label
only after the remaining sections are complete.

## Full target outline

```text
FRONT MATTER
  Cover Page
  Abstract / Executive Summary
  Table of Contents
1. Introduction
  1.1 Background and Motivation
  1.2 Problem Statement
  1.3 Objectives and Scope
  1.4 Proposed Approach
2. Dataset and Exploratory Analysis
  2.1 Dataset Description
  2.2 Data Preparation and Splitting
  2.3 Exploratory Data Analysis
  2.4 Preprocessing
  2.5 Key Findings
3. Methodology
  3.1 Overall System Architecture
  3.2 Baseline Approach
  3.3 Proposed Models / Algorithms
  3.4 Mathematical Formulation
  3.5 Training Strategy
  3.6 Evaluation Metrics
4. Implementation and Experimental Setup
  4.1 Tools and Environment
  4.2 Implementation Details
  4.3 Experimental Configurations
  4.4 Reproducibility
5. Results and Analysis
  5.1 Quantitative Results
  5.2 Training and Validation Analysis
  5.3 Model Comparison
  5.4 Error Analysis
  5.5 Efficiency and Computational Cost
  5.6 Additional Experiments / Ablations
6. Discussion
  6.1 Interpretation of Findings
  6.2 Trade-offs
  6.3 Limitations
7. Conclusion and Future Work
  7.1 Summary of Achievements
  7.2 Future Improvements
References
```
