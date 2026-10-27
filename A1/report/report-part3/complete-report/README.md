# Complete Assignment 1 report

`report.tex` is the main editable LaTeX source. It brings the retained introduction
and data-analysis ideas, the approved fresh Methodology, and new results, error
analysis, discussion, conclusion and appendices into one consistent report.
The complete PDF is also delivered at `output/pdf/assignment1_complete_report.pdf`
from the repository root.

All earlier reports, sources and original experiment artifacts are retained.
`AI_USAGE.md`, original implementation, configurations and data splits are
unchanged. Separate CUDA baseline runs and their configuration were added.
The earlier A1_M1 report was a guide to simple presentation only. Its wording,
architectures and numerical results are not used as this report's experiment.
Established methods are attributed to their primary papers in the bibliography.

## Build the PDF

From this folder:

```powershell
tectonic --keep-logs --keep-intermediates report.tex
```

Alternatively compile `report.tex` with XeLaTeX twice. The ZIP is self-contained
for LaTeX compilation: source files, generated tables and figure PDFs are
included. No Python or model checkpoints are needed merely to compile the PDF.
For Overleaf, upload the source ZIP, choose `report.tex` as the main file and
select XeLaTeX.

## Regenerate evidence and figures

From this folder, with the full repository and Python packages `numpy`,
`matplotlib`, `pymupdf`, `pillow` and `pypandoc_binary` available:

```powershell
python prepare_evidence.py
tectonic --keep-logs --keep-intermediates report.tex
python verify_and_export.py
```

`prepare_evidence.py` checks all five prediction CSVs against the local test
labels, independently recomputes class metrics and confusion matrices, and
verifies saved checkpoint hashes and selection against histories. It creates
tables, charts and `qa/evidence.json`. It seeds the Methodology from the preserved `fresh-methodology` directory only
when the complete report has no Methodology file, preserving subsequent edits.

`verify_and_export.py` checks the document, renders every page, verifies original
file preservation, exports browser HTML and Markdown, and packages editable source.
LaTeX/PDF is the canonical layout; HTML and Markdown use flowing layouts.
The HTML includes embedded figures and MathML equations and can be opened offline.

## Evidence boundary

Classification results and error analysis use the original saved experiments.
For the computational-cost table, Linear and MLP were separately retrained and
evaluated on the GTX 1660 SUPER with CUDA, using the same seed, split and
training settings. CNN/LSTM/Transformer retain their saved CUDA measurements.
All five cost rows now use CUDA on the same GPU, recorded in separate sessions.
The baseline cost repetitions do not replace the original classification scores.
No model-memory measurements are claimed.
Accuracy, macro-F1, per-class metrics and confusion matrices were recomputed.
Losses and timings come from the corresponding saved summaries. Earlier checkpoint reload checks
are identified as earlier evidence in the report. The MLP/LSTM near tie is not
treated as a stable ranking.

`qa/verification.json` records machine checks. Page PNGs and the visual review
record document layout inspection. `preservation.json` records the hashes of
238 files that existed before this complete-report task, including the protected
disclosure file.

## Reproduce the CUDA baseline costs

From `A1/`, run:

```powershell
.\.venv\Scripts\python.exe -m a1.cuda_baselines
```

`configs/m1_cuda.json` explicitly requires CUDA and saves under
`runs/cuda_baselines/{linear,mlp}`. The script trains or validates a completed
matching run, evaluates the selected checkpoints on CUDA, then independently
checks all 20,000 test predictions, metrics, confusion matrices, validation
selection, split/source hashes, CPU/CUDA agreement and plot integrity. The
report evidence step requires these successful checks and records the source
files for every CUDA cost row in `qa/evidence.json`.
