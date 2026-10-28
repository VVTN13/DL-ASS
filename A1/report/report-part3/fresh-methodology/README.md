# Fresh Methodology

This is a new five-page Section 3, written from the current assignment implementation.
`A1/report/A1_M1.pdf` was used only to understand the requested plain writing and
compact presentation. Its prose, model configurations and results were not reused.
The earlier Methodology draft was not used as the writing source.

## Files

- `methodology.pdf`: the complete section for review.
- `methodology.tex`: standalone LaTeX document.
- `sections/03_methodology.tex`: section body for insertion after your retained Section 2.
- `references.tex`: references cited in the new section.
- `methodology_source.zip`: editable source and build instructions.
- `qa/verification.json`: document and saved-configuration checks.
- `qa/page-01.png` through `qa/page-05.png`: rendered pages.

The existing files in `report-part3`, the retained PDF, and `AI_USAGE.md` were
left unchanged. This draft is standalone; it has not been merged into the
retained report. Its page and table numbering are local to this preview.
When inserting the section into your report, let the parent document continue
its counters and merge the bibliography entries into the main bibliography.
The explicit page breaks group the material into five readable pages; they can
be adjusted when integrating with a different page layout.

## Build

Run from this directory:

```powershell
tectonic --keep-logs --keep-intermediates methodology.tex
```

Alternatively, run XeLaTeX twice:

```powershell
xelatex -interaction=nonstopmode -halt-on-error methodology.tex
xelatex -interaction=nonstopmode -halt-on-error methodology.tex
```

The source uses standard LaTeX packages and contains no external image dependency.

## Review performed

The architectures were checked against `A1/a1/models.py`; the split, input order,
training procedure and timing definitions were checked against `data.py`,
`train.py` and `evaluate.py`. Parameter counts and shared settings were compared
with all five saved run summaries. The PDF was compiled and all five rendered
pages were visually inspected. No model training or checkpoint evaluation was
rerun for this writing task.

Primary research references support the dataset, CNN, LSTM, attention,
patch-based image modelling and Adam descriptions. Project-specific settings
come from this repository.
