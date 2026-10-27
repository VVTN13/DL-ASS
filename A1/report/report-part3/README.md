# Section 3: Methodology

`report_with_methodology.pdf` continues the supplied 13-page report with Section 3
and its references. The supplied `Report_Style_Template (11).pdf` is unchanged.
In the combined derivative, the existing body text is retained, the contents
gain Section 3 entries, and footer page totals are updated.

`report.pdf` contains only the new Methodology and references. Printed page
numbers continue from 13, so the same pages can be appended to the retained
report. `report.tex` is its complete editable entry point;
`sections/03_methodology.tex` is the reusable body fragment.

## Outline

1. 3.1 Overall System Architecture: partitions, information flow, and input interfaces.
2. 3.2 Baseline Approach: Linear and two-hidden-layer MLP.
3. 3.3 Proposed Models / Algorithms: implemented CNN, row LSTM, and patch Transformer.
4. 3.4 Mathematical Formulation: logits, cross-entropy, forward computations, and readouts.
5. 3.5 Training Strategy: common settings, validation selection, and reproducibility boundaries.
6. 3.6 Evaluation Metrics: accuracy, class scores, confusion matrices, and timing scopes.

## Source and integration

The wording, diagrams, and organization were developed for this implementation.
`../report/A1_M1.pdf` was consulted for examples of technical detail and report
presentation. Its different architectures, training budget, selection rule,
and results were not adopted. Primary research papers and framework
documentation are cited where the section explains established methods.

To integrate with the original LaTeX project, insert
`sections/03_methodology.tex` after Section 2, copy the two `figures/*.pdf`
assets, and merge the seven `meth_` bibliography entries from
`references_methodology.tex` into the existing bibliography. Keep only one
`thebibliography` environment. The original report should number sections,
figures, tables, and equations automatically; the counter settings in
`report.tex` are only for the standalone continuation. Check relative figure
paths if the source fragment is moved to another directory.

## Rebuild and verification

From `A1`, run:

```powershell
.\report-part3\build.ps1
# Or specify a compiler:
.\report-part3\build.ps1 -Compiler 'C:\path\to\tectonic.exe'
```

The existing `.venv` supplies PyTorch, Matplotlib, PyMuPDF, and Pillow. Figures
can be reused on Overleaf without running Python. Select XeLaTeX and upload
the `.tex` files and figure PDFs with their directory structure intact.

`generate_methodology.py` checks input ordering, forward shapes, parameter counts,
strict checkpoint loading, common settings, and validation-loss selection.
It does not retrain or overwrite model artifacts. `build_and_verify.py` checks
the compiled section and combined PDF, renders all new pages, checks retained
body text and protected file hashes, and creates `methodology_source.zip`.
The ZIP contains editable source, figures, author notes, and the evidence summary.
The supplied original report PDF is needed separately to rebuild the combined PDF.

See `review_notes.md` for the evidence inventory, technical boundaries, and
self-review. Machine checks and page previews are saved in `qa/`.
