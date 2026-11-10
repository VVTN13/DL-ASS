# Assignment 2 experiment artifacts

`final/` is the lightweight evidence package intended for Git. It preserves the
four completed experiments from the model-specific notebooks without requiring
four executable copies of the same training loop. Each model folder contains:

- `config.json`: the reconstruction preset; the runnable presets are in `../configs/`.
- `cv_results.csv`, `cv_summary.csv`, `pooled_validation_metrics.csv`, and
  `heldout_test_metrics.csv`: the three evaluation summaries, as originally printed.
- `confusion_matrices.json`: original fold, pooled-validation and test matrices.
- `history.csv` and `training.log`: original epoch logs (printed precision).
- `environment.txt`: package versions observable in the installation output.
- `provenance.json`: original notebook hash, known hardware, split and checkpoint
  fingerprints, best epochs and explicit gaps in the historical evidence.
- `heldout_test_predictions.csv`: available for CNN, Transformer and CBraMod.
  AttnSleep's CSV remains pending; its aggregate notebook results are included.

These are **retrospective exports**, not newly trained models. Unknown historical
commit/timestamp values remain null. The current corrected loss logger cannot
retroactively recover exact weighted epoch losses from the old printed means.
CSV column values preserve their original precision; no scores were changed.

`final/cv_folds.csv` preserves participant assignments shared by the four full
experiments. The model-specific notebooks remain available locally as original
records. They are not required as separate runnable entry points on GitHub.

New executions of `training.ipynb` create sibling directories named
`<model>_<dataset>_<timestamp>/`, with full machine-recorded manifests,
checkpoints and probability CSVs. Those directories are ignored by Git by
default. Deliberately curate an experiment's small evidence files into a new
versioned location when choosing it for the report; do not overwrite `final/`
with a later rerun. The new-run evaluation-only instructions in `../README.md`
apply to the timestamped run format, not to this retrospective `final/` format.

Large `.pt` / `.pth` weights belong in checkpoint storage or a downloadable
release, with links and SHA-256 hashes recorded in the repository. Existing
weights remain in `../spec_model/`. No upload or publication was performed.
