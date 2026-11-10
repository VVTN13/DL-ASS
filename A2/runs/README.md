# One fixed directory per model and dataset

Select `configs/<model>.json` in `training.ipynb`. Every training execution uses
**`runs/<model>/<dataset>_final/`**, with no date or timestamp to enter.
Rerunning the training cell **replaces that directory's previous results and
weights from scratch**. Other models, other datasets and `../spec_model/` are
untouched. Copy a run yourself before rerunning if you want to retain it.
Do not train the same model/dataset concurrently in the same working directory.

```text
runs/
├── Transformer/
│   └── sleep_waveform_final/
│       ├── config.json
│       ├── run.json
│       ├── fold_1.pt                 # Local only; ignored by Git
│       ├── fold_1.history.csv
│       ├── fold_1.best.json
│       ├── fold_1.setup.json
│       ├── ... fold_2–fold_5
│       ├── history.csv
│       ├── cv_results.csv
│       ├── cv_summary.csv
│       ├── pooled_validation_metrics.csv
│       ├── pooled_validation_confusion_matrix.csv
│       ├── test_metrics.csv
│       ├── test_predictions.csv
│       └── test_confusion_matrix.csv
├── StandardCNN2D/
├── CBraModClassifier/
└── AttnSleep/
```

There is no `checkpoints/` subdirectory. Full uses `sleep_waveform_final`;
small uses `sleep_waveform_small_final`. New runs also record sample identities,
preprocessing, source snapshots and installed versions. All saving and loading
uses `RUN_DIR`.

The five local `.pt` files are needed for pooled validation and five-model
soft-voting. They contain best-validation state dictionaries, not complete
optimizer/scheduler states for resuming training. They do not need to be
published. Git ignores `.pt` and `.pth` under `runs/`; keep the small configs,
metrics and logs. Download the run folder before ending a cloud instance.

To evaluate without training, follow the reload cell in [../README.md](../README.md)
and run the last two notebook cells. They load `RUN_DIR / "fold_1.pt"` through
`fold_5.pt` and overwrite only the evaluation outputs. They do not reset the run.

## Existing results

The supplied CNN, Transformer and CBraMod checkpoints now sit directly in their
`sleep_waveform_final/` directories. All 15 hashes are unchanged. AttnSleep's
historical metrics/logs are present; its checkpoints and prediction CSV have
not yet been supplied locally. Training it will replace its historical run
with the newly measured results.

Historical `run.json` records explicitly identify imported evidence. Unknown
training commit/time remain null; recorded dataset/code hashes were obtained
at import, not from the original cloud machine. Original model notebooks and
printed logs retain their historical paths as provenance.

`../spec_model/` is preserved intact: 15 original checkpoints and 9 CSVs. It is
now ignored and removed from the Git index after the owner confirmed a cloud
backup. Commit and push that removal to update the remote branch; local files
remain. The current notebook never writes there. Old local bundles in
`../releases/` retain their historical
folder layouts and are not the current run layout.
