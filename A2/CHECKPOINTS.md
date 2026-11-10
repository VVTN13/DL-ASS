# Local checkpoints

The notebook writes the five best-validation state dictionaries directly into:

```text
runs/<model>/<dataset>_final/
  fold_1.pt
  fold_2.pt
  fold_3.pt
  fold_4.pt
  fold_5.pt
```

There is no timestamp and no `checkpoints/` subfolder. Rerunning training for the
same model/dataset replaces its old run. Pooled validation and test soft-voting
load these five local files. Evaluation alone does not delete or retrain them.

No `.pt` publication or GitHub Release is needed for this workflow. New weights
under `runs/` are ignored by Git; configs, logs and metrics can be committed.
The [README](README.md) explains how to reproduce the weights from the dataset
and JSON config. Download the complete run folder before terminating a cloud
instance if you want to evaluate later without retraining.

## Existing spec_model

`spec_model/` is kept intact: 15 checkpoints and 9 CSVs restored from HEAD.
Nothing in the canonical notebook writes to this directory. After the owner
confirmed a cloud backup, `git rm --cached -r -- A2/spec_model` removed these
files from the index without deleting the local copies. `/spec_model/` is now
ignored. The removal was committed as `f85a3a4`; local copies remain.
Previously pushed weights remain in older commits;
this does not rewrite or shrink Git history.

The run copies of the 15 CNN, Transformer and CBraMod weights have the same
SHA-256 hashes as their originals. AttnSleep's completed cloud run now supplies
five additional verified weights, full validation/test probability CSVs and
recorded environment/source metadata in `runs/AttnSleep/sleep_waveform_final/`.
Its original timestamped directory is preserved locally. All twenty fold
checkpoints remain local and ignored; their hashes are recorded with each run.

The old ZIP packages in `releases/` are local backups with the previous paths.
They are not required for training, evaluation or submission and have not been
published by this workflow.
