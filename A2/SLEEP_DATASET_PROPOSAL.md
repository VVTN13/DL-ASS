# Dataset proposal: five-stage sleep classification

**Proposed cohort:** the full Sleep Cassette study, with 78 participants and 153 nights.
**Instructor decision:** ____________________
Permitted decisions: Approved; Approved with conditions; Rejected.

The proposal meets the five-class requirement and has substantially more than
5,000 candidate training epochs. The instructor must still approve EEG-derived
spectrograms for the assignment's image-classification category. No model has been
trained as part of this exploration.

## Dataset, source, version, and license

- **Name:** Sleep-EDF Database Expanded, Sleep Cassette cohort.
- **Source:** [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).
- **Version:** 1.0.0, DOI [10.13026/C2X676](https://doi.org/10.13026/C2X676).
  Use this archive version identifier; the dataset description also refers to its
  historical 2018 expansion as version 2.
- **License:** Open Data Commons Attribution License v1.0. Attribute the dataset
  and cite its source publications and PhysioNet as requested on the dataset page.
- **Scope:** all 153 Sleep Cassette nights. Exclude the separate 44-recording Sleep
  Telemetry study, which includes a temazepam/placebo experiment.

The 20-participant subset is useful for an initial pilot, but the full cohort
provides more participant diversity and a larger independent test group. The
main proposal therefore uses all 78 participants; only the signal preview uses
one recording.

## Task, input, output, and annotations

**Task:** classify each complete, non-overlapping 30-second EEG epoch into one of
five sleep stages. This is sleep staging, not diagnosis of a sleep disorder.

**Input:** two EEG channels, Fpz-Cz and Pz-Oz, sampled at 100 Hz. Each epoch starts
as an array of shape `(2, 3000)`. The proposed log-PSD representation uses a
two-second Hann window, 0.5-second hop, and frequencies from 0.5 through 30 Hz,
giving a float32 array of shape `(2, 60, 57)`.

**Output:** one integer label per epoch:

| Label | Stage | Original expert annotation |
|---|---|---|
| 0 | W | Sleep stage W |
| 1 | N1 | Sleep stage 1 |
| 2 | N2 | Sleep stage 2 |
| 3 | N3 | Sleep stage 3 or Sleep stage 4 |
| 4 | REM | Sleep stage R |

The hypnograms provide expert-scored stage intervals, not image masks or boxes.
One interval may span multiple epochs. The source uses Rechtschaffen–Kales
scoring; merging stages 3 and 4 provides the proposed five-class conversion,
without claiming an AASM rescoring.

## Preliminary distribution analysis

The executed [exploration notebook](sleep_exploration.ipynb) verified all 153
hypnograms against the archive's checksum manifest. It retained complete
30-second epochs from 30 minutes before the first scored sleep interval through
30 minutes after the last scored sleep interval. Movement and unscored intervals
were excluded.

| Stage | Candidate epochs | Percentage |
|---|---:|---:|
| W | 65,941 | 33.73% |
| N1 | 21,522 | 11.01% |
| N2 | 69,132 | 35.37% |
| N3 | 13,039 | 6.67% |
| REM | 25,835 | 13.22% |
| **Total** | **195,469** | **100.00%** |

These are annotation-derived candidate counts, before full PSG boundary and
signal-quality checks. All 153 PSG filenames were present during exploration;
one PSG was checksum-verified and used successfully for the spectrogram preview.
Presence alone does not verify every recording's integrity.

N2 is about 5.3 times as frequent as N3. Report per-stage performance and use
training-derived class weights if needed. Preserve the natural validation and
test distributions.

## Splitting and leakage prevention

Keep the original 12 test participants unchanged. The other 66 participants form
a development set for **five-fold StratifiedGroupKFold**, using sleep-stage labels
for stratification and participant identifiers for grouping. Use `shuffle=True`
and `random_state=42`. The original holdout comes from sorting all participant
identifiers, shuffling with Python `random.Random(42)`, and reserving the final 12.

| Split | Participants | W | N1 | N2 | N3 | REM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 66 | 59,966 | 19,441 | 58,754 | 10,529 | 21,847 | **170,537** |
| Test | 12 | 5,975 | 2,081 | 10,378 | 2,510 | 3,988 | 24,932 |

The executed notebook produced these development folds:

| Fold | Training participants | Validation participants | Training epochs | Validation epochs |
|---|---:|---:|---:|---:|
| 1 | 53 | 13 | 137,438 | 33,099 |
| 2 | 53 | 13 | 136,763 | 33,774 |
| 3 | 53 | 13 | 136,914 | 33,623 |
| 4 | 52 | 14 | 136,026 | 34,511 |
| 5 | 53 | 13 | 135,007 | 35,530 |

Each development participant appears in validation exactly once and training
four times. All five classes are present in every training and validation fold,
and every training fold exceeds 5,000 candidate epochs. These counts precede
full signal-quality checks.

**Split unit: participant.** Both nights and all epochs from one person stay
together within each fold. Ordinary StratifiedKFold on epochs would violate this
requirement. StratifiedGroupKFold approximates class balance subject to the group
constraint; exact stage proportions and equal participant counts are not guaranteed.
[Method documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Use the saved [cv_folds.csv](sleep_exploration/cv_folds.csv) when training. Its
`validation_fold` field is 1–5 for development participants and -1 for test
participants. A fold's training set must select **development** participants whose
validation fold differs from that fold; checking the fold number alone would
accidentally include the test set. Package version and settings are recorded in
[cv_metadata.json](sleep_exploration/cv_metadata.json); the saved assignments are
the reproducibility record if splitter behavior changes across library versions.

Reinitialize model and optimizer state for every fold. Fit normalization, class
weights, and other learned preprocessing on that fold's training participants
only. Choose the model and hyperparameters using mean validation macro-F1 over
five folds. Report its fold standard deviation as descriptive variability, not
as a confidence interval from five independent experiments.

After selection, refit on all 66 development participants. Choose the final
training duration from cross-validation, such as the median selected epoch count;
do not use the test set for early stopping. Evaluate the selected final model on
the 12 test participants after training. This is development cross-validation
plus an independent test holdout, not nested cross-validation.

If temporal context is added later, a context window must remain within its
recording and participant. Confidence intervals should resample participants,
rather than treating adjacent epochs as independent people.

## Evaluation and model plan

- **Primary metric:** macro-F1 over all five stages.
- **Secondary metrics:** balanced accuracy, Cohen's kappa, per-stage recall,
  ordinary accuracy, and a five-by-five confusion matrix.
- **Reporting:** mean and standard deviation of validation scores across the five
  development folds, then final held-out test metrics and the distribution of
  scores across test participants. Include a training-derived majority-class baseline.
- **Baseline:** relative EEG band-power features with class-weighted logistic
  regression, followed by a small CNN trained from scratch on spectrograms.
- **Pretrained comparison:** ImageNet-pretrained ResNet-18. Adapt the first
  convolution to the two input channels, replace the classification head with
  five outputs, and compare frozen-feature training with fine-tuning. Natural-image
  pretraining is a hypothesis to test, not an assumed improvement for EEG.

## Compute and storage estimate

The local Sleep Cassette PSG files occupy approximately **7.60 GB**. At the
candidate count above, float32 spectrogram patches alone would occupy about
**5.35 GB**, excluding metadata and intermediate arrays. Allow approximately
20–25 GB of working disk space. Process and save one recording at a time to avoid
holding several full copies of the dataset in RAM.

Start with one 8–16 GB GPU, batches of 32–64, and 20–30 epochs with early stopping.
A provisional planning allowance is **2–6 GPU-hours per neural-model fit**.
Two model configurations across five folds require ten fits, followed by one
final refit of the selected model: provisionally **22–66 GPU-hours**. Additional
hyperparameter configurations or random seeds increase this budget. Start with
one fixed seed and the same folds for both models.
This is an unmeasured estimate, not a benchmark. Measure one full training epoch
on the actual hardware and update the estimate before committing to all runs.

## Subset and preprocessing rules

1. Include the complete Sleep Cassette cohort; do not select participants based
   on model performance. Exclude Sleep Telemetry to retain one study protocol.
2. Pair PSG and hypnogram files by the first six filename characters. Group
   nights by the first five characters, which identify the participant.
3. Apply the fixed five-stage mapping, wake-trimming rule, and complete 30-second
   annotation-aligned epochs described above. Do not manufacture extra samples
   through overlapping windows to satisfy the sample threshold.
4. After approval, verify signal integrity, channel identity, sampling rates,
   annotation alignment, and epoch boundaries. Log every excluded recording or
   epoch with its reason and recalculate the counts.
5. A proposed signal-processing starting point is a 0.3–35 Hz bandpass on each
   continuous recording, followed by log-PSD extraction. Document the final
   filter and quality rules before model comparison. The current preview uses
   unfiltered signals and does not implement this full pipeline.
6. Keep participant, recording, and onset metadata alongside saved `.pt` tensors.
   Use recording-level shards plus the frozen participant/fold manifests for
   manageable RAM use. All models must use the same folds for comparison.

Wake trimming uses expert labels to define a retrospective benchmark. Results on
this benchmark do not establish performance on untrimmed full-day recordings.

## Reproduce the exploration

From the repository root, install `A2/requirements-sleep.txt`, open
`A2/sleep_exploration.ipynb`, select the installed Python environment, and run all
cells. Results are written under `A2/sleep_exploration/`:

- `class_distribution.csv` and `class_distribution.png`
- `participant_split.csv`, `split_counts.csv`, and `night_counts.csv`
- `cv_folds.csv`, `cv_fold_summary.csv`, `cv_class_counts.csv`, and `cv_metadata.json`
- `epoch_preview.png`

The existing dementia-specific `preprocess.py` is not a sleep-staging pipeline.
Implement the full sleep preprocessing after the instructor approves this proposal.

References: [official dataset](https://physionet.org/content/sleep-edfx/1.0.0/)
and [MNE sleep-staging tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html).
