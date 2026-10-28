# Methodology evidence and review

## Evidence inventory

- `../a1/models.py`: all five architectures, layer settings, forward order, and parameter counts.
- `../a1/data.py` and `../split_cache.json`: stratification, normalization, value ordering, and loader policy.
- `../a1/train.py`: cross-entropy, Adam, ten-epoch loop, checkpoint contents, strict validation-loss selection, and timing.
- `../a1/evaluate.py`: inference mode, test metrics, confidence, fingerprints, warm-up, and timing boundaries.
- `../configs/*.json`: common seed, budget, batch size, learning rate, augmentation, and worker count.
- `../runs/m1/*/` and `../runs/m2/*/`: strict checkpoint loads, counts, configurations, device provenance, and ten-row histories.
- `Report_Style_Template (11).pdf`: retained report, numbering, typography, and established preprocessing narrative.
- `../report/A1_M1.pdf`: presentation reference only; its experimental settings are different.
- Primary papers and versioned PyTorch documentation: checked bibliographic identities and mathematical conventions.

## Verification notes and boundaries

No unresolved layer dimensions or common training settings were needed for this
draft. Live model instantiation, a batch forward pass for each architecture,
strict loading of all five checkpoints, and independent history selection checks
support the architecture and protocol claims. These checks are not a new training
experiment or a new full test evaluation.

Historical Linear/MLP summaries record MPS; CNN/LSTM/Transformer summaries record
CUDA. Software versions also differ. The draft explicitly separates these from
the proposed common CUDA rerun. It does not rank historical training speed or
claim measured peak GPU memory. The current evaluator has no independently
repeated timing distribution, and the draft describes the actual aggregate timer.

The supplied report's earlier editorial details and figure references are kept
as supplied. They were not rewritten as part of Section 3. Source PDFs, model
code, checkpoints, and `AI_USAGE.md` remain unchanged.

## Self-review

- Pipeline: training updates, validation selection, and test reporting are separated; all representation paths return ten logits.
- Mathematics: dimensions, logits versus probabilities, image-weighted loss, convolution padding, LSTM biases/state reset, attention dropout, pre-normalization, and normalization-before-pooling agree with code.
- Paragraph flow: each paragraph describes one operation or rationale; subsection order follows system, models, equations, training, and evaluation.
- Terminology: standardized image, row sequence, patch token, logit, selected checkpoint, and macro-F1 are used consistently.
- Reproducibility: seed, split, statistics, optimizer, dropout locations, budget, device provenance, checkpoint contents, and timing boundaries are stated.
- Claims and evidence: this is a comparison of established configurations, with unequal capacity and one seed; no new performance gains, hardware equivalence, or novel architecture is claimed.
- Presentation: all eight new pages were visually inspected, including equations and both diagrams; the combined contents page was inspected. Tables and diagrams stay with their explanations. The final compiler log has no overfull boxes, missing glyphs, or unresolved citations. `qa/verification.json` records the automated checks.
