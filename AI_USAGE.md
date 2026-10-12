# AI Usage Disclosure

If no AI tool was used for a given assignment, state plainly under that section:

> The group declares that no generative AI tool was used in this assignment.

Otherwise, log every use with the fields below (handbook Sec. 5.2).

## Assignment 1

| Field | Value |
|---|---|
| Tool name / version | Claude (Anthropic), via Claude Code |
| Used by | Nguyễn Vũ Long |
| Time / stage | M1 Draft, September 2026 |
| Purpose | 1. Build the Assignment 1 GitHub Pages page: fill in the sections and add CSS for figures and the pipeline diagram.<br>2. Review the repo structure and fix wrong links: `DL-Assignment` to `DL-ASS`, the `AI_USAGE.md` link, and paths in `README.md` and `A1/README.md`.<br>3. Export the notebook figures to `A1/figures/` for use on the page.<br>4. Write `A1/README.md` (setup and reproduction instructions) and `A1/docs/M1_ARCHITECTURE.md`.<br>5. Fix code and environment issues, such as MPS/CUDA/CPU device selection and library version errors. |
| Affected section(s) | `assignment1.html`, `style.css`, `index.html`, `assignment2.html`, `assignment3.html`, `README.md`, `A1/README.md`, `A1/docs/M1_ARCHITECTURE.md`, `A1/figures/`, `A1/a1/` (device/environment fixes) |
| Representative prompt or prompt-log link | "kiểm tra lại structure project, cập nhập page github assignment 1" (check the project structure and update the Assignment 1 GitHub page). Summary: asked Claude to review the repo layout, fix broken links, and fill in the Assignment 1 page from the M1 notebook and run results. |
| How output was edited and verified | Tasks 1–3: every number on the page was checked against `A1/runs/m1/*/summary.json`, `history.csv` and the notebook outputs; repository links were checked against the git remote (`VVTN13/DL-ASS`); figure paths were confirmed to exist in `A1/figures/`; changes were reviewed in PR #2 before merging. Tasks 4–5: the training commands in `A1/README.md` were re-run for both models and the results matched `A1/runs/m1/*/summary.json`. |
| Member responsible for final verification | Nguyễn Vũ Long, Lê Phước Vũ |
| Sources used for verification | `A1/runs/m1/linear/summary.json`, `A1/runs/m1/mlp/summary.json`, `history.csv` files, `A1/dl1-261.ipynb` outputs, `A1/configs/m1.json`, [Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist), [torchvision FashionMNIST docs](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) |

### Assignment 1 CNN work, 11 October 2026

| Field | Value |
|---|---|
| Tool | OpenAI Codex |
| Requested scope | Set up a local virtual environment, implement the custom CNN, train it, and test it locally. |
| Affected files | `A1/a1/models.py`, `data.py`, `train.py`, `evaluate.py`; CNN config, Windows requirements, tests, documentation, run artifacts, and Assignment 1 page. |
| Representative request | "A custom CNN. do this for me first, setup a local .venv and do the CNN complete and then tested it yourself, make sure to do it in a human like way." |
| Verification | Automated checks, full Fashion-MNIST training/test evaluation, checkpoint reload, artifact checks, and figure inspection are recorded in `A1/docs/CNN.md`. |
| Human review | Group members must review the implementation, explanations and results before submission. |

### Assignment 1 LSTM work, 11 October 2026

| Field | Value |
|---|---|
| Tool | OpenAI Codex |
| Requested scope | Implement LSTM next, with a clearly explained image-to-sequence representation. |
| Affected files | `A1/a1/models.py`, `data.py`, `train.py`, `plot_sequence.py`; LSTM config, tests, documentation, diagram, notebook appendix, run artifacts, and Assignment 1 page. |
| Representative request | "LSTM with a clearly explained image-to-sequence representation. Do this next" |
| Verification | Pixel-coordinate and normalization tests, recurrent-gradient and independent-state checks, full 10-epoch CUDA training, validation-selected test evaluation, checkpoint reload and artifact checks. Evidence and limitations are recorded in `A1/docs/LSTM.md`. Layer/state semantics were checked against the PyTorch 2.9 LSTM documentation. |
| Human review | Group members must review the implementation, explanation and measured results before submission. |

## Assignment 2

| Field | Value |
|---|---|
| Tool name / version | |
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
| Tool name / version | |
| Used by | |
| Time / stage | |
| Purpose | |
| Affected section(s) | |
| Representative prompt or prompt-log link | |
| How output was edited and verified | |
| Member responsible for final verification | |
| Sources used for verification | |
