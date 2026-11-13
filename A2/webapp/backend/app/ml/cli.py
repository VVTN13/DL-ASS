"""Run the serving pipeline without the web app.

    python -m app.ml.cli infer recording.edf --model transformer-a2-v1 --out predictions.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

DEFAULT_CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"


def infer(args: argparse.Namespace) -> int:
    from .inference import load_ensemble, predict_proba
    from .preprocessing import EPOCH_SECONDS, preprocess_recording
    from .registry import CLASS_NAMES, load_releases

    releases = load_releases(args.checkpoints)
    if args.model not in releases:
        print(f"Unknown model {args.model!r}. Available: {sorted(releases)}", file=sys.stderr)
        return 2
    release = releases[args.model]
    started = time.perf_counter()
    prepared = preprocess_recording(args.edf)
    probabilities = predict_proba(load_ensemble(release), prepared.epochs)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch_index", "onset_seconds", "stage",
                         *(f"prob_{name}" for name in CLASS_NAMES)])
        for index, row in enumerate(probabilities):
            writer.writerow([index, index * EPOCH_SECONDS, CLASS_NAMES[int(row.argmax())],
                             *(f"{value:.6f}" for value in row)])
    print(f"{release.id}: {prepared.n_epochs} epochs, {prepared.dropped_seconds:g} s dropped, "
          f"{time.perf_counter() - started:.1f} s -> {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("infer", help="Predict a sleep stage for every 30 s epoch.")
    run.add_argument("edf", type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--out", type=Path, default=Path("predictions.csv"))
    run.add_argument("--checkpoints", type=Path,
                     default=Path(os.environ.get("CHECKPOINTS_DIR", DEFAULT_CHECKPOINTS)))
    run.set_defaults(handler=infer)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
