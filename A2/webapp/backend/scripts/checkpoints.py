"""Manage model releases under ``backend/checkpoints``. The ``.pt`` files are never committed.

    python scripts/checkpoints.py package                 # copy from A2/runs, write manifests
    python scripts/checkpoints.py fetch --base-url URL    # download files named in the manifests
    python scripts/checkpoints.py verify                  # check presence and SHA-256
    python scripts/checkpoints.py synthetic --out DIR     # random weights for CI and UI tests

``fetch`` expects ``URL/<release id>/fold_N.pt`` and rejects any file whose hash differs
from the committed manifest. ``synthetic`` releases exercise the full software path but
their predictions are meaningless; they are flagged in the API and the UI.
"""

import argparse
import csv
import json
import shutil
import sys
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ml.preprocessing import CONFIG as PREPROCESSING  # noqa: E402
from app.ml.registry import CLASS_NAMES, load_releases, sha256  # noqa: E402

A2 = BACKEND.parents[1]
DEFAULT_DIR = BACKEND / "checkpoints"
RELEASES = {
    "transformer-a2-v1": {
        "name": "Transformer", "network": "Transformer", "run": "Transformer", "order": 1,
        "input_channels": ["EEG Fpz-Cz", "EEG Pz-Oz"],
    },
    "attnsleep-a2-v1": {
        "name": "AttnSleep", "network": "AttnSleep", "run": "AttnSleep", "order": 2,
        "input_channels": ["EEG Fpz-Cz"],
    },
}


def published_metrics(run_dir: Path) -> dict:
    rows = list(csv.reader((run_dir / "test_metrics.csv").open()))
    values = dict(zip(rows[1][1:], map(float, rows[2][1:]), strict=True))
    return {
        "split": "A2 held-out test participants, 5-model soft voting",
        "accuracy": round(values["Accuracy"], 4),
        "balanced_accuracy": round(values["Balanced Accuracy"], 4),
        "macro_f1": round(values["F1"], 4),
        "kappa": round(values["Kappa"], 4),
        "per_class_f1": {name: round(values[name], 4) for name in CLASS_NAMES},
    }


def manifest(release_id: str, spec: dict, folds: list[dict], **extra) -> dict:
    config = json.loads((A2 / "configs" / f"{spec['run']}.json").read_text())
    return {
        "id": release_id, "name": spec["name"], "order": spec["order"],
        "network": spec["network"],
        "network_kwargs": config["model_kwargs"], "class_names": list(CLASS_NAMES),
        "input_channels": spec["input_channels"], "ensemble": "mean of 5 softmax outputs",
        "preprocessing": PREPROCESSING, "folds": folds, **extra,
    }


def write_manifest(directory: Path, content: dict) -> None:
    (directory / "manifest.json").write_text(json.dumps(content, indent=2) + "\n")


def package(args: argparse.Namespace) -> int:
    for release_id, spec in RELEASES.items():
        run_dir = args.runs / spec["run"] / "sleep_waveform_final"
        target = args.dir / release_id
        target.mkdir(parents=True, exist_ok=True)
        folds = []
        for fold in range(1, 6):
            source = run_dir / f"fold_{fold}.pt"
            shutil.copy2(source, target / source.name)
            folds.append({"file": source.name, "sha256": sha256(source)})
        write_manifest(target, manifest(
            release_id, spec, folds, published_metrics=published_metrics(run_dir),
            source={"run": f"A2/runs/{spec['run']}/sleep_waveform_final",
                    "config": f"A2/configs/{spec['run']}.json"},
        ))
        print(f"packaged {release_id} -> {target}")
    return 0


def fetch(args: argparse.Namespace) -> int:
    for release in load_releases(args.dir).values():
        for fold, path in zip(release.folds, release.fold_paths, strict=True):
            if path.is_file() and sha256(path) == fold["sha256"]:
                continue
            url = f"{args.base_url.rstrip('/')}/{release.id}/{fold['file']}"
            partial = path.with_suffix(".part")
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, partial)  # noqa: S310
            if sha256(partial) != fold["sha256"]:
                partial.unlink()
                print(f"SHA-256 mismatch: {url}", file=sys.stderr)
                return 1
            partial.replace(path)
    return verify(args)


def verify(args: argparse.Namespace) -> int:
    releases = load_releases(args.dir)
    for release in releases.values():
        release.verify()
        print(f"ok {release.id}: {len(release.folds)} folds")
    return 0 if releases else 1


def synthetic(args: argparse.Namespace) -> int:
    import torch

    from app.ml.networks import NETWORKS

    for index, (release_id, spec) in enumerate(RELEASES.items()):
        target = args.out / release_id
        target.mkdir(parents=True, exist_ok=True)
        content = manifest(release_id, spec, [], synthetic=True)
        for fold in range(1, 6):
            torch.manual_seed(1000 * index + fold)
            model = NETWORKS[spec["network"]](**content["network_kwargs"])
            path = target / f"fold_{fold}.pt"
            torch.save(model.state_dict(), path)
            content["folds"].append({"file": path.name, "sha256": sha256(path)})
        write_manifest(target, content)
        print(f"synthetic {release_id} -> {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("package", package), ("fetch", fetch), ("verify", verify)):
        command = commands.add_parser(name)
        command.add_argument("--dir", type=Path, default=DEFAULT_DIR)
        command.set_defaults(handler=handler)
    commands.choices["package"].add_argument("--runs", type=Path, default=A2 / "runs")
    commands.choices["fetch"].add_argument("--base-url", required=True)
    fake = commands.add_parser("synthetic")
    fake.add_argument("--out", type=Path, required=True)
    fake.set_defaults(handler=synthetic)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
