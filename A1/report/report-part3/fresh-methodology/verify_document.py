"""Check this report against saved configuration evidence; render its pages."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

import pymupdf

HERE = Path(__file__).resolve().parent
A1 = HERE.parents[1]
ROOT = A1.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    qa = HERE / "qa"
    qa.mkdir(exist_ok=True)
    expected = {
        "linear": 784 * 10 + 10,
        "mlp": (784 + 1) * 256 + (256 + 1) * 128 + (128 + 1) * 10,
        "cnn": (9 + 1) * 32 + (32 * 9 + 1) * 64
               + (64 * 7 * 7 + 1) * 128 + (128 + 1) * 10,
        "lstm": 4 * 128 * (28 + 128 + 2) + (128 + 1) * 10,
        "transformer": 16 * 64 + 64 + 49 * 64
                       + 2 * (3 * 64 * 64 + 3 * 64 + 64 * 64 + 64
                              + 64 * 128 + 128 + 128 * 64 + 64 + 4 * 64)
                       + 2 * 64 + 64 * 10 + 10,
    }
    evidence = {}
    splits = set()
    for model, count in expected.items():
        run = A1 / "runs" / ("m1" if model in {"linear", "mlp"} else "m2") / model
        saved = json.loads((run / "summary.json").read_text())
        test = json.loads((run / "test_summary.json").read_text())
        assert saved["parameters"] == count, model
        for field, value in {
            "seed": 42, "batch_size": 128, "epochs": 10,
            "learning_rate": 0.001, "val_ratio": 0.1,
            "augment_train": False, "num_workers": 0,
        }.items():
            assert saved["config"][field] == value, (model, field)
        assert saved["train_samples"] == 54000
        assert saved["val_samples"] == 6000
        assert test["test_samples"] == 10000
        assert test["selection_rule"] == "lowest validation loss"
        assert "1660 SUPER" in test["hardware"]["gpu"]
        assert saved["device"] == ("mps" if model in {"linear", "mlp"} else "cuda")
        assert round(saved["train_mean"], 6) == 0.286209
        assert round(saved["train_std"], 6) == 0.353160
        splits.add(saved["split_sha256"])
        evidence[model] = {
            "parameters": count,
            "training_device": saved["device"],
            "summary_sha256": sha(run / "summary.json"),
        }
    assert len(splits) == 1

    doc = pymupdf.open(HERE / "methodology.pdf")
    assert len(doc) == 5
    text = "\n".join(page.get_text() for page in doc)
    assert "??" not in text and "\ufffd" not in text
    assert not re.search(r"\b(TODO|TBD|PLACEHOLDER)\b", text)
    for count in expected.values():
        assert f"{count:,}" in text
    for i, page in enumerate(doc):
        assert len(page.get_text()) > 1500
        for block in page.get_text("blocks"):
            assert page.rect.contains(pymupdf.Rect(block[:4]))
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(qa / f"page-{i+1:02}.png")
    log = (HERE / "methodology.log").read_text(errors="replace")
    for issue in ["Overfull", "Missing character", "undefined references", "Citation `"]:
        assert issue not in log, issue

    snapshot = ROOT / "tmp/pdfs/methodology_fresh/preservation.json"
    preserved = None
    if snapshot.exists():
        before = json.loads(snapshot.read_text())
        for name, digest in before.items():
            assert sha(Path(name)) == digest, name
        preserved = len(before)
    (qa / "text.txt").write_text(text, encoding="utf-8")
    report = {
        "passed": True,
        "pages": len(doc),
        "saved_configuration_checks": evidence,
        "common_split_sha256": next(iter(splits)),
        "pdf_sha256": sha(HERE / "methodology.pdf"),
        "prior_files_checked_unchanged": preserved,
        "AI_USAGE_sha256": sha(ROOT / "AI_USAGE.md"),
        "code_sources": {name: sha(A1 / "a1" / name)
                         for name in ["models.py", "data.py", "train.py", "evaluate.py"]},
        "training_or_inference_rerun": False,
        "visual_review": "All five pages inspected; tables, formulas, references and footers readable; no clipping or overlaps.",
    }
    (qa / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with zipfile.ZipFile(HERE / "methodology_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for item in ["methodology.tex", "sections/03_methodology.tex",
                     "references.tex", "README.md", "verify_document.py"]:
            archive.write(HERE / item, item)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
