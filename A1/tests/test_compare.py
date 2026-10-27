"""Check that export cannot silently mix incompatible experiments."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from a1.compare import RUNS, collect_results, write_comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for model, stage in RUNS.items():
            directory = self.root / stage / model
            directory.mkdir(parents=True)
            checkpoint = model.encode()
            (directory / "best.pt").write_bytes(checkpoint)
            summary = {
                "model": model, "best_epoch": 2, "source_sha256": "train-source",
                "split_sha256": "split", "seed": 42, "train_mean": 0.28, "train_std": 0.35,
                "train_samples": 54000, "val_samples": 6000, "parameters": 100,
                "training_seconds": 10, "device": "cpu",
                "config": {"epochs": 10, "batch_size": 128, "learning_rate": 0.001,
                           "augment_train": False, "val_ratio": 0.1},
            }
            test = {
                "model": model, "checkpoint_epoch": 2, "selection_rule": "lowest validation loss",
                "test_samples": 10000, "source_sha256": "train-source", "split_sha256": "split",
                "seed": 42, "test_source_sha256": "test-source",
                "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
                "test_loss": 0.3, "test_accuracy": 0.9, "test_macro_f1": 0.89,
                "hardware": {"device": "cpu"},
                "inference": {"batch_size": 128, "forward_ms_per_image": 0.01, "evaluation_seconds": 2},
            }
            for name, value in (("summary.json", summary), ("test_summary.json", test)):
                (directory / name).write_text(json.dumps(value), encoding="utf-8")

    def change_test(self, **updates):
        path = self.root / "m2/cnn/test_summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(updates)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_exports_preserve_order_metrics_and_sources(self):
        result = collect_results(self.root)
        self.assertEqual([row["model"] for row in result["results"]], list(RUNS))
        self.assertEqual(len(result["sources_relative_to_runs"]), 10)
        output = self.root / "comparison"
        write_comparison(result, output)
        saved = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        self.assertIn("90.00%", (output / "comparison.md").read_text(encoding="utf-8"))
        self.assertTrue((output / "comparison.png").is_file())

    def test_rejects_mixed_test_dataset(self):
        self.change_test(test_source_sha256="other-test-source")
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            collect_results(self.root)

    def test_rejects_wrong_checkpoint(self):
        (self.root / "m2/cnn/best.pt").write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            collect_results(self.root)

    def test_rejects_epoch_mismatch(self):
        self.change_test(checkpoint_epoch=3)
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            collect_results(self.root)

    def test_rejects_nonfinite_metric(self):
        self.change_test(test_accuracy=float("nan"))
        with self.assertRaisesRegex(ValueError, "invalid test_accuracy"):
            collect_results(self.root)


if __name__ == "__main__":
    unittest.main()
