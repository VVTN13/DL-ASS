"""Small checks for shape handling, optimization and checkpoint reloads."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from a1.models import build_model, count_parameters, input_mode
from a1.train import load_checkpoint


class CNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_image_batches_and_raw_logits(self):
        model = build_model("cnn").eval()
        for batch_size in (1, 8):
            logits = model(torch.randn(batch_size, 1, 28, 28))
            self.assertEqual(tuple(logits.shape), (batch_size, 10))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(batch_size)))
        self.assertEqual(count_parameters(model), 421642)

    def test_optimizer_reaches_convolutions_and_reduces_loss(self):
        model = build_model("cnn")
        images = torch.randn(16, 1, 28, 28)
        labels = torch.arange(16) % 10
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.features[0].weight.detach().clone()
        model.eval()
        initial_loss = criterion(model(images), labels).item()
        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            self.assertTrue(torch.isfinite(model.features[0].weight.grad).all())
            optimizer.step()
        model.eval()
        self.assertLess(criterion(model(images), labels).item(), initial_loss * 0.5)
        self.assertFalse(torch.equal(before, model.features[0].weight))

    def test_checkpoint_roundtrip_and_eval_dropout(self):
        model = build_model("cnn").eval()
        images = torch.randn(4, 1, 28, 28)
        expected = model(images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model_name": "cnn", "state_dict": model.state_dict()}, path)
            restored, _ = load_checkpoint(path)
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
            torch.testing.assert_close(restored(images), restored(images), rtol=0, atol=0)
            self.assertFalse(restored.training)

    def test_existing_models_keep_flattened_inputs(self):
        for name in ("linear", "mlp"):
            self.assertEqual(input_mode(name), "flatten")
            self.assertEqual(tuple(build_model(name)(torch.randn(2, 784)).shape), (2, 10))
        self.assertEqual(input_mode("cnn"), "image")


if __name__ == "__main__":
    unittest.main()
