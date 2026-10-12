"""Check row ordering, batch independence, recurrent gradients and reloads."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from a1.data import FashionMNISTView, to_representation
from a1.models import build_model, count_parameters, input_mode
from a1.train import load_checkpoint


class LSTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_rows_preserve_every_pixel_and_normalization(self):
        # Distinct values on both axes catch an accidental transpose.
        image = torch.arange(784, dtype=torch.float32).reshape(1, 28, 28)
        view = FashionMNISTView([(image, 3)], [0], mean=10, std=2, mode="row_seq")
        sequence, label = view[0]
        self.assertEqual(label, 3)
        self.assertEqual(tuple(sequence.shape), (28, 28))
        for row in range(28):
            torch.testing.assert_close(sequence[row], (image[0, row] - 10) / 2)
        self.assertEqual(sequence[1, 0].item(), 9)
        self.assertEqual(sequence[0, 1].item(), -4.5)
        torch.testing.assert_close(sequence.reshape(1, 28, 28), (image - 10) / 2)
        with self.assertRaises(ValueError):
            to_representation(torch.zeros(3, 28, 28), "row_seq")

    def test_readout_shapes_and_images_do_not_share_state(self):
        model = build_model("lstm").eval()
        self.assertEqual(input_mode("lstm"), "row_seq")
        self.assertEqual(count_parameters(model), 82186)
        images = torch.randn(8, 28, 28)
        with torch.no_grad():
            together = model(images)
            separate = torch.cat([model(image.unsqueeze(0)) for image in images])
            _, (hidden, _) = model.lstm(images)
            torch.testing.assert_close(together, model.classifier(hidden[-1]))
            torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
            model(torch.randn(2, 28, 28))
            torch.testing.assert_close(model(images), together, rtol=0, atol=0)
        self.assertEqual(tuple(together.shape), (8, 10))
        self.assertTrue(torch.isfinite(together).all())
        self.assertFalse(torch.allclose(together.sum(1), torch.ones(8)))
        for invalid in (torch.zeros(2, 784), torch.zeros(2, 1, 28, 28)):
            with self.assertRaises(ValueError):
                model(invalid)

    def test_gradients_reach_early_rows_and_recurrent_weights(self):
        model = build_model("lstm")
        images = torch.randn(16, 28, 28, requires_grad=True)
        labels = torch.arange(16) % 10
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.lstm.weight_hh_l0.detach().clone()
        model.eval()
        initial_loss = criterion(model(images), labels).item()
        model.train()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            images.grad = None
            loss = criterion(model(images), labels)
            loss.backward()
            gradient = model.lstm.weight_hh_l0.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            self.assertGreater(images.grad[:, 0].abs().sum().item(), 0)
            optimizer.step()
        model.eval()
        self.assertLess(criterion(model(images), labels).item(), initial_loss * 0.5)
        self.assertFalse(torch.equal(before, model.lstm.weight_hh_l0))

    def test_checkpoint_reload_and_eval_dropout(self):
        model = build_model("lstm").eval()
        images = torch.randn(4, 28, 28)
        with torch.no_grad():
            expected = model(images)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                torch.save({"model_name": "lstm", "state_dict": model.state_dict()}, path)
                restored, _ = load_checkpoint(path)
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
                self.assertFalse(restored.dropout.training)


if __name__ == "__main__":
    unittest.main()
