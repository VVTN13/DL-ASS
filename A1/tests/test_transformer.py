"""Check patch order, positional information, attention gradients and reloads."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from a1.data import FashionMNISTView, to_representation
from a1.models import build_model, count_parameters, input_mode
from a1.train import load_checkpoint


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_patch_order_normalization_and_exact_reconstruction(self):
        image = torch.arange(784, dtype=torch.float32).reshape(1, 28, 28)
        view = FashionMNISTView([(image, 3)], [0], mean=10, std=2, mode="patch_seq")
        patches, label = view[0]
        self.assertEqual(label, 3)
        self.assertEqual(tuple(patches.shape), (49, 16))
        restored = torch.empty_like(image)
        for row in range(7):
            for col in range(7):
                patch = patches[7 * row + col].reshape(4, 4)
                expected = (image[0, 4 * row:4 * row + 4, 4 * col:4 * col + 4] - 10) / 2
                torch.testing.assert_close(patch, expected, rtol=0, atol=0)
                restored[0, 4 * row:4 * row + 4, 4 * col:4 * col + 4] = patch
        torch.testing.assert_close(restored, (image - 10) / 2, rtol=0, atol=0)
        for invalid in (torch.zeros(3, 28, 28), torch.zeros(28, 28)):
            with self.assertRaises(ValueError):
                to_representation(invalid, "patch_seq")
        for size in (0, -1, 3):
            with self.assertRaises(ValueError):
                to_representation(image, "patch_seq", patch_size=size)

    def test_logits_shapes_batch_independence_and_position_sensitivity(self):
        model = build_model("transformer").eval()
        self.assertEqual(input_mode("transformer"), "patch_seq")
        self.assertEqual(count_parameters(model), 71946)
        patches = torch.randn(4, 49, 16)
        with torch.no_grad():
            logits = model(patches)
            separate = torch.cat([model(image.unsqueeze(0)) for image in patches])
            torch.testing.assert_close(logits, separate, rtol=1e-5, atol=1e-6)
            self.assertFalse(torch.allclose(logits, model(patches.flip(1)), rtol=1e-5, atol=1e-6))
            # Without positions, self-attention + mean pooling is permutation invariant.
            model.position_embedding.zero_()
            torch.testing.assert_close(model(patches), model(patches.flip(1)), rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(logits.shape), (4, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(torch.allclose(logits.sum(1), torch.ones(4)))
        self.assertEqual(tuple(build_model("transformer", num_classes=3)(patches).shape), (4, 3))
        for invalid in (torch.zeros(2, 784), torch.zeros(2, 1, 28, 28), torch.zeros(2, 28, 28)):
            with self.assertRaises(ValueError):
                model(invalid)

    def test_attention_position_and_all_patch_gradients(self):
        model = build_model("transformer")
        self.assertFalse(torch.equal(model.blocks[0].self_attn.in_proj_weight,
                                     model.blocks[1].self_attn.in_proj_weight))
        patches = torch.randn(8, 49, 16, requires_grad=True)
        targets = torch.arange(8)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        model.eval()
        initial = criterion(model(patches), targets).item()
        for _ in range(25):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            patches.grad = None
            criterion(model(patches), targets).backward()
            for parameter in (model.position_embedding, model.patch_embedding.weight,
                              *(block.self_attn.in_proj_weight for block in model.blocks)):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertTrue((patches.grad.abs().sum(dim=2) > 0).all())
            optimizer.step()
        model.eval()
        self.assertLess(criterion(model(patches), targets).item(), initial * 0.5)

    def test_checkpoint_reload_and_dropout(self):
        model = build_model("transformer")
        patches = torch.randn(4, 49, 16)
        with torch.no_grad():
            self.assertFalse(torch.equal(model(patches), model(patches)))
            model.eval()
            expected = model(patches)
            torch.testing.assert_close(model(patches), expected, rtol=0, atol=0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                torch.save({"model_name": "transformer", "state_dict": model.state_dict()}, path)
                restored, _ = load_checkpoint(path)
                torch.testing.assert_close(restored(patches), expected, rtol=0, atol=0)
                self.assertTrue(all(not module.training for module in restored.modules()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_inference_matches_without_changing_global_backend(self):
        fastpath_before = torch.backends.mha.get_fastpath_enabled()
        cpu = build_model("transformer").eval()
        gpu = build_model("transformer").cuda().eval()
        gpu.load_state_dict(cpu.state_dict())
        patches = torch.randn(8, 49, 16) * 3
        with torch.inference_mode():
            expected = cpu(patches)
            actual = gpu(patches.cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(actual.argmax(1), expected.argmax(1)))
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), fastpath_before)


if __name__ == "__main__":
    unittest.main()
