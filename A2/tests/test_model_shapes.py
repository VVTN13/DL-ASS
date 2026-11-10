"""Check classifier/embedding shapes and gradients for the Sleep-EDF pipeline."""

from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from A2 import model_lib


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "pretrained_weights" / "pretrained_weights.pth"
CLASSIFIERS = (
    model_lib.StandardCNN2D,
    model_lib.CBAM_EEG_Net,
    model_lib.ResCBAM_EEG_Net,
    model_lib.ResNet50_CBAM,
    model_lib.Transformer,
    model_lib.ConvLSTMAttentionTransformer,
)


class ModelShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        path = ROOT / "dataset" / "sleep_waveform_small.pt"
        if path.is_file():
            data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            labels = data["labels"].numpy()
            development = np.asarray(data["splits"]) == "development"
            indices = [int(np.flatnonzero(development & (labels == c))[0]) for c in range(5)]
            cls.inputs = data["patches"][indices]
            cls.targets = data["labels"][indices]
        else:
            cls.inputs = torch.randn(5, 2, 30, 200)
            cls.targets = torch.arange(5)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def check_training_step(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        logits = model(self.inputs)
        self.assertEqual(tuple(logits.shape), (5, 5))
        loss = nn.CrossEntropyLoss()(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            single = model(self.inputs[:1])
        self.assertEqual(tuple(single.shape), (1, 5))
        self.assertTrue(torch.isfinite(single).all())

    def test_all_scratch_classifiers_train_on_waveforms_with_default_arguments(self):
        for classifier in CLASSIFIERS:
            with self.subTest(model=classifier.__name__):
                self.check_training_step(classifier())

    @unittest.skipUnless(CHECKPOINT.is_file(), "Local pretrained CBraMod checkpoint is required")
    def test_cbramod_checkpoint_and_waveform_training_step(self):
        self.check_training_step(model_lib.CBraModClassifier(checkpoint_path=CHECKPOINT))

    def test_configurable_channels_classes_and_rectangular_grid(self):
        inputs = torch.randn(2, 3, 60, 57)
        for classifier in CLASSIFIERS:
            with self.subTest(model=classifier.__name__):
                model = classifier(in_channels=3, num_classes=3).eval()
                model_inputs = torch.randn(2, 3, 30, 200) if classifier is model_lib.Transformer else inputs
                with torch.no_grad():
                    logits = model(model_inputs)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all())

    def test_embedding_tokens_match_channels_and_lstm_keeps_time_order(self):
        embeddings = [
            model_lib.ConvEmbedding(in_channels=2, embed_dim=32),
            model_lib.ConvLSTMAttentionEmbedding(in_channels=2, embed_dim=32, lstm_hidden_size=16),
        ]
        for embedding in embeddings:
            with self.subTest(embedding=type(embedding).__name__):
                self.assertEqual(tuple(embedding(self.inputs).shape), (5, 3, 32))

        # Observe the model's actual CNN output and LSTM input at row boundaries.
        hybrid = embeddings[1].eval()
        observed = {}

        def remember_pooled(module, args, output):
            observed["pooled"] = output.detach()

        def remember_temporal(module, args):
            observed["temporal"] = args[0].detach()

        cnn_hook = hybrid.dropout3.register_forward_hook(remember_pooled)
        lstm_hook = hybrid.lstm.register_forward_pre_hook(remember_temporal)
        with torch.no_grad():
            hybrid(self.inputs[:2])
        cnn_hook.remove()
        lstm_hook.remove()
        pooled, temporal = observed["pooled"], observed["temporal"]
        self.assertEqual(tuple(temporal.shape), (4, 350, 8))
        for step in [0, 17, 49, 50, 349]:
            row, column = divmod(step, pooled.shape[-1])
            torch.testing.assert_close(temporal[0, step], pooled[0, :, row, column])
        self.assertEqual(hybrid.lstm.input_size, 8)
        model = model_lib.ConvLSTMAttentionTransformer(img_size=112, lstm_hidden_size=48)
        self.assertEqual(model.patch_embed.lstm.hidden_size, 48)

    def test_transformer_attention_recording_matches_fast_path(self):
        for classifier in (model_lib.Transformer, model_lib.ConvLSTMAttentionTransformer):
            with self.subTest(model=classifier.__name__):
                model = classifier(emb_dim=32, num_heads=4, num_layers=2).eval()
                with torch.no_grad():
                    fast = model(self.inputs[:2])
                    recorded, matrices = model(self.inputs[:2], record_attn=True)
                torch.testing.assert_close(recorded, fast, rtol=1e-4, atol=1e-5)
                self.assertEqual(len(matrices), 2)
                tokens = 31 if classifier is model_lib.Transformer else 3
                for matrix in matrices:
                    self.assertEqual(tuple(matrix.shape), (2, 4, tokens, tokens))
                    if classifier is model_lib.Transformer:
                        self.assertTrue((matrix >= 0).all())
                        torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 4, tokens))

    def test_temporal_embedding_preserves_channels_and_second_order(self):
        embedding = model_lib.TemporalConvEmbedding(in_channels=2, embed_dim=32).eval()
        observed = {}

        def remember_short(module, args):
            observed["short"] = args[0].detach().clone()

        def remember_long(module, args):
            observed["long"] = args[0].detach().clone()

        hooks = [embedding.short_branch[0].register_forward_pre_hook(remember_short),
                 embedding.long_branch[0].register_forward_pre_hook(remember_long)]
        inputs = self.inputs[:2]
        with torch.no_grad():
            original = embedding(inputs)
        for hook in hooks:
            hook.remove()
        self.assertEqual(tuple(original.shape), (2, 31, 32))
        for patches in observed.values():
            self.assertEqual(tuple(patches.shape), (60, 2, 200))
            for batch in range(2):
                for second in range(30):
                    torch.testing.assert_close(patches[batch * 30 + second], inputs[batch, :, second])

        # A local waveform change must affect only that second before attention.
        changed = inputs.clone()
        changed[0, 1, 11] += torch.linspace(-1, 1, 200)
        with torch.no_grad():
            updated = embedding(changed)
        untouched = torch.ones(31, dtype=torch.bool)
        untouched[12] = False  # CLS is token 0; second 11 is token 12.
        torch.testing.assert_close(updated[0, untouched], original[0, untouched])
        torch.testing.assert_close(updated[1], original[1])
        self.assertFalse(torch.allclose(updated[0, 12], original[0, 12]))

        with self.assertRaisesRegex(AssertionError, "200 Hz"):
            embedding(torch.zeros(1, 2, 30, 100))

    def test_attention_and_residual_helpers_accept_small_channel_counts(self):
        inputs = self.inputs[:2]
        attention = model_lib.CBAM(in_planes=2)
        output = attention(inputs)
        self.assertEqual(output.shape, inputs.shape)
        output.square().mean().backward()
        for parameter in attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        residual = model_lib.ResidualCBAMBlock(2, 2, dropout_p=0.0)
        self.assertEqual(tuple(residual(inputs).shape), (2, 2, 15, 100))


if __name__ == "__main__":
    unittest.main()
