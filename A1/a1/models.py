"""Small classifiers for the Fashion-MNIST comparison."""

import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        return self.classifier(x.flatten(start_dim=1))


class MLPClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class CNNClassifier(nn.Module):
    """Learn local features before classifying the pooled image."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LSTMClassifier(nn.Module):
    """Read 28 image rows top to bottom, then classify the final hidden state.

    Input is (batch, rows=28, pixel_features=28), supplied by ``row_seq``.
    A fresh zero hidden/cell state is used for each image on every forward call.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, batch_first=True)
        # Single-layer LSTMs have no inter-layer dropout; apply it to the readout.
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (28, 28):
            raise ValueError("LSTM expects (batch, 28 rows, 28 pixel features) from row_seq")
        _, (hidden, _) = self.lstm(x)
        # hidden: (layers=1, batch, 128), even with batch_first=True.
        return self.classifier(self.dropout(hidden[-1]))


class TransformerClassifier(nn.Module):
    """Classify 49 ordered 4x4 patches using two self-attention blocks.

    ``patch_seq`` supplies (batch, patches=49, pixels=16), in raster order.
    Learned positions distinguish patches; mean pooling reads all output tokens.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_embedding = nn.Linear(16, 64)
        self.position_embedding = nn.Parameter(torch.empty(1, 49, 64))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(0.2)
        # Construct each block separately so they start with independent weights.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, dropout=0.2,
                activation="gelu", batch_first=False, norm_first=True,
            )
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (49, 16):
            raise ValueError("Transformer expects (batch, 49 patches, 16 pixels) from patch_seq")
        tokens = self.dropout(self.patch_embedding(x) + self.position_embedding)
        # Sequence-first blocks use the standard attention path. On this CUDA
        # stack, fused batch-first inference drifts from the float32 CPU result.
        tokens = tokens.transpose(0, 1)  # (49 patches, batch, 64 features)
        # No causal or padding mask: all 49 patches of each image are present.
        for block in self.blocks:
            tokens = block(tokens)
        tokens = tokens.transpose(0, 1)
        return self.classifier(self.dropout(self.norm(tokens).mean(dim=1)))


def input_mode(name: str):
    if name == "cnn":
        return "image"
    if name in {"linear", "mlp"}:
        return "flatten"
    if name == "lstm":
        return "row_seq"
    if name == "transformer":
        return "patch_seq"
    raise ValueError(f"Unknown model {name!r}")


def build_model(name: str, num_classes: int = 10):
    if name == "linear":
        return LinearClassifier(num_classes)
    if name == "mlp":
        return MLPClassifier(num_classes)
    if name == "cnn":
        return CNNClassifier(num_classes)
    if name == "lstm":
        return LSTMClassifier(num_classes)
    if name == "transformer":
        return TransformerClassifier(num_classes)
    raise ValueError(f"Unknown model {name!r}; choose 'linear', 'mlp', 'cnn', 'lstm', or 'transformer'")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
