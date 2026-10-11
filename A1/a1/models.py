"""Small classifiers for the Fashion-MNIST comparison."""

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


def input_mode(name: str):
    if name == "cnn":
        return "image"
    if name in {"linear", "mlp"}:
        return "flatten"
    raise ValueError(f"Unknown model {name!r}")


def build_model(name: str, num_classes: int = 10):
    if name == "linear":
        return LinearClassifier(num_classes)
    if name == "mlp":
        return MLPClassifier(num_classes)
    if name == "cnn":
        return CNNClassifier(num_classes)
    raise ValueError(f"Unknown model {name!r}; choose 'linear', 'mlp', or 'cnn'")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
