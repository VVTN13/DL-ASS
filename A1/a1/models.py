"""The two classifiers used in M1."""

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


def build_model(name: str, num_classes: int = 10):
    if name == "linear":
        return LinearClassifier(num_classes)
    if name == "mlp":
        return MLPClassifier(num_classes)
    raise ValueError(f"Unknown model {name!r}; choose 'linear' or 'mlp'")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
