"""Agreement between predicted and reference stages, computed from one confusion matrix."""

import numpy as np

N_CLASSES = 5


def confusion_matrix(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Rows are reference stages, columns are predicted stages."""
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(matrix, (reference, predicted), 1)
    return matrix


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def summarize(matrix: np.ndarray) -> dict:
    """Metrics follow scikit-learn's defaults so they match the A2 report.

    Macro-F1 averages over classes present in the reference or the predictions; balanced
    accuracy averages recall over classes present in the reference.
    """
    total = int(matrix.sum())
    if not total:
        raise ValueError("No scored epochs to evaluate.")
    true_positive = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    per_class = []
    for index in range(N_CLASSES):
        f1 = _ratio(2 * true_positive[index], support[index] + predicted[index])
        per_class.append({
            "precision": _ratio(true_positive[index], predicted[index]),
            "recall": _ratio(true_positive[index], support[index]),
            "f1": f1,
            "support": int(support[index]),
        })
    observed = true_positive.sum() / total
    expected = float((support * predicted).sum()) / total ** 2
    f1_values = [row["f1"] for row in per_class if row["f1"] is not None]
    recalls = [row["recall"] for row in per_class if row["recall"] is not None]
    return {
        "accuracy": float(observed),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "kappa": float((observed - expected) / (1 - expected)) if expected < 1 else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "n_scored_epochs": total,
    }
