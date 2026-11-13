import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from app.services.metrics import confusion_matrix, summarize


@pytest.mark.parametrize("classes", [5, 3])  # 3: two stages never occur
def test_metrics_match_scikit_learn(classes):
    rng = np.random.default_rng(classes)
    reference = rng.integers(0, classes, 500)
    predicted = np.where(rng.random(500) < 0.7, reference, rng.integers(0, classes, 500))
    summary = summarize(confusion_matrix(reference, predicted))

    assert summary["n_scored_epochs"] == 500
    assert summary["accuracy"] == pytest.approx(accuracy_score(reference, predicted))
    assert summary["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(reference, predicted))
    assert summary["macro_f1"] == pytest.approx(f1_score(reference, predicted, average="macro"))
    assert summary["kappa"] == pytest.approx(cohen_kappa_score(reference, predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        reference, predicted, labels=range(classes))
    for index in range(classes):
        row = summary["per_class"][index]
        assert (row["precision"], row["recall"], row["f1"], row["support"]) == pytest.approx(
            (precision[index], recall[index], f1[index], support[index]))


def test_absent_classes_are_reported_as_missing_not_zero():
    reference = np.array([0, 0, 2, 2])
    summary = summarize(confusion_matrix(reference, reference))
    assert summary["macro_f1"] == 1.0
    assert summary["per_class"][1] == {"precision": None, "recall": None, "f1": None,
                                       "support": 0}
    assert summary["confusion_matrix"][2][2] == 2


def test_no_scored_epochs_is_an_error():
    with pytest.raises(ValueError):
        summarize(np.zeros((5, 5), dtype=int))
