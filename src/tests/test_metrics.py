from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def test_accuracy():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    result = accuracy_score(
        y_true,
        y_pred
    )

    assert result == 1.0


def test_precision():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    result = precision_score(
        y_true,
        y_pred
    )

    assert result == 1.0


def test_recall():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    result = recall_score(
        y_true,
        y_pred
    )

    assert result == 1.0


def test_f1_score():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    result = f1_score(
        y_true,
        y_pred
    )

    assert result == 1.0