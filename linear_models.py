import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _to_numpy(data):
    return np.asarray(data, dtype=float)


def _standardize_fit(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0

    return (X - mean) / scale, mean, scale


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _classification_metrics(y_true, probabilities):
    y_true = np.asarray(y_true).ravel().astype(int)
    probabilities = np.asarray(probabilities).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = 0.5

    return ClassificationMetrics(
        roc_auc=float(auc),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        recall=float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        f1=float(
            f1_score(y_true, predictions, zero_division=0)
        ),
    )


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = _to_numpy(data.train_features)
    X_test = _to_numpy(data.test_features)
    y_train = _to_numpy(data.train_target).reshape(-1)

    X_scaled, x_mean, x_scale = _standardize_fit(X_train)

    y_mean = y_train.mean()
    y_scale = y_train.std()

    if y_scale == 0:
        y_scale = 1.0

    y_scaled = (y_train - y_mean) / y_scale

    weights = np.zeros(X_scaled.shape[1])
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        predictions = X_scaled @ weights + bias
        errors = predictions - y_scaled

        loss = np.mean(errors ** 2)
        loss_history.append(float(loss))

        grad_weights = (
            2 / len(y_scaled)
        ) * (X_scaled.T @ errors)

        grad_bias = 2 * np.mean(errors)

        weights -= data.learning_rate * grad_weights
        bias -= data.learning_rate * grad_bias

    original_weights = y_scale * weights / x_scale

    original_bias = (
        y_mean
        + y_scale * bias
        - np.dot(original_weights, x_mean)
    )

    predictions = X_test @ original_weights + original_bias

    return RegressionResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(
    data: ClassificationInput,
) -> ClassificationResult:

    X_train = _to_numpy(data.train_features)
    X_test = _to_numpy(data.test_features)

    y_train = (
        _to_numpy(data.train_target)
        .reshape(-1)
        .astype(int)
    )

    if not np.all(np.isin(np.unique(y_train), [0, 1])):
        raise ValueError(
            "Целевая переменная должна содержать классы 0 и 1."
        )

    X_scaled, x_mean, x_scale = _standardize_fit(X_train)

    weights = np.zeros(X_scaled.shape[1])
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        probabilities = _sigmoid(X_scaled @ weights + bias)

        eps = 1e-12

        loss = -np.mean(
            y_train * np.log(probabilities + eps)
            + (1 - y_train) * np.log(1 - probabilities + eps)
        )

        loss_history.append(float(loss))

        errors = probabilities - y_train

        grad_weights = (
            X_scaled.T @ errors
        ) / len(y_train)

        grad_bias = np.mean(errors)

        weights -= data.learning_rate * grad_weights
        bias -= data.learning_rate * grad_bias

    original_weights = weights / x_scale

    original_bias = float(
        bias - np.dot(original_weights, x_mean)
    )

    test_probabilities = _sigmoid(
        X_test @ original_weights + original_bias
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    return ClassificationResult(
        weights=original_weights,
        bias=original_bias,
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


def compare_classical_models(
    data: ClassificationInput,
) -> BaselineComparisonResult:

    X_train = _to_numpy(data.train_features)
    X_test = _to_numpy(data.test_features)

    y_train = (
        _to_numpy(data.train_target)
        .reshape(-1)
        .astype(int)
    )

    y_test = (
        _to_numpy(data.test_target)
        .reshape(-1)
        .astype(int)
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    tree = DecisionTreeClassifier(
        random_state=42
    )

    n_neighbors = min(5, len(X_train))

    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=n_neighbors
        ),
    )

    logistic.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    logistic_classes = logistic.named_steps[
        "logisticregression"
    ].classes_

    tree_classes = tree.classes_

    knn_classes = knn.named_steps[
        "kneighborsclassifier"
    ].classes_

    logistic_probabilities = logistic.predict_proba(
        X_test
    )[:, np.where(logistic_classes == 1)[0][0]]

    tree_probabilities = tree.predict_proba(
        X_test
    )[:, np.where(tree_classes == 1)[0][0]]

    knn_probabilities = knn.predict_proba(
        X_test
    )[:, np.where(knn_classes == 1)[0][0]]

    return BaselineComparisonResult(
        logistic_regression=_classification_metrics(
            y_test,
            logistic_probabilities,
        ),
        decision_tree=_classification_metrics(
            y_test,
            tree_probabilities,
        ),
        knn=_classification_metrics(
            y_test,
            knn_probabilities,
        ),
    )
