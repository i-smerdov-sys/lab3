import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.linear_models import (
    RegressionInput,
    ClassificationInput,
)

from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
)

student = pd.read_csv("Student_Performance.csv")

student["Extracurricular Activities"] = (
    student["Extracurricular Activities"]
    .map({"Yes": 1, "No": 0})
)

X_reg = student.drop(
    columns=["Performance Index"]
).to_numpy(dtype=float)

y_reg = student[
    "Performance Index"
].to_numpy(dtype=float)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42,
    )
)

reg_data = RegressionInput(
    train_features=X_reg_train,
    train_target=y_reg_train,
    test_features=X_reg_test,
    test_target=y_reg_test,
)

reg_result = train_linear_regression(reg_data)

reg_mse = mean_squared_error(
    y_reg_test,
    reg_result.predictions,
)

german = pd.read_csv(
    "german.csv",
    sep=";",
)

target_column = "Creditability"

X_clf = german.drop(
    columns=[target_column]
).to_numpy(dtype=float)

y_clf = german[
    target_column
].to_numpy(dtype=int)

if not np.all(np.isin(np.unique(y_clf), [0, 1])):
    raise ValueError(
        "Creditability должна содержать только классы 0 и 1."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf,
)

clf_data = ClassificationInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
)

own_result = train_logistic_classifier(clf_data)

own_metrics = {
    "roc_auc": float(
        roc_auc_score(
            y_test,
            own_result.probabilities,
        )
    ),
    "accuracy": float(
        accuracy_score(
            y_test,
            own_result.predictions,
        )
    ),
    "precision": float(
        precision_score(
            y_test,
            own_result.predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            own_result.predictions,
            zero_division=0,
        )
    ),
    "f1": float(
        f1_score(
            y_test,
            own_result.predictions,
            zero_division=0,
        )
    ),
}

baseline_result = compare_classical_models(
    clf_data
)

results = {
    "linear_regression": {
        "test_mse": float(reg_mse),
    },
    "classification": {
        "own_logistic": own_metrics,
        "logistic_regression": asdict(
            baseline_result.logistic_regression
        ),
        "decision_tree": asdict(
            baseline_result.decision_tree
        ),
        "knn": asdict(
            baseline_result.knn
        ),
    },
}

with open(
    "experiment_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False,
    )

print(
    json.dumps(
        results,
        indent=4,
        ensure_ascii=False,
    )
)

print("Student Performance:", student.shape)
print("German Credit:", german.shape)