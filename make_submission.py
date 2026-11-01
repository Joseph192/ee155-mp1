"""Train the selected random forest and write a Kaggle submission CSV."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from improved_tree import make_features
from random_forest_sweep import build_forest_preprocessor


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    submission = pd.read_csv("data/sample_submission.csv")
    if len(test) != len(submission) or not test["ID"].equals(submission["ID"]):
        raise ValueError("Test IDs do not match the sample submission")

    train_features = make_features(train)
    test_features = make_features(test)
    preprocessor = build_forest_preprocessor(train_features.columns)
    train_x = preprocessor.fit_transform(train_features)
    test_x = preprocessor.transform(test_features)
    labels = train["Popularity_Type"].eq("High").to_numpy(dtype=int)

    model = RandomForestClassifier(
        n_estimators=800, max_depth=None, min_samples_leaf=3,
        max_features="sqrt", criterion="entropy", class_weight="balanced",
        random_state=155, n_jobs=4,
    )
    model.fit(train_x, labels)
    probabilities = model.predict_proba(test_x)[:, list(model.classes_).index(1)]
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Predictions must be finite probabilities")

    submission["Popularity_Type"] = probabilities
    submission.to_csv("submission.csv", index=False)
    print(f"Wrote submission.csv with {len(submission)} rows")


if __name__ == "__main__":
    main()
