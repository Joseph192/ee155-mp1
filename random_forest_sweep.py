"""Sweep random forests on the same track-grouped validation task."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from improved_tree import make_features


SEED = 155
CATEGORICAL = ["release_weekday", "time_signature", "mode"]


def build_forest_preprocessor(columns):
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), [c for c in columns if c not in CATEGORICAL]),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


def prepared_folds(features, labels, groups, seed):
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(features, labels, groups)
    for train_idx, valid_idx in folds:
        preprocessor = build_forest_preprocessor(features.columns)
        yield (
            preprocessor.fit_transform(features.iloc[train_idx]), labels[train_idx],
            preprocessor.transform(features.iloc[valid_idx]), labels[valid_idx],
        )


def score(folds, depth, leaf, max_features, trees, criterion="gini", class_weight=None, max_samples=None):
    aucs = []
    for train_x, train_y, valid_x, valid_y in folds:
        model = RandomForestClassifier(
            n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
            max_features=max_features, criterion=criterion,
            class_weight=class_weight, max_samples=max_samples,
            random_state=SEED, n_jobs=4,
        )
        model.fit(train_x, train_y)
        aucs.append(roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1]))
    return np.mean(aucs), aucs


def main():
    frame = pd.read_csv("data/train.csv")
    features = make_features(frame)
    labels = frame["Popularity_Type"].eq("High").to_numpy(dtype=int)
    groups = frame["uri"].fillna(pd.Series("missing_track_" + frame.index.astype(str), index=frame.index)).to_numpy()
    folds = list(prepared_folds(features, labels, groups, SEED))

    results = []
    for depth, leaf, max_features in product((None, 8, 16), (1, 3, 10), ("sqrt", 0.7, 1.0)):
        auc, _ = score(folds, depth, leaf, max_features, 200)
        results.append((auc, depth, leaf, max_features))
        print(f"depth={depth} leaf={leaf} features={max_features} trees=200 AUC={auc:.4f}", flush=True)

    _, depth, leaf, max_features = max(results, key=lambda row: row[0])
    print(f"Best shape: depth={depth}, leaf={leaf}, features={max_features}", flush=True)
    tree_results = []
    for trees in (100, 200, 400, 800):
        auc, _ = score(folds, depth, leaf, max_features, trees)
        tree_results.append((auc, trees))
        print(f"trees={trees} AUC={auc:.4f}", flush=True)

    trees = max(tree_results)[1]
    option_results = []
    for criterion, class_weight, max_samples in product(("gini", "entropy"), (None, "balanced"), (None, 0.7)):
        auc, _ = score(folds, depth, leaf, max_features, trees, criterion, class_weight, max_samples)
        option_results.append((auc, criterion, class_weight, max_samples))
        print(f"criterion={criterion} weight={class_weight} sample={max_samples} AUC={auc:.4f}", flush=True)

    _, criterion, class_weight, max_samples = max(option_results, key=lambda row: row[0])
    print(f"Best forest: depth={depth}, leaf={leaf}, features={max_features}, trees={trees}, criterion={criterion}, weight={class_weight}, sample={max_samples}", flush=True)
    for seed in (2026, 42, 777):
        auc, fold_aucs = score(list(prepared_folds(features, labels, groups, seed)), depth, leaf, max_features, trees, criterion, class_weight, max_samples)
        print(f"split_seed={seed} AUC={auc:.4f} folds={[round(x, 4) for x in fold_aucs]}", flush=True)


if __name__ == "__main__":
    main()
