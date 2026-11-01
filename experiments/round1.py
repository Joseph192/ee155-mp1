"""Round 1: feature variants x model families, all on the same grouped folds."""

import sys

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from common import build_forest_preprocessor, load, oof, report
from features import build


def forest(cls, **kw):
    def fit_predict(train_x, train_y, valid_x, *_):
        prep = build_forest_preprocessor(train_x.columns)
        model = cls(n_estimators=500, random_state=155, n_jobs=10, **kw).fit(prep.fit_transform(train_x), train_y)
        return model.predict_proba(prep.transform(valid_x))[:, 1]
    return fit_predict


def hgb(**kw):
    def fit_predict(train_x, train_y, valid_x, *_):
        model = HistGradientBoostingClassifier(early_stopping=False, random_state=155, **kw).fit(train_x, train_y)
        return model.predict_proba(valid_x)[:, 1]
    return fit_predict


MODELS = {
    "rf": forest(RandomForestClassifier, min_samples_leaf=3, criterion="entropy", class_weight="balanced"),
    "et": forest(ExtraTreesClassifier, min_samples_leaf=2, max_features=0.5, criterion="entropy", class_weight="balanced"),
    "hgb": hgb(max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.03, max_iter=300, l2_regularization=1.0),
}


def main():
    train, test, y, groups = load()
    variants = sys.argv[1:] or ["base", "dup", "dup_near", "yearrel", "date_count", "all"]
    for variant in variants:
        features, _ = build(train, test, variant)
        for name, fit_predict in MODELS.items():
            report(f"r1_{variant}_{name}", oof(fit_predict, features, y, groups), y, groups)


if __name__ == "__main__":
    main()
