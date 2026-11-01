"""Test release-cohort features with nested track-grouped target encoding."""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_sweep import engineer, preprocessor


def date_keys(frame):
    dates = frame.track_album_release_date.astype("string").fillna("unknown")
    return pd.DataFrame({"year": dates.str[:4], "year_month": dates.str[:7], "date": dates})


def encode(reference, labels, query, smoothing):
    result = {}
    prior = np.mean(labels)
    for column in reference:
        stats = pd.DataFrame({"key": reference[column].to_numpy(), "label": labels}).groupby("key").label.agg(["sum", "count"])
        rate = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        result[column + "_rate"] = query[column].map(rate).fillna(prior).to_numpy()
        result[column + "_count"] = query[column].map(stats["count"]).fillna(0).to_numpy()
    return np.column_stack(list(result.values()))


def crossfit(train, labels, groups, valid, smoothing):
    encoded = np.empty((len(train), 2 * train.shape[1]))
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=911)
    for fit, held in inner.split(train, labels, groups):
        assert set(groups[fit]).isdisjoint(groups[held])
        encoded[held] = encode(train.iloc[fit], labels[fit], train.iloc[held], smoothing)
    return encoded, encode(train, labels, valid, smoothing)


def self_check():
    ref = pd.DataFrame({"date": ["a", "a", "b"]})
    query = pd.DataFrame({"date": ["a", "unseen"]})
    actual = encode(ref, np.array([1, 0, 1]), query, 2)
    assert np.allclose(actual, [[(1 + 4 / 3) / 4, 2], [2 / 3, 0]])
    # Unique categories must use the inner-training prior, never their own label.
    keys = pd.DataFrame({"date": list("abcdefghijkl")})
    y = np.array([0, 1] * 6)
    encoded, _ = crossfit(keys, y, np.arange(12), keys.iloc[:1], 10)
    assert (encoded[:, 1] == 0).all()
    assert ((encoded[:, 0] > 0) & (encoded[:, 0] < 1)).all()


def main():
    self_check()
    frame = pd.read_csv("data/train.csv")
    features, categorical = engineer(frame, "baseline")
    keys = date_keys(frame)
    y = frame.Popularity_Type.eq("High").to_numpy(dtype=int)
    groups = frame.uri.fillna(pd.Series("missing_" + frame.index.astype(str), index=frame.index)).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=155).split(features, y, groups))
    for variant in ("counts", "rates", "both"):
        for smoothing in ((10,) if variant == "counts" else (2, 10, 50)):
            scores = {"forest": [], "boosted": []}
            for train, valid in folds:
                transform = preprocessor(features, categorical)
                train_x = transform.fit_transform(features.iloc[train])
                valid_x = transform.transform(features.iloc[valid])
                train_e, valid_e = crossfit(keys.iloc[train], y[train], groups[train], keys.iloc[valid], smoothing)
                if variant == "counts":
                    train_e, valid_e = train_e[:, 1::2], valid_e[:, 1::2]
                elif variant == "rates":
                    train_e, valid_e = train_e[:, ::2], valid_e[:, ::2]
                train_x, valid_x = np.column_stack([train_x, train_e]), np.column_stack([valid_x, valid_e])
                for name, model in (
                    ("forest", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, criterion="entropy", class_weight="balanced", random_state=155, n_jobs=4)),
                    ("boosted", HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.03, max_iter=300, early_stopping=False, random_state=155)),
                ):
                    model.fit(train_x, y[train])
                    scores[name].append(roc_auc_score(y[valid], model.predict_proba(valid_x)[:, 1]))
            for model, aucs in scores.items():
                result = {"variant": variant, "smoothing": smoothing, "model": model, "seed": 155, "mean_auc": float(np.mean(aucs)), "fold_aucs": aucs}
                with open("cohort_results.jsonl", "a") as output:
                    output.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
