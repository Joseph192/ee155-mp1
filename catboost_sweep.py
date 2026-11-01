"""CatBoost date-feature sweep; stop training on an inner grouped split."""

import argparse
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_sweep import engineer


def cat_features(frame, variant):
    features, categorical = engineer(frame, "baseline")
    if variant in ("year", "dates", "full"):
        features["year_category"] = features.release_year
        categorical.append("year_category")
    if variant in ("dates", "full"):
        dates = frame.track_album_release_date.astype("string").fillna("unknown")
        features["release_month_category"] = dates.str[5:7]
        features["release_year_month"] = dates.str[:7]
        features["release_date_category"] = dates
        categorical += ["release_month_category", "release_year_month", "release_date_category"]
    if variant == "full":
        features["key"] = frame.key
        categorical.append("key")
    for column in categorical:
        features[column] = features[column].astype("string").fillna("unknown").astype(str)
    for column in features.columns.difference(categorical):
        features[column] = features[column].astype(float)
    return features, categorical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[155])
    parser.add_argument("--variants", nargs="+", choices=["baseline", "year", "dates", "full"], default=["baseline", "year", "dates", "full"])
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--l2", nargs="+", type=float, default=[3, 10])
    args = parser.parse_args()
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise SystemExit("Install the approved dependency: .venv/bin/python -m pip install catboost")
    frame = pd.read_csv("data/train.csv")
    y = frame.Popularity_Type.eq("High").to_numpy(dtype=int)
    groups = frame.uri.fillna(pd.Series("missing_" + frame.index.astype(str), index=frame.index)).to_numpy()
    for variant, seed, depth, l2 in product(args.variants, args.seeds, args.depths, args.l2):
        features, categorical = cat_features(frame, variant)
        aucs, rounds = [], []
        outer = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        for train, valid in outer.split(features, y, groups):
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=911)
            fit, stop = next(inner.split(features.iloc[train], y[train], groups[train]))
            assert set(groups[train]).isdisjoint(groups[valid])
            assert set(groups[train[fit]]).isdisjoint(groups[train[stop]])
            parameters = dict(depth=depth, l2_leaf_reg=l2, learning_rate=0.03,
                              loss_function="Logloss", eval_metric="AUC", cat_features=categorical,
                              random_seed=155, thread_count=4, verbose=False, allow_writing_files=False)
            probe = CatBoostClassifier(iterations=3000, **parameters)
            probe.fit(features.iloc[train[fit]], y[train[fit]],
                      eval_set=(features.iloc[train[stop]], y[train[stop]]), early_stopping_rounds=150)
            count = probe.get_best_iteration() + 1
            assert 1 <= count <= 3000
            model = CatBoostClassifier(iterations=count, **parameters)
            model.fit(features.iloc[train], y[train])
            prediction = model.predict_proba(features.iloc[valid])[:, 1]
            assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
            aucs.append(roc_auc_score(y[valid], prediction))
            rounds.append(count)
        result = dict(variant=variant, seed=seed, depth=depth, l2=l2,
                      mean_auc=float(np.mean(aucs)), fold_aucs=aucs, iterations=rounds)
        with open("catboost_results.jsonl", "a") as output:
            output.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
