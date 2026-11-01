"""Test separate older/recent-release forests against a shared forest."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_sweep import engineer, preprocessor


def fit_predict(features, categorical, labels, train, valid):
    transform = preprocessor(features, categorical)
    tx = transform.fit_transform(features.iloc[train])
    vx = transform.transform(features.iloc[valid])
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=1,
                                   max_features=0.5, criterion="entropy",
                                   class_weight="balanced", random_state=155, n_jobs=4)
    model.fit(tx, labels[train])
    return model.predict_proba(vx)[:, list(model.classes_).index(1)]


def main(seed=155):
    frame = pd.read_csv("data/train.csv")
    features, categorical = engineer(frame, "year_both")
    labels = frame.Popularity_Type.eq("High").to_numpy(dtype=int)
    groups = frame.uri.fillna(pd.Series("missing_" + frame.index.astype(str), index=frame.index)).to_numpy()
    years = features.release_year.to_numpy(dtype=float, na_value=np.nan)
    scores = {}
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for train, valid in splitter.split(features, labels, groups):
        assert set(groups[train]).isdisjoint(groups[valid])
        baseline = fit_predict(features, categorical, labels, train, valid)
        scores.setdefault("baseline", []).append(roc_auc_score(labels[valid], baseline))
        for cutoff in (2015, 2020, 2023):
            recent = years >= cutoff
            specialist = baseline.copy()
            for flag in (False, True):
                local_train = train[recent[train] == flag]
                mask = recent[valid] == flag
                if mask.any() and len(np.unique(labels[local_train])) == 2:
                    specialist[mask] = fit_predict(features, categorical, labels, local_train, valid[mask])
            for weight in (0.25, 0.5, 1.0):
                prediction = weight * specialist + (1 - weight) * baseline
                assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
                scores.setdefault(f"cutoff{cutoff}_weight{weight}", []).append(roc_auc_score(labels[valid], prediction))
        print("Completed fold", flush=True)
    with open("era_expert_results.jsonl", "a") as output:
        for name, aucs in sorted(scores.items(), key=lambda item: -np.mean(item[1])):
            result = dict(variant=name, seed=seed, mean_auc=float(np.mean(aucs)), fold_aucs=aucs)
            output.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[155])
    for seed in parser.parse_args().seeds:
        main(seed)
