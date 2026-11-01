"""Test audio clusters, era-relative sound, and neighbor/forest blends."""

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import NUMERIC
from feature_sweep import engineer, preprocessor


def main():
    frame = pd.read_csv("data/train.csv")
    features, categorical = engineer(frame, "baseline")
    audio = [c for c in NUMERIC if c != "release_year"]
    y = frame.Popularity_Type.eq("High").to_numpy(dtype=int)
    groups = frame.uri.fillna(pd.Series("missing_" + frame.index.astype(str), index=frame.index)).to_numpy()
    scores = {}

    def record(name, truth, prediction):
        scores.setdefault(name, []).append(roc_auc_score(truth, prediction))

    for train, valid in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=155).split(features, y, groups):
        assert set(groups[train]).isdisjoint(groups[valid])
        transform = preprocessor(features, categorical)
        tx = transform.fit_transform(features.iloc[train])
        vx = transform.transform(features.iloc[valid])
        audio_transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        ta = audio_transform.fit_transform(features.iloc[train][audio])
        va = audio_transform.transform(features.iloc[valid][audio])
        baseline_prediction = None
        for variant in ("baseline", "era_relative", "clusters_8", "clusters_32"):
            train_x, valid_x = tx, vx
            if variant == "era_relative":
                # Compare each track to training songs from its five-year era.
                era = features.release_year // 5
                means = features.iloc[train][audio].groupby(era.iloc[train]).mean()
                fallback = features.iloc[train][audio].mean()
                train_mean = means.reindex(era.iloc[train]).reset_index(drop=True).fillna(fallback)
                valid_mean = means.reindex(era.iloc[valid]).reset_index(drop=True).fillna(fallback)
                train_delta = features.iloc[train][audio].reset_index(drop=True) - train_mean
                valid_delta = features.iloc[valid][audio].reset_index(drop=True) - valid_mean
                impute = SimpleImputer(strategy="median")
                train_x = np.column_stack([tx, impute.fit_transform(train_delta)])
                valid_x = np.column_stack([vx, impute.transform(valid_delta)])
            elif variant.startswith("clusters"):
                clusters = KMeans(n_clusters=int(variant.split("_")[1]), n_init=10, random_state=155)
                train_x = np.column_stack([tx, clusters.fit_transform(ta)])
                valid_x = np.column_stack([vx, clusters.transform(va)])
            assert np.isfinite(train_x).all() and np.isfinite(valid_x).all()
            model = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, criterion="entropy", class_weight="balanced", random_state=155, n_jobs=4)
            model.fit(train_x, y[train])
            prediction = model.predict_proba(valid_x)[:, 1]
            record(variant, y[valid], prediction)
            if variant == "baseline":
                baseline_prediction = prediction
        year_transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        train_year = year_transform.fit_transform(features.iloc[train][["release_fractional_year"]])
        valid_year = year_transform.transform(features.iloc[valid][["release_fractional_year"]])
        for year_weight in (0, 1, 3, 10):
            for neighbors in (5, 15, 40, 100):
                model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
                model.fit(np.column_stack([ta, year_weight * train_year]), y[train])
                prediction = model.predict_proba(np.column_stack([va, year_weight * valid_year]))[:, 1]
                for weight in (0.25, 0.5, 1.0):
                    record(f"knn_year{year_weight}_k{neighbors}_weight{weight}", y[valid], weight * prediction + (1 - weight) * baseline_prediction)
        print("Completed fold", flush=True)
    with open("similarity_results.jsonl", "a") as output:
        for name, aucs in sorted(scores.items(), key=lambda item: np.mean(item[1]), reverse=True):
            result = {"variant": name, "seed": 155, "mean_auc": float(np.mean(aucs)), "fold_aucs": aucs}
            output.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
