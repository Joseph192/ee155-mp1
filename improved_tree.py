"""Validate a date-enriched boosted tree on several track-grouped splits."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from data_preprocessing import NUMERIC


def make_features(frame):
    dates = frame["track_album_release_date"].astype("string")
    features = frame.copy()
    features["release_year"] = pd.to_numeric(dates.str[:4], errors="coerce")
    month = pd.to_numeric(dates.str.extract(r"^\d{4}-(\d{2})", expand=False), errors="coerce")
    features["release_day"] = pd.to_numeric(dates.str.extract(r"^\d{4}-\d{2}-(\d{2})$", expand=False), errors="coerce").fillna(0)
    features["release_fractional_year"] = features["release_year"] + (month.fillna(1) - 1) / 12 + (features["release_day"].clip(lower=1) - 1) / 365
    features["release_weekday"] = pd.to_datetime(dates.where(dates.str.len() == 10), errors="coerce").dt.dayofweek
    return features[[c for c in NUMERIC if c != "release_year"] + [
        "release_year", "release_day", "release_fractional_year",
        "release_weekday", "time_signature", "mode",
    ]]


def main():
    frame = pd.read_csv("data/train.csv")
    features = make_features(frame)
    labels = frame["Popularity_Type"].eq("High").to_numpy(dtype=int)
    groups = frame["uri"].fillna(pd.Series("missing_track_" + frame.index.astype(str), index=frame.index)).to_numpy()
    for seed in (155, 2026, 42, 777):
        scores = []
        folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(features, labels, groups)
        for train_idx, valid_idx in folds:
            model = HistGradientBoostingClassifier(
                max_leaf_nodes=31, min_samples_leaf=5, learning_rate=0.03,
                max_iter=300, early_stopping=False, random_state=155,
                categorical_features=["release_weekday", "time_signature", "mode"],
            )
            model.fit(features.iloc[train_idx], labels[train_idx])
            scores.append(roc_auc_score(labels[valid_idx], model.predict_proba(features.iloc[valid_idx])[:, 1]))
        print(f"split_seed={seed} AUC={np.mean(scores):.4f} folds={[round(s, 4) for s in scores]}")


if __name__ == "__main__":
    main()
