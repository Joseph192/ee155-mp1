"""Compare engineered features using track-grouped, fold-fitted validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from improved_tree import make_features
from random_forest_sweep import CATEGORICAL


VARIANTS = (
    "baseline", "year_onehot", "year_both", "year_only_categorical",
    "calendar", "date_precision", "audio_flags", "audio_interactions",
    "musical", "all", "duration_precision", "audio_precision", "precision_year",
)


def engineer(frame, variant):
    features = make_features(frame)
    categorical = list(CATEGORICAL)
    if variant in ("year_onehot", "year_both", "year_only_categorical", "all", "precision_year"):
        features["year_category"] = features["release_year"]
        categorical.append("year_category")
        if variant in ("year_onehot", "year_only_categorical"):
            features = features.drop(columns="release_year")
        if variant == "year_only_categorical":
            features = features.drop(columns="release_fractional_year")
    dates = frame["track_album_release_date"].astype("string")
    month = pd.to_numeric(dates.str.extract(r"^\d{4}-(\d{2})", expand=False), errors="coerce")
    if variant in ("calendar", "all"):
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)
        features["release_quarter"] = (month - 1) // 3
        categorical.append("release_quarter")
    if variant in ("date_precision", "all"):
        features["date_precision"] = dates.str.len().astype(float)
        features["january_first"] = dates.str.endswith("-01-01").fillna(False).astype(int)
        features["month_first"] = dates.str.endswith("-01").fillna(False).astype(int)
    if variant in ("audio_flags", "all"):
        features["instrumental"] = (frame.instrumentalness > 0.5).astype(int)
        features["spoken"] = (frame.speechiness > 0.33).astype(int)
        features["live"] = (frame.liveness > 0.8).astype(int)
        features["audio_missing"] = frame.energy.isna().astype(int)
    if variant in ("audio_interactions", "all"):
        features["energy_dance"] = frame.energy * frame.danceability
        features["energy_valence"] = frame.energy * frame.valence
        features["energy_acoustic"] = frame.energy * frame.acousticness
        features["valence_minus_energy"] = frame.valence - frame.energy
        features["loudness_per_energy"] = frame.loudness / (frame.energy + 0.05)
        features["estimated_beats"] = frame.duration_ms * frame.tempo / 60000
        features["vocal_dance"] = (1 - frame.instrumentalness) * frame.danceability
    if variant in ("musical", "all"):
        features["key_mode"] = frame.key + 12 * frame["mode"]
        categorical.append("key_mode")
        features["key_sin"] = np.sin(2 * np.pi * frame.key / 12)
        features["key_cos"] = np.cos(2 * np.pi * frame.key / 12)
        features["tempo_folded"] = frame.tempo / (2 ** np.floor(np.log2(frame.tempo.clip(lower=1) / 90)))
    if variant in ("duration_precision", "audio_precision", "precision_year"):
        features["duration_remainder"] = frame.duration_ms.mod(1000)
        for modulus in (10, 100, 1000):
            features[f"duration_multiple_{modulus}"] = frame.duration_ms.mod(modulus).eq(0).astype(int)
    if variant in ("audio_precision", "precision_year"):
        features["tempo_integer_distance"] = (frame.tempo - frame.tempo.round()).abs()
        for column in ("tempo", "loudness", "instrumentalness", "speechiness", "liveness"):
            features[column + "_two_decimals"] = (frame[column] * 100).round(6).mod(1).eq(0).astype(int)
        features["instrumentalness_zero"] = frame.instrumentalness.eq(0).astype(int)
    return features, categorical


def preprocessor(features, categorical):
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), [c for c in features if c not in categorical]),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value=-1),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), categorical),
    ])


def evaluate(frame, variant, seed, trees, max_features, drop=(), leaf=3):
    features, categorical = engineer(frame, variant)
    features = features.drop(columns=list(drop))
    categorical = [c for c in categorical if c in features]
    labels = frame.Popularity_Type.eq("High").to_numpy(dtype=int)
    groups = frame.uri.fillna(pd.Series("missing_" + frame.index.astype(str), index=frame.index))
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs = []
    for train, valid in splitter.split(features, labels, groups):
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[valid])
        transform = preprocessor(features, categorical)
        train_x = transform.fit_transform(features.iloc[train])
        valid_x = transform.transform(features.iloc[valid])
        assert np.isfinite(train_x).all() and np.isfinite(valid_x).all()
        model = RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=leaf, max_features=max_features,
            criterion="entropy", class_weight="balanced", random_state=155, n_jobs=4,
        )
        model.fit(train_x, labels[train])
        prediction = model.predict_proba(valid_x)[:, 1]
        aucs.append(roc_auc_score(labels[valid], prediction))
    return {"variant": variant, "seed": seed, "trees": trees,
            "max_features": max_features, "drop": list(drop), "leaf": leaf,
            "mean_auc": float(np.mean(aucs)), "fold_aucs": aucs}


def self_check():
    frame = pd.read_csv("data/train.csv").head(10)
    frame.loc[0, "track_album_release_date"] = "1999"
    frame.loc[1, "track_album_release_date"] = "2000-02-29"
    frame.loc[2, "track_album_release_date"] = None
    for variant in VARIANTS:
        features, categorical = engineer(frame, variant)
        transform = preprocessor(features, categorical)
        assert np.isfinite(transform.fit_transform(features)).all()
        unseen = frame.iloc[[0]].copy()
        unseen["track_album_release_date"] = "2099-12-31"
        assert np.isfinite(transform.transform(engineer(unseen, variant)[0])).all()
        if variant == "year_only_categorical":
            assert "release_year" not in features and "release_fractional_year" not in features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[155])
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--leaf", type=int, default=3)
    parser.add_argument("--output", default="feature_results.jsonl")
    args = parser.parse_args()
    self_check()
    frame = pd.read_csv("data/train.csv")
    max_features = "sqrt" if args.max_features == "sqrt" else float(args.max_features)
    for variant in args.variants:
        for seed in args.seeds:
            result = evaluate(frame, variant, seed, args.trees, max_features, args.drop, args.leaf)
            with Path(args.output).open("a") as output:
                output.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
