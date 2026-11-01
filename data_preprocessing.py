"""Fold-safe preprocessing for the song popularity classifier."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC = [
    "speechiness", "danceability", "duration_ms", "energy",
    "instrumentalness", "valence", "tempo", "loudness",
    "acousticness", "liveness", "release_year",
]
CATEGORICAL = ["key", "time_signature", "release_month"]


def add_release_date_parts(frame):
    frame = frame.copy()
    dates = frame["track_album_release_date"].astype("string")
    frame["release_year"] = pd.to_numeric(dates.str[:4], errors="coerce")
    frame["release_month"] = dates.str.extract(r"^\d{4}-(0[1-9]|1[0-2])", expand=False).fillna("unknown")
    return frame


def build_preprocessor():
    """Return an unfitted transformer; fit it only on each training fold."""
    return Pipeline([
        ("date_parts", FunctionTransformer(add_release_date_parts)),
        ("features", ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
            ("mode", SimpleImputer(strategy="most_frequent"), ["mode"]),
        ])),
    ])
