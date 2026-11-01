"""Train-only rebuild of the reference features: counts and medians come from training rows only.

A query row (validation fold in CV, test at submission) is compared against the training rows and never against
other query rows, so no test-set data informs any feature. Writes ../submission_train_only.csv.
--drop-conflicts removes tracks labelled both High and Low from the training side only (validation/test untouched)
and writes ../submission_train_only_noconflict.csv.
Run with experiments/.venv/bin/python (CatBoost).
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import ROOT, SEEDS, load, oof, report
from features import reference_features
from round3 import blend_models
from round7 import CONFIGS as CAT


def conflicting(uris, labels):
    """Mask of rows whose track appears with both labels among these rows."""
    per_track = pd.Series(labels).groupby(np.asarray(uris)).nunique()
    return pd.Series(np.asarray(uris)).map(per_track).fillna(1).gt(1).to_numpy()


def with_reference(fit_predict, raw, drop_conflicts=False):
    """oof() passes old-feature frames; rebuild both sides here with the training fold as the only reference."""
    def wrapped(_tx, ty, _vx, ti, vi):
        if drop_conflicts:
            keep = ~conflicting(raw.uri.iloc[ti], ty)  # decided from the training fold's own labels only
            ti, ty = ti[keep], ty[keep]
        ref = raw.iloc[ti]
        return fit_predict(reference_features(ref, ref, True), ty, reference_features(ref, raw.iloc[vi], False), ti, vi)
    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-conflicts", action="store_true")
    drop = parser.parse_args().drop_conflicts
    suffix = "_noconflict" if drop else ""
    assert conflicting(["a", "a", "b", "c", "c"], [1, 0, 1, 1, 1]).tolist() == [True, True, False, False, False]

    train, test, y, groups = load()
    raw = train.drop(columns="Popularity_Type")
    blend = lambda tx, ty, vx, *_: blend_models(tx, ty, vx, with_hgb=True)
    preds = {"blend": oof(with_reference(blend, raw, drop), raw, y, groups),
             "cat": oof(with_reference(CAT["r7_cat_d6_lr01_2000"], raw, drop), raw, y, groups)}
    report(f"train_only{suffix}_blend3_hgb", preds["blend"], y, groups)
    report(f"train_only{suffix}_cat", preds["cat"], y, groups)
    mix = {s: 0.7 * rankdata(preds["blend"][s]) + 0.3 * rankdata(preds["cat"][s]) for s in SEEDS}
    report(f"train_only{suffix}_blend+cat0.3", mix, y, groups)

    submission = pd.read_csv(ROOT / "data/sample_submission.csv")
    if not test["ID"].equals(submission["ID"]):
        raise ValueError("Test IDs do not match the sample submission")
    keep = ~conflicting(raw.uri, y) if drop else np.ones(len(y), bool)
    print(f"training rows used: {keep.sum()} of {len(y)}")
    ref, ty = raw[keep], y[keep]
    tx = reference_features(ref, ref, True)
    vx = reference_features(ref, test.drop(columns="ID"), False)
    scores = (0.7 * rankdata(blend(tx, ty, vx)) + 0.3 * rankdata(CAT["r7_cat_d6_lr01_2000"](tx, ty, vx))) / len(vx)
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Predictions must be finite values in [0, 1]")
    submission["Popularity_Type"] = scores
    path = ROOT / f"submission_train_only{suffix}.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} with {len(submission)} rows")


if __name__ == "__main__":
    main()
