"""Round-3 blend + 30% CatBoost with album-mate features. Writes ../submission_mates.csv.
--sim uses round-14 similarity-weighted mates (CV 0.8551) and writes ../submission_mates_sim.csv.
--drop-conflicts trains without tracks labelled both High and Low (CV 0.8549); adds _noconflict to the file name.
--dedup (with --drop-conflicts) also keeps one row per repeated track; adds _dedup.

Features count over train+test; mate rates use train labels only (cross-fitted for train rows).
Run with experiments/.venv/bin/python (CatBoost).
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import ROOT, load
from features import build
from round3 import blend_models
from round7 import CONFIGS as CAT
from round12 import add_mates
from round14 import add_sim_mates
from round4 import embed
from train_only import conflicting


def main():
    train, test, y, groups = load()
    submission = pd.read_csv(ROOT / "data/sample_submission.csv")
    if not test["ID"].equals(submission["ID"]):
        raise ValueError("Test IDs do not match the sample submission")
    tx, vx = build(train, test, "all")
    # Reuse the CV wrapper: train rows = all of train, query rows = test, appended after train.
    feats = pd.concat([tx, vx], ignore_index=True)
    dates = pd.concat([train.track_album_release_date, test.track_album_release_date], ignore_index=True).fillna("?").to_numpy()
    all_groups = np.concatenate([groups, test.uri.fillna("test_missing").to_numpy()])
    sim, drop, dedup = "--sim" in sys.argv, "--drop-conflicts" in sys.argv, "--dedup" in sys.argv
    keep = ~conflicting(groups, y) if drop else np.ones(len(y), bool)  # counts still include them, as in CV
    if dedup:
        keep &= ~pd.Series(groups).duplicated().to_numpy()
    ti, vi = np.arange(len(train))[keep], np.arange(len(train), len(feats))
    y = y[keep]
    print(f"training rows used: {len(ti)} of {len(train)}")
    wrap = (lambda m: add_sim_mates(m, dates, all_groups, embed(feats))) if sim else (lambda m: add_mates(m, dates, all_groups))
    run = lambda model: wrap(model)(feats.iloc[ti], y, feats.iloc[vi], ti, vi)
    blend = run(lambda a, b, c, *_: blend_models(a, b, c, with_hgb=True))
    cat = run(CAT["r7_cat_d6_lr01_2000"])
    scores = (0.7 * rankdata(blend) + 0.3 * rankdata(cat)) / len(vi)
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Predictions must be finite values in [0, 1]")
    submission["Popularity_Type"] = scores
    path = ROOT / f"submission_mates{'_sim' if sim else ''}{'_noconflict' if drop else ''}{'_dedup' if dedup else ''}.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} with {len(submission)} rows")


if __name__ == "__main__":
    main()
