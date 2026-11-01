"""Round-5 best: 70% round-3 blend rank + 30% CatBoost rank. Writes ../submission_blend_cat.csv.

Run with experiments/.venv/bin/python (needs CatBoost).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import ROOT, load
from features import build
from round3 import blend_models
from round5 import catboost


def main():
    train, test, y, _ = load()
    submission = pd.read_csv(ROOT / "data/sample_submission.csv")
    if not test["ID"].equals(submission["ID"]):
        raise ValueError("Test IDs do not match the sample submission")
    train_x, test_x = build(train, test, "all")
    # Same mixing as r5_r3blend+cat0.3: re-rank each component, then weight.
    blend = rankdata(blend_models(train_x, y, test_x, with_hgb=True))
    cat = rankdata(catboost(train_x, y, test_x))
    scores = (0.7 * blend + 0.3 * cat) / len(test_x)
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Predictions must be finite values in [0, 1]")
    submission["Popularity_Type"] = scores
    path = ROOT / "submission_blend_cat.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} with {len(submission)} rows")


if __name__ == "__main__":
    main()
