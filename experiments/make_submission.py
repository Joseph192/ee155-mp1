"""Train the round-3 blend on all training data and write experiments/submission.csv."""

import numpy as np
import pandas as pd

from common import ROOT, load
from features import build
from round3 import blend_models


def main():
    train, test, y, _ = load()
    submission = pd.read_csv(ROOT / "data/sample_submission.csv")
    if not test["ID"].equals(submission["ID"]):
        raise ValueError("Test IDs do not match the sample submission")
    train_x, test_x = build(train, test, "all")
    scores = blend_models(train_x, y, test_x, with_hgb=True)
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Predictions must be finite values in [0, 1]")
    submission["Popularity_Type"] = scores  # rank-averaged scores; AUC only needs the ordering
    path = ROOT / "experiments/submission.csv"
    submission.to_csv(path, index=False)
    print(f"Wrote {path} with {len(submission)} rows")


if __name__ == "__main__":
    main()
