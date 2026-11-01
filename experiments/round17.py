"""Round 17: logistic regression (with similarity-weighted mates) added to the best ensemble (round 14: blend + 30% CatBoost).
Run with .venv/bin/python.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import OUT, SEEDS, build_forest_preprocessor, fold_auc, load, oof, report
from features import build
from round2 import single_auc
from round4 import embed
from round14 import add_sim_mates


def logistic(*steps, C=1.0):
    def fit_predict(tx, ty, vx, *_):
        pipe = make_pipeline(build_forest_preprocessor(tx.columns), StandardScaler(), *steps, LogisticRegression(C=C, max_iter=5000))
        return pipe.fit(tx, ty).predict_proba(vx)[:, 1]
    return fit_predict


def main():
    train, test, y, groups = load()
    tx, vx = build(train, test, "all")
    singles = (tx.uri_count == 1).to_numpy()
    z = embed(pd.concat([tx, vx], ignore_index=True))[:len(train)]
    dates = train.track_album_release_date.fillna("?").to_numpy()
    best = {s: 0.7 * rankdata(np.load(OUT / "r14_sim_blend3_hgb.npz")[str(s)]) / len(y)
               + 0.3 * rankdata(np.load(OUT / "r14_sim_cat.npz")[str(s)]) / len(y) for s in SEEDS}

    def show(name, preds):
        aucs = [fold_auc(preds[s], y, groups, s) for s in SEEDS]
        print(f"{name:36s} mean={np.mean(aucs):.4f} " + " ".join(f"{s}={a:.4f}" for s, a in zip(SEEDS, aucs))
              + f"  singles={single_auc(preds, y, groups, singles):.4f}", flush=True)

    show("current best (blend + 30% cat)", best)
    for name, model in (("lr_C0.1", logistic(C=0.1)), ("lr_C1", logistic(C=1.0)),
                        ("lr_spline_C0.3", logistic(SplineTransformer(n_knots=6), C=0.3))):
        lr = oof(add_sim_mates(model, dates, groups, z), tx, y, groups)
        report(f"r17_sim_{name}", lr, y, groups)
        print("  rank corr with best:", round(np.corrcoef(rankdata(lr[155]), rankdata(best[155]))[0, 1], 3))
        for w in (0.05, 0.1, 0.15, 0.2, 0.3):
            show(f"  best + {w:.0%} {name}", {s: (1 - w) * best[s] + w * rankdata(lr[s]) / len(y) for s in SEEDS})


if __name__ == "__main__":
    main()
