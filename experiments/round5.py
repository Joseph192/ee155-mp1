"""Round 5: cross-fitted neighbour-label features, CatBoost, and CatBoost in the round-3 blend.

Needs CatBoost: run with experiments/.venv/bin/python (venv shares the system sklearn/pandas).
"""

import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

from common import SEEDS, OUT, load, oof, report
from features import build
from round2 import single_auc
from round3 import blend_models
from round4 import embed, neighbour_labels


def crossfit_neighbours(z, y, g, ti, vi):
    """Training rows get rates from an inner grouped split, so no row's features come from models that saw it."""
    train_nn = pd.DataFrame(index=range(len(ti)), columns=["nn30_rate", "nn100_rate"], dtype=float)
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=911)
    for fit, held in inner.split(ti, y[ti], g[ti]):
        rates = neighbour_labels(z[ti][fit], y[ti][fit], g[ti][fit], z[ti][held], g[ti][held], ks=(30, 100))
        train_nn.iloc[held] = rates[["nn30_rate", "nn100_rate"]].to_numpy()
    valid_nn = neighbour_labels(z[ti], y[ti], g[ti], z[vi], g[vi], ks=(30, 100))[["nn30_rate", "nn100_rate"]]
    return train_nn, valid_nn


def catboost(tx, ty, vx, *_, seed=155):
    model = CatBoostClassifier(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                               auto_class_weights="Balanced", random_seed=seed, verbose=0, thread_count=10)
    return model.fit(tx, ty).predict_proba(vx)[:, 1]


def main():
    train, test, y, groups = load()
    features, test_features = build(train, test, "all")
    z = embed(pd.concat([features, test_features], ignore_index=True))[:len(train)]
    singles = (features.uri_count == 1).to_numpy()
    variants = sys.argv[1:] or ["nn_crossfit", "catboost", "blend_cat"]

    def run(name, fit_predict):
        preds = oof(fit_predict, features, y, groups)
        report(name, preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)
        return preds

    if "nn_crossfit" in variants:
        def nn_fit(tx, ty, vx, ti, vi):
            tr_nn, va_nn = crossfit_neighbours(z, y, groups, ti, vi)
            return blend_models(pd.concat([tx.reset_index(drop=True), tr_nn], axis=1), ty,
                                pd.concat([vx.reset_index(drop=True), va_nn.reset_index(drop=True)], axis=1), model_seeds=(155,))
        run("r5_nn_crossfit", nn_fit)
    if "catboost" in variants:
        run("r5_catboost", catboost)
    if "blend_cat" in variants:
        r3 = np.load(OUT / "r3_blend_3seeds_hgb.npz")
        cat = np.load(OUT / "r5_catboost.npz")
        for w in (0.15, 0.3, 0.5):
            blend = {s: (1 - w) * rankdata(r3[str(s)]) + w * rankdata(cat[str(s)]) for s in SEEDS}
            report(f"r5_r3blend+cat{w}", blend, y, groups)
            print(f"{'':40s} singles_only={single_auc(blend, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
