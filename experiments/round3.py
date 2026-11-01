"""Round 3: seed-averaged ET+RF(+HGB) blend, and test-set pseudo-labelling."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from common import OUT, SEEDS, build_forest_preprocessor, load, oof, report
from features import build
from round2 import single_auc


def blend_models(train_x, train_y, valid_x, model_seeds=(155, 1, 2), with_hgb=False, weight=None):
    prep = build_forest_preprocessor(train_x.columns)
    tx, vx = prep.fit_transform(train_x), prep.transform(valid_x)
    ranks = []
    for s in model_seeds:
        for model in (
            ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, criterion="entropy", class_weight="balanced", random_state=s, n_jobs=10),
            RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt", criterion="entropy", class_weight="balanced", random_state=s, n_jobs=10),
        ):
            ranks.append(rankdata(model.fit(tx, train_y, sample_weight=weight).predict_proba(vx)[:, 1]) / len(vx))
    if with_hgb:
        hgb = HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.03, max_iter=300, l2_regularization=1.0, early_stopping=False, random_state=155)
        ranks.append(rankdata(hgb.fit(tx, train_y, sample_weight=weight).predict_proba(vx)[:, 1]) / len(vx))
    return np.mean(ranks, axis=0)


def main():
    train, test, y, groups = load()
    features, test_features = build(train, test, "all")
    singles = (features.uri_count == 1).to_numpy()

    def run(name, fit_predict):
        preds = oof(fit_predict, features, y, groups)
        report(name, preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)

    run("r3_blend_3seeds", lambda tx, ty, vx, *_: blend_models(tx, ty, vx))
    run("r3_blend_3seeds_hgb", lambda tx, ty, vx, *_: blend_models(tx, ty, vx, with_hgb=True))

    def pseudo(tx, ty, vx, *_):
        # Label the confident ends of the unlabeled pool (test + this fold's validation features), weight them down.
        pool = pd.concat([test_features, vx], ignore_index=True)
        score = blend_models(tx, ty, pool, model_seeds=(155,))
        lo, hi = np.quantile(score, [0.25, 0.85])
        keep = (score <= lo) | (score >= hi)
        x = pd.concat([tx, pool[keep]], ignore_index=True)
        labels = np.concatenate([ty, (score[keep] >= hi).astype(int)])
        weight = np.concatenate([np.ones(len(ty)), np.full(keep.sum(), 0.5)])
        return blend_models(x, labels, vx, weight=weight)

    run("r3_pseudo", pseudo)

    # Label-noise check: how often does a repeated track disagree with itself?
    rep = pd.DataFrame({"g": groups, "y": y})[~singles].groupby("g").y.agg(["min", "max"])
    print(f"repeated tracks: {len(rep)}, with conflicting labels: {(rep['min'] != rep['max']).sum()}")


if __name__ == "__main__":
    main()
