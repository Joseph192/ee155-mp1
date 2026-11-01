"""Round 7 (idea 7): CatBoost tuning and a pairwise ranking loss. Run with .venv/bin/python."""

import numpy as np
from catboost import CatBoost, CatBoostClassifier, Pool

from common import load, oof, report
from features import build
from round2 import single_auc


def cat(**kw):
    def fit_predict(tx, ty, vx, *_):
        params = dict(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=5, auto_class_weights="Balanced",
                      random_seed=155, verbose=0, thread_count=5)
        params.update(kw)
        return CatBoostClassifier(**params).fit(tx, ty).predict_proba(vx)[:, 1]
    return fit_predict


def pairlogit(iterations=800, depth=6, group_size=64):
    def fit_predict(tx, ty, vx, *_):
        # Ranking loss needs query groups: random groups of ~64 rows, so pairs are sampled across the whole fold.
        order = np.random.default_rng(155).permutation(len(tx))
        group = np.arange(len(tx)) // group_size
        pool = Pool(tx.iloc[order], ty[order], group_id=group)
        model = CatBoost(dict(loss_function="PairLogit", iterations=iterations, learning_rate=0.03, depth=depth,
                              l2_leaf_reg=5, random_seed=155, verbose=0, thread_count=5))
        return model.fit(pool).predict(vx)
    return fit_predict


CONFIGS = {
    "r7_cat_d4": cat(depth=4, iterations=1200),
    "r7_cat_d8": cat(depth=8, iterations=600),
    "r7_cat_d6_lr01_2000": cat(learning_rate=0.01, iterations=2000),
    "r7_cat_d6_l2_20": cat(l2_leaf_reg=20, iterations=1200),
    "r7_cat_d6_unweighted": cat(auto_class_weights=None),
    "r7_cat_d6_rsm05": cat(rsm=0.5, iterations=1200),
    "r7_pairlogit_d6": pairlogit(),
    "r7_pairlogit_d4": pairlogit(iterations=1200, depth=4),
}


def main():
    train, test, y, groups = load()
    features, _ = build(train, test, "all")
    singles = (features.uri_count == 1).to_numpy()
    for name, fit_predict in CONFIGS.items():
        preds = oof(fit_predict, features, y, groups)
        report(name, preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
