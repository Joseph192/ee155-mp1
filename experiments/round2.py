"""Round 2: tune the best families on the 'all' features, add model diversity, blend OOF predictions.

Also scores on single-occurrence tracks only, since test has ~1% duplicated tracks versus ~6% in train.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import SEEDS, build_forest_preprocessor, folds, load, oof, report
from features import build
from round1 import forest


def single_auc(preds, y, groups, singles):
    return np.mean([np.mean([roc_auc_score(y[v][singles[v]], p[v][singles[v]]) for _, v in folds(y, groups, s)]) for s, p in preds.items()])


def scaled(model):
    def fit_predict(train_x, train_y, valid_x, *_):
        pipe = make_pipeline(build_forest_preprocessor(train_x.columns), StandardScaler(), *model())
        return pipe.fit(train_x, train_y).predict_proba(valid_x)[:, 1]
    return fit_predict


def main():
    train, test, y, groups = load()
    features, _ = build(train, test, "all")
    singles = (features.uri_count == 1).to_numpy()
    results = {}

    def run(name, fit_predict):
        preds = oof(fit_predict, features, y, groups)
        results[name] = preds
        report(name, preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)

    for leaf in (1, 3):
        for mf in (0.3, 0.5, 0.8):
            run(f"r2_et_leaf{leaf}_mf{mf}", forest(ExtraTreesClassifier, min_samples_leaf=leaf, max_features=mf, criterion="entropy", class_weight="balanced"))
    for mf in ("sqrt", 0.5):
        run(f"r2_rf_mf{mf}", forest(RandomForestClassifier, min_samples_leaf=3, max_features=mf, criterion="entropy", class_weight="balanced"))
    run("r2_logit_spline", scaled(lambda: [SplineTransformer(n_knots=6), LogisticRegression(C=0.3, max_iter=3000)]))
    run("r2_knn50", scaled(lambda: [KNeighborsClassifier(50, weights="distance")]))

    best = sorted(results, key=lambda k: -np.mean([roc_auc_score(y, p) for p in results[k].values()]))
    print("ranked:", best[:5])
    for combo in ([best[0], "r2_rf_mfsqrt"], [best[0], "r2_rf_mfsqrt", "r2_logit_spline"], [best[0], "r2_rf_mfsqrt", "r2_knn50"]):
        blend = {s: np.mean([rankdata(results[k][s]) for k in combo], axis=0) for s in SEEDS}
        report("r2_blend_" + "+".join(k.replace("r2_", "") for k in combo), blend, y, groups)
        print(f"{'':40s} singles_only={single_auc(blend, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
