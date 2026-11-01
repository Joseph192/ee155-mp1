"""Round 18: soft voting (probability averaging) vs the best model's rank averaging, same members and features.
Members: ET x3 seeds, RF x3 seeds, HGB, CatBoost, all with similarity-weighted mates. Run with .venv/bin/python.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from common import OUT, SEEDS, build_forest_preprocessor, fold_auc, load, oof, report
from features import build
from round2 import single_auc
from round4 import embed
from round7 import CONFIGS as CAT
from round14 import add_sim_mates


def member(kind):
    """Probabilities from one member family; forests averaged over the same 3 seeds as blend_models."""
    def fit_predict(tx, ty, vx, *_):
        prep = build_forest_preprocessor(tx.columns)
        a, b = prep.fit_transform(tx), prep.transform(vx)
        if kind == "hgb":
            return HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.03, max_iter=300,
                                                  l2_regularization=1.0, early_stopping=False, random_state=155).fit(a, ty).predict_proba(b)[:, 1]
        cls, kw = {"et": (ExtraTreesClassifier, dict(max_features=0.3)), "rf": (RandomForestClassifier, dict(max_features="sqrt"))}[kind]
        return np.mean([cls(500, min_samples_leaf=3, criterion="entropy", class_weight="balanced", random_state=s, n_jobs=10, **kw)
                        .fit(a, ty).predict_proba(b)[:, 1] for s in (155, 1, 2)], axis=0)
    return fit_predict


def main():
    train, test, y, groups = load()
    tx, vx = build(train, test, "all")
    singles = (tx.uri_count == 1).to_numpy()
    z = embed(pd.concat([tx, vx], ignore_index=True))[:len(train)]
    dates = train.track_album_release_date.fillna("?").to_numpy()
    p = {k: oof(add_sim_mates(member(k), dates, groups, z), tx, y, groups) for k in ("et", "rf", "hgb")}
    p["cat"] = {s: v for s, v in np.load(OUT / "r14_sim_cat.npz").items()}  # CatBoost OOF probabilities from round 14
    p["cat"] = {s: p["cat"][str(s)] for s in SEEDS}
    for k in p:
        report(f"r18_sim_{k}_prob", p[k], y, groups)
    r = lambda a: rankdata(a) / len(a)
    best = {s: 0.7 * r(np.load(OUT / "r14_sim_blend3_hgb.npz")[str(s)]) + 0.3 * r(p["cat"][s]) for s in SEEDS}

    def show(name, preds):
        aucs = [fold_auc(preds[s], y, groups, s) for s in SEEDS]
        print(f"{name:44s} mean={np.mean(aucs):.4f} " + " ".join(f"{s}={a:.4f}" for s, a in zip(SEEDS, aucs))
              + f"  singles={single_auc(preds, y, groups, singles):.4f}", flush=True)

    print("mean predicted probability per member:", {k: round(float(np.mean(p[k][155])), 3) for k in p})
    show("best: rank average (blend + 30% cat)", best)
    soft = {
        "soft: equal 4 families": {s: np.mean([p[k][s] for k in p], axis=0) for s in SEEDS},
        "soft: ET+RF+HGB (3:3:1)": {s: (3 * p["et"][s] + 3 * p["rf"][s] + p["hgb"][s]) / 7 for s in SEEDS},
        "soft: same weights as best": {s: 0.7 * (3 * p["et"][s] + 3 * p["rf"][s] + p["hgb"][s]) / 7 + 0.3 * p["cat"][s] for s in SEEDS},
        "soft: ET+RF+CatBoost equal": {s: (p["et"][s] + p["rf"][s] + p["cat"][s]) / 3 for s in SEEDS},
    }
    for name, preds in soft.items():
        show(name, preds)
    top = soft["soft: same weights as best"]
    for w in (0.2, 0.5):
        show(f"best + {w:.0%} soft-voting ensemble", {s: (1 - w) * best[s] + w * r(top[s]) for s in SEEDS})


if __name__ == "__main__":
    main()
