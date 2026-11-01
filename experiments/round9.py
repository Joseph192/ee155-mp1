"""Round 9 (ideas 4-5): combine saved OOF predictions.

Greedy ensemble selection (Caruana 2004) picks weights on layout 155 and is checked on 2026/42.
Stacking fits a logistic meta-learner on two folds' OOF and scores the third, per layout.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from common import OUT, SEEDS, fold_auc, folds, load
from features import build
from round2 import single_auc

POOL = ["r3_blend_3seeds_hgb", "r2_et_leaf3_mf0.3", "r2_rf_mfsqrt", "r1_all_hgb", "r5_catboost",
        "r7_cat_d6_lr01_2000", "r7_cat_d8", "r7_pairlogit_d6", "r6_plr_e20", "r6_plr_e40_drop0.5_wd1e-2",
        "r8_residual_net_1.0", "r6_skmlp"]


def main():
    train, test, y, groups = load()
    singles = (build(train, test, "all")[0].uri_count == 1).to_numpy()
    ranks = {n: {s: rankdata(np.load(OUT / f"{n}.npz")[str(s)]) / len(y) for s in SEEDS} for n in POOL}
    score = lambda pred, s: fold_auc(pred, y, groups, s)

    # Greedy selection with replacement on layout 155 only.
    chosen, current = [], np.zeros(len(y))
    for _ in range(15):
        best = max(POOL, key=lambda n: score(current * len(chosen) / (len(chosen) + 1) + ranks[n][155] / (len(chosen) + 1), 155))
        chosen.append(best)
        current = np.mean([ranks[n][155] for n in chosen], axis=0)
    weights = {n: chosen.count(n) / len(chosen) for n in set(chosen)}
    print("greedy weights:", {k: round(v, 2) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])})
    blend = {s: sum(w * ranks[n][s] for n, w in weights.items()) for s in SEEDS}
    print("greedy  " + " ".join(f"{s}={score(blend[s], s):.4f}" for s in SEEDS) + f"  held-out mean={np.mean([score(blend[s], s) for s in SEEDS[1:]]):.4f}  singles={single_auc(blend, y, groups, singles):.4f}")

    for name, members in (("stack_all", POOL), ("stack_trees_cat_nn", ["r3_blend_3seeds_hgb", "r7_cat_d6_lr01_2000", "r6_plr_e20", "r8_residual_net_1.0"])):
        stacked = {}
        for s in SEEDS:
            x = np.column_stack([ranks[n][s] for n in members])
            pred = np.empty(len(y))
            for tr, va in folds(y, groups, s):
                pred[va] = LogisticRegression(C=1.0, max_iter=1000).fit(x[tr], y[tr]).predict_proba(x[va])[:, 1]
            stacked[s] = pred
        print(f"{name:7s} " + " ".join(f"{s}={score(stacked[s], s):.4f}" for s in SEEDS) + f"  mean={np.mean([score(stacked[s], s) for s in SEEDS]):.4f}  singles={single_auc(stacked, y, groups, singles):.4f}")

    ref = {s: 0.7 * ranks["r3_blend_3seeds_hgb"][s] + 0.3 * ranks["r5_catboost"][s] for s in SEEDS}
    print("current best (r3 + 30% cat) " + " ".join(f"{s}={score(ref[s], s):.4f}" for s in SEEDS) + f"  held-out mean={np.mean([score(ref[s], s) for s in SEEDS[1:]]):.4f}")


if __name__ == "__main__":
    main()
